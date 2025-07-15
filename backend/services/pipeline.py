import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import json
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import orjson
# Using OpenAI through wrapper to avoid proxy issues
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from database.manager import db_manager
from utils.ncbi_utils import extract_gene_mentions, map_to_gene_id
from utils.bean_data import function_schema, answer_bean_query

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# --- Load Models ---
bge_model = SentenceTransformer(settings.bge_model)
tokenizer = AutoTokenizer.from_pretrained(settings.pubmedbert_model)
pub_model = AutoModel.from_pretrained(settings.pubmedbert_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pub_model.to(device).eval()

# --- RAG Context Loading ---
def load_rag_text_jsonl(path: Path) -> Dict[str, str]:
    rag_lookup = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = orjson.loads(line)
                doi = record.get("doi") or record.get("source", "").replace(".pdf", "")
                rag = record.get("summary", "")
                if doi and rag:
                    rag_lookup[doi.strip()] = rag.strip()
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                record = orjson.loads(line)
                doi = record.get("doi") or record.get("source", "").replace(".pdf", "")
                rag = record.get("summary", "")
                if doi and rag:
                    rag_lookup[doi.strip()] = rag.strip()
    return rag_lookup

RAG_LOOKUP = load_rag_text_jsonl(Path(settings.rag_file))

def get_rag_context_from_dois(dois: List[str]) -> Tuple[str, List[str]]:
    context_blocks = []
    confirmed_dois = []

    for i, doi in enumerate(dois, 1):
        if doi in RAG_LOOKUP:
            summary = RAG_LOOKUP[doi]
            context_blocks.append(f"[{i}] Source: {doi}\n{summary}")
            confirmed_dois.append(doi)

    return "\n\n".join(context_blocks), confirmed_dois

# --- Embedding Functions ---
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)

def embed_query_pubmedbert(query: str) -> List[float]:
    encoded = tokenizer(
        query, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = pub_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy().tolist()[0]

def query_pinecone(index_name: str, vector: List[float]):
    return pc.Index(index_name).query(vector=vector, top_k=settings.top_k, include_metadata=True)

def normalize_scores(matches):
    scores = [m["score"] for m in matches]
    if not scores or max(scores) == min(scores):
        return {m["metadata"].get("doi", f"{m['id']}"): 0.0 for m in matches}
    norm = MinMaxScaler().fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    return {
        m["metadata"].get("doi", f"{m['id']}"): norm[i] for i, m in enumerate(matches)
    }

def combine_scores(bge_scores: dict, pub_scores: dict, alpha: float = settings.alpha) -> dict:
    all_sources = set(bge_scores) | set(pub_scores)
    combined = {}
    for src in all_sources:
        bge_val = bge_scores.get(src, 0.0)
        pub_val = pub_scores.get(src, 0.0)
        score = alpha * bge_val + (1 - alpha) * pub_val
        if score > 0.05:
            combined[src] = score
    return combined

# --- Question Processing ---
def is_genetics_question(question: str, api_key: str) -> bool:
    """
    Determine if a question is about genetics/molecular biology using OpenAI.
    Now requires user-provided API key.
    """
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classifier that determines if a question is about genetics, molecular biology, "
                        "gene function, protein analysis, or genomics. Respond with only 'true' or 'false'.\n\n"
                        "Questions about yield data, cultivar performance, trial results, location comparisons, "
                        "or statistical analysis of agricultural data should be classified as 'false'.\n\n"
                        "Questions about genes, proteins, molecular mechanisms, genetic markers, "
                        "or biological processes should be classified as 'true'."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result == "true"
    except Exception as e:
        print(f"Error in genetics classification: {e}")
        return False

def query_openai(context: str, source_list: List[str], question: str, conversation_history: List[Dict] = None, api_key: str = None) -> str:
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Adjust system prompt based on whether this is a follow-up to bean data analysis
    system_content = (
        "You are a dry bean genetics and genomics research platform. Your goal is to provide expert-level, "
                "evidence-backed answers to plant science questions.\n"
                "Prioritize high information density and clarity over brevity. Be thorough and precise in explaining "
                "genetic traits, gene functions, and cultivar-level differences.\n\n"

                "Format answers in clean, professional markdown:\n"
                "- Use **bold** for key findings, metrics, or gene names\n"
                "- Use *italics* for scientific terms and species names\n"
                "- Use bullet points (•) for lists\n"
                "- Use tables where helpful\n"
                "- Separate sections with headers when there's a topic shift\n"
                "- Include inline citations like [1], [2] to reference the provided context\n"
                "- DO NOT include a references section at the end - references will be handled separately\n\n"

                "Avoid vague statements. If context is lacking, say so, then supplement with well-established knowledge "
                "clearly labeled as general background.\n\n"

                "Focus on providing comprehensive scientific information without including reference lists."
    )
    
    # If this is a follow-up to successful bean data analysis, adjust the prompt
    if "We successfully analyzed the bean data" in question:
        system_content = (
            "You are a dry bean genetics and genomics research platform. The user has already completed "
            "a successful data analysis with charts and visualizations. Your role is to provide complementary "
            "research context from scientific literature about the biological and genetic factors underlying "
            "the analysis.\n\n"
            
            "Focus on:\n"
            "- Genetic mechanisms related to the traits being analyzed\n"
            "- Breeding implications and cultivar development insights\n"
            "- Research findings that explain the biological basis of the data patterns\n"
            "- Molecular markers and genomic studies relevant to the analysis\n\n"
            
            "Format answers in clean, professional markdown with inline citations [1], [2] to reference sources.\n"
            "Do NOT repeat the data analysis or charts - focus on research insights that complement the completed analysis."
        )
    
    messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current context and question
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using the context provided. Include the bracketed numbers inline.",
    })

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()

def query_openai_stream(context: str, source_list: List[str], question: str, conversation_history: List[Dict] = None, api_key: str = None):
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Adjust system prompt based on whether this is a follow-up to bean data analysis
    system_content = (
        "You are a dry bean genetics and genomics research platform. Your goal is to provide expert-level, "
                "evidence-backed answers to plant science questions.\n"
                "Prioritize high information density and clarity over brevity. Be thorough and precise in explaining "
                "genetic traits, gene functions, and cultivar-level differences.\n\n"

                "Format answers in clean, professional markdown:\n"
                "- Use **bold** for key findings, metrics, or gene names\n"
                "- Use *italics* for scientific terms and species names\n"
                "- Use bullet points (•) for lists\n"
                "- Use tables where helpful\n"
                "- Separate sections with headers when there's a topic shift\n"
                "- Include inline citations like [1], [2] to reference the provided context\n"
                "- DO NOT include a references section at the end - references will be handled separately\n\n"

                "Avoid vague statements. If context is lacking, say so, then supplement with well-established knowledge "
                "clearly labeled as general background.\n\n"

                "Focus on providing comprehensive scientific information without including reference lists."
    )
    
    # If this is a follow-up to successful bean data analysis, adjust the prompt
    if "We successfully analyzed the bean data" in question:
        system_content = (
            "You are a dry bean genetics and genomics research platform. The user has already completed "
            "a successful data analysis with charts and visualizations. Your role is to provide complementary "
            "research context from scientific literature about the biological and genetic factors underlying "
            "the analysis.\n\n"
            
            "Focus on:\n"
            "- Genetic mechanisms related to the traits being analyzed\n"
            "- Breeding implications and cultivar development insights\n"
            "- Research findings that explain the biological basis of the data patterns\n"
            "- Molecular markers and genomic studies relevant to the analysis\n\n"
            
            "Format answers in clean, professional markdown with inline citations [1], [2] to reference sources.\n"
            "Do NOT repeat the data analysis or charts - focus on research insights that complement the completed analysis."
        )
    
    messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current context and question
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using the context provided. Include the bracketed numbers inline.",
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages, 
            temperature=0.2, 
            stream=True,
            timeout=60  # 60 second timeout
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"❌ OpenAI streaming error: {e}")
        yield f"\n\n*Error generating response: {str(e)}*\n\n"

def generate_suggested_questions(
    answer: str,
    sources: List[str] | None = None,
    genes: List[dict] | None = None,
    full_markdown_table: str | None = None
) -> List[str]:
    """Generates a list of suggested follow-up questions based on the provided answer and data."""

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found, skipping suggested questions generation")
        return []
    
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)

    prompt = (
        "Based on the following assistant response (answer and potentially data/sources), "
        "generate a concise list of 3-5 relevant follow-up questions that a user might ask. "
        "Format the response as a simple comma-separated list of questions. "
        "Ensure questions are natural-sounding and directly relate to the information provided."
        "Avoid questions that simply ask for a summary or restatement."
        "Example: 'Tell me more about X, What is the significance of Y, Are there other sources on Z?'\n\n"
        f"Assistant Answer: {answer}\n\n"
    )

    if sources:
        prompt += f"Sources: {', '.join(sources)}\n\n"
    if genes:
        gene_names = [gene['name'] for gene in genes if 'name' in gene]
        prompt += f"Genes mentioned: {', '.join(gene_names)}\n\n"
    if full_markdown_table:
         # Include table data if it's not too long, otherwise just mention its presence
         if len(full_markdown_table) < 1000:
             prompt += f"Bean Data Table Provided:\n{full_markdown_table}\n\n"
         else:
             prompt += "Bean data table was provided.\n\n"

    prompt += "Suggested Questions:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for generating concise follow-up questions."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=150
        )
        suggested_questions_text = response.choices[0].message.content.strip()
        # Split the comma-separated list into a Python list
        return [q.strip() for q in suggested_questions_text.split(',') if q.strip()]
    except Exception as e:
        print(f"Error generating suggested questions: {e}")
        return []

def answer_question_stream(question: str, conversation_history: List[Dict] = None, api_key: str = None):
    """
    Stream the answer to a question with progress updates.
    Now requires user-provided API key.
    """
    # Immediately show thinking indicator
    yield {"type": "progress", "data": {"step": "thinking", "detail": "Thinking..."}}
    
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Initialize bean data variables
    bean_chart_data = {}
    bean_full_md = ""
    bean_data_found = False
    
    # Add current question to conversation history for context
    if conversation_history is None:
        conversation_history = []
    
    current_conversation = conversation_history + [{"role": "user", "content": question}]
    
    # Check if this is a genetics question
    is_genetic = is_genetics_question(question, api_key)
    print(f"🧪 Is this a genetics question? {is_genetic}")
    
    yield {"type": "progress", "data": {"step": "analysis", "detail": "Analyzing question type"}}

    if not is_genetic:
        yield {"type": "progress", "data": {"step": "dataset", "detail": "Checking cultivar database"}}
        
        # Check for bean data keywords - broader detection for data analysis
        bean_keywords = ["yield", "maturity", "cultivar", "variety", "performance", "bean", "production", "steam", "lighthouse", "seal"]
        chart_keywords = ["chart", "plot", "graph", "visualization", "visualize", "show me", "create", "generate", "table", "display"]
        
        # Trigger bean data analysis for relevant questions
        has_bean_keywords = any(keyword in question.lower() for keyword in bean_keywords)
        explicitly_wants_chart = any(keyword in question.lower() for keyword in chart_keywords)
        
        if has_bean_keywords:
            # Let GPT decide whether to call the bean function
            function_call_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a dry bean research platform with access to Ontario bean trial data. "
                        "ALWAYS call the query_bean_data function when the user asks for:\n"
                        "- Bean performance data (yield, maturity, etc.)\n"
                        "- Charts, plots, graphs, or visualizations of bean data\n"
                        "- Cultivar comparisons or analysis\n"
                        "- Questions about specific bean varieties\n"
                        "- Questions about trial results or research station data\n"
                        "- Any question that mentions bean characteristics, locations, or years\n\n"
                        "The user's question mentions bean-related terms, so you should call the function."
                    )
                }
            ]
            
            # Add conversation history for context
            if conversation_history:
                function_call_messages.extend(conversation_history)
            
            function_call_messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=function_call_messages,
                functions=[function_schema],
                function_call="auto",
            )

            choice = response.choices[0]
            
            if choice.finish_reason == "function_call":
                yield {"type": "progress", "data": {"step": "processing", "detail": "Processing cultivar data"}}
                
                call = choice.message.function_call
                
                if call.name == "query_bean_data":
                    args = json.loads(call.arguments)
                    args['original_question'] = question
                    args['api_key'] = api_key
                    
                    preview, full_md, chart_data = answer_bean_query(args)
                    
                    if preview and not preview.strip().startswith("## 🔍 **Dataset Query Results**\n\nNo matching"):
                        yield {"type": "progress", "data": {"step": "dataset_success", "detail": "Found matching data"}}
                        
                        # Generate natural language summary
                        yield {"type": "progress", "data": {"step": "generation", "detail": "Creating analysis summary"}}
                        
                        summary_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a dry bean research analyst reporting to PhD researchers. "
                                        "CRITICAL: Never provide 'steps for analysis' or tell researchers what they should do. "
                                        "Instead, present direct analytical findings, statistical results, and conclusions. "
                                        "This dataset contains Ontario bean trial data from research stations (WOOD, WINC, STHM, etc.) - NOT global country data. "
                                        "Do not refer to this as 'sample data' - this is the complete dataset available. "
                                        "Provide direct analytical results with specific numbers, statistical significance where relevant, and evidence-based conclusions. "
                                        "Use **bold** for key findings and quantitative results. "
                                        "If global/world data is requested, state clearly that only Ontario research station data is available and provide the available analysis."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on the question '{question}', analyze this data:\n\n{preview}"
                                }
                            ],
                            temperature=0.3,
                        )
                        
                        final_answer = summary_response.choices[0].message.content.strip()
                        
                        # Stream the complete answer
                        for char in final_answer:
                            yield {"type": "content", "data": char}
                        
                        # Store bean data for later metadata
                        bean_chart_data = chart_data
                        bean_full_md = full_md
                        bean_data_found = True
                        
                        # Continue with research literature search
                        yield {"type": "progress", "data": {"step": "literature_search", "detail": "Searching research papers for additional insights"}}
                        
                        # Add transition to literature search
                        transition_text = "\n\n---\n\n## 📚 **Related Research Literature**\n\nSearching scientific publications for additional context and insights...\n\n"
                        for char in transition_text:
                            yield {"type": "content", "data": char}
                    else:
                        # No data found, fall back to literature search
                        yield {"type": "progress", "data": {"step": "fallback", "detail": "No data found, searching literature"}}
                        bean_chart_data = {}
                        bean_full_md = ""
            else:
                yield {"type": "progress", "data": {"step": "generation", "detail": "Proceeding to literature search"}}
                
                # Add transition to literature search
                transition_text = "\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                for char in transition_text:
                    yield {"type": "content", "data": char}
        else:
            # No bean keywords, proceed directly to research papers
            yield {"type": "progress", "data": {"step": "generation", "detail": "Proceeding to literature search"}}
            
            # Add transition to literature search
            transition_text = "\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
            for char in transition_text:
                yield {"type": "content", "data": char}

    # --- Genetics question flow ---
    yield {"type": "progress", "data": {"step": "embeddings", "detail": "Processing semantic embeddings"}}
    
    bge_vec = bge_model.encode(question, normalize_embeddings=True).tolist()
    pub_vec = embed_query_pubmedbert(question)
    
    yield {"type": "progress", "data": {"step": "search", "detail": "Searching literature database"}}
    
    bge_res = query_pinecone(settings.bge_index_name, bge_vec)
    pub_res = query_pinecone(settings.pubmedbert_index_name, pub_vec)

    bge_scores = normalize_scores(bge_res["matches"])
    pub_scores = normalize_scores(pub_res["matches"])
    combined_scores = combine_scores(bge_scores, pub_scores)

    top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
    top_dois = [src for src, _ in top_sources]
    
    yield {"type": "progress", "data": {"step": "papers", "detail": f"Found {len(top_dois)} relevant papers"}}
    
    print("🔎 Top DOIs from Pinecone:", top_dois)

    context, source_list = get_rag_context_from_dois(top_dois)
    
    yield {"type": "progress", "data": {"step": "generation", "detail": "Synthesizing findings with AI"}}

    # Stream the response
    # If we have bean data, modify the question to include context
    rag_question = question
    if bean_data_found:
        rag_question = f"We successfully analyzed the bean data for: '{question}'. Now provide additional research context from scientific literature about the genetic and biological factors related to this analysis."
    
    for chunk in query_openai_stream(context, source_list, rag_question, conversation_history, api_key):
        yield {"type": "content", "data": chunk}

    yield {"type": "progress", "data": {"step": "genes", "detail": "Analyzing genetic elements"}}

    # Get the full response for gene extraction (use same modified question as streaming)
    rag_question = question
    if bean_data_found:
        rag_question = f"We successfully analyzed the bean data for: '{question}'. Now provide additional research context from scientific literature about the genetic and biological factors related to this analysis."
    
    full_response = query_openai(context, source_list, rag_question, conversation_history, api_key)
    
    # Extract genes from the complete answer
    print("🧬 Extracting gene mentions...")
    gene_mentions, db_hits, gpt_hits = extract_gene_mentions(full_response)
    print(f"Found gene mentions: {gene_mentions}")

    # Map genes to their summaries with preview URLs
    gene_summaries = []
    for gene in gene_mentions:
        # Try to get gene ID from NCBI database
        gene_id = map_to_gene_id(gene)
        if gene_id:
            # Get summary from NCBI database
            from utils.ncbi_utils import get_gene_summary
            gene_summary = get_gene_summary(gene_id)
            preview_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"
            gene_summaries.append({
                "name": gene,
                "summary": f"![NCBI Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                "link": preview_url,
                "source": "NCBI Gene Database",
                "description": gene_summary or f"Gene ID: {gene_id}"
            })
        else:
            # Try to get UniProt information
            from utils.ncbi_utils import get_uniprot_info
            uniprot_info = get_uniprot_info(gene)
            if uniprot_info:
                preview_url = f"https://www.uniprot.org/uniprotkb/{uniprot_info['entry']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![UniProt Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "UniProt Protein Database",
                    "description": uniprot_info['protein_names'] or f"UniProt Entry: {uniprot_info['entry']}"
                })
            else:
                # Gene not found in databases, add basic info
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- Gene identifier: {gene}\n- Source: Literature mention",
                    "source": "Literature Mention",
                    "description": f"Gene mentioned in research literature: {gene}",
                    "not_found": True
                })

    genes = gene_summaries

    yield {"type": "progress", "data": {"step": "finalizing", "detail": "Completing analysis"}}

    # Include bean data in metadata if available
    try:
        final_chart_data = bean_chart_data if bean_data_found else {}
        final_full_md = bean_full_md if bean_data_found else ""
    except:
        final_chart_data = {}
        final_full_md = ""

    yield {
        "type": "metadata",
        "data": {
            "sources": source_list,
            "genes": genes,
            "full_markdown_table": final_full_md,
            "chart_data": final_chart_data,
            "suggested_questions": []
        }
    }

def answer_question(question: str, conversation_history: List[Dict] = None, api_key: str = None) -> Tuple[str, List[str], List[dict], str]:
    is_genetic = is_genetics_question(question, api_key)
    print(f"🧪 Is this a genetics question? {is_genetic}")
    
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Initialize transition_message
    transition_message = ""

    if not is_genetic:
        # Check for bean data keywords - broader detection for data analysis
        bean_keywords = ["yield", "maturity", "cultivar", "variety", "performance", "bean", "production", "steam", "lighthouse", "seal"]
        chart_keywords = ["chart", "plot", "graph", "visualization", "visualize", "show me", "create", "generate", "table", "display"]
        
        # Trigger bean data analysis for relevant questions
        has_bean_keywords = any(keyword in question.lower() for keyword in bean_keywords)
        explicitly_wants_chart = any(keyword in question.lower() for keyword in chart_keywords)
                
        if has_bean_keywords:
            try:
                # Let GPT decide whether to call the bean function
                function_call_messages = [
                    {
                        "role": "system",
                        "content": "You are a dry bean research platform. If the user asks for bean performance data, charts, or cultivar analysis, call the appropriate function."
                    }
                ]
                
                # Add conversation history for context
                if conversation_history:
                    function_call_messages.extend(conversation_history)
                
                function_call_messages.append({"role": "user", "content": question})
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=function_call_messages,
                    functions=[function_schema],
                    function_call="auto",
                )

                choice = response.choices[0]
                if choice.finish_reason == "function_call":
                    call = choice.message.function_call
                    if call.name == "query_bean_data":
                        args = json.loads(call.arguments)
                        args['original_question'] = question
                        args['api_key'] = api_key
                        
                        preview, full_md, chart_data = answer_bean_query(args)
                        
                        if preview and len(preview) > 20:  # Valid response
                            # Add transition message for research papers
                            transition_message = "## 🔍 **Dataset Analysis Results**\n\n" + preview + "\n\n---\n\n## 📚 **Related Research Literature**\n\nSearching scientific publications for additional context and insights...\n\n"
                        else:
                            # Fallback to research papers with transition message
                            transition_message = "## 🔍 **Dataset Search Results**\n\nNo specific data found in our cultivar performance dataset for this query.\n\n---\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                            print("🔄 Bean data insufficient, falling back to research papers...")
                else:
                    # GPT decided not to use function, add transition message
                    transition_message = "## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                    
            except Exception as e:
                print(f"❌ Bean data query failed: {e}")
                # Error fallback with transition message
                transition_message = "## 🔍 **Dataset Search**\n\nEncountered an issue accessing the cultivar dataset.\n\n---\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
        else:
            # No bean keywords, proceed directly to research papers
            transition_message = ""
    else:
        # Genetics question, no transition needed
        transition_message = ""

    # --- RAG pipeline for research papers ---
    print("🔬 Proceeding with research paper search...")
    bge_vec = bge_model.encode(question, normalize_embeddings=True).tolist()
    pub_vec = embed_query_pubmedbert(question)
    
    print("🔎 Querying Pinecone...")
    bge_matches = query_pinecone(settings.bge_index_name, bge_vec)
    pub_matches = query_pinecone(settings.pubmedbert_index_name, pub_vec)
    print("✅ Pinecone queries completed.")

    bge_scores = normalize_scores(bge_matches["matches"])
    pub_scores = normalize_scores(pub_matches["matches"])
    combined_scores = combine_scores(bge_scores, pub_scores)

    top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
    top_dois = [src for src, _ in top_sources]
    print("🔎 Top DOIs from Pinecone:", top_dois)

    combined_context, confirmed_dois = get_rag_context_from_dois(top_dois)
    if not combined_context.strip():
        print("⚠️ No RAG matches found for top DOIs.")
        return "No matching papers found in RAG corpus.", top_dois, [], ""

    final_answer = query_openai(combined_context, top_dois, question, conversation_history, api_key)
    print("✅ Generated answer with context.")

    # Add transition message if needed
    if transition_message:
        final_answer = transition_message + final_answer

    # Extract genes from the complete answer
    print("🧬 Extracting gene mentions...")
    gene_mentions, db_hits, gpt_hits = extract_gene_mentions(final_answer)
    print(f"Found gene mentions: {gene_mentions}")

    # Map genes to their summaries with preview URLs
    gene_summaries = []
    for gene in gene_mentions:
        # Try to get gene ID from NCBI database
        gene_id = map_to_gene_id(gene)
        if gene_id:
            # Get summary from NCBI database
            from utils.ncbi_utils import get_gene_summary
            gene_summary = get_gene_summary(gene_id)
            preview_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"
            gene_summaries.append({
                "name": gene,
                "summary": f"![NCBI Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                "link": preview_url,
                "source": "NCBI Gene Database",
                "description": gene_summary or f"Gene ID: {gene_id}"
            })
        else:
            # Try to get UniProt information
            from utils.ncbi_utils import get_uniprot_info
            uniprot_info = get_uniprot_info(gene)
            if uniprot_info:
                preview_url = f"https://www.uniprot.org/uniprotkb/{uniprot_info['entry']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![UniProt Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "UniProt Protein Database",
                    "description": uniprot_info['protein_names'] or f"UniProt Entry: {uniprot_info['entry']}"
                })
            else:
                # Gene not found in databases, add basic info
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- Gene identifier: {gene}\n- Source: Literature mention",
                    "source": "Literature Mention",
                    "description": f"Gene mentioned in research literature: {gene}",
                    "not_found": True
                })

    print(f"✅ Gene extraction completed. Found {len(gene_summaries)} genes.")
    return final_answer, confirmed_dois, gene_summaries, "" 