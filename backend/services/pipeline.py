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
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from utils.ncbi_utils import extract_gene_mentions, map_to_gene_id, load_gene_db, load_uniprot_db
from utils.bean_data import function_schema, answer_bean_query

# Load environment variables
load_dotenv()

# --- Config ---
BGE_INDEX_NAME = "dry-bean-bge-abstract"
PUBMEDBERT_INDEX_NAME = "dry-bean-pubmedbert-abstract"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOP_K = 8
ALPHA = 0.6

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Load Models ---
bge_model = SentenceTransformer(BGE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
pub_model = AutoModel.from_pretrained(PUBMEDBERT_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pub_model.to(device).eval()

# Load gene databases
GENE_DB_PATH = os.getenv("GENE_DB_PATH", "../data/NCBI_Filtered_Data_Enriched.xlsx")
UNIPROT_DB_PATH = os.getenv("UNIPROT_DB_PATH", "../data/uniprotkb_Phaseolus_vulgaris.xlsx")
RAG_FILE = os.getenv("RAG_FILE", "../data/summaries.jsonl")

# Load gene data
load_gene_db(GENE_DB_PATH)
print(f"Loaded gene database from {GENE_DB_PATH}")

# Load UniProt data
load_uniprot_db(UNIPROT_DB_PATH)
print(f"Loaded UniProt database from {UNIPROT_DB_PATH}")

# Load RAG data
try:
    with open(RAG_FILE, 'r', encoding='utf-8') as f:
        rag_data = [json.loads(line) for line in f]
except UnicodeDecodeError:
    # Fallback to latin-1 if utf-8 fails
    with open(RAG_FILE, 'r', encoding='latin-1') as f:
        rag_data = [json.loads(line) for line in f]

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

RAG_LOOKUP = load_rag_text_jsonl(Path(RAG_FILE))

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
    return pc.Index(index_name).query(vector=vector, top_k=TOP_K, include_metadata=True)

def normalize_scores(matches):
    scores = [m["score"] for m in matches]
    if not scores or max(scores) == min(scores):
        return {m["metadata"].get("doi", f"{m['id']}"): 0.0 for m in matches}
    norm = MinMaxScaler().fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    return {
        m["metadata"].get("doi", f"{m['id']}"): norm[i] for i, m in enumerate(matches)
    }

def combine_scores(bge_scores: dict, pub_scores: dict, alpha: float = ALPHA) -> dict:
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
    api_key = api_key or os.getenv("OPENAI_API_KEY")  # fallback to env var
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    
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
    api_key = api_key or os.getenv("OPENAI_API_KEY")  # fallback to env var
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a dry bean genetics and genomics main platform. Your goal is to provide expert-level, "
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
            ),
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
    api_key = api_key or os.getenv("OPENAI_API_KEY")  # fallback to env var
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a dry bean genetics and genomics main platform. Your goal is to provide expert-level, "
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
            ),
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
    
    client = OpenAI(api_key=api_key)

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
    api_key = api_key or os.getenv("OPENAI_API_KEY")  # fallback to env var
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    # Create client with user-provided API key
    client = OpenAI(api_key=api_key)
    
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
        
        # Let GPT decide whether to call the bean function
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dry bean main platform. If the user asks for bean performance "
                        "data (like yield, maturity, cultivar names), you should call the appropriate function."
                    ),
                },
                {"role": "user", "content": question},
            ],
            functions=[function_schema],
            function_call="auto",
        )

        choice = response.choices[0]
        if choice.finish_reason == "function_call":
            yield {"type": "progress", "data": {"step": "processing", "detail": "Processing cultivar data"}}
            
            call = choice.message.function_call
            if call.name == "query_bean_data":
                args = json.loads(call.arguments)
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
                                    "You are a dry bean genetics research assistant. The user asked a question about cultivar performance data, "
                                    "and we found matching data in our dataset. Your job is to provide a comprehensive, natural language summary "
                                    "of the data results, formatted in clean professional markdown.\n\n"
                                    
                                    "Guidelines:\n"
                                    "- Be thorough and analytical in your interpretation\n"
                                    "- Use **bold** for key metrics, findings, and cultivar names\n"
                                    "- Use bullet points for lists and comparisons\n"
                                    "- Include specific numbers and statistics from the data\n"
                                    "- Provide actionable insights for researchers\n"
                                    "- Reference data directly (e.g., 'The data shows...')\n"
                                    "- Don't just repeat the table - interpret and explain patterns\n"
                                    "- If charts are included, reference them appropriately\n\n"
                                    
                                    "Answer the user's original question directly using the data provided."
                                ),
                            }
                        ] + current_conversation + [
                            {
                                "role": "assistant",
                                "content": f"Based on your question, I found relevant cultivar performance data:\n\n{preview}"
                            },
                            {
                                "role": "user",
                                "content": f"Please provide a comprehensive analysis of this data to answer my original question: '{question}'"
                            }
                        ],
                        temperature=0.3,
                    )
                    
                    final_answer = summary_response.choices[0].message.content.strip()
                    
                    # Stream the complete answer
                    for char in final_answer:
                        yield {"type": "content", "data": char}
                    
                    # Send metadata
                    yield {
                        "type": "metadata",
                        "data": {
                            "sources": [],
                            "genes": [],
                            "full_markdown_table": full_md,
                            "chart_data": chart_data,
                            "suggested_questions": []
                        }
                    }
                    return
                else:
                    # No data found, fall back to literature search
                    yield {"type": "progress", "data": {"step": "fallback", "detail": "No data found, searching literature"}}

        # If no function call or no data found, fall back to natural response
        yield {"type": "progress", "data": {"step": "generation", "detail": "Generating response"}}
        
        response_text = response.choices[0].message.content.strip()
        for char in response_text:
            yield {"type": "content", "data": char}
        
        yield {
            "type": "metadata", 
            "data": {
                "sources": [], 
                "genes": [], 
                "full_markdown_table": "", 
                "suggested_questions": []
            }
        }
        return

    # --- Genetics question flow ---
    yield {"type": "progress", "data": {"step": "embeddings", "detail": "Processing semantic embeddings"}}
    
    bge_vec = bge_model.encode(question, normalize_embeddings=True).tolist()
    pub_vec = embed_query_pubmedbert(question)
    
    yield {"type": "progress", "data": {"step": "search", "detail": "Searching literature database"}}
    
    bge_res = query_pinecone(BGE_INDEX_NAME, bge_vec)
    pub_res = query_pinecone(PUBMEDBERT_INDEX_NAME, pub_vec)

    bge_scores = normalize_scores(bge_res["matches"])
    pub_scores = normalize_scores(pub_res["matches"])
    combined_scores = combine_scores(bge_scores, pub_scores)

    top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_dois = [src for src, _ in top_sources]
    
    yield {"type": "progress", "data": {"step": "papers", "detail": f"Found {len(top_dois)} relevant papers"}}
    
    print("🔎 Top DOIs from Pinecone:", top_dois)

    context, source_list = get_rag_context_from_dois(top_dois)
    
    yield {"type": "progress", "data": {"step": "generation", "detail": "Synthesizing findings with AI"}}

    # Stream the response
    for chunk in query_openai_stream(context, source_list, question, conversation_history, api_key):
        yield {"type": "content", "data": chunk}

    yield {"type": "progress", "data": {"step": "genes", "detail": "Analyzing genetic elements"}}

    # Get the full response for gene extraction
    full_response = query_openai(context, source_list, question, conversation_history, api_key)
    
    # Extract genes from the complete answer
    print("🧬 Extracting gene mentions...")
    gene_mentions, db_hits, gpt_hits = extract_gene_mentions(full_response)
    print(f"Found gene mentions: {gene_mentions}")

    # Map genes to their summaries with preview URLs
    gene_summaries = []
    for gene in gene_mentions:
        # db_hits now contains GPT genes that were validated against databases
        # gpt_hits contains all GPT genes
        # So we don't need the possible_flag logic anymore since all genes come from GPT
        possible_flag = False
        gene_info = map_to_gene_id(gene)
        if gene_info:
            if gene_info["source"] == "NCBI":
                preview_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_info['gene_id']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![NCBI Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "NCBI Gene Database" if not possible_flag else "NCBI Gene Database (Possible Match)",
                    "description": gene_info['description']
                })
            elif gene_info["source"] == "UniProt":
                preview_url = f"https://www.uniprot.org/uniprotkb/{gene_info['entry']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![UniProt Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "UniProt Protein Database" if not possible_flag else "UniProt Protein Database (Possible Match)",
                    "description": gene_info['description']
                })
            elif gene_info["source"] == "GPT-4o":
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- {gene_info['description']}\n- Generated by AI analysis",
                    "source": "AI Analysis" if not possible_flag else "AI Analysis (Possible Match)",
                    "description": gene_info['description'],
                    "generated": True
                })
            else:
                source_label = gene_info['source'] if not possible_flag else f"Possible {gene_info['source']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- Description: `{gene_info['description']}`\n- Source: {source_label}",
                    "source": source_label,
                    "description": gene_info['description']
                })
        else:
            # Gene identified but not found in any database
            gene_summaries.append({
                "name": gene,
                "summary": f"- Genetic element mentioned in context\n- No database match found",
                "source": "Literature Reference" if not possible_flag else "Literature Reference (Possible Match)",
                "description": f"This genetic element was identified in the research context but could not be matched to existing databases.",
                "not_found": True
            })

    genes = gene_summaries

    yield {"type": "progress", "data": {"step": "finalizing", "detail": "Completing analysis"}}

    yield {
        "type": "metadata",
        "data": {
            "sources": source_list,
            "genes": genes,
            "full_markdown_table": "",
            "suggested_questions": []
        }
    }

def answer_question(question: str, conversation_history: List[Dict] = None) -> Tuple[str, List[str], List[dict], str]:
    is_genetic = is_genetics_question(question, os.getenv("OPENAI_API_KEY"))
    print(f"🧪 Is this a genetics question? {is_genetic}")
    
    # Initialize transition_message
    transition_message = ""

    if not is_genetic:
        # Check if it might be bean data query first - use broader, more flexible keywords
        bean_data_keywords = [
            "average", "mean", "maximum", "minimum", "highest", "lowest", 
            "yield", "performance", "analysis", "compare", "cultivar", "variety",
            "chart", "plot", "graph", "visualization", "show", "display", "data",
            "production", "producing", "rank", "ranking"
        ]
        keywords_found = [keyword for keyword in bean_data_keywords if keyword in question.lower()]
        
        # Also check if this is a follow-up to previous bean data query
        has_bean_context = False
        if conversation_history:
            for msg in conversation_history[-3:]:  # Check last 3 messages
                if msg.get('role') == 'assistant' and msg.get('content'):
                    content = msg['content'].lower()
                    if any(indicator in content for indicator in [
                        'scatter plot', 'bean data analysis', 'cultivar', 'yield', 'maturity',
                        'white bean', 'coloured bean', 'location', 'trial', 'dataset', 'filter:'
                    ]):
                        has_bean_context = True
                        break
        
        # Check for follow-up requests (more flexible)
        followup_terms = [
            'chart', 'plot', 'graph', 'visualization', 'same question', 'different', 'another', 
            'show me', 'generate', 'create', 'make', 'display', 'with', 'using', 'for'
        ]
        is_followup = has_bean_context and any(term in question.lower() for term in followup_terms)
        
        print(f"🔍 Has bean data keywords? {bool(keywords_found)} - Found: {keywords_found}")
        print(f"🔍 Has bean context from history? {has_bean_context}")
        print(f"🔍 Is follow-up? {is_followup}")
        
        if keywords_found or is_followup:
            try:
                # Analyze conversation history for bean data context
                has_bean_data_context = False
                if conversation_history:
                    for msg in conversation_history[-4:]:  # Check last 4 messages
                        if msg.get('role') == 'assistant' and msg.get('content'):
                            content = msg['content'].lower()
                            if any(indicator in content for indicator in [
                                'scatter plot', 'bean data analysis', 'cultivar', 'yield', 'maturity',
                                'white bean', 'coloured bean', 'location', 'trial', 'dataset'
                            ]):
                                has_bean_data_context = True
                                break
                
                # Enhanced system prompt that considers conversation context
                system_prompt = (
                    "You are a dry bean research platform. If the user asks for bean performance "
                    "data (like yield, maturity, cultivar names), you should call the appropriate function. "
                    "IMPORTANT: For comparison requests involving specific cultivars (like OAC Seal), rankings, "
                    "or yield comparisons, always use the bean data function to provide specific numerical data "
                    "from the Merged_Bean_Dataset rather than generic responses. "
                    "When users ask about 'production' or 'dry bean production', ALWAYS use analysis_column='Yield' since production equals yield per hectare. "
                    "For ranking questions mentioning cultivars, ALWAYS extract the cultivar name and use analysis_type='compare'. "
                    "CRITICAL PARAMETER MAPPING: "
                    "- bean_type: Use for 'white bean' or 'coloured bean' (NOT for major/minor) "
                    "- trial_group: Use for 'major' or 'minor' trials (NOT for bean color) "
                    "- When users mention 'minor bean types' or 'major bean types', they mean trial_group='minor' or trial_group='major' "
                    "- When users mention bean colors or varieties, use bean_type parameter "
                    "CRITICAL: When you see 'OAC Seal' or any cultivar name in the user query, you MUST include cultivar='OAC Seal' parameter. "
                    "EXAMPLE: For 'compare with OAC Seal' you MUST include cultivar='OAC Seal' in your function call. "
                    "EXAMPLE: For 'vs OAC Steam' you MUST include cultivar='OAC Steam' in your function call. "
                    "For visualization requests, analyze the user's intent and data context to choose the most "
                    "appropriate analysis_type and parameters. Guidelines: "
                    "- For ranking/comparison requests (e.g., 'rank countries vs OAC Seal', 'compare production with OAC Seal'), "
                    "  use analysis_type='compare' with cultivar='OAC Seal' or the specific cultivar name mentioned "
                    "- When users mention specific cultivar names (OAC Seal, Seal, OAC Steam, etc.), ALWAYS include cultivar parameter "
                    "- For cultivar performance questions, use analysis_type='cultivar_analysis' "
                    "- When comparing global production to specific cultivars, use the cultivar parameter "
                    "- For general chart/visualization requests (including pie charts, bar charts, etc.), use analysis_type='visualization' "
                    "- When user requests a specific chart type (pie chart, bar chart, line chart, histogram, area chart, scatter plot), "
                    "  use analysis_type='visualization' AND set chart_type parameter to the specific type: "
                    "  'pie', 'bar', 'line', 'histogram', 'area', or 'scatter' "
                    "- For specific scatter plot requests, you can also use analysis_type='scatter' "
                    "- For location-focused analysis, use analysis_type='location_analysis' "
                    "- For cultivar-focused analysis, use analysis_type='cultivar_analysis' "
                    "- For year-over-year trends, use analysis_type='yearly_average' or 'trend' "
                    "- For basic statistics, use analysis_type='average', 'max', 'min', etc. "
                    "Always include relevant filters like bean_type, year, location, and cultivar based on the user's request. "
                )
                
                if has_bean_data_context:
                    system_prompt += (
                        "IMPORTANT: The conversation history shows previous bean data queries. "
                        "For follow-up visualization requests, extract the context and filters from the "
                        "conversation history (bean_type, year, location, etc.) and apply them to create "
                        "a meaningful visualization. When user asks for a specific chart type (like 'pie chart', "
                        "'bar chart'), use analysis_type='visualization' and set chart_type to the requested type. "
                    )
                
                # Let GPT decide whether to call the bean function (same as non-streaming version)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        # Add conversation history if available
                        *(conversation_history or []),
                        {"role": "user", "content": question},
                    ],
                    functions=[function_schema],
                    function_call="auto",
                )

                choice = response.choices[0]
                if choice.finish_reason == "function_call":
                    call = choice.message.function_call
                    if call.name == "query_bean_data":
                        args = json.loads(call.arguments)
                        # Add the original question for cultivar detection
                        args['original_question'] = question
                        
                        # Map "production" to "Yield" if analysis_column is not set or contains production
                        if 'analysis_column' not in args or not args.get('analysis_column'):
                            if 'production' in question.lower():
                                args['analysis_column'] = 'Yield'
                        elif args.get('analysis_column') and 'production' in args['analysis_column'].lower():
                            args['analysis_column'] = 'Yield'
                        
                        preview, full_md, chart_data = answer_bean_query(args)
                        
                        if preview and len(preview) > 20:  # Valid response
                            return preview, [], [], full_md
                        else:
                            # Fallback to research papers with transition message
                            transition_message = "## 🔍 Dataset Search Results\n\nNo specific data found in our cultivar performance dataset for this query.\n\n---\n\n## 📚 Research Literature Search\n\nSearching scientific publications for relevant information...\n\n"
                            print("🔄 Bean data insufficient, falling back to research papers...")
                else:
                    # GPT decided not to use function, add transition message
                    transition_message = "## 🔍 Query Analysis\n\nThis query appears to be outside our structured dataset scope.\n\n---\n\n## 📚 Research Literature Search\n\nSearching scientific publications for relevant information...\n\n"
                    
            except Exception as e:
                print(f"❌ Bean data query failed: {e}")
                # Error fallback with transition message
                transition_message = "## 🔍 Dataset Search\n\nEncountered an issue accessing the cultivar dataset.\n\n---\n\n## 📚 Research Literature Search\n\nSearching scientific publications for relevant information...\n\n"
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
    bge_matches = query_pinecone(BGE_INDEX_NAME, bge_vec)
    pub_matches = query_pinecone(PUBMEDBERT_INDEX_NAME, pub_vec)
    print("✅ Pinecone queries completed.")

    bge_scores = normalize_scores(bge_matches["matches"])
    pub_scores = normalize_scores(pub_matches["matches"])
    combined_scores = combine_scores(bge_scores, pub_scores)

    top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_dois = [src for src, _ in top_sources]
    print("🔎 Top DOIs from Pinecone:", top_dois)

    combined_context, confirmed_dois = get_rag_context_from_dois(top_dois)
    if not combined_context.strip():
        print("⚠️ No RAG matches found for top DOIs.")
        return "No matching papers found in RAG corpus.", top_dois, [], ""

    final_answer = query_openai(combined_context, top_dois, question, conversation_history, os.getenv("OPENAI_API_KEY"))
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
        # db_hits now contains GPT genes that were validated against databases
        # gpt_hits contains all GPT genes
        # So we don't need the possible_flag logic anymore since all genes come from GPT
        possible_flag = False
        gene_info = map_to_gene_id(gene)
        if gene_info:
            if gene_info["source"] == "NCBI":
                preview_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_info['gene_id']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![NCBI Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "NCBI Gene Database" if not possible_flag else "NCBI Gene Database (Possible Match)",
                    "description": gene_info['description']
                })
            elif gene_info["source"] == "UniProt":
                preview_url = f"https://www.uniprot.org/uniprotkb/{gene_info['entry']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![UniProt Preview](https://api.screenshotmachine.com/?key=demo&url={preview_url}&dimension=400x300)",
                    "link": preview_url,
                    "source": "UniProt Protein Database" if not possible_flag else "UniProt Protein Database (Possible Match)",
                    "description": gene_info['description']
                })
            elif gene_info["source"] == "GPT-4o":
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- {gene_info['description']}\n- Generated by AI analysis",
                    "source": "AI Analysis" if not possible_flag else "AI Analysis (Possible Match)",
                    "description": gene_info['description'],
                    "generated": True
                })
            else:
                source_label = gene_info['source'] if not possible_flag else f"Possible {gene_info['source']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"- Description: `{gene_info['description']}`\n- Source: {source_label}",
                    "source": source_label,
                    "description": gene_info['description']
                })
        else:
            # Gene identified but not found in any database
            gene_summaries.append({
                "name": gene,
                "summary": f"- Genetic element mentioned in context\n- No database match found",
                "source": "Literature Reference" if not possible_flag else "Literature Reference (Possible Match)",
                "description": f"This genetic element was identified in the research context but could not be matched to existing databases.",
                "not_found": True
            })

    print(f"✅ Gene extraction completed. Found {len(gene_summaries)} genes.")
    return final_answer, confirmed_dois, gene_summaries, "" 