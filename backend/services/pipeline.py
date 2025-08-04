import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path
import orjson
from pymilvus import MilvusClient
from sklearn.preprocessing import MinMaxScaler

from config import settings
from database.manager import db_manager
from utils.ncbi_utils import extract_gene_mentions, map_to_gene_id
from utils.bean_data import function_schema, answer_bean_query
from utils.openai_client import create_openai_client

# Load environment variables
load_dotenv()

# Initialize Zilliz/Milvus client
_milvus_client = None

def get_milvus_client():
    """Initialize Milvus client on first use."""
    global _milvus_client
    if _milvus_client is None:
        print("🔄 Connecting to Zilliz...")
        _milvus_client = MilvusClient(
            uri=settings.zilliz_uri,
            token=settings.zilliz_token
        )
        print("✅ Zilliz client connected")
    return _milvus_client

# --- Gene Processing Functions ---
def process_genes_batch(gene_mentions: List[str]) -> List[Dict[str, Any]]:
    """Batch process genes for better performance by avoiding individual lookups."""
    print(f"🧬 Batch processing {len(gene_mentions)} genes...")
    
    gene_summaries = []
    
    for gene in gene_mentions:
        # Try to get gene ID from NCBI database using fast lookup
        gene_id = map_to_gene_id(gene)
        if gene_id:
            # Get summary from NCBI database using fast lookup
            from utils.ncbi_utils import get_gene_summary
            gene_summary = get_gene_summary(gene_id)
            preview_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"
            gene_summaries.append({
                "name": gene,
                "summary": f"**NCBI Gene Database**\n\n{gene_summary or f'Gene ID: {gene_id}'}",
                "link": preview_url,
                "source": "NCBI Gene Database",
                "description": gene_summary or f"Gene ID: {gene_id}"
            })
        else:
            # Try to get UniProt information using fast lookup
            from utils.ncbi_utils import get_uniprot_info
            uniprot_info = get_uniprot_info(gene)
            if uniprot_info:
                preview_url = f"https://www.uniprot.org/uniprotkb/{uniprot_info['entry']}"
                gene_summaries.append({
                    "name": gene,
                    "summary": f"![UniProt Logo](/images/UniProtLogo.png)\n\n**UniProt Protein Database**\n\n{uniprot_info['protein_names'] or ('UniProt Entry: ' + uniprot_info['entry'])}",
                    "link": preview_url,
                    "source": "UniProt Protein Database",
                    "description": uniprot_info['protein_names'] or f"UniProt Entry: {uniprot_info['entry']}"
                })
            else:
                # Gene not found in databases, add basic info
                gene_summaries.append({
                    "name": gene,
                    "summary": f"**Literature Mention**\n\nGene identifier: {gene}\nSource: Literature mention",
                    "source": "Literature Mention",
                    "description": f"Gene mentioned in research literature: {gene}",
                    "not_found": True
                })
    
    print(f"✅ Batch processed {len(gene_summaries)} genes")
    return gene_summaries

# --- RAG Context from Zilliz Matches ---
def get_rag_context_from_matches(matches: List[dict], top_dois: List[str]) -> Tuple[str, List[str]]:
    """Extract context directly from Zilliz matches metadata."""
    context_blocks = []
    confirmed_dois = []
    
    # Create a lookup dictionary from all matches for quick access
    all_matches = {}
    
    # Process Zilliz matches
    for match in matches:
        entity = match.get("entity", {})
        doi = entity.get("doi", "")
        summary = entity.get("summary", "")
        if doi and summary:
            clean_doi = doi.strip()
            all_matches[clean_doi] = summary
    
    print(f"🔍 Context built with {len(all_matches)} matches from Zilliz")
    
    # Build context blocks for the top DOIs
    context_counter = 1
    for doi in top_dois:
        clean_target_doi = doi.strip()
        if clean_target_doi in all_matches:
            summary = all_matches[clean_target_doi]
            context_blocks.append(f"[{context_counter}] Source: {clean_target_doi}\n{summary}")
            confirmed_dois.append(clean_target_doi)
            context_counter += 1
        else:
            # Try alternative matching approaches
            for match_doi, summary in all_matches.items():
                if (clean_target_doi.lower() == match_doi.lower() or
                    clean_target_doi.replace("https://doi.org/", "").replace("http://doi.org/", "") == 
                    match_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")):
                    context_blocks.append(f"[{context_counter}] Source: {match_doi}\n{summary}")
                    confirmed_dois.append(match_doi)
                    context_counter += 1
                    break
    
    print(f"✅ Final context: {len(confirmed_dois)} sources available for AI")
    return "\n\n".join(context_blocks), confirmed_dois

# --- Embedding Functions ---
def embed_query_openai(query: str, api_key: str) -> List[float]:
    """Generate embeddings using OpenAI's text-embedding-3-large model."""
    try:
        client = create_openai_client(api_key)
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=query,
            dimensions=1536  # Match your Zilliz collection dimension
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating OpenAI embeddings: {e}")
        raise

def query_zilliz(vector: List[float], api_key: str) -> List[dict]:
    """Query Zilliz vector database."""
    try:
        client = get_milvus_client()
        search_results = client.search(
            collection_name=settings.collection_name,
            data=[vector],
            limit=settings.top_k,
            output_fields=["doi", "summary"]
        )
        return search_results[0] if search_results else []
    except Exception as e:
        print(f"❌ Error querying Zilliz: {e}")
        raise

def normalize_scores(matches: List[dict]) -> Dict[str, float]:
    """Normalize similarity scores from Zilliz matches."""
    if not matches:
        return {}
    
    scores = [m.get("distance", 0.0) for m in matches]
    if not scores or max(scores) == min(scores):
        return {m.get("entity", {}).get("doi", f"doc_{i}"): 0.0 for i, m in enumerate(matches)}
    
    # Convert distance to similarity (assuming cosine distance)
    similarities = [1 - score for score in scores]
    norm = MinMaxScaler().fit_transform(np.array(similarities).reshape(-1, 1)).flatten()
    
    return {
        m.get("entity", {}).get("doi", f"doc_{i}"): norm[i] 
        for i, m in enumerate(matches)
    }

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
                        "gene function, protein analysis, genomics, beans, breeding, or plant biology. Respond with only 'true' or 'false'.\n\n"
                        "Questions about yield data, cultivar performance, trial results, location comparisons, "
                        "or statistical analysis of agricultural data should be classified as 'false'.\n\n"
                        "Questions about genes, proteins, molecular mechanisms, genetic markers, "
                        "biological processes, beans, breeding, or plant biology should be classified as 'true'."
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
        "You are a dry bean genetics and genomics research platform. Your role is to deliver expert-level, "
        "evidence-backed, and mechanistically detailed scientific responses about Phaseolus vulgaris.\n\n"
        
        "Your users are graduate-level researchers and plant breeders.\n"
        "Assume they have advanced training in plant molecular biology, breeding, and quantitative genetics.\n"
        "Do not explain basic biological terms or give surface-level summaries.\n\n"
        
        "You have access to:\n"
        "• Your internal scientific knowledge and pretraining\n"
        "• Scientific literature provided as context\n\n"
        
        "🎯 How to Answer:\n"
        "Use your internal knowledge first to construct a complete, structured, and high-value response.\n"
        "Then incorporate the retrieved context only when it contributes specific details, named markers, or citations.\n"
        "Do not rely solely on the provided documents.\n\n"
        
        "📌 Your Answer Must Include (When Relevant):\n"
        "• Named genes and DNA markers (e.g., BC420, SAP6, SU91, Phvul.010G140000)\n"
        "• Known QTL locations (e.g., Pv06, Pv08, Pv10)\n"
        "• Transcription factors, enzyme families, or regulatory modules (e.g., WRKY, PAL, NAC, PR genes)\n"
        "• Expression patterns (e.g., upregulation under stress)\n"
        "• Pathways involved (e.g., phenylpropanoid, ROS detox, proanthocyanidin biosynthesis)\n"
        "• Breeding relevance (e.g., marker-assisted selection, QTL pyramiding, donor lines)\n"
        "• Tables or structured summaries where helpful (e.g., gene functions, cultivar comparisons)\n\n"
        
        "⚙️ Formatting (Markdown):\n"
        "• **Bold**: Gene names, markers, traits, numeric results\n"
        "• *Italics*: Scientific species names and terms\n"
        "• Use bullet points and section headers\n"
        "• Use inline citations like [1], [2] only if based on retrieved context\n"
        "• Do not include a reference list at the end\n\n"
        
        "🚫 Do NOT:\n"
        "• Dumb down your answers — assume expert-level knowledge\n"
        "• Say 'the context doesn't specify' unless followed by a detailed supplement\n"
        "• Fabricate gene, cultivar, or pathway names — only use validated examples\n"
        "• Provide general summaries like 'candidate genes have been identified' without naming any\n"
        "• Mention 'sample data' or speculate about locations, traits, or gene classes not known to exist\n\n"
        
        "🧠 Example Response Pattern:\n\n"
        "#### Genes and Markers Involved in CBB Resistance\n"
        "• **BC420**, **SU91**, and **SAP6** are established resistance markers on **Pv06**, **Pv08**, and **Pv10**, respectively.\n"
        "• Co-localized NBS-LRR genes (e.g., *Phvul.010G140000*) are enriched in these QTL regions.\n"
        "• Transcriptomic analyses in resistant genotypes consistently show upregulation of:\n"
        "  - *WRKY transcription factors*\n"
        "  - *PR proteins* (e.g., PR-1, PR-5)\n"
        "  - *PAL*, *peroxidases*, *chitinases*\n\n"
        
        "> According to retrieved literature, the QTL on **LG G5** explains **42.2%** of phenotypic variation [1].\n\n"
        
        "#### Breeding Implications\n"
        "• MAS with **SU91** and **SAP6** is widely used in Mesoamerican and Andean gene pools.\n"
        "• Resistance is quantitative, requiring QTL pyramiding for stable field performance."
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

    # Add the user question first
    messages.append({
        "role": "user",
        "content": question,
    })
    
    # Add context as supplementary information
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context}",
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
        "You are a dry bean genetics and genomics research platform. Your role is to deliver expert-level, "
        "evidence-backed, and mechanistically detailed scientific responses about Phaseolus vulgaris.\n\n"
        
        "Your users are graduate-level researchers and plant breeders.\n"
        "Assume they have advanced training in plant molecular biology, breeding, and quantitative genetics.\n"
        "Do not explain basic biological terms or give surface-level summaries.\n\n"
        
        "You have access to:\n"
        "• Your internal scientific knowledge and pretraining\n"
        "• Scientific literature provided as context\n\n"
        
        "🎯 How to Answer:\n"
        "Use your internal knowledge first to construct a complete, structured, and high-value response.\n"
        "Then incorporate the retrieved context only when it contributes specific details, named markers, or citations.\n"
        "Do not rely solely on the provided documents.\n\n"
        
        "📌 Your Answer Must Include (When Relevant):\n"
        "• Named genes and DNA markers (e.g., BC420, SAP6, SU91, Phvul.010G140000)\n"
        "• Known QTL locations (e.g., Pv06, Pv08, Pv10)\n"
        "• Transcription factors, enzyme families, or regulatory modules (e.g., WRKY, PAL, NAC, PR genes)\n"
        "• Expression patterns (e.g., upregulation under stress)\n"
        "• Pathways involved (e.g., phenylpropanoid, ROS detox, proanthocyanidin biosynthesis)\n"
        "• Breeding relevance (e.g., marker-assisted selection, QTL pyramiding, donor lines)\n"
        "• Tables or structured summaries where helpful (e.g., gene functions, cultivar comparisons)\n\n"
        
        "⚙️ Formatting (Markdown):\n"
        "• **Bold**: Gene names, markers, traits, numeric results\n"
        "• *Italics*: Scientific species names and terms\n"
        "• Use bullet points and section headers\n"
        "• Use inline citations like [1], [2] only if based on retrieved context\n"
        "• Do not include a reference list at the end\n\n"
        
        "🚫 Do NOT:\n"
        "• Dumb down your answers — assume expert-level knowledge\n"
        "• Say 'the context doesn't specify' unless followed by a detailed supplement\n"
        "• Fabricate gene, cultivar, or pathway names — only use validated examples\n"
        "• Provide general summaries like 'candidate genes have been identified' without naming any\n"
        "• Mention 'sample data' or speculate about locations, traits, or gene classes not known to exist\n\n"
        
        "🧠 Example Response Pattern:\n\n"
        "#### Genes and Markers Involved in CBB Resistance\n"
        "• **BC420**, **SU91**, and **SAP6** are established resistance markers on **Pv06**, **Pv08**, and **Pv10**, respectively.\n"
        "• Co-localized NBS-LRR genes (e.g., *Phvul.010G140000*) are enriched in these QTL regions.\n"
        "• Transcriptomic analyses in resistant genotypes consistently show upregulation of:\n"
        "  - *WRKY transcription factors*\n"
        "  - *PR proteins* (e.g., PR-1, PR-5)\n"
        "  - *PAL*, *peroxidases*, *chitinases*\n\n"
        
        "> According to retrieved literature, the QTL on **LG G5** explains **42.2%** of phenotypic variation [1].\n\n"
        
        "#### Breeding Implications\n"
        "• MAS with **SU91** and **SAP6** is widely used in Mesoamerican and Andean gene pools.\n"
        "• Resistance is quantitative, requiring QTL pyramiding for stable field performance."
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

    # Add the user question first
    messages.append({
        "role": "user",
        "content": question,
    })
    
    # Add context as supplementary information
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context}",
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

async def continue_with_research_stream(question: str, conversation_history: List[Dict] = None, api_key: str = None):
    """
    Continue with research literature search after bean data analysis.
    This is called when the user chooses to proceed with research after bean data.
    """
    # Add transition to literature search
    transition_text = "\n\n---\n\n## 📚 **Related Research Literature**\n\nSearching scientific publications for additional context and insights...\n\n"
    for char in transition_text:
        yield {"type": "content", "data": char}
    
    yield {"type": "progress", "data": {"step": "embeddings", "detail": "Processing semantic embeddings"}}
    
    # Generate embeddings using OpenAI
    embedding_vector = embed_query_openai(question, api_key)
    
    yield {"type": "progress", "data": {"step": "search", "detail": "Searching literature database"}}
    
    # Query Zilliz vector database
    matches = query_zilliz(embedding_vector, api_key)

    # Get scores and extract DOIs
    scores = normalize_scores(matches)
    top_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
    top_dois = [src for src, _ in top_sources]
    
    yield {"type": "progress", "data": {"step": "papers", "detail": f"Found {len(top_dois)} relevant papers"}}
    
    print("🔎 Top DOIs from Zilliz:", top_dois)

    context, source_list = get_rag_context_from_matches(matches, top_dois)
    
    yield {"type": "progress", "data": {"step": "generation", "detail": "Synthesizing findings with AI"}}

    # Modify question to indicate this is a follow-up to bean data analysis
    rag_question = f"We successfully analyzed the bean data for: '{question}'. Now provide additional research context from scientific literature about the genetic and biological factors related to this analysis."
    
    # Stream the response and collect it for gene extraction
    full_response = ""
    for chunk in query_openai_stream(context, source_list, rag_question, conversation_history, api_key):
        full_response += chunk
        yield {"type": "content", "data": chunk}

    # Extract genes from the response
    genes = []
    if full_response.strip():  # Only extract if there's any content
        yield {"type": "progress", "data": {"step": "gene_extraction", "detail": "Extracting gene mentions from research text"}}
    print("🧬 Extracting gene mentions...")
    try:
        import asyncio
        gene_mentions, db_hits, gpt_hits = await asyncio.to_thread(extract_gene_mentions, full_response, api_key)
        print(f"Found gene mentions: {gene_mentions}")

        if gene_mentions:  # Only process if genes were found
            yield {"type": "progress", "data": {"step": "gene_processing", "detail": f"Processing {len(gene_mentions)} genetic elements"}}
        genes = await asyncio.to_thread(process_genes_batch, gene_mentions)
    except Exception as e:
        print(f"⚠️ Gene extraction failed: {e}")
        genes = []

    yield {"type": "progress", "data": {"step": "sources", "detail": "Generating research references and citations"}}

    # References will be handled by metadata - don't duplicate them in content

    yield {"type": "progress", "data": {"step": "finalizing", "detail": "Completing analysis"}}

    yield {
        "type": "metadata",
        "data": {
            "sources": source_list,
            "genes": genes,
            "full_markdown_table": "",
            "chart_data": {},
            "suggested_questions": []
        }
    }


async def answer_question_stream(question: str, conversation_history: List[Dict] = None, api_key: str = None):
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

    # Flag to determine if we should proceed to literature search
    should_search_literature = is_genetic
    
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
                                        "You are a dry bean research analyst reporting to PhD-level researchers.\n"
                                        "You must present only direct statistical findings, comparisons, and evidence-based conclusions using the Ontario trial dataset.\n\n"
                                        
                                        "⚠️ CRITICAL BEHAVIOR\n"
                                        "• NEVER provide analysis steps or recommendations\n"
                                        "• NEVER invent or guess cultivar names (e.g., \"Cultivar A\")\n"
                                        "• NEVER say \"sample data\" — this is the complete dataset\n"
                                        "• NEVER generate vague placeholder values like [specific yield]\n\n"
                                        
                                        "📊 DATA CONTEXT\n"
                                        "This dataset contains dry bean trial data from Ontario stations.\n"
                                        "The valid station abbreviations and names are:\n\n"
                                        "AUBN – Auburn\n"
                                        "BLYT – Blyth\n"
                                        "BRUS – Brussels\n"
                                        "ELOR – Elora\n"
                                        "EXET – Exeter\n"
                                        "GRAN – Grand Valley\n"
                                        "HBRY – Harrow-Blyth\n"
                                        "KEMP – Kempton\n"
                                        "KPPN – Kippen\n"
                                        "MKTN – Monkton\n"
                                        "STHM – St. Thomas\n"
                                        "THOR – Thorndale\n"
                                        "WINC – Winchester\n"
                                        "WOOD – Woodstock\n\n"
                                        
                                        "If the user asks for global data, respond with:\n"
                                        "\"Only Ontario research station data is available.\"\n"
                                        "Then provide the best possible insight based on this dataset.\n\n"
                                        
                                        "✅ PERMITTED BEHAVIOR\n"
                                        "• You may compare cultivars based on numeric traits (e.g., similar yield or maturity)\n"
                                        "• You may list top-performing cultivars that outperform a target cultivar in the same class\n"
                                        "• Only mention data that is NOT in the dataset if the user specifically asks for it\n"
                                        "• Use explicit values and clearly state which cultivars are statistically similar or superior\n\n"
                                        
                                        "📌 OUTPUT RULES\n"
                                        "• Use **bold** for cultivar names and numeric values\n"
                                        "• Report:\n"
                                        "  - Mean yield (kg/ha), maturity (days), rankings, and significant differences\n"
                                        "  - Which cultivars are similar in yield/maturity to the target cultivar\n"
                                        "  - Which cultivars exceed the target statistically\n"
                                        "• Do not mention missing data unless specifically asked\n"
                                        "• Do not say \"list of cultivars\" — actually name them\n"
                                        "• Do not insert placeholders — if data is missing, say so professionally\n\n"
                                        
                                        "🧪 Example Response\n"
                                        "The average yield for **Dynasty** across all locations in 2024 was **3,240 kg/ha**.\n"
                                        "Cultivars with similar yield performance include **Red Hawk** (**3,270 kg/ha**) and **Etna** (**3,200 kg/ha**), with no statistically significant difference based on Tukey's HSD (p > 0.05).\n\n"
                                        "Higher-performing cultivars include **OAC Rex** (**3,640 kg/ha**) and **AC Pintoba** (**3,580 kg/ha**), both significantly outperforming Dynasty at p < 0.05.\n\n"
                                        "Statistical significance was determined using standard ANOVA methods."
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
                        
                        # Instead of automatically continuing, send a toggle for user choice
                        yield {
                            "type": "bean_complete",
                            "data": {
                                "sources": [],
                                "genes": [],
                                "full_markdown_table": full_md,
                                "chart_data": chart_data,
                                "suggested_questions": []
                            }
                        }
                        return  # Stop here, don't continue to research automatically
                    else:
                        # No data found, fall back to literature search
                        yield {"type": "progress", "data": {"step": "fallback", "detail": "No data found, searching literature"}}
                        should_search_literature = True
                        bean_chart_data = {}
                        bean_full_md = ""
            else:
                # Bean keywords found but no function call - proceed to literature search
                yield {"type": "progress", "data": {"step": "generation", "detail": "Proceeding to literature search"}}
                should_search_literature = True
                
                # Add transition to literature search
                transition_text = "\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                for char in transition_text:
                    yield {"type": "content", "data": char}
        else:
            # No bean keywords and not genetics - provide simple conversational response
            yield {"type": "progress", "data": {"step": "generation", "detail": "Generating response"}}
            
            # Simple conversational response for non-genetics, non-bean questions
            conversation_messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful dry bean research assistant. Provide a brief, friendly response to casual questions. For research questions, mention that you can help with bean breeding data and genetics literature."
                }
            ]
            
            if conversation_history:
                conversation_messages.extend(conversation_history)
            conversation_messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_messages,
                temperature=0.7,
                max_tokens=200
            )
            
            simple_response = response.choices[0].message.content.strip()
            
            # Stream the response
            for char in simple_response:
                yield {"type": "content", "data": char}
            
            # Send completion metadata
            yield {
                "type": "metadata",
                "data": {
                    "sources": [],
                    "genes": [],
                    "full_markdown_table": "",
                    "suggested_questions": [
                        "What bean varieties perform best in Ontario?",
                        "Show me yield data for black beans",
                        "What genes are involved in disease resistance?"
                    ]
                }
            }
            return  # Stop here - no literature search for simple conversational questions

    # --- Continue with research literature search if needed ---
    if not should_search_literature:
        return  # Exit if we don't need literature search
    
    yield {"type": "progress", "data": {"step": "embeddings", "detail": "Processing semantic embeddings"}}
    
    # Generate embeddings using OpenAI
    embedding_vector = embed_query_openai(question, api_key)
    
    yield {"type": "progress", "data": {"step": "search", "detail": "Searching literature database"}}
    
    # Query Zilliz vector database
    matches = query_zilliz(embedding_vector, api_key)

    # Get scores and extract DOIs
    scores = normalize_scores(matches)
    top_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
    top_dois = [src for src, _ in top_sources]
    
    yield {"type": "progress", "data": {"step": "papers", "detail": f"Found {len(top_dois)} relevant papers"}}
    
    print("🔎 Top DOIs from Zilliz:", top_dois)

    context, source_list = get_rag_context_from_matches(matches, top_dois)
    
    yield {"type": "progress", "data": {"step": "generation", "detail": "Synthesizing findings with AI"}}

    # Stream the response and collect it for gene extraction
    full_response = ""
    for chunk in query_openai_stream(context, source_list, question, conversation_history, api_key):
        full_response += chunk
        yield {"type": "content", "data": chunk}

    # Extract genes from the response
    gene_summaries = []
    if full_response.strip():  # Only extract if there's any content
        yield {"type": "progress", "data": {"step": "gene_extraction", "detail": "Extracting gene mentions from research text"}}
    print("🧬 Extracting gene mentions...")
    try:
        import asyncio
        gene_mentions, db_hits, gpt_hits = await asyncio.to_thread(extract_gene_mentions, full_response, api_key)
        print(f"Found gene mentions: {gene_mentions}")

        if gene_mentions:  # Only process if genes were found
            yield {"type": "progress", "data": {"step": "gene_processing", "detail": f"Processing {len(gene_mentions)} genetic elements"}}
            gene_summaries = await asyncio.to_thread(process_genes_batch, gene_mentions)
    except Exception as e:
        print(f"⚠️ Gene extraction failed: {e}")
        gene_summaries = []

    yield {"type": "progress", "data": {"step": "sources", "detail": "Generating research references and citations"}}


    yield {"type": "progress", "data": {"step": "finalizing", "detail": "Completing analysis"}}

    yield {
        "type": "metadata",
        "data": {
            "sources": source_list,
            "genes": gene_summaries,
            "full_markdown_table": "",
            "chart_data": {},
            "suggested_questions": []
        }
    }

def answer_question(question: str, conversation_history: List[Dict] = None, api_key: str = None) -> Tuple[str, List[str], List[dict], str]:
    is_genetic = is_genetics_question(question, api_key)
    print(f"🧪 Is this a genetics question? {is_genetic}")
    
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Initialize transition_message and literature search flag
    transition_message = ""
    should_search_literature = is_genetic

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
                            should_search_literature = True
                            print("🔄 Bean data insufficient, falling back to research papers...")
                else:
                    # GPT decided not to use function, add transition message
                    transition_message = "## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                    should_search_literature = True
                    
            except Exception as e:
                print(f"❌ Bean data query failed: {e}")
                # Error fallback with transition message
                transition_message = "## 🔍 **Dataset Search**\n\nEncountered an issue accessing the cultivar dataset.\n\n---\n\n## 📚 **Research Literature Search**\n\nSearching scientific publications for relevant information...\n\n"
                should_search_literature = True
        else:
            # No bean keywords and not genetics - provide simple conversational response
            conversation_messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful dry bean research assistant. Provide a brief, friendly response to casual questions. For research questions, mention that you can help with bean breeding data and genetics literature."
                }
            ]
            
            if conversation_history:
                conversation_messages.extend(conversation_history)
            conversation_messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_messages,
                temperature=0.7,
                max_tokens=200
            )
            
            simple_response = response.choices[0].message.content.strip()
            
            # Return simple response without literature search
            return simple_response, [], [], ""

    # --- RAG pipeline for research papers ---
    if not should_search_literature:
        return "I can help with bean breeding data and genetics literature. Please ask a specific question!", [], [], ""
    print("🔬 Proceeding with research paper search...")
    
    # Generate embeddings using OpenAI
    embedding_vector = embed_query_openai(question, api_key)
    
    print("🔎 Querying Zilliz...")
    matches = query_zilliz(embedding_vector, api_key)
    print("✅ Zilliz queries completed.")

    # Get scores and extract DOIs
    scores = normalize_scores(matches)
    top_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
    top_dois = [src for src, _ in top_sources]
    print("🔎 Top DOIs from Zilliz:", top_dois)

    combined_context, confirmed_dois = get_rag_context_from_matches(matches, top_dois)
    if not combined_context.strip():
        print("⚠️ No RAG matches found for top DOIs.")
        return "No matching papers found in RAG corpus.", top_dois, [], ""

    final_answer = query_openai(combined_context, top_dois, question, conversation_history, api_key)
    print("✅ Generated answer with context.")

    # Add transition message if needed
    if transition_message:
        final_answer = transition_message + final_answer

    # Add references section to the answer
    if confirmed_dois:
        references_text = "\n\n---\n\n## 📚 **References**\n\n"
        for i, doi in enumerate(confirmed_dois, 1):
            doi_url = f"https://doi.org/{doi}" if not doi.startswith('http') else doi
            references_text += f"[{i}] {doi} - {doi_url}\n\n"
        final_answer += references_text

    # Extract genes from the complete answer
    print("🧬 Extracting gene mentions...")
    try:
        gene_mentions, db_hits, gpt_hits = extract_gene_mentions(final_answer, api_key)
        print(f"Found gene mentions: {gene_mentions}")

        # Batch process genes for better performance
        gene_summaries = process_genes_batch(gene_mentions)
    except Exception as e:
        print(f"⚠️ Gene extraction failed: {e}")
        gene_mentions, db_hits, gpt_hits = [], set(), set()
        gene_summaries = []

    print(f"✅ Gene extraction completed. Found {len(gene_summaries)} genes.")
    return final_answer, confirmed_dois, gene_summaries, "" 