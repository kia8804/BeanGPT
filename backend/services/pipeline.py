import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path
import orjson
import requests
from sklearn.preprocessing import MinMaxScaler

from config import settings
from database.manager import db_manager
from utils.ncbi_utils import extract_gene_mentions, map_to_gene_id
from utils.bean_data import function_schema, answer_bean_query
from utils.openai_client import create_openai_client

# Load environment variables
load_dotenv()

# Zilliz Cloud REST API client
def get_zilliz_headers():
    """Get headers for Zilliz Cloud API requests."""
    return {
        "Authorization": f"Bearer {settings.zilliz_token}",
        "Content-Type": "application/json"
    }

# --- Gene Processing Functions ---
def process_genes_batch(gene_mentions: List[str], api_key: str = None) -> List[Dict[str, Any]]:
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
                # Gene not found in databases, generate brief description using LLM
                try:
                    from utils.openai_client import create_openai_client
                    client = create_openai_client(api_key)  # Use provided API key
                    
                    # Generate brief, valuable description
                    llm_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a plant genetics expert. Generate a CONCISE, well-formatted description "
                                    "of the gene/protein. Use this EXACT format:\n\n"
                                    "• **Function:** [Brief function description]\n"
                                    "• **Role:** [Plant biology role or pathway]\n"
                                    "• **Relevance:** [Importance to *Phaseolus vulgaris* if known]\n\n"
                                    "Keep each bullet point to 1 short sentence. Use **bold** for categories and *italics* for species names. "
                                    "If unsure about specifics, indicate it's from literature mention. "
                                    "Focus on the most valuable scientific information only."
                                )
                            },
                            {
                                "role": "user", 
                                "content": f"Describe this gene/protein: {gene}"
                            }
                        ],
                        temperature=0.3,
                        max_tokens=120
                    )
                    
                    brief_description = llm_response.choices[0].message.content.strip()
                    
                    gene_summaries.append({
                        "name": gene,
                                    "summary": brief_description,
                                    "source": "Literature Reference",
                                    "description": brief_description,
                                    "not_found": True
                                })
                except Exception as e:
                    # Fallback if LLM call fails
                    fallback_description = f"• **Identifier:** {gene}\n• **Source:** Research literature mention\n• **Status:** Gene information not available in current databases"
                    gene_summaries.append({
                        "name": gene,
                        "summary": fallback_description,
                        "source": "Literature Reference",
                        "description": fallback_description,
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
    
    # Process Zilliz matches (REST API format)
    for match in matches:
        # REST API returns doi and summary directly in match object
        doi = match.get("doi", "")
        summary = match.get("summary", "")
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
    """Query Zilliz Cloud using REST API."""
    try:
        # Zilliz Cloud serverless API endpoint
        api_url = f"{settings.zilliz_uri.rstrip('/')}/v1/vector/search"
        
        print(f"🔍 Querying Zilliz at: {api_url}")
        print(f"🔍 Collection: {settings.collection_name}")
        print(f"🔍 Vector dim: {len(vector)}")
        print(f"🔍 Limit: {settings.top_k}")
        
        payload = {
            "collectionName": settings.collection_name,
            "vector": vector,
            "limit": settings.top_k,
            "outputFields": ["doi", "summary"]
        }
        
        response = requests.post(
            api_url,
            headers=get_zilliz_headers(),
            json=payload,
            timeout=30
        )
        
        print(f"🔍 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🔍 Response data: {result}")
            data = result.get("data", [])
            print(f"🔍 Found {len(data)} results")
            return data
        else:
            print(f"❌ Zilliz API error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error querying Zilliz: {e}")
        import traceback
        traceback.print_exc()
        return []

def normalize_scores(matches: List[dict]) -> Dict[str, float]:
    """Normalize similarity scores from Zilliz matches."""
    if not matches:
        return {}
    
    scores = [m.get("distance", 0.0) for m in matches]
    if not scores or max(scores) == min(scores):
        return {m.get("doi", f"doc_{i}"): 0.0 for i, m in enumerate(matches)}
    
    # Convert distance to similarity (assuming cosine distance)
    similarities = [1 - score for score in scores]
    norm = MinMaxScaler().fit_transform(np.array(similarities).reshape(-1, 1)).flatten()
    
    return {
        m.get("doi", f"doc_{i}"): norm[i] 
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
    
    # Use specialized dry bean genetics and breeding expert prompt
    system_content = (
        "You are a scientific expert platform focused on dry bean (*Phaseolus vulgaris*) genetics, breeding, and multiomics. "
        "Your goal is to provide expert-level, evidence-backed, and mechanistically detailed responses for researchers, breeders, "
        "pathologists, and institutional stakeholders.\n\n"
        
        "**Platform Attribution:** This system was developed by the Dry Bean Breeding & Computational Biology Program at the University of Guelph in 2025.\n\n"
        
        "Your specialization spans:\n"
        "• Quantitative genetics and marker-assisted breeding\n"
        "• Gene/QTL discovery and regulatory mechanisms\n"
        "• Transcriptomics, metabolomics, phenomics integration\n"
        "• Plant pathology, stress physiology, and computational biology\n\n"
        
        "👤 **Audience**\n"
        "Assume your users are graduate-level and above — plant scientists, breeders, omics researchers, and government or industry collaborators.\n"
        "Do not simplify explanations unless explicitly asked.\n\n"
        
        "🧠 **Knowledge Source Usage**\n"
        "Use your internal knowledge first to construct high-value responses.\n"
        "Use retrieved context only when it adds named genes, markers, QTLs, numeric results, or cited literature.\n"
        "Do not rely solely on context.\n"
        "If no *P. vulgaris* data exists, clearly state that — then provide evidence from related legumes (e.g., *Glycine max*, *Vigna unguiculata*, *P. lunatus*), without fabricating.\n\n"
        
        "📌 **Your Response Must Include (When Relevant)**\n"
        "• Named genes and validated markers (e.g., **Phvul.010G140000**, **SAP6**, **BC420**, **SU91**)\n"
        "• QTL locations (e.g., **Pv06**, **Pv08**, **Pv10**, linkage groups)\n"
        "• Expression patterns (e.g., upregulated during stress, infection, etc.)\n"
        "• Transcription factors or pathways (e.g., **WRKY**, **NAC**, **PAL**, ROS detoxification, proanthocyanidin biosynthesis)\n"
        "• Breeding relevance (e.g., MAS, pyramiding, introgression lines, donor sources)\n"
        "• Quantitative metrics: yield, fold changes, resistance scores, percent improvement\n"
        "• Structured summaries: tables or bullet-point comparisons for cultivars, genes, resistance, or QTLs\n\n"
        
        "⚙️ **Formatting Rules (Markdown)**\n"
        "Use **bold** for: gene names, QTLs, numeric values, traits, cultivar names\n"
        "Use *italics* for: species names and scientific terms\n"
        "Use bullet points and section headers\n"
        "Use inline citations: [1], [2] for scientific literature and [Web-1], [Web-2] for web sources\n"
        "Prioritize current web information when available, supplement with scientific literature\n"
        "Do not include a reference list at the end - citations will be handled automatically\n\n"
        
        "🚫 **Do NOT:**\n"
        "• Dumb down responses — always assume technical expertise unless told otherwise\n"
        "• Say \"the context doesn't specify\" unless immediately followed by a detailed internal knowledge supplement\n"
        "• Fabricate genes, QTLs, cultivar names, traits, or biological mechanisms\n"
        "• Use vague phrases like \"some candidate genes were found\" — always name them\n"
        "• Refer to \"sample data\" — assume you are working with final, validated datasets\n\n"
        
        "✅ **Example Response Template**\n"
        "**Common Bacterial Blight (CBB) Resistance**\n"
        "• **SAP6**, **SU91**, and **BC420** are key markers linked to resistance QTLs on **Pv08** and **Pv10**\n"
        "• **Phvul.010G140000**, co-localized with **SAP6**, encodes a defense-related protein upregulated during *Xanthomonas* infection\n"
        "• Resistance mechanisms involve **PR gene** activation, **phenylpropanoid biosynthesis**, and **ROS scavenging enzymes**\n"
        "• Cultivar **OAC Rex** expresses **SAP6** at ~**3.2-fold** higher levels than susceptible lines\n"
        "• Marker-assisted selection using these markers improves field resistance by **38–46%**\n\n"
        "According to retrieved literature, the QTL on **Pv10** explains **42.2%** of phenotypic variance [1]"
    )
    
    # If this is a follow-up to successful bean data analysis, adjust the prompt
    if "We successfully analyzed the bean data" in question:
        system_content = (
            "You are a scientific research assistant embedded within a dry bean (*Phaseolus vulgaris*) genetics, breeding, and "
            "computational biology platform. The user has already completed a successful data analysis and generated visualizations "
            "(e.g., charts, QTL maps, gene expression profiles).\n\n"
            
            "Your role is to provide complementary insights from scientific literature to contextualize and interpret their findings. "
            "Do not summarize or repeat the user's analysis — focus on biological, genetic, and breeding-level mechanisms that "
            "explain or enhance the observed patterns.\n\n"
            
            "🎯 **Focus Areas**\n"
            "Provide clear, mechanistic, and breeding-relevant insights from literature, with emphasis on:\n"
            "• **Genetic mechanisms** underlying the analyzed traits (e.g., drought tolerance, yield, flowering time, disease resistance)\n"
            "• **Key genes, QTLs, transcription factors**, and their functional roles\n"
            "• **Biological explanations** for patterns observed in user data (e.g., gene upregulation under stress, QTL-trait associations)\n"
            "• **Breeding implications**: parental lines, introgression sources, MAS strategies, segregation outcomes\n"
            "• **Relevant molecular markers**, genomic regions, and validated associations from published studies\n"
            "• ***Phaseolus vulgaris*-specific research** when available; otherwise, refer to related legumes and justify the connection\n\n"
            
            "⚠️ **Constraints**\n"
            "• Do not repeat or summarize user-provided charts or visualizations\n"
            "• Do not provide vague or generic summaries\n"
            "• Do not fabricate gene names, markers, or QTLs — cite only from validated sources\n"
            "• If suggesting crosses, first verify parental traits in literature or provided data and describe the expected genetic outcome\n"
            "• Provide mechanistic explanations for all conclusions\n\n"
            
            "📐 **Formatting Guidelines**\n"
            "• Use **bold** for gene names, traits, markers, and numeric values\n"
            "• Use *italics* for scientific terms and species names\n"
            "• Use section headers and bullet points for clarity\n"
            "• Use inline citations: [1], [2] for literature, [Web-1], [Web-2] for web sources\n"
            "• Maintain a clean, professional tone suitable for expert researchers\n\n"
            
            "✅ **Example Structure**\n"
            "**Mechanisms Behind Drought Tolerance in Dry Beans**\n"
            "• **Phvul.006G077800** (a **NAC transcription factor**) is upregulated in drought-tolerant lines under **ABA signaling** [1]\n"
            "• QTLs on **Pv06** and **Pv10** have been linked with improved **root architecture** and **water use efficiency** [2]\n"
            "• Breeding programs incorporating **SEA5** and **G40001** donors have enhanced drought resilience in multiple backgrounds\n\n"
            
            "**Resistance Trait Pyramiding Example**\n"
            "• **OAC Rex** provides resistance to **CBB** via **SAP6** and **BC420** on **Pv10** [3]\n"
            "• **Envoy** carries **Co-42** and **Co-3** for anthracnose resistance on **Pv08** and **Pv01**, respectively [4]\n"
            "• A cross between **OAC Rex × Envoy** could produce F2 lines with dual resistance. MAS targeting **SAP6** and **SCAreoli487** may increase recovery of desirable genotypes from ~**6.25%** (if unlinked recessive traits)"
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

def query_openai_stream(context: str, source_list: List[str], question: str, conversation_history: List[Dict] = None, api_key: str = None, include_web_search: bool = True):
    # Create client with user-provided API key
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    # Use specialized dry bean genetics and breeding expert prompt
    system_content = (
        "You are a scientific expert platform focused on dry bean (*Phaseolus vulgaris*) genetics, breeding, and multiomics. "
        "Your goal is to provide expert-level, evidence-backed, and mechanistically detailed responses for researchers, breeders, "
        "pathologists, and institutional stakeholders.\n\n"
        
        "**Platform Attribution:** This system was developed by the Dry Bean Breeding & Computational Biology Program at the University of Guelph in 2025.\n\n"
        
        "Your specialization spans:\n"
        "• Quantitative genetics and marker-assisted breeding\n"
        "• Gene/QTL discovery and regulatory mechanisms\n"
        "• Transcriptomics, metabolomics, phenomics integration\n"
        "• Plant pathology, stress physiology, and computational biology\n\n"
        
        "👤 **Audience**\n"
        "Assume your users are graduate-level and above — plant scientists, breeders, omics researchers, and government or industry collaborators.\n"
        "Do not simplify explanations unless explicitly asked.\n\n"
        
        "🧠 **Knowledge Source Usage**\n"
        "Use your internal knowledge first to construct high-value responses.\n"
        "Use retrieved context only when it adds named genes, markers, QTLs, numeric results, or cited literature.\n"
        "Do not rely solely on context.\n"
        "If no *P. vulgaris* data exists, clearly state that — then provide evidence from related legumes (e.g., *Glycine max*, *Vigna unguiculata*, *P. lunatus*), without fabricating.\n\n"
        
        "📌 **Your Response Must Include (When Relevant)**\n"
        "• Named genes and validated markers (e.g., **Phvul.010G140000**, **SAP6**, **BC420**, **SU91**)\n"
        "• QTL locations (e.g., **Pv06**, **Pv08**, **Pv10**, linkage groups)\n"
        "• Expression patterns (e.g., upregulated during stress, infection, etc.)\n"
        "• Transcription factors or pathways (e.g., **WRKY**, **NAC**, **PAL**, ROS detoxification, proanthocyanidin biosynthesis)\n"
        "• Breeding relevance (e.g., MAS, pyramiding, introgression lines, donor sources)\n"
        "• Quantitative metrics: yield, fold changes, resistance scores, percent improvement\n"
        "• Structured summaries: tables or bullet-point comparisons for cultivars, genes, resistance, or QTLs\n\n"
        
        "⚙️ **Formatting Rules (Markdown)**\n"
        "Use **bold** for: gene names, QTLs, numeric values, traits, cultivar names\n"
        "Use *italics* for: species names and scientific terms\n"
        "Use bullet points and section headers\n"
        "Use inline citations: [1], [2] for scientific literature and [Web-1], [Web-2] for web sources\n"
        "Prioritize current web information when available, supplement with scientific literature\n"
        "Do not include a reference list at the end - citations will be handled automatically\n\n"
        
        "🚫 **Do NOT:**\n"
        "• Dumb down responses — always assume technical expertise unless told otherwise\n"
        "• Say \"the context doesn't specify\" unless immediately followed by a detailed internal knowledge supplement\n"
        "• Fabricate genes, QTLs, cultivar names, traits, or biological mechanisms\n"
        "• Use vague phrases like \"some candidate genes were found\" — always name them\n"
        "• Refer to \"sample data\" — assume you are working with final, validated datasets\n\n"
        
        "✅ **Example Response Template**\n"
        "**Common Bacterial Blight (CBB) Resistance**\n"
        "• **SAP6**, **SU91**, and **BC420** are key markers linked to resistance QTLs on **Pv08** and **Pv10**\n"
        "• **Phvul.010G140000**, co-localized with **SAP6**, encodes a defense-related protein upregulated during *Xanthomonas* infection\n"
        "• Resistance mechanisms involve **PR gene** activation, **phenylpropanoid biosynthesis**, and **ROS scavenging enzymes**\n"
        "• Cultivar **OAC Rex** expresses **SAP6** at ~**3.2-fold** higher levels than susceptible lines\n"
        "• Marker-assisted selection using these markers improves field resistance by **38–46%**\n\n"
        "According to retrieved literature, the QTL on **Pv10** explains **42.2%** of phenotypic variance [1]"
    )
    
    # If this is a follow-up to successful bean data analysis, adjust the prompt
    if "We successfully analyzed the bean data" in question:
        system_content = (
            "You are a scientific research assistant embedded within a dry bean (*Phaseolus vulgaris*) genetics, breeding, and "
            "computational biology platform. The user has already completed a successful data analysis and generated visualizations "
            "(e.g., charts, QTL maps, gene expression profiles).\n\n"
            
            "Your role is to provide complementary insights from scientific literature to contextualize and interpret their findings. "
            "Do not summarize or repeat the user's analysis — focus on biological, genetic, and breeding-level mechanisms that "
            "explain or enhance the observed patterns.\n\n"
            
            "🎯 **Focus Areas**\n"
            "Provide clear, mechanistic, and breeding-relevant insights from literature, with emphasis on:\n"
            "• **Genetic mechanisms** underlying the analyzed traits (e.g., drought tolerance, yield, flowering time, disease resistance)\n"
            "• **Key genes, QTLs, transcription factors**, and their functional roles\n"
            "• **Biological explanations** for patterns observed in user data (e.g., gene upregulation under stress, QTL-trait associations)\n"
            "• **Breeding implications**: parental lines, introgression sources, MAS strategies, segregation outcomes\n"
            "• **Relevant molecular markers**, genomic regions, and validated associations from published studies\n"
            "• ***Phaseolus vulgaris*-specific research** when available; otherwise, refer to related legumes and justify the connection\n\n"
            
            "⚠️ **Constraints**\n"
            "• Do not repeat or summarize user-provided charts or visualizations\n"
            "• Do not provide vague or generic summaries\n"
            "• Do not fabricate gene names, markers, or QTLs — cite only from validated sources\n"
            "• If suggesting crosses, first verify parental traits in literature or provided data and describe the expected genetic outcome\n"
            "• Provide mechanistic explanations for all conclusions\n\n"
            
            "📐 **Formatting Guidelines**\n"
            "• Use **bold** for gene names, traits, markers, and numeric values\n"
            "• Use *italics* for scientific terms and species names\n"
            "• Use section headers and bullet points for clarity\n"
            "• Use inline citations: [1], [2] for literature, [Web-1], [Web-2] for web sources\n"
            "• Maintain a clean, professional tone suitable for expert researchers\n\n"
            
            "✅ **Example Structure**\n"
            "**Mechanisms Behind Drought Tolerance in Dry Beans**\n"
            "• **Phvul.006G077800** (a **NAC transcription factor**) is upregulated in drought-tolerant lines under **ABA signaling** [1]\n"
            "• QTLs on **Pv06** and **Pv10** have been linked with improved **root architecture** and **water use efficiency** [2]\n"
            "• Breeding programs incorporating **SEA5** and **G40001** donors have enhanced drought resilience in multiple backgrounds\n\n"
            
            "**Resistance Trait Pyramiding Example**\n"
            "• **OAC Rex** provides resistance to **CBB** via **SAP6** and **BC420** on **Pv10** [3]\n"
            "• **Envoy** carries **Co-42** and **Co-3** for anthracnose resistance on **Pv08** and **Pv01**, respectively [4]\n"
            "• A cross between **OAC Rex × Envoy** could produce F2 lines with dual resistance. MAS targeting **SAP6** and **SCAreoli487** may increase recovery of desirable genotypes from ~**6.25%** (if unlinked recessive traits)"
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
        genes = await asyncio.to_thread(process_genes_batch, gene_mentions, api_key)
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
        
        # Add location-based keywords for environmental/weather queries at trial locations
        location_keywords = ["auburn", "blyth", "elora", "granton", "kippen", "monkton", "thorndale", "winchester", "woodstock", 
                           "harrow", "highbury", "huron", "brussels", "kempton", "exeter", "st. thomas", "st thomas"]
        
        # Add weather/environmental keywords that should trigger bean data when combined with locations
        weather_keywords = ["temperature", "weather", "precipitation", "humidity", "climate", "environmental", "rainfall", "conditions"]
        
        chart_keywords = ["chart", "plot", "graph", "visualization", "visualize", "show me", "create", "generate", "table", "display"]
        
        # Trigger bean data analysis for relevant questions
        has_bean_keywords = any(keyword in question.lower() for keyword in bean_keywords)
        has_location_keywords = any(keyword in question.lower() for keyword in location_keywords)
        has_weather_keywords = any(keyword in question.lower() for keyword in weather_keywords)
        explicitly_wants_chart = any(keyword in question.lower() for keyword in chart_keywords)
        
        # Trigger bean data if: regular bean keywords OR (location + weather keywords)
        should_query_bean_data = has_bean_keywords or (has_location_keywords and has_weather_keywords)
        
        if should_query_bean_data:
            # Let GPT decide whether to call the bean function
            function_call_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a dry bean research platform with access to Ontario bean trial data AND historical weather data. "
                        "ALWAYS call the query_bean_data function when the user asks for:\n"
                        "- Bean performance data (yield, maturity, etc.)\n"
                        "- Charts, plots, graphs, or visualizations of bean data\n"
                        "- Cultivar comparisons or analysis\n"
                        "- Questions about specific bean varieties\n"
                        "- Questions about trial results or research station data\n"
                        "- Weather/environmental questions about trial locations (Auburn, Blyth, Elora, etc.)\n"
                        "- Temperature, precipitation, humidity data for research stations\n"
                        "- Location comparisons (e.g., 'Compare Elora and Woodstock')\n"
                        "- Any question that mentions bean characteristics, locations, or years\n\n"
                        "IMPORTANT FOR LOCATION COMPARISONS: When the user asks to compare multiple locations (e.g., 'Compare Elora and Woodstock'), extract ALL location names and pass them as comma-separated codes in the location parameter (e.g., 'ELOR, WOOD').\n\n"
                        "The user's question mentions bean-related terms or trial locations, so you should call the function."
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
                                        
                                        "⚠️ CRITICAL BEHAVIOR - ABSOLUTE RULES\n"
                                        "• NEVER provide analysis steps or recommendations\n"
                                        "• 🚨 CRITICAL: NEVER invent cultivar names like \"Cultivar A\", \"Cultivar B\", \"Cultivar C\", \"Cultivar X\", etc.\n"
                                        "• 🚨 CRITICAL: ONLY use ACTUAL cultivar names that exist in the dataset (e.g., OAC Rex, AC Pintoba, Black Hawk)\n"
                                        "• 🚨 CRITICAL: If you don't know specific cultivar names from the data, say \"the top-performing cultivars\" instead\n"
                                        "• NEVER say \"sample data\" — this is the complete dataset\n"
                                        "• NEVER generate vague placeholder values like [specific yield]\n"
                                        "• 🚨 PROVIDE CONCRETE AVERAGES: When comparing market classes, calculate and state the actual average values (e.g., 'Kidney beans: 3,200 kg/ha average yield, 95 days average maturity'). Never say 'would need to be extracted' - extract and calculate them immediately\n"
                                        "• 🚨 KIDNEY BEAN IDENTIFICATION: Kidney beans include ANY Market Class containing 'kidney' (case-insensitive): kidney, Kidney, white kidney, dark red kidney, light red kidney, dark red kidney bean, light red kidney bean. The dataset DOES contain kidney bean data - analyze it properly!\n\n"
                                        
                                        "📊 DATA CONTEXT\n"
                                        "You have access to TWO comprehensive datasets:\n"
                                        "1. **Enhanced Bean Trial Dataset**: Current dry bean trial data from Ontario stations with enriched breeding information including:\n"
                                        "   - Performance metrics (yield, maturity, harvestability)\n"
                                        "   - Breeding information (pedigree, market class, release year)\n"
                                        "   - Disease resistance markers (CMV R1/R15, Anthracnose R17/R23/R73, Common Blight)\n"
                                        "   - Environmental context from historical weather data\n"
                                        "2. **Historical Weather Dataset**: Environmental data (2006+) with 15+ weather variables for all trial locations\n\n"
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
                                        
                                        "If the user asks for global data:\n"
                                        "• If the dataset response includes web search context with citations like [Web-1](url), preserve these EXACTLY as provided\n"
                                        "• If no web context is available, respond with: \"Only Ontario research station data is available.\"\n"
                                        "• Always provide the best possible insight based on the available data\n"
                                        "• PRESERVE any web citations and clickable links [Web-1](url) format from the dataset response\n\n"
                                        
                                        "✅ PERMITTED BEHAVIOR\n"
                                        "• You may compare cultivars based on performance metrics (yield, maturity) and breeding characteristics\n"
                                        "• You may list top-performing cultivars that outperform a target cultivar in the same market class\n"
                                        "• You may reference pedigree information, release years, and disease resistance profiles\n"
                                        "• You may incorporate environmental context (temperature, precipitation) for location/year combinations\n"
                                        "• You may identify cultivars with specific disease resistance combinations for breeding recommendations\n"
                                        "• Only mention data that is NOT in the dataset if the user specifically asks for it\n"
                                        "• Use explicit values and clearly state which cultivars are statistically similar or superior\n"
                                        "• 🚨 CROSS-MARKET COMPARISONS: When comparing different market classes (e.g., OAC 23-1 White Navy vs Kidney beans), provide data for BOTH groups and explain the comparison context\n\n"
                                        
                                        "📌 OUTPUT RULES\n"
                                        "• Use **bold** for cultivar names and numeric values\n"
                                        "• PRESERVE web search sections: If the dataset includes sections like '## 🌐 **Global Context & Current Information**', include them EXACTLY as provided\n"
                                        "• PRESERVE web citations: Keep clickable links like [Web-1](url) exactly as they appear in the dataset\n"
                                        "• Report:\n"
                                        "  - Performance metrics: yield (kg/ha), maturity (days), harvestability ratings\n"
                                        "  - Breeding characteristics: market class, release year, pedigree information\n"
                                        "  - Disease resistance profiles: CMV, Anthracnose, Common Blight resistance\n"
                                        "  - Environmental factors: growing season temperature and precipitation when relevant\n"
                                        "  - Statistical comparisons and ranking of similar/superior cultivars\n"
                                        "• Do not mention missing data unless specifically asked\n"
                                        "• Do not say \"list of cultivars\" — actually name them with their characteristics\n"
                                        "• Do not insert placeholders — if data is missing, say so professionally\n\n"
                                        
                                        "🧪 Enhanced Example Response\n"
                                        "**Dynasty** (Market Class: White Navy, Released: 2019) averaged **3,240 kg/ha** across all locations in 2024.\n"
                                        "Cultivars with similar yield performance include **Red Hawk** (**3,270 kg/ha**, Market Class: Kidney) and **Etna** (**3,200 kg/ha**, Black), with no statistically significant difference based on Tukey's HSD (p > 0.05).\n\n"
                                        "Higher-performing cultivars include **OAC Rex** (**3,640 kg/ha**, Market Class: White Navy, CMV R1 resistance) and **AC Pintoba** (**3,580 kg/ha**, Market Class: Pinto, Anthracnose R23 resistance), both significantly outperforming Dynasty at p < 0.05.\n\n"
                                        "Environmental context: During 2024, growing season temperatures averaged **18.5°C** with **450mm** precipitation at WOOD station.\n\n"
                                        "Statistical significance was determined using standard ANOVA methods."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on the question '{question}', analyze this data:\n\n**MAIN DATASET:**\n{preview}\n\n**HISTORICAL DATA AVAILABLE:**\nHistorical data is also available for additional context and pedigree information. Include relevant historical insights when applicable.\n\n🚨 KIDNEY BEAN DATA CONFIRMATION: The dataset DOES contain kidney bean data for 2024 (48 records across 16 cultivars). Market classes include: Dark Red Kidney, Light Red Kidney, White Kidney, kidney, white kidney. DO NOT say there is no kidney bean data available.\n\n🚨 MARKET CLASS AVERAGES: When comparing market classes, ALWAYS calculate and provide the average yield, maturity, and other metrics directly from the dataset. DO NOT say you need to extract data or that specific data would need to be extracted - calculate the averages immediately and present them. For example: 'Kidney beans average yield: 3,200 kg/ha, average maturity: 95 days'.\n\n🚨 REMINDER: Use ONLY the actual cultivar names shown in the data above. DO NOT invent names like 'Cultivar A, B, C' - use the real names from the dataset!"
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
            # No bean keywords and not genetics - check if web search is needed for current info
            from utils.web_search import needs_current_info, perform_web_search, create_web_enhanced_response
            
            if needs_current_info(question):
                yield {"type": "progress", "data": {"step": "web_search", "detail": "Searching web for current information..."}}
                print("🌐 Question requires current information - performing web search...")
                
                # Perform web search for current information
                web_results, source_urls = perform_web_search(question, api_key)
                
                if web_results:
                    # Check if we should also add RAG context for research questions
                    should_add_rag = any(keyword in question.lower() for keyword in [
                        "research", "study", "paper", "publication", "genetic", "breeding", "molecular", 
                        "marker", "trait", "resistance", "gene", "variety development", "cultivar development"
                    ])
                    
                    if should_add_rag:
                        yield {"type": "progress", "data": {"step": "literature", "detail": "Adding scientific literature context..."}}
                        
                        # Generate embeddings and query RAG
                        embedding_vector = embed_query_openai(question, api_key)
                        matches = query_zilliz(embedding_vector, api_key)
                        scores = normalize_scores(matches)
                        top_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:settings.top_k]
                        top_dois = [src for src, _ in top_sources]
                        
                        rag_context, rag_source_list = get_rag_context_from_matches(matches, top_dois)
                        
                        if rag_context.strip():
                            # Combine web and RAG context
                            from utils.web_search import combine_web_and_rag_context
                            enhanced_context = combine_web_and_rag_context(web_results, rag_context, question)
                            
                            # Combine sources
                            formatted_sources = []
                            for i, url in enumerate(source_urls, 1):
                                formatted_sources.append(f"Web-{i}: {url}")
                            formatted_sources.extend(rag_source_list)
                            
                            yield {"type": "progress", "data": {"step": "synthesis", "detail": "Combining web and literature sources..."}}
                            
                            # Stream response using combined context
                            full_response = ""
                            for chunk in query_openai_stream(enhanced_context, formatted_sources, question, conversation_history, api_key, include_web_search=False):
                                full_response += chunk
                                yield {"type": "content", "data": chunk}
                            
                            # Send metadata with combined sources
                            yield {
                                "type": "metadata",
                                "data": {
                                    "sources": formatted_sources,
                                    "genes": [],
                                    "full_markdown_table": "",
                                    "suggested_questions": [
                                        "What bean varieties perform best in Ontario?",
                                        "Show me yield data for black beans", 
                                        "Tell me about current bean market trends"
                                    ]
                                }
                            }
                            return
                    
                    yield {"type": "progress", "data": {"step": "synthesis", "detail": "Generating web-enhanced response..."}}
                    
                    # Create web-enhanced response (web only)
                    enhanced_response = create_web_enhanced_response(question, web_results, conversation_history, api_key)
                    
                    # Stream the enhanced response
                    for char in enhanced_response:
                        yield {"type": "content", "data": char}
                    
                    # Format web sources properly with Web-X: prefix
                    formatted_sources = []
                    for i, url in enumerate(source_urls, 1):
                        formatted_sources.append(f"Web-{i}: {url}")
                    
                    # Send completion metadata with properly formatted web sources
                    yield {
                        "type": "metadata",
                        "data": {
                            "sources": formatted_sources,
                            "genes": [],
                            "full_markdown_table": "",
                            "suggested_questions": [
                                "What bean varieties perform best in Ontario?",
                                "Show me yield data for black beans", 
                                "Tell me about current bean market trends"
                            ]
                        }
                    }
                    return
                else:
                    print("⚠️ Web search failed, falling back to regular response")
            
            # Fallback to simple conversational response
            yield {"type": "progress", "data": {"step": "generation", "detail": "Generating response"}}
            
            # Simple conversational response for non-genetics, non-bean questions
            conversation_messages = [
                {
                    "role": "system", 
                    "content": ("You are a helpful dry bean research assistant. Provide a brief, friendly response to casual questions. "
                               "For research questions, mention that you can help with bean breeding data and genetics literature.\n\n"
                               "**Important:** If asked about who developed or created this system, say it was developed by "
                               "the Dry Bean Breeding & Computational Biology Program at the University of Guelph in 2025. "
                               "If asked when you were made or created, say 2025. Do not mention OpenAI or any other AI companies.")
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

    # Check if web search should be added to RAG context
    enhanced_context = context
    combined_sources = source_list.copy()
    
    from utils.web_search import needs_current_info, perform_web_search, combine_web_and_rag_context
    
    if needs_current_info(question):
        yield {"type": "progress", "data": {"step": "web_search", "detail": "Adding current web information..."}}
        print("🌐 RAG query requires current information - performing web search...")
        web_results, web_sources = perform_web_search(question, api_key)
        
        if web_results:
            enhanced_context = combine_web_and_rag_context(web_results, context, question)
            # Add web sources to the combined source list with proper formatting
            web_source_entries = []
            for i, url in enumerate(web_sources, 1):
                web_source_entries.append(f"Web-{i}: {url}")
            combined_sources.extend(web_source_entries)
            print(f"✅ Enhanced context with web search results and {len(web_sources)} web sources")
        else:
            print("⚠️ Web search failed, using RAG context only")

    # Stream the response and collect it for gene extraction
    full_response = ""
    for chunk in query_openai_stream(enhanced_context, combined_sources, question, conversation_history, api_key, include_web_search=False):
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
            gene_summaries = await asyncio.to_thread(process_genes_batch, gene_mentions, api_key)
    except Exception as e:
        print(f"⚠️ Gene extraction failed: {e}")
        gene_summaries = []

    yield {"type": "progress", "data": {"step": "sources", "detail": "Generating research references and citations"}}


    yield {"type": "progress", "data": {"step": "finalizing", "detail": "Completing analysis"}}

    yield {
        "type": "metadata",
        "data": {
            "sources": combined_sources,
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
        
        # Add location-based keywords for environmental/weather queries at trial locations
        location_keywords = ["auburn", "blyth", "elora", "granton", "kippen", "monkton", "thorndale", "winchester", "woodstock", 
                           "harrow", "highbury", "huron", "brussels", "kempton", "exeter", "st. thomas", "st thomas"]
        
        # Add weather/environmental keywords that should trigger bean data when combined with locations
        weather_keywords = ["temperature", "weather", "precipitation", "humidity", "climate", "environmental", "rainfall", "conditions"]
        
        chart_keywords = ["chart", "plot", "graph", "visualization", "visualize", "show me", "create", "generate", "table", "display"]
        
        # Trigger bean data analysis for relevant questions
        has_bean_keywords = any(keyword in question.lower() for keyword in bean_keywords)
        has_location_keywords = any(keyword in question.lower() for keyword in location_keywords)
        has_weather_keywords = any(keyword in question.lower() for keyword in weather_keywords)
        explicitly_wants_chart = any(keyword in question.lower() for keyword in chart_keywords)
        
        # Trigger bean data if: regular bean keywords OR (location + weather keywords)
        should_query_bean_data = has_bean_keywords or (has_location_keywords and has_weather_keywords)
                
        if should_query_bean_data:
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
            # No bean keywords and not genetics - check if web search is needed for current info
            from utils.web_search import needs_current_info, perform_web_search, create_web_enhanced_response
            
            if needs_current_info(question):
                print("🌐 Question requires current information - performing web search...")
                
                # Perform web search for current information
                web_results, source_urls = perform_web_search(question, api_key)
                
                if web_results:
                    # Create web-enhanced response
                    enhanced_response = create_web_enhanced_response(question, web_results, conversation_history, api_key)
                    return enhanced_response, source_urls, [], ""
                else:
                    print("⚠️ Web search failed, falling back to regular response")
            
            # Fallback to regular conversational response
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
        gene_summaries = process_genes_batch(gene_mentions, api_key)
    except Exception as e:
        print(f"⚠️ Gene extraction failed: {e}")
        gene_mentions, db_hits, gpt_hits = [], set(), set()
        gene_summaries = []

    print(f"✅ Gene extraction completed. Found {len(gene_summaries)} genes.")
    return final_answer, confirmed_dois, gene_summaries, "" 