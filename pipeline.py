import os, re, requests, sys, torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
import orjson
import json

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler

sys.path.append(r"C:\Users\mirka\Documents\Guelph Research\streamlit")
from ncbi_utils import extract_gene_mentions, map_to_gene_id, load_gene_db
from bean_data_module import function_schema, answer_bean_query

sys.stdout.reconfigure(encoding="utf-8")

# --- Config ---
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    "pcsk_3QK1aQ_2ybAXUtxbGHBCeboEwva7W5tXTLQP3hHvvrA6XYX21hrDKCg2y9sDLTemr7v5hy",
)
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "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",
)
BGE_INDEX_NAME = "dry-bean-bge-abstract"
PUBMEDBERT_INDEX_NAME = "dry-bean-pubmedbert-abstract"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOP_K = 5
ALPHA = 0.6

# --- Load Once ---
bge_model = SentenceTransformer(BGE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
pub_model = AutoModel.from_pretrained(PUBMEDBERT_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pub_model.to(device).eval()

# Load gene data
load_gene_db(
    r"C:\Users\mirka\Documents\Guelph Research\NCBI_Filtered_Data_Enriched.xlsx"
)

# Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Load RAG Text from JSONL ---
# RAG_FILE = Path(
#     r"C:\Users\mirka\Documents\Guelph Research\AbstractPineconeMethod\rag_sections_gpt4o.jsonl"
# )

RAG_FILE = Path(r"C:\Users\mirka\Documents\Guelph Research\summaries.jsonl")


def load_rag_text_jsonl(path):
    rag_lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = orjson.loads(line)
            doi = record.get("doi") or record.get("source", "").replace(".pdf", "")
            rag = record.get("summary", "")
            if doi and rag:
                rag_lookup[doi.strip()] = rag.strip()
    return rag_lookup


RAG_LOOKUP = load_rag_text_jsonl(RAG_FILE)


def get_rag_context_from_dois(dois: list[str]) -> Tuple[str, list[str]]:
    context_blocks = []
    confirmed_dois = []

    for i, doi in enumerate(dois, 1):
        if doi in RAG_LOOKUP:
            summary = RAG_LOOKUP[doi]
            context_blocks.append(f"[{i}] Source: {doi}\n{summary}")
            confirmed_dois.append(doi)  # Always push the real DOI

    return "\n\n".join(context_blocks), confirmed_dois


# --- Embedding + Retrieval ---
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


def is_genetics_question(question: str) -> bool:
    prompt = f"""
    Decide if the question is about genetics or molecular biology in dry beans or plants.

    Say true if it's about genes, gene expression, resistance genes, molecular traits, or genetic mapping.
    Say false if it's only about general agriculture, yield, soil, or farming without genetics.
    Only return true or false.

    Question:
    {question}
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return "true" in answer


def query_openai(context: str, source_list: list[str], question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a plant molecular biology assistant. Always prioritize the provided context as your main knowledge source. "
                "Use it to answer scientific questions in detail. Only add background knowledge if it is well-established and clearly complements the context, never as a substitute. "
                "Your answers should be specific, accurate, and well-structured. Use bullet points, paragraph breaks, or sections to organize information clearly. "
                "Cite the sources using bracketed references like [1], [2], and highlight gene/protein names in **bold** or `monospace` format when present. "
                "Avoid vague statements, and never make up gene names, study details, or data. If the context lacks sufficient information, say so directly and explain what is known more generally in the field."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using the context provided. Include the bracketed numbers inline.",
        },
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.2
    )
    print("\n\n==== CONTEXT PASSED TO GPT ====\n")
    print(context)
    print("\n\n==== END OF CONTEXT ====\n")
    return response.choices[0].message.content.strip()


def extract_gene_names_with_summaries(answer_text: str) -> List[dict]:
    prompt = f"""
    From the following text, extract all gene or protein names mentioned.
    For each one, give a 1-2 bullet-point scientific summary explaining:
    - Its function or role (e.g., transcription factor, stress response, etc.)
    - Any known relevance to drought resistance, if applicable

    Be concise, and structure your response like this Python list of dictionaries:

    [
      {{
        "name": "Asr1",
        "summary": "- Encodes a protein involved in abscisic acid response\n- Plays a role in drought stress tolerance"
      }},
      ...
    ]

    Text:
    {answer_text}
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a biomedical gene annotation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    try:
        raw = response.choices[0].message.content.strip()
        return eval(raw) if raw.startswith("[") else []
    except:
        return []


def answer_question(question: str) -> Tuple[str, List[str], List[dict], str]:
    is_genetic = is_genetics_question(question)
    print(f"🧪 Is this a genetics question? {is_genetic}")

    if not is_genetic:
        # Let GPT decide whether to call the bean function
        response = openai_client.chat.completions.create(
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
            call = choice.message.function_call
            if call.name == "query_bean_data":
                args = json.loads(call.arguments)
                preview, full_md = answer_bean_query(args)
                return preview, [], [], full_md

        return response.choices[0].message.content.strip(), [], [], ""

    # --- Genetics question flow ---
    bge_vec = bge_model.encode(question, normalize_embeddings=True).tolist()
    pub_vec = embed_query_pubmedbert(question)
    bge_res = query_pinecone(BGE_INDEX_NAME, bge_vec)
    pub_res = query_pinecone(PUBMEDBERT_INDEX_NAME, pub_vec)

    bge_scores = normalize_scores(bge_res["matches"])
    pub_scores = normalize_scores(pub_res["matches"])
    combined_scores = combine_scores(bge_scores, pub_scores)

    top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
        :TOP_K
    ]
    top_dois = [src for src, _ in top_sources]
    print("🔎 Top DOIs from Pinecone:", top_dois)
    print("🗂 RAG_LOOKUP keys (sample):", list(RAG_LOOKUP.keys())[:5])

    combined_context, confirmed_dois = get_rag_context_from_dois(top_dois)
    if not combined_context.strip():
        print("⚠️ No RAG matches found for top DOIs.")
        return "No matching papers found in RAG corpus.", top_dois, [], ""

    final_answer = query_openai(combined_context, top_dois, question)

    mentions = extract_gene_mentions(final_answer)
    mapped_genes = [map_to_gene_id(m) for m in mentions]
    gene_summaries = [
        {
            "name": gene["mention"],
            "summary": f"- Matched to `{gene['matched_text']}` ({gene['matched_type']})\n"
            f"- NCBI GeneID: `{gene['GeneID']}`\n"
            f"- [NCBI Link](https://www.ncbi.nlm.nih.gov/gene/{gene['GeneID']})",
        }
        for gene in mapped_genes
        if gene["GeneID"]
    ]

    return final_answer, confirmed_dois, gene_summaries, ""


if __name__ == "__main__":
    while True:
        try:
            print(
                "\n🔬 Paste the GPT answer you'd like to analyze (or type 'exit' to quit):\n"
            )
            answer = input("> ")

            if answer.lower().strip() in {"exit", "quit"}:
                print("Goodbye!")
                break

            mentions = extract_gene_mentions(answer)
            mapped = [map_to_gene_id(m) for m in mentions]

            gene_summaries = [
                {
                    "name": gene["mention"],
                    "summary": f"- Matched to `{gene['matched_text']}` ({gene['matched_type']})\n"
                    f"- NCBI GeneID: `{gene['GeneID']}`\n"
                    f"- [NCBI Link](https://www.ncbi.nlm.nih.gov/gene/{gene['GeneID']})",
                }
                for gene in mapped
                if gene["GeneID"]
            ]

            if gene_summaries:
                print("\n🧬 Gene Mentions & NCBI Summaries:")
                for gene in gene_summaries:
                    print(f"• {gene['name']}")
                    print(gene["summary"])
                    print()
            else:
                print("\n🧬 No direct NCBI matches for genes found in answer.")
                suggestions_shown = False

                for m in mapped:
                    suggestion = m.get("did_you_mean")
                    if suggestion and suggestion["score"] >= 60:
                        suggestions_shown = True
                        print(
                            f"❔ '{m['mention']}' → Did you mean '{suggestion['text']}' (score: {suggestion['score']:.1f})?"
                        )
                        print(f"   🔗 {suggestion['ncbi_link']}")

                if not suggestions_shown:
                    print("No strong gene name suggestions available either.")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
