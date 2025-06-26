# ncbi_utils.py
import pandas as pd
import json
from rapidfuzz import process, fuzz
from openai import OpenAI

# === Globals ===
gene_choices = {}
client = OpenAI(
    api_key="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"
)  # You can inject this dynamically too
score_cutoff = 70


def load_gene_db(csv_path: str):
    """Load NCBI gene list from an Excel file and build a lookup dictionary."""
    global gene_choices
    df = pd.read_excel(csv_path, sheet_name=0)
    df = df.rename(
        columns={"NCBI GeneID": "GeneID", "Symbol": "Symbol", "Description": "GeneName"}
    )

    gene_choices = {}
    for _, row in df.iterrows():
        gene_id = row["GeneID"]
        symbol = (
            str(row.get("Symbol")).strip().lower()
            if pd.notna(row.get("Symbol"))
            else None
        )
        name = (
            str(row.get("GeneName")).strip().lower()
            if pd.notna(row.get("GeneName"))
            else None
        )
        if symbol:
            gene_choices[symbol] = (gene_id, "Symbol")
        if name:
            gene_choices[name] = (gene_id, "GeneName")


def extract_gene_mentions(text: str) -> list[str]:
    """Ask GPT to extract gene-like entities from input text."""
    system_prompt = (
        "You are a biology parser. Extract any gene symbols or full gene names "
        "from the text. These may appear in full, abbreviated, lowercase, or paraphrased. "
        'Return only a raw JSON array of strings like: ["Asr1", "DREB2A"]. No markdown, no preamble, no explanations.'
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    raw_output = resp.choices[0].message.content.strip()
    print("\n==== RAW GPT OUTPUT ====\n", raw_output)
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").strip()
        if raw_output.lower().startswith("json"):
            raw_output = raw_output[4:].strip()

    try:
        return json.loads(raw_output)
    except Exception as e:
        print("❌ Error parsing GPT output:", e)
        return []


def map_to_gene_id(mention: str) -> dict:
    """Map a gene mention to the closest known NCBI entry using fuzzy matching."""
    mention_norm = mention.lower()
    match, score, _ = process.extractOne(
        mention_norm, gene_choices.keys(), scorer=fuzz.ratio
    ) or (None, 0, None)

    result = {
        "mention": mention,
        "matched_text": None,
        "matched_type": None,
        "GeneID": None,
        "score": score,
        "did_you_mean": None,
    }

    if match and score >= score_cutoff:
        gene_id, matched_type = gene_choices[match]
        result.update(
            {
                "matched_text": match,
                "matched_type": matched_type,
                "GeneID": gene_id,
            }
        )

    # Always suggest the closest possible match
    closest_match, closest_score, _ = process.extractOne(
        mention_norm, gene_choices.keys(), scorer=fuzz.ratio
    )
    if closest_match:
        gene_id, match_type = gene_choices[closest_match]
        result["did_you_mean"] = {
            "text": closest_match,
            "score": closest_score,
            "GeneID": gene_id,
            "type": match_type,
            "ncbi_link": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
        }

    return result
