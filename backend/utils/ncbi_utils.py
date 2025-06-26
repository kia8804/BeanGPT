import re
import pandas as pd
from typing import List, Dict, Optional
import json
from openai import OpenAI
import os

# Global gene databases
GENE_DB = None
UNIPROT_DB = None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_gene_db(path: str):
    """Load the NCBI gene database from Excel file."""
    global GENE_DB
    GENE_DB = pd.read_excel(path)
    print(f"Loaded {len(GENE_DB)} genes from {path}")

def load_uniprot_db(path: str):
    """Load the UniProt database from Excel file."""
    global UNIPROT_DB
    UNIPROT_DB = pd.read_excel(path)
    print(f"Loaded {len(UNIPROT_DB)} UniProt entries from {path}")

def extract_gene_mentions(text: str) -> tuple[list[str], set[str], set[str]]:
    """Ask GPT to extract actual gene names and molecular markers from input text.

    This function first uses GPT to identify potential genetic elements from the text,
    then validates those candidates against the local NCBI and UniProt databases.
    This approach ensures we only consider genes that are actually mentioned in the text.
    """
    system_prompt = (
        "Based on the user's input, extract and list only the molecular entities—which may include genes, specific metabolites, transcriptomic elements, proteins, enzymes, or any other uniquely named molecular markers—directly implicated in Phaseolus vulgaris research.\n\n"
        "If a gene is not specified but a QTL or trait is mentioned, respond with: \"No specified gene identified, but the QTL named <QTL_name> is identified.\"\n\n"
        "STRICTLY INCLUDE ONLY:\n"
        "- Exact gene names or IDs following standard or widely accepted nomenclature (e.g., PvP5CS1, Phvul.001G123400, etc.).\n"
        "- Named transcription factors with specific identifiers (e.g., MYB123, bZIP45, NAC072, WRKY33, etc.; not case sensitive).\n"
        "- Gene family names when asked about specifically (e.g., MYB, WRKY, NAC, HSP, LEA when directly queried).\n"
        "- Specifically named proteins with unique identifiers (e.g., aquaporin PIP2;1, catalase CAT1, HSP70, etc.).\n"
        "- Named enzyme genes with identifiers (e.g., SOD1, APX2, RBOH, etc.).\n"
        "- Specific, named metabolites or phytochemicals (e.g., abscisic acid, proline, raffinose, etc.) directly linked to the molecular genetics of P. vulgaris.\n"
        "- Named non-coding RNAs or transcriptome elements where a unique identifier or locus is given (e.g., miR156, Phvul.TCONS_00058903, etc.).\n\n"
        "STRICTLY EXCLUDE:\n"
        "- Breeding lines, varieties, cultivars, accessions, or germplasm names (e.g., SEA 5, AND 277, BAT477, etc.).\n"
        "- Full or partial species names or abbreviations (e.g., Phaseolus vulgaris, P. vulgaris).\n"
        "- Generic, non-specific terms like QTL, general molecular markers, yield, drought tolerance, resistance gene, root traits, pod traits, etc.\n"
        "- Chromosomal locations, linkage groups, or genome build references (e.g., Pv04, chromosome 3, LG5, etc.).\n"
        "- Any population type or breeding/population code (e.g., RIL, F2, BC, DH, accession numbers).\n"
        "- Classes or families of molecules, genes, proteins, metabolites, or traits without a unique identifier (e.g., aquaporin without identifier, NAC without number, flavonoid if not a specific compound, LEA proteins without gene name).\n\n"
        "Return a JSON array of strings, where each string is a validated, specifically named gene, metabolite, transcript, protein, enzyme, or molecular marker. If nothing meeting the criteria is found, return an empty array.\n\n"
        "Be conservative but include commonly queried gene families (MYB, WRKY, NAC, HSP, LEA, etc.) when directly asked about. When in doubt about specific identifiers, exclude, but include gene family names when they are the main subject of the query.\n\n"
        "IMPORTANT: Only extract terms that are explicitly mentioned in the provided text. Do not infer or add terms that are not directly present."
    )

    try:
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
        
        # Clean up the output if it's wrapped in markdown code blocks
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").strip()
            if raw_output.lower().startswith("json"):
                raw_output = raw_output[4:].strip()

        # Parse the JSON array
        gpt_genes = json.loads(raw_output)
        print(f"GPT extracted genes: {gpt_genes}")
        
        # Now validate GPT results against databases and find additional matches
        db_validated_genes = []
        for gene in gpt_genes:
            # Check if this gene exists in our databases
            if is_gene_in_databases(gene):
                db_validated_genes.append(gene)
        
        print(f"Database validated genes: {db_validated_genes}")
        
        # Return: all_genes, db_validated_set, gpt_original_set
        return gpt_genes, set(db_validated_genes), set(gpt_genes)
    except Exception as e:
        print("❌ Error in gene extraction:", e)
        return [], set(), set()

def is_gene_in_databases(gene_name: str) -> bool:
    """Check if a gene name exists in either NCBI or UniProt databases."""
    if GENE_DB is None and UNIPROT_DB is None:
        return False
    
    # Check NCBI database
    if GENE_DB is not None:
        # Check Symbol column primarily
        if "Symbol" in GENE_DB.columns:
            if not GENE_DB[GENE_DB["Symbol"].str.lower() == gene_name.lower()].empty:
                return True
        
        # Check other columns for partial matches
        for col in GENE_DB.columns:
            if col.lower() != "geneid":
                if not GENE_DB[GENE_DB[col].astype(str).str.contains(gene_name, case=False, na=False)].empty:
                    return True
    
    # Check UniProt database
    if UNIPROT_DB is not None:
        for col in UNIPROT_DB.columns[1:]:  # Skip Entry column
            if "protein" not in col.lower():  # Skip protein name columns
                if not UNIPROT_DB[UNIPROT_DB[col].astype(str).str.contains(gene_name, case=False, na=False)].empty:
                    return True
    
    return False

def map_to_gene_id(gene_name: str) -> Optional[Dict]:
    """Map a gene name to its NCBI entry first, then UniProt entry if not found."""
    if GENE_DB is None:
        raise ValueError("Gene database not loaded. Call load_gene_db() first.")
    
    # Step 1: Try NCBI database first
    # Try exact match first
    exact_match = GENE_DB[GENE_DB['Symbol'] == gene_name]
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return {
            "name": gene_name,
            "description": str(row.get('GeneID', 'Unknown')),
            "source": "NCBI",
            "gene_id": str(row.get('GeneID', 'Unknown')),
            "symbol": str(row.get('Symbol', gene_name))
        }
    
    # Try case-insensitive match
    case_insensitive = GENE_DB[GENE_DB['Symbol'].str.lower() == gene_name.lower()]
    if not case_insensitive.empty:
        row = case_insensitive.iloc[0]
        return {
            "name": gene_name,
            "description": str(row.get('GeneID', 'Unknown')),
            "source": "NCBI",
            "gene_id": str(row.get('GeneID', 'Unknown')),
            "symbol": str(row.get('Symbol', gene_name))
        }
    
    # Try partial match (escape regex special characters)
    escaped_gene_name = re.escape(gene_name)
    partial_match = GENE_DB[GENE_DB['Symbol'].str.contains(escaped_gene_name, case=False, na=False, regex=True)]
    if not partial_match.empty:
        row = partial_match.iloc[0]
        return {
            "name": gene_name,
            "description": str(row.get('GeneID', 'Unknown')),
            "source": "NCBI",
            "gene_id": str(row.get('GeneID', 'Unknown')),
            "symbol": str(row.get('Symbol', gene_name))
        }
    
    # Step 2: If not found in NCBI, try UniProt database
    if UNIPROT_DB is None:
        print("⚠️ UniProt database not loaded, skipping UniProt lookup")
        return generate_gene_description_with_gpt(gene_name)
    
    # Get all columns except the first one (Entry column)
    search_columns = UNIPROT_DB.columns[1:].tolist()
    
    for column in search_columns:
        # Try exact match in this column
        exact_match = UNIPROT_DB[UNIPROT_DB[column].astype(str).str.lower() == gene_name.lower()]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            entry_value = str(row.iloc[0])  # First column (Entry)
            return {
                "name": gene_name,
                "description": entry_value,
                "source": "UniProt",
                "entry": entry_value,
                "matched_column": column,
                "matched_value": str(row[column])
            }
        
        # Try partial match in this column (escape regex special characters)
        escaped_gene_name = re.escape(gene_name)
        partial_match = UNIPROT_DB[UNIPROT_DB[column].astype(str).str.contains(escaped_gene_name, case=False, na=False, regex=True)]
        if not partial_match.empty:
            row = partial_match.iloc[0]
            entry_value = str(row.iloc[0])  # First column (Entry)
            return {
                "name": gene_name,
                "description": entry_value,
                "source": "UniProt",
                "entry": entry_value,
                "matched_column": column,
                "matched_value": str(row[column])
            }
    
    # If not found in either database, generate description with GPT-4o
    return generate_gene_description_with_gpt(gene_name)

def generate_gene_description_with_gpt(gene_name: str) -> Optional[Dict]:
    """Generate a brief description for a gene using GPT-4o when not found in databases."""
    try:
        prompt = f"""
        Provide a brief, scientific description for the gene or molecular marker "{gene_name}" in the context of plant biology, specifically dry beans (Phaseolus vulgaris) if applicable.

        Include:
        - What type of gene/protein it is
        - Its primary function or role
        - Relevance to plant biology or agriculture (if any)

        Keep it concise (2-3 sentences max). If this doesn't appear to be a real gene name, indicate that it may be a gene identifier or locus name.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a plant genetics expert. Provide accurate, concise gene descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        description = response.choices[0].message.content.strip()
        
        return {
            "name": gene_name,
            "description": description,
            "source": "GPT-4o",
            "generated": True
        }
        
    except Exception as e:
        print(f"❌ Error generating description for {gene_name}: {e}")
        return None 