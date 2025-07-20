import re
import pandas as pd
from typing import List, Dict, Optional
import json
# Using OpenAI through wrapper to avoid proxy issues
import os
from database.manager import db_manager

# OpenAI client will be initialized in functions as needed

def extract_gene_mentions(text: str) -> tuple[list[str], set[str], set[str]]:
    """Ask GPT to extract actual gene names and molecular markers from input text.

    This function first uses GPT to identify potential genetic elements from the text,
    then validates those candidates against the local NCBI and UniProt databases.
    This approach ensures we only consider genes that are actually mentioned in the text.
    """
    # Initialize OpenAI client (simple approach for research use)
    from .openai_client import create_openai_client
    client = create_openai_client()
    
    system_prompt = (
        "Based on the user's input, extract and list only the molecular entities—which may include genes, specific metabolites, transcriptomic elements, proteins, enzymes, or any other uniquely named molecular markers—directly implicated in Phaseolus vulgaris research.\n\n"
        "If a gene is not specified but a QTL or trait is mentioned, respond with: \"No specified gene identified, but the QTL named <QTL_name> is identified.\"\n\n"
        "STRICTLY INCLUDE ONLY:\n"
        "- Exact gene names or IDs following standard or widely accepted nomenclature (e.g., PvP5CS1, Phvul.001G123400, etc.).\n"
        "- Named transcription factors with specific identifiers (e.g., MYB123, bZIP45, NAC072, WRKY33, etc.; not case sensitive).\n"
        "- Gene family names when asked about specifically (e.g., MYB, WRKY, NAC, HSP, LEA when directly queried).\n"
        "- Specifically named proteins with unique identifiers (e.g., aquaporin PIP2;1, catalase CAT1, HSP70, etc.).\n"
        "- Named enzyme genes with identifiers (e.g., SOD1, APX2, RBOH, etc.).\n"
        "- Well-known gene names in plant biology, even if short, when explicitly mentioned as genes (e.g., Asp gene, Phg gene, Fin gene, etc.).\n"
        "- Specific metabolites or compounds when mentioned as direct research targets (e.g., chlorophyll, anthocyanin, etc.).\n"
        "- Quantitative trait loci (QTL) when specifically named (e.g., QTL-1, disease-resistance QTL, etc.).\n"
        "- Genetic markers or SNPs when mentioned by name (e.g., SNP123, marker_ABC, etc.).\n\n"
        "EXCLUDE:\n"
        "- General descriptive terms (e.g., 'disease resistance', 'stress tolerance', 'high yield').\n"
        "- Common words (e.g., 'gene', 'protein', 'enzyme' without specific identifiers).\n"
        "- Trait names without specific gene identifiers (e.g., 'drought tolerance', 'yield', 'maturity').\n"
        "- Functional categories (e.g., 'transcription factor', 'metabolic pathway').\n"
        "- Anatomical or developmental terms (e.g., 'root', 'leaf', 'flower').\n"
        "- General biological processes (e.g., 'photosynthesis', 'respiration' without specific gene names).\n\n"
        "RESPONSE FORMAT: Return a JSON array of strings, each representing a specific molecular entity. If no specific entities are found, return an empty array []."
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
    """Check if a gene exists in either NCBI or UniProt databases."""
    try:
        # Check NCBI database
        gene_db = db_manager.gene_db
        ncbi_matches = gene_db[
            gene_db['FullGeneName'].str.contains(gene_name, case=False, na=False) |
            gene_db['GeneID'].astype(str).str.contains(gene_name, case=False, na=False) |
            gene_db['Symbol'].str.contains(gene_name, case=False, na=False)
        ]
        
        if not ncbi_matches.empty:
            return True
            
        # Check UniProt database
        uniprot_db = db_manager.uniprot_db
        uniprot_matches = uniprot_db[
            uniprot_db['Gene Names'].str.contains(gene_name, case=False, na=False) |
            uniprot_db['Entry Name'].str.contains(gene_name, case=False, na=False) |
            uniprot_db['Protein names'].str.contains(gene_name, case=False, na=False)
        ]
        
        return not uniprot_matches.empty
        
    except Exception as e:
        print(f"Error checking gene in databases: {e}")
        return False


def map_to_gene_id(gene_name: str) -> Optional[str]:
    """Map a gene name to its Gene ID using the database manager."""
    try:
        gene_db = db_manager.gene_db
        
        # Try exact match first
        exact_match = gene_db[gene_db['FullGeneName'] == gene_name]
        if not exact_match.empty:
            return str(exact_match.iloc[0]['GeneID'])
        
        # Try case-insensitive match
        case_match = gene_db[gene_db['FullGeneName'].str.lower() == gene_name.lower()]
        if not case_match.empty:
            return str(case_match.iloc[0]['GeneID'])
        
        # Try symbol match
        symbol_match = gene_db[gene_db['Symbol'].str.lower() == gene_name.lower()]
        if not symbol_match.empty:
            return str(symbol_match.iloc[0]['GeneID'])
        
        return None
        
    except Exception as e:
        print(f"Error mapping gene to ID: {e}")
        return None


def get_gene_summary(gene_id: str) -> Optional[str]:
    """Get gene summary from the database manager."""
    try:
        gene_db = db_manager.gene_db
        
        # Find the gene by ID
        gene_match = gene_db[gene_db['GeneID'].astype(str) == gene_id]
        if not gene_match.empty:
            return gene_match.iloc[0]['Description']
        
        return None
        
    except Exception as e:
        print(f"Error getting gene summary: {e}")
        return None


def get_uniprot_info(gene_name: str) -> Optional[Dict]:
    """Get UniProt information for a gene."""
    try:
        uniprot_db = db_manager.uniprot_db
        
        # Search in gene names and protein names
        matches = uniprot_db[
            uniprot_db['Gene Names'].str.contains(gene_name, case=False, na=False) |
            uniprot_db['Entry Name'].str.contains(gene_name, case=False, na=False) |
            uniprot_db['Protein names'].str.contains(gene_name, case=False, na=False)
        ]
        
        if not matches.empty:
            first_match = matches.iloc[0]
            return {
                'entry': first_match['Entry'],
                'entry_name': first_match['Entry Name'],
                'protein_names': first_match['Protein names'],
                'gene_names': first_match['Gene Names'],
                'organism': first_match.get('Organism', 'Phaseolus vulgaris')
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting UniProt info: {e}")
        return None 