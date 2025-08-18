from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import os
import numpy as np
from services.pipeline import get_openai_client
import json

router = APIRouter()

# Load and cache the gene databases
def load_gene_databases():
    try:
        ncbi_data = pd.read_csv('data/NCBI_Filtered_Data_Enriched.csv')
        uniprot_data = pd.read_csv('data/uniprotkb_Phaseolus_vulgaris.csv')
        return ncbi_data, uniprot_data
    except Exception as e:
        print(f"Error loading gene databases: {e}")
        return None, None

ncbi_data, uniprot_data = load_gene_databases()

class GeneSearchRequest(BaseModel):
    query: str
    api_key: Optional[str] = None

class GeneReference(BaseModel):
    title: str
    url: str

class GeneResult(BaseModel):
    id: str
    name: str
    source: str
    description: str
    aliases: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    references: Optional[List[GeneReference]] = None

def search_bean_databases(query: str):
    results = []
    
    # Normalize query
    query_lower = query.lower().strip()
    
    # Search NCBI database
    if ncbi_data is not None:
        ncbi_matches = ncbi_data[
            ncbi_data['Gene_Name'].str.lower().str.contains(query_lower, na=False) |
            ncbi_data['Description'].str.lower().str.contains(query_lower, na=False)
        ]
        
        for _, row in ncbi_matches.iterrows():
            results.append({
                'id': str(row['Gene_ID']),
                'name': row['Gene_Name'],
                'source': 'NCBI',
                'description': row['Description'],
                'aliases': row.get('Aliases', '').split(';') if pd.notna(row.get('Aliases')) else [],
                'references': [
                    {'title': 'View in NCBI', 'url': f"https://www.ncbi.nlm.nih.gov/gene/{row['Gene_ID']}"}
                ]
            })
    
    # Search UniProt database
    if uniprot_data is not None:
        uniprot_matches = uniprot_data[
            uniprot_data['Entry'].str.lower().str.contains(query_lower, na=False) |
            uniprot_data['Protein names'].str.lower().str.contains(query_lower, na=False)
        ]
        
        for _, row in uniprot_matches.iterrows():
            results.append({
                'id': row['Entry'],
                'name': row['Entry name'],
                'source': 'UniProt',
                'description': row['Protein names'],
                'functions': [row['Function']] if pd.notna(row.get('Function')) else [],
                'references': [
                    {'title': 'View in UniProt', 'url': f"https://www.uniprot.org/uniprot/{row['Entry']}"}
                ]
            })
    
    return results

def generate_ai_description(query: str, api_key: str):
    client = get_openai_client(api_key)
    
    try:
        # First, check if this might be a human gene
        check_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a gene expert. Determine if this query is likely referring to a human gene or a bean (Phaseolus vulgaris) gene. Respond in JSON format with fields: is_human_gene (boolean), explanation (string)."},
                {"role": "user", "content": f"Is this likely a human gene: {query}"}
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={ "type": "json_object" }
        )
        
        check_result = check_response.choices[0].message.content
        check_data = json.loads(check_result)
        
        # Generate the description with context from the check
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": """You are a specialized gene research assistant focused on Phaseolus vulgaris (common bean) genetics.
                Generate a detailed, PhD-level description of the gene or gene family, focusing on its role in bean biology.
                If this appears to be a human gene, suggest related bean genes or homologs.
                
                Response must be valid JSON with the following structure:
                {
                    "sections": [
                        {
                            "title": "Gene Overview",
                            "content": "Technical overview of gene structure, family, and key characteristics"
                        },
                        {
                            "title": "Molecular Function",
                            "content": "Detailed description of molecular mechanisms and biochemical functions"
                        },
                        {
                            "title": "Expression & Regulation",
                            "content": "Expression patterns, regulatory mechanisms, and pathway interactions"
                        },
                        {
                            "title": "Phenotypic Effects",
                            "content": "Impact on plant development, stress responses, or other phenotypes"
                        },
                        {
                            "title": "Research Context",
                            "content": "Current research status and significance in bean biology"
                        }
                    ],
                    "molecular_details": {
                        "domains": ["List of key protein domains"],
                        "interactions": ["Known protein-protein or molecular interactions"],
                        "pathways": ["Associated metabolic or signaling pathways"]
                    },
                    "is_bean_specific": true/false,
                    "suggested_bean_genes": ["PvGene1", "PvGene2"],
                    "technical_notes": "Additional technical details relevant for researchers"
                }
                
                Guidelines:
                1. Use proper gene nomenclature (e.g., PvNAC1, not NAC1)
                2. Include specific molecular mechanisms and pathways
                3. Reference protein domains and structural features
                4. Discuss regulatory networks and interactions
                5. Focus on bean-specific research context
                6. Use technical, PhD-level terminology
                7. Avoid generic descriptions - be specific and detailed
                8. Include quantitative data where relevant (e.g., expression fold changes)"""},
                {"role": "user", "content": f"Gene query: {query}\nPreliminary analysis: {check_result}"}
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={ "type": "json_object" }
        )
        
        result = response.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        print(f"Error generating AI description: {e}")
        print(f"Response content: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
        return None

@router.post("/gene-search")
async def search_genes(request: GeneSearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Search query is required")
    
    # Search databases first
    database_results = search_bean_databases(request.query)
    
    # If no results found and API key provided, generate AI description
    if not database_results and request.api_key:
        ai_result = generate_ai_description(request.query, request.api_key)
        if ai_result:
            # Create a synthetic result from AI description
            result_data = {
                'id': 'ai_generated',
                'name': request.query,
                'source': 'AI Analysis',
                'is_bean_specific': ai_result.get('is_bean_specific', False),
                'suggested_bean_genes': ai_result.get('suggested_bean_genes', [])
            }
            
            # Add structured sections if available
            if 'sections' in ai_result:
                result_data['sections'] = ai_result['sections']
            if 'molecular_details' in ai_result:
                result_data['molecular_details'] = ai_result['molecular_details']
            if 'technical_notes' in ai_result:
                result_data['technical_notes'] = ai_result['technical_notes']
            
            # Fallback for simple description
            if 'description' in ai_result and 'sections' not in ai_result:
                result_data['description'] = ai_result['description']
                
            database_results.append(result_data)
    
    return database_results
