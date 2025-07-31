"""
Configuration management for the BeanGPT backend.
Simple approach using environment variables.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        # API Keys (required)
        self.openai_api_key = self._get_required_env("OPENAI_API_KEY")
        self.pinecone_api_key = self._get_required_env("PINECONE_API_KEY")
        
        # Model Configuration
        self.bge_model = os.getenv("BGE_MODEL", "BAAI/bge-base-en-v1.5")
        self.pubmedbert_model = os.getenv("PUBMEDBERT_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # Vector Database Configuration
        self.bge_index_name = os.getenv("BGE_INDEX_NAME", "bge-production")
        self.pubmedbert_index_name = os.getenv("PUBMEDBERT_INDEX_NAME", "pubmed-production")
        
        # Search Configuration
        self.top_k = int(os.getenv("TOP_K", "8"))
        self.alpha = float(os.getenv("ALPHA", "0.6"))
        
        # Data Paths
        self.gene_db_path = os.getenv("GENE_DB_PATH", "../data/NCBI_Filtered_Data_Enriched.xlsx")
        self.uniprot_db_path = os.getenv("UNIPROT_DB_PATH", "../data/uniprotkb_Phaseolus_vulgaris.xlsx")

        self.merged_data_path = os.getenv("MERGED_DATA_PATH", "../data/Merged_Bean_Dataset.xlsx")
        
        # API Configuration
        self.cors_origins = self._parse_list(os.getenv("CORS_ORIGINS", "http://localhost:5173"))
        self.api_prefix = os.getenv("API_PREFIX", "/api")
    
    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable or raise an error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse a comma-separated string into a list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",")]


# Global settings instance
settings = Settings() 