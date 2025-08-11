"""
Configuration settings for the BeanGPT backend.
Loads settings from environment variables with sensible defaults.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", None)  # Optional - users provide via UI
        
        # Zilliz Configuration
        self.zilliz_uri = self._get_required_env("ZILLIZ_URI")
        self.zilliz_token = self._get_required_env("ZILLIZ_TOKEN")
        
        # Model Configuration
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.collection_name = os.getenv("ZILLIZ_COLLECTION_NAME", "openai-embeddings")
        
        # Search Configuration
        self.top_k = int(os.getenv("TOP_K", "8"))
        self.alpha = float(os.getenv("ALPHA", "0.6"))
        
        # Data Paths
        self.gene_db_path = os.getenv("GENE_DB_PATH", "../data/NCBI_Filtered_Data_Enriched.xlsx")
        self.uniprot_db_path = os.getenv("UNIPROT_DB_PATH", "../data/uniprotkb_Phaseolus_vulgaris.xlsx")
        self.merged_data_path = os.getenv("MERGED_DATA_PATH", "../data/merged_bean_data.xlsx")
        self.historical_data_path = os.getenv("HISTORICAL_DATA_PATH", "../data/Historical_Bean_Data.xlsx")
        
        # API Configuration
        self.cors_origins = self._parse_list(os.getenv("CORS_ORIGINS", "http://localhost:5173"))
        self.api_prefix = os.getenv("API_PREFIX", "/api")
        
        # Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable or raise an error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse a comma-separated string into a list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

# Global settings instance
settings = Settings()