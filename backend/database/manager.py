"""
Database manager for handling all database operations without global state.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import json
import orjson
from config import settings
from exceptions import DatabaseError


class DatabaseManager:
    """Manages all database operations with lazy loading and proper error handling."""
    
    def __init__(self):
        self._gene_db: Optional[pd.DataFrame] = None
        self._uniprot_db: Optional[pd.DataFrame] = None
        self._bean_data: Optional[pd.DataFrame] = None
        self._rag_lookup: Optional[Dict[str, str]] = None
    
    @property
    def gene_db(self) -> pd.DataFrame:
        """Lazy-loaded NCBI gene database."""
        if self._gene_db is None:
            self._load_gene_db()
        return self._gene_db
    
    @property
    def uniprot_db(self) -> pd.DataFrame:
        """Lazy-loaded UniProt database."""
        if self._uniprot_db is None:
            self._load_uniprot_db()
        return self._uniprot_db
    
    @property
    def bean_data(self) -> pd.DataFrame:
        """Lazy-loaded bean trial data."""
        if self._bean_data is None:
            self._load_bean_data()
        return self._bean_data
    
    @property
    def rag_lookup(self) -> Dict[str, str]:
        """Lazy-loaded RAG context lookup."""
        if self._rag_lookup is None:
            self._load_rag_lookup()
        return self._rag_lookup
    
    def _load_gene_db(self) -> None:
        """Load the NCBI gene database."""
        try:
            self._gene_db = pd.read_excel(settings.gene_db_path)
            print(f"✅ Loaded {len(self._gene_db)} genes from {settings.gene_db_path}")
        except FileNotFoundError:
            raise DatabaseError(f"Gene database not found at {settings.gene_db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load gene database: {str(e)}")
    
    def _load_uniprot_db(self) -> None:
        """Load the UniProt database."""
        try:
            self._uniprot_db = pd.read_excel(settings.uniprot_db_path)
            print(f"✅ Loaded {len(self._uniprot_db)} UniProt entries from {settings.uniprot_db_path}")
        except FileNotFoundError:
            raise DatabaseError(f"UniProt database not found at {settings.uniprot_db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load UniProt database: {str(e)}")
    
    def _load_bean_data(self) -> None:
        """Load the bean trial data."""
        try:
            df = pd.read_excel(settings.merged_data_path)
            
            # Clean and process the data
            df = df[
                ~df["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
            ]
            
            # Convert columns to appropriate types
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
            df["Maturity"] = pd.to_numeric(df["Maturity"], errors="coerce")
            
            # Optional: Add lowercase columns for easier filtering
            if 'bean_type' in df.columns:
                df['bean_type'] = df['bean_type'].astype(str).str.lower()
            if 'trial_group' in df.columns:
                df['trial_group'] = df['trial_group'].astype(str).str.lower()
            
            self._bean_data = df
            print(f"✅ Loaded {len(self._bean_data)} bean trial records from {settings.merged_data_path}")
            
        except FileNotFoundError:
            raise DatabaseError(f"Bean dataset not found at {settings.merged_data_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load bean dataset: {str(e)}")
    
    def _load_rag_lookup(self) -> None:
        """Load the RAG context lookup."""
        try:
            rag_lookup = {}
            
            # Try UTF-8 first, fallback to latin-1
            encodings = ['utf-8', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(settings.rag_file, 'r', encoding=encoding) as f:
                        for line in f:
                            record = orjson.loads(line)
                            doi = record.get("doi") or record.get("source", "").replace(".pdf", "")
                            rag = record.get("summary", "")
                            if doi and rag:
                                rag_lookup[doi.strip()] = rag.strip()
                    break
                except UnicodeDecodeError:
                    continue
            
            self._rag_lookup = rag_lookup
            print(f"✅ Loaded {len(self._rag_lookup)} RAG entries from {settings.rag_file}")
            
        except FileNotFoundError:
            raise DatabaseError(f"RAG file not found at {settings.rag_file}")
        except Exception as e:
            raise DatabaseError(f"Failed to load RAG lookup: {str(e)}")
    
    def is_gene_in_databases(self, gene_name: str) -> bool:
        """Check if a gene name exists in either NCBI or UniProt databases."""
        try:
            # Check NCBI database
            if not self.gene_db.empty:
                ncbi_matches = self.gene_db[
                    self.gene_db['Symbol'].str.contains(gene_name, case=False, na=False) |
                    self.gene_db['FullGeneName'].str.contains(gene_name, case=False, na=False) |
                    self.gene_db['Description'].str.contains(gene_name, case=False, na=False)
                ]
                if not ncbi_matches.empty:
                    return True
            
            # Check UniProt database
            if not self.uniprot_db.empty:
                uniprot_matches = self.uniprot_db[
                    self.uniprot_db['Gene Names'].str.contains(gene_name, case=False, na=False) |
                    self.uniprot_db['Protein names'].str.contains(gene_name, case=False, na=False)
                ]
                if not uniprot_matches.empty:
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking gene in databases: {e}")
            return False
    
    def get_rag_context_from_dois(self, dois: list[str]) -> tuple[str, list[str]]:
        """Get RAG context from DOIs."""
        context_blocks = []
        confirmed_dois = []
        
        for i, doi in enumerate(dois, 1):
            if doi in self.rag_lookup:
                summary = self.rag_lookup[doi]
                context_blocks.append(f"[{i}] Source: {doi}\n{summary}")
                confirmed_dois.append(doi)
        
        return "\n\n".join(context_blocks), confirmed_dois


# Global instance - this is acceptable as it's a singleton pattern
db_manager = DatabaseManager() 