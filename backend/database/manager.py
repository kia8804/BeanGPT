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
import os


class DatabaseManager:
    """Manages all database operations with lazy loading and proper error handling."""
    
    def __init__(self):
        self._gene_db: Optional[pd.DataFrame] = None
        self._uniprot_db: Optional[pd.DataFrame] = None
        self._bean_data: Optional[pd.DataFrame] = None
        self._bean_historical_data: Optional[pd.DataFrame] = None

        
        # Efficient lookup indices for gene operations
        self._gene_id_lookup: Optional[Dict[str, str]] = None  # gene_name -> gene_id
        self._gene_summary_lookup: Optional[Dict[str, str]] = None  # gene_id -> summary
        self._gene_symbol_lookup: Optional[Dict[str, str]] = None  # symbol -> gene_id
        self._uniprot_lookup: Optional[Dict[str, Dict]] = None  # gene_name -> uniprot_info
        self._gene_existence_cache: Optional[set] = None  # set of all known gene names
    
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
        """Lazy-loaded bean trial data (main merged data)."""
        if self._bean_data is None:
            self._load_bean_data()
        return self._bean_data
    
    @property
    def bean_historical_data(self) -> pd.DataFrame:
        """Lazy-loaded historical bean data."""
        if self._bean_historical_data is None:
            self._load_bean_data()
        return self._bean_historical_data
    

    
    def _build_gene_indices(self) -> None:
        """Build efficient lookup indices for gene operations."""
        if self._gene_id_lookup is not None:
            return  # Already built
            
        print("🔨 Building gene lookup indices...")
        
        # Initialize lookup dictionaries
        self._gene_id_lookup = {}
        self._gene_summary_lookup = {}
        self._gene_symbol_lookup = {}
        self._gene_existence_cache = set()
        
        # Build NCBI gene indices
        for _, row in self.gene_db.iterrows():
            gene_id = str(row['GeneID'])
            full_name = str(row['FullGeneName']).strip()
            symbol = str(row['Symbol']).strip()
            description = str(row['Description']).strip()
            
            # Build lookups (case-insensitive)
            if full_name and full_name.lower() != 'nan':
                self._gene_id_lookup[full_name.lower()] = gene_id
                self._gene_existence_cache.add(full_name.lower())
                
            if symbol and symbol.lower() != 'nan':
                self._gene_symbol_lookup[symbol.lower()] = gene_id
                self._gene_existence_cache.add(symbol.lower())
                
            if description and description.lower() != 'nan':
                self._gene_summary_lookup[gene_id] = description
        
        print(f"✅ Built gene indices: {len(self._gene_id_lookup)} name mappings, {len(self._gene_symbol_lookup)} symbol mappings")
    
    def _build_uniprot_indices(self) -> None:
        """Build efficient lookup indices for UniProt operations."""
        if self._uniprot_lookup is not None:
            return  # Already built
            
        print("🔨 Building UniProt lookup indices...")
        
        self._uniprot_lookup = {}
        
        # Build UniProt indices
        for _, row in self.uniprot_db.iterrows():
            entry = str(row['Entry']).strip()
            entry_name = str(row['Entry Name']).strip()
            protein_names = str(row['Protein names']).strip()
            gene_names = str(row['Gene Names']).strip()
            organism = str(row.get('Organism', 'Phaseolus vulgaris')).strip()
            
            # Create UniProt info object
            uniprot_info = {
                'entry': entry,
                'entry_name': entry_name,
                'protein_names': protein_names,
                'gene_names': gene_names,
                'organism': organism
            }
            
            # Index by various gene name variations
            if gene_names and gene_names.lower() != 'nan':
                # Split gene names by common separators
                for gene_name in gene_names.replace(';', ' ').replace(',', ' ').split():
                    gene_name = gene_name.strip().lower()
                    if gene_name:
                        self._uniprot_lookup[gene_name] = uniprot_info
                        self._gene_existence_cache.add(gene_name)
            
            # Also index by entry name
            if entry_name and entry_name.lower() != 'nan':
                self._uniprot_lookup[entry_name.lower()] = uniprot_info
                self._gene_existence_cache.add(entry_name.lower())
        
        print(f"✅ Built UniProt indices: {len(self._uniprot_lookup)} gene mappings")
    
    def _load_gene_db(self) -> None:
        """Load the NCBI gene database with optimized format preference."""
        try:
            # Try to load from optimized CSV format first
            csv_path = settings.gene_db_path.replace('.xlsx', '.csv')
            if os.path.exists(csv_path):
                print(f"📈 Loading from optimized CSV format: {csv_path}")
                self._gene_db = pd.read_csv(csv_path)
            else:
                print(f"📊 Loading from Excel format: {settings.gene_db_path}")
                self._gene_db = pd.read_excel(settings.gene_db_path)
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for future loads: {csv_path}")
                    self._gene_db.to_csv(csv_path, index=False)
                    print(f"✅ Saved optimized format: {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
                    
            print(f"✅ Loaded {len(self._gene_db)} genes from gene database")
            
            # Build indices immediately after loading
            self._build_gene_indices()
        except FileNotFoundError:
            raise DatabaseError(f"Gene database not found at {settings.gene_db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load gene database: {str(e)}")
    
    def _load_uniprot_db(self) -> None:
        """Load the UniProt database with optimized format preference."""
        try:
            # Try to load from optimized CSV format first
            csv_path = settings.uniprot_db_path.replace('.xlsx', '.csv')
            if os.path.exists(csv_path):
                print(f"📈 Loading from optimized CSV format: {csv_path}")
                self._uniprot_db = pd.read_csv(csv_path)
            else:
                print(f"📊 Loading from Excel format: {settings.uniprot_db_path}")
                self._uniprot_db = pd.read_excel(settings.uniprot_db_path)
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for future loads: {csv_path}")
                    self._uniprot_db.to_csv(csv_path, index=False)
                    print(f"✅ Saved optimized format: {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
                    
            print(f"✅ Loaded {len(self._uniprot_db)} UniProt entries from database")
            
            # Build indices immediately after loading
            self._build_uniprot_indices()
        except FileNotFoundError:
            raise DatabaseError(f"UniProt database not found at {settings.uniprot_db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load UniProt database: {str(e)}")
    
    def _load_bean_data(self) -> None:
        """Load the bean trial data from multiple sheets with optimized format preference."""
        try:
            # Define CSV paths for both sheets
            merged_csv_path = settings.merged_data_path.replace('.xlsx', '_merged_data.csv')
            historical_csv_path = settings.merged_data_path.replace('.xlsx', '_historical.csv')
            
            # Load Merged Data sheet
            if os.path.exists(merged_csv_path):
                print(f"📈 Loading Merged Data from optimized CSV format: {merged_csv_path}")
                df_merged = pd.read_csv(merged_csv_path)
            else:
                print(f"📊 Loading Merged Data sheet from Excel format: {settings.merged_data_path}")
                df_merged = pd.read_excel(settings.merged_data_path, sheet_name='Merged Data')
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for Merged Data: {merged_csv_path}")
                    df_merged.to_csv(merged_csv_path, index=False)
                    print(f"✅ Saved optimized format: {merged_csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
            
            # Load Historical sheet
            if os.path.exists(historical_csv_path):
                print(f"📈 Loading Historical data from optimized CSV format: {historical_csv_path}")
                df_historical = pd.read_csv(historical_csv_path)
            else:
                print(f"📊 Loading Historical sheet from Excel format: {settings.merged_data_path}")
                df_historical = pd.read_excel(settings.merged_data_path, sheet_name='Historical')
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for Historical data: {historical_csv_path}")
                    df_historical.to_csv(historical_csv_path, index=False)
                    print(f"✅ Saved optimized format: {historical_csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
            
            # Process the main merged data
            # Clean and process the data
            df_merged = df_merged[
                ~df_merged["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
            ]
            
            # Convert columns to appropriate types for merged data
            if "Year" in df_merged.columns:
                df_merged["Year"] = pd.to_numeric(df_merged["Year"], errors="coerce")
            if "Yield" in df_merged.columns:
                df_merged["Yield"] = pd.to_numeric(df_merged["Yield"], errors="coerce")
            if "Maturity" in df_merged.columns:
                df_merged["Maturity"] = pd.to_numeric(df_merged["Maturity"], errors="coerce")
            
            # Optional: Add lowercase columns for easier filtering
            if 'bean_type' in df_merged.columns:
                df_merged['bean_type'] = df_merged['bean_type'].astype(str).str.lower()
            if 'trial_group' in df_merged.columns:
                df_merged['trial_group'] = df_merged['trial_group'].astype(str).str.lower()
            
            # Process historical data
            if "Cultivar Name" in df_historical.columns:
                df_historical = df_historical[
                    ~df_historical["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
                ]
            
            # Convert numeric columns in historical data if they exist
            for col in df_historical.columns:
                if col.lower() in ['year', 'yield', 'maturity']:
                    df_historical[col] = pd.to_numeric(df_historical[col], errors="coerce")
            
            self._bean_data = df_merged
            self._bean_historical_data = df_historical
            
            print(f"✅ Loaded {len(self._bean_data)} main bean trial records from Merged Data sheet")
            print(f"✅ Loaded {len(self._bean_historical_data)} historical bean records from Historical sheet")
            
        except FileNotFoundError:
            raise DatabaseError(f"Bean dataset not found at {settings.merged_data_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load bean dataset: {str(e)}")
    

    
    def is_gene_in_databases(self, gene_name: str) -> bool:
        """Check if a gene name exists in either NCBI or UniProt databases using fast lookup."""
        try:
            # Ensure indices are built
            if self._gene_existence_cache is None:
                self._build_gene_indices()
                self._build_uniprot_indices()
            
            # Fast O(1) lookup
            return gene_name.lower() in self._gene_existence_cache
            
        except Exception as e:
            print(f"⚠️ Error checking gene in databases: {e}")
            return False
    
    def map_to_gene_id(self, gene_name: str) -> Optional[str]:
        """Fast gene name to ID mapping using lookup indices."""
        try:
            # Ensure indices are built
            if self._gene_id_lookup is None:
                self._build_gene_indices()
            
            gene_lower = gene_name.lower()
            
            # Try exact name match first
            if gene_lower in self._gene_id_lookup:
                return self._gene_id_lookup[gene_lower]
            
            # Try symbol match
            if gene_lower in self._gene_symbol_lookup:
                return self._gene_symbol_lookup[gene_lower]
            
            return None
            
        except Exception as e:
            print(f"Error mapping gene to ID: {e}")
            return None
    
    def get_gene_summary(self, gene_id: str) -> Optional[str]:
        """Fast gene summary lookup using lookup indices."""
        try:
            # Ensure indices are built
            if self._gene_summary_lookup is None:
                self._build_gene_indices()
            
            return self._gene_summary_lookup.get(gene_id)
            
        except Exception as e:
            print(f"Error getting gene summary: {e}")
            return None
    
    def get_uniprot_info(self, gene_name: str) -> Optional[Dict]:
        """Fast UniProt information lookup using lookup indices."""
        try:
            # Ensure indices are built
            if self._uniprot_lookup is None:
                self._build_uniprot_indices()
            
            return self._uniprot_lookup.get(gene_name.lower())
            
        except Exception as e:
            print(f"Error getting UniProt info: {e}")
            return None
    



# Global instance - this is acceptable as it's a singleton pattern
db_manager = DatabaseManager() 