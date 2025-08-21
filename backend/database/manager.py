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
        self._historical_data: Optional[pd.DataFrame] = None
        self._climate_data: Optional[pd.DataFrame] = None

        
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
        """Lazy-loaded bean trial data."""
        if self._bean_data is None:
            self._load_bean_data()
        return self._bean_data
    
    @property
    def historical_data(self) -> pd.DataFrame:
        """Lazy-loaded historical weather data."""
        if self._historical_data is None:
            self._load_historical_data()
        return self._historical_data
    
    @property
    def climate_data(self) -> pd.DataFrame:
        """Lazy-loaded climate decade data with RCP scenarios."""
        if self._climate_data is None:
            self._load_climate_data()
        return self._climate_data
    

    
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
        """Load the updated bean trial data with enhanced column mapping."""
        try:
            # Try to load from optimized CSV format first
            csv_path = settings.merged_data_path.replace('.xlsx', '.csv')
            if os.path.exists(csv_path):
                print(f"📈 Loading from optimized CSV format: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                print(f"📊 Loading from Excel format: {settings.merged_data_path}")
                df = pd.read_excel(settings.merged_data_path)
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for future loads: {csv_path}")
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Saved optimized format: {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
            
            # Handle new column structure (Merged_Bean_data_update.xlsx format)
            column_mapping = {
                'Name': 'Cultivar Name',  # Map 'Name' to 'Cultivar Name' for compatibility
                'Grow Year': 'Year',       # Map 'Grow Year' to 'Year' for compatibility
                'Harvestability': 'Dir Harv Suit'  # Map new to old column name
            }
            
            # Apply column mapping for backward compatibility
            df = df.rename(columns=column_mapping)
            
            # Clean and process the data - handle both old and new formats
            cultivar_col = 'Cultivar Name' if 'Cultivar Name' in df.columns else 'Name'
            if cultivar_col in df.columns:
                df = df[
                    ~df[cultivar_col].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
                ]
            
            # Convert columns to appropriate types
            year_col = 'Year' if 'Year' in df.columns else 'Grow Year'
            if year_col in df.columns:
                df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
                if 'Year' not in df.columns:  # Ensure 'Year' column exists for compatibility
                    df['Year'] = df[year_col]
                    
            if 'Yield' in df.columns:
                df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
            if 'Maturity' in df.columns:
                df["Maturity"] = pd.to_numeric(df["Maturity"], errors="coerce")
            
            # Process enriched data columns
            if 'Released Year' in df.columns:
                df['Released Year'] = pd.to_numeric(df['Released Year'], errors="coerce")
            
            # Ensure Cultivar Name column exists for compatibility
            if 'Cultivar Name' not in df.columns and 'Name' in df.columns:
                df['Cultivar Name'] = df['Name']
            
            # Optional: Add lowercase columns for easier filtering
            if 'bean_type' in df.columns:
                df['bean_type'] = df['bean_type'].astype(str).str.lower()
            if 'trial_group' in df.columns:
                df['trial_group'] = df['trial_group'].astype(str).str.lower()
            
            self._bean_data = df
            print(f"✅ Loaded {len(self._bean_data)} bean trial records with {len(df.columns)} columns")
            print(f"📊 Enhanced data includes: {', '.join([col for col in ['Pedigree', 'Market Class', 'Released Year'] if col in df.columns])}")
            
        except FileNotFoundError:
            raise DatabaseError(f"Bean dataset not found at {settings.merged_data_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load bean dataset: {str(e)}")
    
    def _load_historical_data(self) -> None:
        """Load the historical weather data for environmental context."""
        try:
            # Try to load from optimized CSV format first
            csv_path = settings.historical_data_path.replace('.xlsx', '.csv')
            if os.path.exists(csv_path):
                print(f"📈 Loading historical data from optimized CSV format: {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                print(f"📊 Loading historical data from Excel format: {settings.historical_data_path}")
                df = pd.read_excel(settings.historical_data_path)
                
                # Create optimized CSV for future loads
                try:
                    print(f"💾 Creating optimized CSV for future loads: {csv_path}")
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Saved optimized format: {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized format: {e}")
            
            # Process the historical data
            # Convert date column to proper datetime
            if 'Year_Historical' in df.columns:
                df['Year_Historical'] = pd.to_datetime(df['Year_Historical'], errors='coerce')
                df['Year'] = df['Year_Historical'].dt.year  # Extract year for easier merging
                df['Month'] = df['Year_Historical'].dt.month
                df['Day'] = df['Year_Historical'].dt.day
            
            # Convert coordinate columns to numeric
            for coord_col in ['longitude', 'latitude']:
                if coord_col in df.columns:
                    df[coord_col] = pd.to_numeric(df[coord_col], errors='coerce')
            
            # Convert weather variables to numeric
            weather_columns = [col for col in df.columns if col not in 
                             ['Location', 'longitude', 'latitude', 'Unnamed: 3', 'Year_Historical', 'Year', 'Month', 'Day']]
            for col in weather_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any completely empty rows and unnecessary columns
            df = df.dropna(how='all')
            if 'Unnamed: 3' in df.columns:
                df = df.drop('Unnamed: 3', axis=1)
            
            self._historical_data = df
            print(f"✅ Loaded {len(self._historical_data)} historical weather records with {len(weather_columns)} weather variables")
            print(f"📊 Weather data covers: {df['Year'].min():.0f}-{df['Year'].max():.0f}")
            print(f"📍 Historical locations: {', '.join(sorted(df['Location'].dropna().unique()))}")
            
        except FileNotFoundError:
            raise DatabaseError(f"Historical dataset not found at {settings.historical_data_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to load historical dataset: {str(e)}")

    def _load_climate_data(self) -> None:
        """Load the climate decade data with RCP scenarios for future climate projections."""
        try:
            csv_path = "../data/climate_decade.csv"
            print(f"📈 Loading climate data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Process the climate data
            # Convert decade to numeric
            df['Decade'] = pd.to_numeric(df['Decade'], errors='coerce')
            
            # Convert climate variables to numeric
            for col in ['Precipitation', 'Tmin', 'Tmax']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean scenario names for easier filtering
            df['Scenario'] = df['Scenario'].str.replace('RCP_', 'RCP ')
            
            # Map scenario names to descriptive labels
            scenario_mapping = {
                'RCP 2.5': 'Best Case (RCP 2.5)',
                'RCP 4.5': 'Normal Case (RCP 4.5)', 
                'RCP 8.5': 'Worst Case (RCP 8.5)'
            }
            df['Scenario_Description'] = df['Scenario'].map(scenario_mapping).fillna(df['Scenario'])
            
            # Drop any rows with missing critical data
            df = df.dropna(subset=['Decade', 'Location', 'Scenario'])
            
            self._climate_data = df
            print(f"✅ Loaded {len(self._climate_data)} climate records")
            print(f"📊 Climate data covers decades: {df['Decade'].min():.0f}-{df['Decade'].max():.0f}")
            print(f"📍 Climate locations: {', '.join(sorted(df['Location'].dropna().unique()))}")
            print(f"🌡️ Climate scenarios: {', '.join(sorted(df['Scenario_Description'].dropna().unique()))}")
            
        except FileNotFoundError:
            raise DatabaseError(f"Climate dataset not found at data/climate_decade.csv")
        except Exception as e:
            raise DatabaseError(f"Failed to load climate dataset: {str(e)}")

    def get_historical_data_for_location_year(self, location: str, year: int, aggregate: str = 'growing_season') -> pd.DataFrame:
        """Get historical weather data for a specific location and year with optional aggregation."""
        try:
            hist_data = self.historical_data
            
            # Filter by location and year
            filtered = hist_data[
                (hist_data['Location'] == location) & 
                (hist_data['Year'] == year)
            ].copy()
            
            if filtered.empty:
                return pd.DataFrame()
            
            if aggregate == 'growing_season':
                # Aggregate data for typical bean growing season (May-September)
                growing_season = filtered[
                    (filtered['Month'] >= 5) & (filtered['Month'] <= 9)
                ]
                if not growing_season.empty:
                    # Calculate seasonal averages for weather variables
                    weather_cols = [col for col in growing_season.columns if col not in 
                                  ['Location', 'longitude', 'latitude', 'Year_Historical', 'Year', 'Month', 'Day']]
                    aggregated = growing_season[weather_cols].mean().to_frame().T
                    aggregated['Location'] = location
                    aggregated['Year'] = year
                    aggregated['Period'] = 'Growing Season (May-Sep)'
                    return aggregated
                    
            elif aggregate == 'annual':
                # Annual averages
                weather_cols = [col for col in filtered.columns if col not in 
                              ['Location', 'longitude', 'latitude', 'Year_Historical', 'Year', 'Month', 'Day']]
                aggregated = filtered[weather_cols].mean().to_frame().T
                aggregated['Location'] = location
                aggregated['Year'] = year
                aggregated['Period'] = 'Annual'
                return aggregated
            
            return filtered
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def get_climate_data_for_location_decade(self, location: str, decade: int, scenario: str = 'RCP 4.5') -> pd.DataFrame:
        """Get climate data for a specific location, decade, and scenario."""
        try:
            climate_data = self.climate_data
            
            # Handle location mapping if needed (Elora might not be in climate data)
            location_mapping = {
                'Elora': 'Fergus',  # Closest location to Elora in climate data
                'Fergus': 'Fergus',
                'Woodstock': 'Woodstock',
                'St. Thomas': 'St. Thomas',
                'Thorndale': 'Thorndale',
                'Exeter': 'Exeter',
                'Kempton': 'Kempton',
                'Kemptonton': 'Kemptonton',
                'Timmins': 'Timmins'
            }
            
            # Map location if available
            mapped_location = location_mapping.get(location, location)
            
            # Filter by location and decade
            filtered = climate_data[
                (climate_data['Location'] == mapped_location) & 
                (climate_data['Decade'] == decade)
            ].copy()
            
            # Filter by scenario if specified
            if scenario:
                # Handle both "RCP 4.5" and "RCP_4.5" formats
                scenario_clean = scenario.replace(' ', '_').replace('RCP_', 'RCP ')
                filtered = filtered[filtered['Scenario'] == scenario_clean]
            
            return filtered
            
        except Exception as e:
            print(f"Error getting climate data: {e}")
            return pd.DataFrame()

    def get_climate_comparison(self, location: str, decade1: int, decade2: int, scenario: str = 'RCP 4.5') -> Dict:
        """Compare climate data between two decades for a location."""
        try:
            data1 = self.get_climate_data_for_location_decade(location, decade1, scenario)
            data2 = self.get_climate_data_for_location_decade(location, decade2, scenario)
            
            if data1.empty or data2.empty:
                return {}
            
            comparison = {
                'location': location,
                'scenario': scenario,
                'decade1': decade1,
                'decade2': decade2,
                'precipitation_change': float(data2['Precipitation'].iloc[0] - data1['Precipitation'].iloc[0]),
                'tmin_change': float(data2['Tmin'].iloc[0] - data1['Tmin'].iloc[0]),
                'tmax_change': float(data2['Tmax'].iloc[0] - data1['Tmax'].iloc[0]),
                'data1': data1.iloc[0].to_dict(),
                'data2': data2.iloc[0].to_dict()
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error getting climate comparison: {e}")
            return {}
    
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