"""
Simplified bean data analysis with single chart generation function.
Replaces all the complex chart type logic with GPT-4o intelligence.
"""

import pandas as pd
import re
import os
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from .simple_plotly import create_smart_chart

# Path to the merged bean dataset file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MERGED_DATA_PATH = os.getenv("MERGED_DATA_PATH", os.path.join(PROJECT_ROOT, "data", "Merged_Bean_Dataset.xlsx"))

# ---- Load merged dataset ----
def load_all_trials() -> pd.DataFrame:
    """Load the bean trial data from Excel."""
    try:
        df = pd.read_excel(MERGED_DATA_PATH)
        # Ensure consistency and correct data types
        df = df[
            ~df["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
        ]
        # Convert relevant columns to numeric, coercing errors to NaN
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
        df["Maturity"] = pd.to_numeric(df["Maturity"], errors="coerce")
        
        # Optional: Add lowercase bean_type and trial_group for easier filtering if they exist
        if 'bean_type' in df.columns:
            df['bean_type'] = df['bean_type'].astype(str).str.lower()
        if 'trial_group' in df.columns:
            df['trial_group'] = df['trial_group'].astype(str).str.lower()

        print(f"Loaded {len(df)} rows from {MERGED_DATA_PATH}")
        return df
    except FileNotFoundError:
        print(f"Error: Merged bean dataset not found at {MERGED_DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading bean data from {MERGED_DATA_PATH}: {e}")
        return pd.DataFrame()

# Load the full dataset once when the module is imported
df_trials = load_all_trials()

def answer_bean_query(args: Dict) -> Tuple[str, str, Dict]:
    """
    SIMPLIFIED VERSION: Analyze bean data with optional chart generation.
    Only creates charts when explicitly requested.
    """

    # Check if data was loaded successfully
    if df_trials.empty:
        return "Bean trial data could not be loaded.", "", {}

    # Extract API key for chart generation
    api_key = args.get('api_key')
    if not api_key:
        print("⚠️ No API key provided for chart generation")

    # Debug: Print the arguments received
    print(f"🔍 Bean query args received: {args}")
    
    # NO FILTERING - Pass full dataset to GPT always
    df = df_trials.copy()
    print(f"📊 Passing FULL dataset to GPT: {len(df)} rows")

    # Get the original question for analysis
    original_question = args.get("original_question", "")
    
    # Add analysis details based on the question - dynamically detect cultivar names
    def find_mentioned_cultivars(question_text, df):
        """Find cultivar names mentioned in the question by checking against actual dataset."""
        mentioned_cultivars = []
        question_lower = question_text.lower()
        
        # Get unique cultivar names from the dataset
        unique_cultivars = df['Cultivar Name'].dropna().unique()
        
        for cultivar in unique_cultivars:
            # Convert to string first (in case cultivar names are integers)
            cultivar_str = str(cultivar)
            cultivar_lower = cultivar_str.lower()
            cultivar_words = cultivar_lower.split()
            
            # Check if the full cultivar name or key parts are mentioned
            if (cultivar_lower in question_lower or 
                any(word in question_lower for word in cultivar_words if len(word) > 3)):
                mentioned_cultivars.append(cultivar)
        
        return mentioned_cultivars
    
    mentioned_cultivars = find_mentioned_cultivars(original_question, df)
    
    # General dynamic disambiguation system
    def detect_and_resolve_ambiguity(question, args, df):
        """
        Detect ambiguous references and attempt to resolve them using context.
        Returns (resolved_entities, needs_clarification, clarification_message)
        """
        import re
        
        # Detect potential ambiguous patterns dynamically
        ambiguous_patterns = [
            r'\b(this|that|these|those)\s+(\w+)',  # "this cultivar", "that location"
            r'\bit\b',  # standalone "it"
            r'\bthe\s+(one|same|previous|last|first)\b',  # "the same", "the previous"
        ]
        
        found_ambiguous = []
        for pattern in ambiguous_patterns:
            matches = re.findall(pattern, question.lower())
            found_ambiguous.extend(matches)
        
        if not found_ambiguous:
            return {}, False, ""
        
        # Try to resolve using function parameters (GPT's interpretation)
        resolved_params = {}
        for param, value in args.items():
            if value and param != 'original_question' and param != 'api_key':
                resolved_params[param] = value
        
        # If we have resolved parameters, validate them against the dataset
        validation_errors = []
        if resolved_params:
            for param, value in resolved_params.items():
                if param == 'cultivar':
                    if not df[df['Cultivar Name'].str.contains(str(value), case=False, na=False)].empty:
                        continue
                    else:
                        available = df['Cultivar Name'].dropna().unique()
                        validation_errors.append(f"Cultivar '{value}' not found. Available: {', '.join([str(c) for c in available[:10]])}")
                elif param == 'location':
                    if str(value).upper() in df['Location'].unique():
                        continue
                    else:
                        available = df['Location'].dropna().unique()
                        validation_errors.append(f"Location '{value}' not found. Available: {', '.join(available)}")
                elif param == 'year':
                    if int(value) in df['Year'].dropna().unique():
                        continue
                    else:
                        available = sorted(df['Year'].dropna().unique())
                        validation_errors.append(f"Year {value} not found. Available: {min(available)}-{max(available)}")
        
        if validation_errors:
            clarification = "**🤔 Reference Issue:**\n\n" + "\n".join(validation_errors) + "\n\n"
            return {}, True, clarification
        
        if resolved_params:
            return resolved_params, False, ""
        
        # If no parameters resolved, ask for clarification
        clarification = "**🤔 Clarification Needed:**\n\n"
        clarification += "Your question contains ambiguous references that I need help understanding. "
        clarification += "Could you please be more specific?\n\n"
        
        # Provide context-aware suggestions based on available data
        clarification += "**Available options:**\n"
        if 'Cultivar Name' in df.columns:
            cultivars = df['Cultivar Name'].dropna().unique()
            clarification += f"• **Cultivars:** {', '.join([str(c) for c in cultivars[:10]])}\n"
        if 'Location' in df.columns:
            locations = df['Location'].dropna().unique()
            clarification += f"• **Locations:** {', '.join(locations)}\n"
        if 'Year' in df.columns:
            years = sorted(df['Year'].dropna().unique())
            clarification += f"• **Years:** {min(years)}-{max(years)}\n"
        
        clarification += "\n**Example:** Instead of 'plot this cultivar', try 'plot Dynasty yield over time'\n\n"
        
        return {}, True, clarification
    
    # Apply general disambiguation BEFORE chart generation
    resolved_entities, needs_clarification, clarification_msg = detect_and_resolve_ambiguity(original_question, args, df)
    
    if needs_clarification:
        response = f"## 📊 **Bean Data Analysis Results**\n\n"
        response += clarification_msg
        return response, "", {}
                
    # Update mentioned_cultivars based on resolved entities
    if 'cultivar' in resolved_entities and not mentioned_cultivars:
        cultivar_matches = df[df['Cultivar Name'].str.contains(resolved_entities['cultivar'], case=False, na=False)]['Cultivar Name'].unique()
        if len(cultivar_matches) > 0:
            mentioned_cultivars = [cultivar_matches[0]]
    
    # Check if user explicitly wants a chart/visualization
    chart_keywords = ["chart", "plot", "graph", "visualization", "visualize", "show me", "create", "generate"]
    wants_chart = any(keyword in original_question.lower() for keyword in chart_keywords)
    
    # Extract basic analysis parameters
    analysis_type = args.get("analysis_type", "analysis")
    analysis_column = args.get("analysis_column", "Yield")
    chart_type = args.get("chart_type", None)

    # CONDITIONAL CHART GENERATION - Only if explicitly requested
    chart_data = {}
    
    if wants_chart and api_key and len(df) > 0:
        print("🎯 Chart explicitly requested - generating visualization")
        
        # Build enhanced prompt with resolved context
        context_additions = []
        if mentioned_cultivars:
            context_additions.append(f"Focus on cultivar: {mentioned_cultivars[0]}")
        if resolved_entities:
            for param, value in resolved_entities.items():
                if param == 'cultivar':
                    context_additions.append(f"Specific cultivar: {value}")
                elif param == 'location':
                    context_additions.append(f"Specific location: {value}")
                elif param == 'year':
                    context_additions.append(f"Specific year: {value}")
        
        context_str = ". ".join(context_additions) if context_additions else ""
        
        # Build a comprehensive prompt for GPT-4o with resolved context
        if chart_type:
            chart_prompt = f"User request: {original_question}. {context_str}. Create a {chart_type} chart based on this request. Handle all filtering, grouping, and styling as needed."
        elif analysis_type == "scatter":
            chart_prompt = f"User request: {original_question}. {context_str}. Create a scatter plot based on this request. Handle all filtering, grouping, and styling as needed."
        else:
            chart_prompt = f"User request: {original_question}. {context_str}. Create the most appropriate visualization based on this request. Handle all filtering, grouping, and styling as needed."
        
        chart_data = create_smart_chart(df, chart_prompt, api_key, context_str)
    elif wants_chart:
        print("🎯 Chart requested but no API key available")
    else:
        print("📝 No chart requested - providing data analysis only")

    # Build response focused on data analysis
    response = f"## 📊 **Bean Data Analysis Results**\n\n"
    response += f"**Dataset:** {len(df)} total records available for analysis\n\n"
    
    for cultivar_name in mentioned_cultivars:
        cultivar_data = df[df['Cultivar Name'] == cultivar_name]
        if not cultivar_data.empty:
            response += f"**{cultivar_name} Variety Analysis:**\n"
            response += f"- Found {len(cultivar_data)} records for {cultivar_name} variety\n"
            response += f"- Average yield: {cultivar_data['Yield'].mean():.1f} kg/ha\n"
            response += f"- Yield range: {cultivar_data['Yield'].min():.1f} - {cultivar_data['Yield'].max():.1f} kg/ha\n"
            response += f"- Average maturity: {cultivar_data['Maturity'].mean():.1f} days\n"
            response += f"- Years tested: {', '.join(map(str, sorted(cultivar_data['Year'].unique())))}\n"
            response += f"- Locations tested: {', '.join(cultivar_data['Location'].unique())}\n\n"
    
    if wants_chart:
        response += f"**Visualization:** {'Chart generated based on your request' if chart_data else 'Chart generation requested but unavailable'}\n\n"
    else:
        response += f"**Analysis Type:** Data analysis (no visualization requested)\n\n"

    # Show a small sample of relevant data
    response += "### 📋 **Sample Data:**\n\n"
    display_cols = [c for c in ["Year", "Location", "Cultivar Name", "Yield", "Maturity", "bean_type"] if c in df.columns]
    if display_cols:
        # Show relevant data if specific cultivar mentioned, otherwise general sample
        if mentioned_cultivars:
            # Use the first mentioned cultivar for sample data
            cultivar_name = mentioned_cultivars[0]
            cultivar_data = df[df['Cultivar Name'] == cultivar_name]
            if not cultivar_data.empty:
                sample_df = cultivar_data[display_cols].head(5)
                response += sample_df.to_markdown(index=False)
                response += f"\n\n*Showing {cultivar_name} variety data ({len(sample_df)} of {len(cultivar_data)} {cultivar_name} records)*"
            else:
                sample_df = df[display_cols].head(5)
                response += sample_df.to_markdown(index=False)
                response += f"\n\n*Sample of {len(sample_df)} rows from {len(df)} total records*"
        else:
            sample_df = df[display_cols].head(5)
            response += sample_df.to_markdown(index=False)
            response += f"\n\n*Sample of {len(sample_df)} rows from {len(df)} total records*"
    
    return response, "", chart_data
                        
# ---- GPT-compatible JSON Schema (unchanged) ----
function_schema = {
    "name": "query_bean_data",
    "description": "Query dry bean trial data with filtering, analysis, and visualization options",
    "parameters": {
        "type": "object",
        "properties": {
            "year": {"type": "integer", "description": "Single year to filter by"},
            "year_start": {"type": "integer", "description": "Start year for range filtering"},
            "year_end": {"type": "integer", "description": "End year for range filtering"},
            "location": {"type": "string", "description": "Location code (e.g., WOOD, ELOR, HARR)"},
            "bean_type": {"type": "string", "description": "Type of bean: 'white bean' or 'coloured bean'"},
            "trial_group": {"type": "string", "description": "Trial group classification: 'major' (main trials) or 'minor' (secondary trials)"},
            "cultivar": {"type": "string", "description": "Cultivar name or partial name to search for"},
            "min_yield": {"type": "number", "description": "Minimum yield threshold"},
            "max_maturity": {"type": "number", "description": "Maximum maturity days"},
            "sort": {"type": "string", "enum": ["highest", "lowest"], "description": "Sort order for results"},
            "limit": {"type": "integer", "description": "Maximum number of results to return"},
            "analysis_type": {
                "type": "string",
                "enum": ["average", "sum", "count", "max", "min", "median", "std", "similar", "compare", "yearly_average", "trend", "visualization", "scatter", "location_analysis", "cultivar_analysis"],
                "description": "Type of analysis to perform"
            },
            "analysis_column": {"type": "string", "description": "Column to analyze (Yield, Maturity, etc.)"},
            "chart_type": {"type": "string", "enum": ["pie", "bar", "line", "histogram", "area", "scatter"], "description": "Specific chart type for visualization"}
        }
    }
} 