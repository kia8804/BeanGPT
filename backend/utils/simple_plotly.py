"""
Simplified Plotly chart generation using GPT-4o.
Based on the user's tested dynamic_graph_server_plotly.py approach.
"""

import ast
import json
import re
import traceback
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

def strip_md_fences(code: str) -> str:
    """Remove ``` fences and any explanatory text, extracting only Python code."""
    # Look for code blocks first
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, code, re.DOTALL)
    
    if matches:
        # Take the first code block found
        code = matches[0].strip()
    else:
        # If no code blocks, try to extract lines that look like Python code
        lines = code.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip explanatory text, look for actual Python code
            if (line.strip().startswith(('#', 'import ', 'from ', 'df', 'fig', 'steam_', 'filtered_')) or
                'go.Figure' in line or 'px.' in line or '=' in line and not line.strip().startswith('To ') and not line.strip().startswith('This ')):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip() and not line.strip().startswith(('To ', 'This ', 'The ', 'We ', 'If ', 'However')):
                code_lines.append(line)
            elif line.strip() == '':
                if in_code:
                    code_lines.append(line)
        
        if code_lines:
            code = '\n'.join(code_lines)
    
    return code.strip()

def sample_rows(df: pd.DataFrame, n: int = 15) -> list:
    """Return a stratified sample of rows, not just head()."""
    if df.empty:
        return []
    return df.sample(n=min(n, len(df)), random_state=42).to_dict(orient="records")

def numeric_cols(df: pd.DataFrame) -> list:
    """Return numeric column names only (int / float)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def ask_llm_for_plotly(prompt: str, df: pd.DataFrame, api_key: str) -> str:
    """Generate Plotly chart code using OpenAI - exactly like your tested version."""
    client = OpenAI(api_key=api_key)
    
    cols = list(df.columns)
    num_cols = numeric_cols(df)
    rows = sample_rows(df)

    system_msg = (
        "CRITICAL INSTRUCTION: Output ONLY executable Python code. No explanations, no text, no markdown formatting, no comments about what you're doing.\n\n"
        
        "You are a code generator for a dry bean research platform. Generate ONLY raw Python code that:\n"
        "- Uses the existing DataFrame `df` (never create sample data or overwrite df)\n"
        "- DATASET CONTEXT: This is Ontario research station data with locations like WOOD, WINC, STHM, AUBN, etc.\n"
        "- DO NOT filter for 'Ontario' - ALL data is already from Ontario research stations\n"
        "- Location column contains research station codes (WOOD, WINC, etc.), not country names\n"
        "- Bean types in dataset: 'White Bean' (includes navy beans), 'Coloured Bean'\n"
        "- IMPORTANT: Navy beans = White beans in this dataset. Use 'White Bean' not 'navy bean'\n"
        "- This is NOT global data - it's all Ontario bean trial data from different research stations\n"
        "- CRITICAL: If user asks for global/world/country data that doesn't exist, create a chart showing available Ontario station data instead\n"
        "- When global data is requested but unavailable, create an informative title like 'Ontario Research Station Comparison (Global Data Not Available)'\n"
        "- IMPORTANT: Before applying complex filters, check if the requested data actually exists in the dataset\n"
        "- If filtering for specific conditions, first check if those values exist in the relevant columns\n"
        "- If the requested conditions don't exist, create a simple chart with available data and a descriptive title explaining what's shown\n"
        "- Creates exactly ONE Plotly figure assigned to variable `fig`\n" 
        "- Uses plotly.graph_objects as go and plotly.express as px\n"
        "- Never calls fig.show()\n"
        "- Includes professional styling with update_layout()\n"
        "- Sets height=450, width=900 for proper display\n"
        "- Uses clear axis labels with units (e.g., 'Yield (kg/ha)', 'Maturity (days)')\n"
        "- Includes descriptive title that reflects what data is actually shown\n"
        "- For highlighting specific items:\n"
        "  * Bar charts: use bright colors (red/orange) and line borders: marker=dict(color='red', line=dict(color='black', width=2))\n"
        "  * Scatter plots: use large markers (size=15-20) and bright colors: marker=dict(size=15, color='red', line=dict(color='black', width=2))\n"
        "  * Line charts: use thick lines (width=4) and bright colors: line=dict(color='red', width=4)\n"
        "- Uses hover templates for interactivity\n\n"
        
        "EXAMPLE - Check data before filtering:\n"
        "# Always check what values actually exist before filtering\n"
        "print('Available bean types:', df['bean_type'].unique() if 'bean_type' in df.columns else 'No bean_type column')\n"
        "print('Available cultivars:', df['Cultivar Name'].unique()[:10])  # Show first 10\n"
        "print('Available years:', sorted(df['Year'].unique()) if 'Year' in df.columns else 'No Year column')\n"
        "\n"
        "# For navy beans, use 'White Bean' (the actual value in dataset)\n"
        "if 'White Bean' in df['bean_type'].values:\n"
        "    bean_df = df[df['bean_type'] == 'White Bean']\n"
        "else:\n"
        "    bean_df = df  # Use all data if no bean type filtering possible\n"
        "\n"
        "# Check if specific cultivar exists\n"
        "if 'Steam' in bean_df['Cultivar Name'].values:\n"
        "    filtered_df = bean_df[bean_df['Cultivar Name'] == 'Steam']\n"
        "else:\n"
        "    filtered_df = bean_df  # Show all data if specific cultivar not found\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Yield']))\n"
        "fig.update_layout(title='Chart Title', height=450, width=900)\n\n"
        
        "EXAMPLE - Global request with Ontario data:\n"
        "# User asks for global country comparison but we only have Ontario stations\n"
        "stations_df = df.groupby('Location')['Yield'].mean().reset_index()\n"
        "fig = go.Bar(x=stations_df['Location'], y=stations_df['Yield'])\n"
        "fig = go.Figure(data=fig)\n"
        "fig.update_layout(title='Ontario Research Stations Yield Comparison (Global Data Not Available)', height=450, width=900)\n\n"
        
        "EXAMPLE - Navy bean comparison (use 'White Bean' not 'navy bean'):\n"
        "# Filter for white beans (which includes navy beans in this dataset)\n"
        "white_beans = df[df['bean_type'] == 'White Bean'] if 'bean_type' in df.columns else df\n"
        "# Group by cultivar and calculate mean yield\n"
        "grouped_df = white_beans.groupby('Cultivar Name')['Yield'].mean().reset_index()\n"
        "# Highlight specific cultivar\n"
        "colors = ['red' if x == 'Steam' else 'blue' for x in grouped_df['Cultivar Name']]\n"
        "fig = go.Figure(data=go.Bar(x=grouped_df['Cultivar Name'], y=grouped_df['Yield'], \n"
        "                            marker=dict(color=colors, line=dict(color='black', width=1))))\n"
        "fig.update_layout(title='White Bean Cultivar Yield Comparison (Navy Beans)', height=450, width=900)\n\n"
        
        "OUTPUT ONLY THE PYTHON CODE. NO OTHER TEXT."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Available columns: {cols}, numeric columns: {num_cols}"},
        {"role": "user", "content": f"Sample data: {rows[:3]}"},  # Reduce sample size
        {"role": "user", "content": f"Generate Python code for: {prompt}"},
        {"role": "user", "content": "RESPOND WITH ONLY PYTHON CODE - NO EXPLANATORY TEXT"},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

def run_generated_code(code: str, df: pd.DataFrame) -> go.Figure:
    """Execute the LLM-generated code and return the Plotly Figure."""
    safe_globals = {
        "df": df, 
        "go": go, 
        "px": px,
        "pd": pd,
        "np": __import__("numpy"),
        "plotly": __import__("plotly")
    }
    local_ns: Dict = {}

    code = strip_md_fences(code)
    code = re.sub(r"\bfig\.show\(\s*\)", "", code)  # remove any fig.show()

    print("─" * 60, "\nLLM-generated Plotly code:\n", code, "\n", "─" * 60)

    try:
        ast.parse(code, mode="exec")  # syntax check
        exec(code, safe_globals, local_ns)  # run
        
        # Check if any filtered dataframes in the code resulted in empty data
        for var_name, var_value in local_ns.items():
            if isinstance(var_value, pd.DataFrame) and var_name.endswith('_df') and var_value.empty:
                print(f"⚠️ Warning: {var_name} is empty after filtering")
                
                # Provide debugging information
                print("🔍 Available data for debugging:")
                if 'bean_type' in df.columns:
                    print(f"  Bean types: {df['bean_type'].unique()}")
                if 'Cultivar Name' in df.columns:
                    print(f"  Cultivars: {df['Cultivar Name'].unique()[:10]}...")
                if 'Year' in df.columns:
                    print(f"  Years: {sorted(df['Year'].unique())}")
                
                # Create an informative error figure with helpful context
                available_info = []
                if 'bean_type' in df.columns:
                    available_info.append(f"Available bean types: {', '.join(df['bean_type'].unique())}")
                if 'Cultivar Name' in df.columns:
                    cultivar_count = len(df['Cultivar Name'].unique())
                    available_info.append(f"Available cultivars: {cultivar_count} varieties")
                if 'Year' in df.columns:
                    years = sorted(df['Year'].unique())
                    available_info.append(f"Available years: {min(years)}-{max(years)}")
                
                fig = go.Figure()
                error_text = "No data found matching the specified criteria.<br><br>"
                error_text += "<br>".join(available_info)
                error_text += "<br><br>Tip: Navy beans are stored as 'White Bean' in this dataset."
                
                fig.add_annotation(
                    text=error_text,
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="red")
                )
                fig.update_layout(
                    title="No Data Found - Check Filter Criteria",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=450, width=900
                )
                return fig
        
        # Preferred: a variable named `fig`
        fig = local_ns.get("fig") or safe_globals.get("fig")

        # Fallback: first Figure object we can find
        if fig is None:
            for v in list(local_ns.values()) + list(safe_globals.values()):
                if isinstance(v, go.Figure):
                    fig = v
                    break

        if not isinstance(fig, go.Figure):
            raise RuntimeError("No plotly Figure was produced.")
        
        # Check if the figure has any data traces (handle both plots and tables)
        has_data = False
        if fig.data:
            for trace in fig.data:
                # Check for scatter/line plots with x/y data
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                    has_data = True
                    break
                # Check for tables with cell data
                elif hasattr(trace, 'cells') and trace.cells is not None:
                    if hasattr(trace.cells, 'values') and trace.cells.values is not None:
                        # Check if table has actual data (not just headers)
                        if any(len(col) > 0 for col in trace.cells.values if col is not None):
                            has_data = True
                            break
                # Check for other chart types (bar, histogram, etc.)
                elif hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                    has_data = True
                    break
        
        if not has_data:
            print("⚠️ Warning: Generated figure has no data points")
            # Create an informative error figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data points to display.<br>The filtering criteria may be too restrictive or the requested data may not exist in this dataset.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="No Data Points Found",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=450, width=900
            )
        
        return fig

    except Exception as e:
        traceback.print_exc()
        
        # Handle common Plotly property errors with specific guidance
        if "Invalid property specified for object of type" in str(e):
            if "Marker: 'size'" in str(e) and "bar" in str(e).lower():
                error_msg = "Chart generation failed: 'size' property not valid for bar charts. Use color and line properties instead."
            elif "invalid property" in str(e).lower():
                error_msg = f"Chart generation failed: Invalid Plotly property used. {str(e)}"
            else:
                error_msg = f"Chart generation failed: Plotly configuration error. {str(e)}"
        else:
            error_msg = f"Generated code failed → {e}"
            
        raise RuntimeError(error_msg) from e

def create_smart_chart(df: pd.DataFrame, user_request: str, api_key: str, context: str = "") -> Dict:
    """
    SINGLE FUNCTION to create any chart - let GPT-4o decide everything!
    This replaces all 19+ specific chart functions.
    """
    try:
        if df.empty:
            return {
                "type": "error",
                "error": "No data available for visualization",
                "title": "Chart Generation Error"
            }
        
        # Build comprehensive prompt for GPT-4o
        prompt = f"{user_request}"
        if context:
            prompt += f" Context: {context}"
        
        # Let GPT-4o analyze the data and create the perfect chart
        code = ask_llm_for_plotly(prompt, df, api_key)
        
        # Execute the code to create the figure
        fig = run_generated_code(code, df)
        
        # Convert to JSON for frontend
        fig_json = fig.to_json()
        
        # Extract title from the generated figure if available
        chart_title = "Data Visualization"
        if fig.layout and fig.layout.title and fig.layout.title.text:
            chart_title = fig.layout.title.text
        
        return {
            "type": "plotly",
            "data": json.loads(fig_json),
            "title": chart_title,
            "generated_code": code
        }
        
    except RuntimeError as e:
        error_msg = str(e)
        
        # Try to auto-fix common issues and retry once
        if "'size' property not valid for bar charts" in error_msg:
            print("🔄 Attempting to fix bar chart marker size issue...")
            try:
                # Add specific instruction to avoid size property for bar charts
                fixed_prompt = f"{prompt}\n\nIMPORTANT: For bar charts, DO NOT use 'size' property in marker dict. Use only 'color' and 'line' properties."
                fixed_code = ask_llm_for_plotly(fixed_prompt, df, api_key)
                fig = run_generated_code(fixed_code, df)
                fig_json = fig.to_json()
                
                chart_title = "Data Visualization"
                if fig.layout and fig.layout.title and fig.layout.title.text:
                    chart_title = fig.layout.title.text
                
                return {
                    "type": "plotly",
                    "data": json.loads(fig_json),
                    "title": chart_title,
                    "generated_code": fixed_code
                }
            except Exception as retry_error:
                error_msg = f"Chart generation failed even after auto-fix attempt: {str(retry_error)}"
        
        print(f"Error creating chart: {error_msg}")
        return {
            "type": "error",
            "error": error_msg,
            "title": "Chart Generation Error"
        }
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        return {
            "type": "error",
            "error": f"Failed to generate chart: {str(e)}",
            "title": "Chart Generation Error"
        } 