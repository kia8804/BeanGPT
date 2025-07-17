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
# Using OpenAI through wrapper to avoid proxy issues
from exceptions import DataProcessingError
from utils.openai_client import create_openai_client

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

def generate_plotly_code(client, prompt: str, df: pd.DataFrame) -> str:
    """Generate Plotly code using GPT-4o with dynamic dataset awareness."""
    cols = list(df.columns)
    num_cols = numeric_cols(df)
    rows = sample_rows(df)

    # Dynamically analyze the dataset
    categorical_cols = [col for col in cols if col not in num_cols]
    date_cols = [col for col in cols if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'year' in col.lower()]
    
    # Get unique values for categorical columns (limited to first 10 for brevity)
    categorical_info = {}
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        unique_vals = df[col].unique()
        if len(unique_vals) <= 20:  # Only show if reasonable number of unique values
            categorical_info[col] = list(unique_vals)[:10]

    system_msg = (
        "CRITICAL INSTRUCTION: Output ONLY executable Python code. No explanations, no text, no markdown formatting, no comments about what you're doing.\n\n"
        
        "You are a dynamic chart generator capable of creating ANY type of visualization. Generate ONLY raw Python code that:\n"
        "- Uses the existing DataFrame `df` (never create sample data or overwrite df)\n"
        "- ALWAYS check if columns exist before using them\n"
        "- ALWAYS check if values exist in columns before filtering\n"
        "- If requested data doesn't exist, create a chart with available data and informative title\n"
        "- When user requests global/world/country data but only local data is available, create a clear table showing the data limitation\n"
        "- Creates exactly ONE Plotly figure assigned to variable `fig`\n" 
        "- Uses plotly.graph_objects as go and plotly.express as px\n"
        "- Never calls fig.show()\n"
        "- Includes professional styling with update_layout()\n"
        "- Sets height=450, width=900 for proper display\n"
        "- Uses clear, descriptive axis labels and titles\n"
        "- Can create ANY chart type including but not limited to:\n"
        "  * Basic: bar, line, scatter, pie, histogram, box plots\n"
        "  * Advanced: heatmaps, treemaps, sunburst, radar charts\n"
        "  * Statistical: regression lines, correlation matrices, distribution plots\n"
        "  * 3D: surface plots, 3D scatter plots, mesh plots\n"
        "  * Tables: formatted data tables with go.Table()\n"
        "  * Custom: any creative visualization requested by user\n"
        "- For statistical analysis (regression, correlation, etc.), use pandas and numpy as needed\n"
        "- IMPORTANT: Use modern pandas syntax - NO .append() method (deprecated), use pd.concat() instead\n"
        "- CRITICAL: Initialize ALL variables outside if-blocks to avoid NameError/scoping issues\n"
        "- CRITICAL: Extract cultivar names from user request dynamically, don't hardcode specific cultivars\n"
        "- For highlighting specific items:\n"
        "  * Bar charts: use bright colors (red/orange) and line borders: marker=dict(color='red', line=dict(color='black', width=2))\n"
        "  * Scatter plots: use large markers (size=15-20) and bright colors: marker=dict(size=15, color='red', line=dict(color='black', width=2))\n"
        "  * Line charts: use thick lines (width=4) and bright colors: line=dict(color='red', width=4)\n"
        "- Uses hover templates for interactivity\n"
        "- Before filtering, always print available values to help user understand the data\n"
        "- CRITICAL: When user mentions cultivar names, extract them dynamically from the request\n"
        "- EXAMPLE: 'OAC Steam' → search for 'OAC Steam', 'Steam' → search for 'Steam'\n"
        "- DO NOT assume or hardcode cultivar names like 'OAC Seal'\n\n"
        
        "EXAMPLE - Dynamic data checking:\n"
        "# Always check what columns and values exist before filtering\n"
        "print('Available columns:', df.columns.tolist())\n"
        "print('Data shape:', df.shape)\n"
        "\n"
        "# Check if specific columns exist before using them\n"
        "if 'category_col' in df.columns:\n"
        "    print('Available categories:', df['category_col'].unique()[:10])\n"
        "    filtered_df = df[df['category_col'] == 'specific_value'] if 'specific_value' in df['category_col'].values else df\n"
        "else:\n"
        "    filtered_df = df\n"
        "\n"
        "# Use the first available numeric column for y-axis if specific column not found\n"
        "numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
        "y_col = numeric_cols[0] if numeric_cols else df.columns[0]\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[y_col]))\n"
        "fig.update_layout(title='Dynamic Chart Title', height=450, width=900)\n\n"
        
        "EXAMPLE - Dynamic table generation with dark mode support:\n"
        "# Create a summary table with available data\n"
        "if len(df.columns) >= 2:\n"
        "    group_col = df.columns[0]  # Use first column for grouping\n"
        "    value_col = df.select_dtypes(include=['number']).columns[0] if df.select_dtypes(include=['number']).columns.any() else df.columns[1]\n"
        "    \n"
        "    table_data = df.groupby(group_col)[value_col].mean().reset_index() if df[group_col].nunique() < 50 else df.head(20)\n"
        "    \n"
        "    fig = go.Figure(data=[go.Table(\n"
        "        header=dict(values=list(table_data.columns), \n"
        "                   fill_color='#4A90E2', \n"
        "                   font=dict(color='white', size=12),\n"
        "                   align='left'),\n"
        "        cells=dict(values=[table_data[col] for col in table_data.columns], \n"
        "                   fill_color='#F8F9FA', \n"
        "                   font=dict(color='black', size=11),\n"
        "                   align='left')\n"
        "    )])\n"
        "    fig.update_layout(title='Data Summary Table', height=450, width=900, \n"
        "                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')\n"
        "else:\n"
        "    fig = go.Figure()\n"
        "    fig.add_annotation(text='Insufficient data for table generation', x=0.5, y=0.5)\n"
        "    fig.update_layout(title='No Data Available', height=450, width=900)\n\n"
        
        "EXAMPLE - Adding rows to DataFrame (MODERN PANDAS):\n"
        "# DO NOT use .append() - it's deprecated! Use pd.concat() instead\n"
        "base_data = df.groupby('category')['value'].mean().reset_index()\n"
        "# To add a new row, create a new DataFrame and concatenate\n"
        "new_row = pd.DataFrame({'category': ['New Category'], 'value': [123.45]})\n"
        "combined_data = pd.concat([base_data, new_row], ignore_index=True)\n"
        "# Or add multiple rows at once\n"
        "additional_rows = pd.DataFrame({\n"
        "    'category': ['Cat1', 'Cat2'], \n"
        "    'value': [100, 200]\n"
        "})\n"
        "final_data = pd.concat([base_data, additional_rows], ignore_index=True)\n"
        "# For tables, use dark mode friendly colors:\n"
        "fig = go.Figure(data=[go.Table(\n"
        "    header=dict(values=['Category', 'Value'], fill_color='#4A90E2', font=dict(color='white', size=12)),\n"
        "    cells=dict(values=[final_data['category'], final_data['value']], fill_color='#F8F9FA', font=dict(color='black', size=11))\n"
        ")])\n"
        "fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')\n\n"
        
        "EXAMPLE - Dynamic scatter plot:\n"
        "# Create scatter plot with available numeric columns\n"
        "numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
        "if len(numeric_cols) >= 2:\n"
        "    x_col, y_col = numeric_cols[0], numeric_cols[1]\n"
        "    fig = go.Figure()\n"
        "    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers'))\n"
        "    fig.update_layout(title=f'{y_col} vs {x_col}', xaxis_title=x_col, yaxis_title=y_col, height=450, width=900)\n"
        "else:\n"
        "    fig = go.Figure()\n"
        "    fig.add_annotation(text='Need at least 2 numeric columns for scatter plot', x=0.5, y=0.5)\n"
        "    fig.update_layout(title='Insufficient Numeric Data', height=450, width=900)\n\n"
        
        "EXAMPLE - Linear regression:\n"
        "# Create scatter plot with regression line\n"
        "import numpy as np\n"
        "numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
        "if len(numeric_cols) >= 2:\n"
        "    x_col, y_col = numeric_cols[0], numeric_cols[1]\n"
        "    # Remove NaN values\n"
        "    clean_df = df[[x_col, y_col]].dropna()\n"
        "    x_vals, y_vals = clean_df[x_col], clean_df[y_col]\n"
        "    \n"
        "    # Calculate regression line\n"
        "    z = np.polyfit(x_vals, y_vals, 1)\n"
        "    p = np.poly1d(z)\n"
        "    \n"
        "    fig = go.Figure()\n"
        "    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data'))\n"
        "    fig.add_trace(go.Scatter(x=x_vals, y=p(x_vals), mode='lines', name='Regression Line'))\n"
        "    fig.update_layout(title=f'Linear Regression: {y_col} vs {x_col}', height=450, width=900)\n\n"
        
        "EXAMPLE - Heatmap/Correlation matrix:\n"
        "# Create correlation heatmap\n"
        "numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
        "if len(numeric_cols) >= 2:\n"
        "    corr_matrix = df[numeric_cols].corr()\n"
        "    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, \n"
        "                                    x=corr_matrix.columns, \n"
        "                                    y=corr_matrix.columns,\n"
        "                                    colorscale='RdBu'))\n"
        "    fig.update_layout(title='Correlation Matrix', height=450, width=900)\n\n"
        
        "EXAMPLE - Handling global data request with local data and specific cultivar:\n"
        "# When user asks for global/world/country comparison but only local data is available\n"
        "# IMPORTANT: Initialize variables outside if-blocks to avoid scoping issues\n"
        "if 'Location' in df.columns and 'Yield' in df.columns:\n"
        "    location_data = df.groupby('Location').agg({'Yield': 'mean', 'Year': 'nunique'}).reset_index()\n"
        "    location_data = location_data.sort_values('Yield', ascending=False)\n"
        "    \n"
        "    # Initialize cultivar data - MUST be outside any if-blocks\n"
        "    cultivar_avg_yield = None\n"
        "    cultivar_name = None\n"
        "    \n"
        "    # CRITICAL: Extract cultivar names dynamically from user request\n"
        "    # DO NOT hardcode specific cultivars like 'OAC Seal'\n"
        "    if 'Cultivar Name' in df.columns:\n"
        "        available_cultivars = df['Cultivar Name'].unique()\n"
        "        request_words = str(prompt).lower().split()\n"
        "        \n"
        "        # Look for cultivar names mentioned in user request\n"
        "        user_request_lower = str(prompt).lower()  # prompt contains the user request\n"
        "        for cultivar in available_cultivars:\n"
        "            cultivar_str = str(cultivar).lower()\n"
        "            # Check if cultivar name appears in request\n"
        "            if cultivar_str in user_request_lower:\n"
        "                cultivar_name = cultivar\n"
        "                print(f'Found cultivar in request: {cultivar_name}')\n"
        "                break\n"
        "            # Also check for partial matches (e.g., 'Steam' matches 'Steam')\n"
        "            elif any(word in cultivar_str for word in request_words if len(word) > 3):\n"
        "                cultivar_name = cultivar\n"
        "                print(f'Found cultivar by partial match: {cultivar_name}')\n"
        "                break\n"
        "    \n"
        "    # Check for specific cultivar data - only if cultivar was found\n"
        "    if cultivar_name and 'Cultivar Name' in df.columns:\n"
        "        # Try exact match first\n"
        "        cultivar_data = df[df['Cultivar Name'] == cultivar_name]\n"
        "        if cultivar_data.empty and cultivar_name:\n"
        "            # Try case-insensitive match\n"
        "            cultivar_data = df[df['Cultivar Name'].str.contains(cultivar_name, case=False, na=False)]\n"
        "        \n"
        "        if not cultivar_data.empty:\n"
        "            cultivar_avg_yield = cultivar_data['Yield'].mean()\n"
        "            print(f'Found {len(cultivar_data)} records for {cultivar_name}, avg yield: {cultivar_avg_yield:.1f}')\n"
        "        else:\n"
        "            print(f'No data found for cultivar: {cultivar_name}')\n"
        "    elif not cultivar_name:\n"
        "        print('No specific cultivar mentioned in request')\n"
        "    \n"
        "    # Create comparison column - now cultivar_avg_yield is always defined\n"
        "    comparison_vals = []\n"
        "    for yield_val in location_data['Yield']:\n"
        "        if cultivar_avg_yield is not None:\n"
        "            if yield_val > cultivar_avg_yield:\n"
        "                comparison_vals.append(f'Higher than {cultivar_name}')\n"
        "            elif yield_val < cultivar_avg_yield:\n"
        "                comparison_vals.append(f'Lower than {cultivar_name}')\n"
        "            else:\n"
        "                comparison_vals.append(f'Equal to {cultivar_name}')\n"
        "        else:\n"
        "            comparison_vals.append('No cultivar data')\n"
        "    \n"
        "    # Add summary rows\n"
        "    locations = [f'Ontario Station: {loc}' for loc in location_data['Location']] + ['NOTE: Global data not available']\n"
        "    yields = list(location_data['Yield'].round(1)) + ['Only Ontario research station data']\n"
        "    comparisons = comparison_vals + ['No international comparison possible']\n"
        "    \n"
        "    # Add cultivar row if data exists\n"
        "    if cultivar_avg_yield is not None:\n"
        "        locations.append(f'{cultivar_name} (Ontario avg)')\n"
        "        yields.append(round(cultivar_avg_yield, 1))\n"
        "        comparisons.append('Reference cultivar')\n"
        "    \n"
        "    # Create informative table\n"
        "    fig = go.Figure(data=[go.Table(\n"
        "        header=dict(values=['Data Available', 'Average Yield (kg/ha)', 'Comparison'], \n"
        "                   fill_color='#4A90E2', \n"
        "                   font=dict(color='white', size=12),\n"
        "                   align='left'),\n"
        "        cells=dict(values=[locations, yields, comparisons], \n"
        "                   fill_color='#F8F9FA', \n"
        "                   font=dict(color='black', size=11),\n"
        "                   align='left')\n"
        "    )])\n"
        "    fig.update_layout(title='Ontario Research Station Data (Global Data Not Available)', \n"
        "                      height=450, width=900, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')\n\n"
        
        "OUTPUT ONLY THE PYTHON CODE. NO OTHER TEXT."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Dataset info:\n- Columns: {cols}\n- Numeric columns: {num_cols}\n- Categorical columns: {categorical_cols}\n- Date columns: {date_cols}"},
        {"role": "user", "content": f"Sample categorical values: {categorical_info}"},
        {"role": "user", "content": f"Sample data (first 3 rows): {rows[:3]}"},
        {"role": "user", "content": f"User request: {prompt}"},
        {"role": "user", "content": "CRITICAL: Extract any cultivar names mentioned in the user request and use them in your analysis"},
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
    # Create safe execution environment for research use
    safe_globals = {
        "df": df, 
        "go": go, 
        "px": px,
        "pd": pd,
        "np": __import__("numpy"),
        "plotly": __import__("plotly"),
        "print": print,
        "len": len,
        "range": range,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "enumerate": enumerate,
        "zip": zip,
    }
    local_ns: Dict = {}

    code = strip_md_fences(code)
    code = re.sub(r"\bfig\.show\(\s*\)", "", code)  # remove any fig.show()

    print("─" * 60, "\nLLM-generated Plotly code:\n", code, "\n", "─" * 60)

    try:
        # Simple execution for research use
        exec(code, safe_globals, local_ns)
        
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
            raise DataProcessingError("No plotly Figure was produced.")
        
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
            
        raise DataProcessingError(error_msg) from e

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
        
        # Create OpenAI client
        client = create_openai_client(api_key)
        
        # Build comprehensive prompt for GPT-4o
        prompt = f"{user_request}"
        if context:
            prompt += f" Context: {context}"
        
        # Let GPT-4o analyze the data and create the perfect chart
        code = generate_plotly_code(client, prompt, df)
        
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
                fixed_code = generate_plotly_code(client, fixed_prompt, df)
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
        error_msg = str(e)
        
        # Check for deprecated pandas methods and attempt auto-fix
        if "has no attribute 'append'" in error_msg or "append" in error_msg:
            print("🔄 Attempting to fix deprecated pandas .append() method...")
            try:
                # Add specific instruction to use pd.concat() instead of .append()
                fixed_prompt = f"{prompt}\n\nCRITICAL: DO NOT use DataFrame.append() method (deprecated). Use pd.concat() instead. Example: pd.concat([df1, df2], ignore_index=True)"
                fixed_code = generate_plotly_code(client, fixed_prompt, df)
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
                error_msg = f"Chart generation failed even after pandas syntax fix attempt: {str(retry_error)}"
        
        # Check for variable scoping issues and attempt auto-fix
        elif "is not defined" in error_msg or "NameError" in error_msg:
            print("🔄 Attempting to fix variable scoping/NameError issue...")
            try:
                # Add specific instruction about variable initialization
                fixed_prompt = f"{prompt}\n\nCRITICAL: Initialize ALL variables outside if-blocks to avoid NameError. Example: var = None (before if-block), then set var = value (inside if-block)."
                fixed_code = generate_plotly_code(client, fixed_prompt, df)
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
                error_msg = f"Chart generation failed even after scoping fix attempt: {str(retry_error)}"
        
        print(f"Error creating chart: {error_msg}")
        return {
            "type": "error",
            "error": f"Failed to generate chart: {error_msg}",
            "title": "Chart Generation Error"
        } 