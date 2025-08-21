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
    
    # Get unique values for categorical columns  
    categorical_info = {}
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns for prompting context only
        unique_vals = df[col].unique()
        if len(unique_vals) <= 50:  # Show more values for better context
            categorical_info[col] = list(unique_vals)  # Show all unique values

    system_msg = (
        "CRITICAL INSTRUCTION: Output ONLY executable Python code. No explanations, no text, no markdown formatting, no comments about what you're doing.\n\n"
        "🚨 MARKET CLASS VISUALIZATION RULES:\n"
        "When user mentions market classes or comparisons involving market classes:\n"
        "1. ALWAYS show average yield across ALL LOCATIONS for each market class\n"
        "2. Group data by market class (e.g., Dark Red Kidney, Light Red Kidney, White Kidney, etc.)\n"
        "3. Calculate the mean yield for each market class across all locations and years\n"
        "4. Create a comprehensive bar chart showing ALL market classes with their average yields\n"
        "5. If a specific cultivar is mentioned (e.g., Dynasty), highlight it prominently:\n"
        "   - Show Dynasty as a separate bar in RED\n"
        "   - Show its market class average in a different color\n"
        "   - Include other market classes for context\n"
        "6. Use professional titles like: 'Average Yield by Market Class (All Locations)'\n"
        "7. Sort bars by yield (highest to lowest) for better readability\n\n"
        
        "You are an INTELLIGENT chart generator that creates MEANINGFUL visualizations. Generate ONLY raw Python code that:\n"
        "- Uses the existing DataFrame `df` (NEVER create sample data, NEVER overwrite df, NEVER create new DataFrames with sample data)\n"
        "- The `df` variable contains the FULL REAL DATASET with thousands of records - USE IT DIRECTLY\n"
        "- ALWAYS check if columns exist before using them\n"
        "- ALWAYS check if values exist in columns before filtering\n"
        "- If requested data doesn't exist, create a chart with available data and informative title\n"
        "- When user requests global/world/country data but only local data is available, create a clear table showing the data limitation\n"
        "- Creates exactly ONE Plotly figure assigned to variable `fig`\n" 
        "\n"
        "🚨 CRITICAL CHART INTELLIGENCE RULES:\n"
        "- NEVER create single-value charts (1 bar, 1 point, etc.) - they are USELESS\n"
        "- If user asks about specific cultivars, ALWAYS add comparative context:\n"
        "  * Compare to other cultivars in same bean type\n"
        "  * Compare to overall averages\n"
        "  * Show trends over time if years available\n"
        "  * Compare performance across locations\n"
        "- For yield questions: show top performers vs requested cultivars\n"
        "- For single cultivar questions: show it alongside 5-10 similar cultivars\n"
        "- Make charts tell a STORY, not just show isolated data points\n"
        "- Add context like 'vs. average', 'vs. top performers', 'over time'\n"
        "- 🚨 MARKET CLASS ANALYSIS: When user mentions market classes or bean types:\n"
        "  * Show comprehensive view of ALL market classes with their average yields across all locations\n"
        "  * Group by market class (Dark Red Kidney, Light Red Kidney, White Kidney, etc.)\n"
        "  * Calculate mean yield for each market class across all locations and years\n"
        "  * If specific cultivar mentioned (e.g., Dynasty), highlight it as separate RED bar alongside its market class\n"
        "  * Sort by yield (highest to lowest) for professional presentation\n"
        "  * Title: 'Average Yield by Market Class (All Locations)' or similar\n"
        "  * This provides comprehensive market intelligence, not just simple comparisons\n\n" 
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
        
        "EXAMPLE - Robust pie chart for yield comparison:\n"
        "# Always check what data exists before creating pie chart\n"
        "print('Available columns:', df.columns.tolist())\n"
        "print('Data shape:', df.shape)\n"
        "\n"
        "# Check for specific year and provide fallback\n"
        "if 'Year' in df.columns:\n"
        "    print('Available years:', sorted(df['Year'].unique()))\n"
        "    year_data = df[df['Year'] == 2024] if 2024 in df['Year'].values else df\n"
        "    if year_data.empty:\n"
        "        print('No 2024 data, using most recent year')\n"
        "        latest_year = df['Year'].max()\n"
        "        year_data = df[df['Year'] == latest_year]\n"
        "        year_label = f' ({latest_year})'\n"
        "    else:\n"
        "        year_label = ' (2024)'\n"
        "else:\n"
        "    year_data = df\n"
        "    year_label = ''\n"
        "\n"
        "# Create pie chart with total yields by bean type\n"
        "if 'bean_type' in year_data.columns and 'Yield' in year_data.columns:\n"
        "    # Group by bean type and sum total yields\n"
        "    pie_data = year_data.groupby('bean_type')['Yield'].sum().reset_index()\n"
        "    print(f'Pie chart data: {pie_data}')\n"
        "    \n"
        "    if not pie_data.empty and pie_data['Yield'].sum() > 0:\n"
        "        fig = go.Figure(data=[go.Pie(\n"
        "            labels=pie_data['bean_type'],\n"
        "            values=pie_data['Yield'],\n"
        "            marker=dict(colors=['#FF6347', '#4682B4', '#32CD32', '#FFD700'])\n"
        "        )])\n"
        "        fig.update_layout(title=f'Total Yield by Bean Type{year_label}', height=450, width=900)\n"
        "    else:\n"
        "        fig = go.Figure()\n"
        "        fig.add_annotation(text='No yield data available for pie chart', x=0.5, y=0.5)\n"
        "        fig.update_layout(title='No Yield Data', height=450, width=900)\n"
        "else:\n"
        "    fig = go.Figure()\n"
        "    fig.add_annotation(text='Missing bean_type or Yield columns', x=0.5, y=0.5)\n"
        "    fig.update_layout(title='Missing Required Data', height=450, width=900)\n\n"
        
        "EXAMPLE - Smart comparative chart (NOT single-value):\n"
        "# If user asks about Black Velvet yield, don't show just 1 bar!\n"
        "# Instead, show it compared to other black bean cultivars\n"
        "print('Available columns:', df.columns.tolist())\n"
        "\n"
        "target_cultivar = 'Black Velvet'  # Extract from user request\n"
        "if 'Cultivar Name' in df.columns and 'Yield' in df.columns and 'Market Class' in df.columns:\n"
        "    # Get black bean cultivars using Market Class (proper filtering)\n"
        "    black_beans = df[df['Market Class'].str.contains('Black', case=False, na=False)]\n"
        "    black_cultivars = black_beans['Cultivar Name'].unique()\n"
        "    \n"
        "    if len(black_cultivars) > 1:\n"
        "        # Calculate average yield per cultivar\n"
        "        avg_yields = black_beans.groupby('Cultivar Name')['Yield'].mean().reset_index()\n"
        "        avg_yields = avg_yields.sort_values('Yield', ascending=False)\n"
        "        \n"
        "        # Highlight the target cultivar\n"
        "        colors = ['red' if x == target_cultivar else 'lightblue' for x in avg_yields['Cultivar Name']]\n"
        "        \n"
        "        fig = go.Figure(data=[go.Bar(\n"
        "            x=avg_yields['Cultivar Name'],\n"
        "            y=avg_yields['Yield'],\n"
        "            marker=dict(color=colors, line=dict(color='black', width=1))\n"
        "        )])\n"
        "        fig.update_layout(\n"
        "            title=f'{target_cultivar} vs Other Black Bean Cultivars - Average Yield',\n"
        "            xaxis_title='Cultivar',\n"
        "            yaxis_title='Average Yield (kg/ha)',\n"
        "            height=450, width=900\n"
        "        )\n"
        "    else:\n"
        "        # Not enough cultivars for comparison, return None for text-only\n"
        "        fig = None\n"
        "else:\n"
        "    fig = None\n\n"
        
        "EXAMPLE - Cross-market comparison with color coding:\n"
        "# For cross-market comparisons (e.g., Navy vs Kidney beans) with proper color distinction\n"
        "print('Available columns:', df.columns.tolist())\n"
        "print('Data shape:', df.shape)\n"
        "\n"
        "# CRITICAL EXAMPLE: OAC 23-1 vs Kidney beans cross-market comparison\n"
        "# This MUST show BOTH the cultivar (OAC 23-1) AND kidney beans on same chart\n"
        "print('Available columns:', df.columns.tolist())\n"
        "print('Data shape:', df.shape)\n"
        "\n"
        "# Extract from user request: OAC 23-1 vs Kidney beans in 2024\n"
        "if 'Year' in df.columns and 'Market Class' in df.columns and 'Cultivar Name' in df.columns:\n"
        "    # Filter for 2024 data\n"
        "    data_2024 = df[df['Year'] == 2024] if 2024 in df['Year'].values else df\n"
        "    \n"
        "    # Get OAC 23-1 data (specific cultivar)\n"
        "    oac_23_1_data = data_2024[data_2024['Cultivar Name'] == 'OAC 23-1']\n"
        "    \n"
        "    # Get kidney beans data (market class - MUST include all kidney variations)\n"
        "    kidney_data = data_2024[data_2024['Market Class'].str.contains('kidney', case=False, na=False)]\n"
        "    \n"
        "    print(f'OAC 23-1 records found: {len(oac_23_1_data)}')\n"
        "    print(f'Kidney bean records found: {len(kidney_data)}')\n"
        "    \n"
        "    # MUST show both if either exists\n"
        "    if not oac_23_1_data.empty or not kidney_data.empty:\n"
        "        fig = go.Figure()\n"
        "        \n"
        "        # Add OAC 23-1 data (RED) - by location average\n"
        "        if not oac_23_1_data.empty:\n"
        "            oac_avg = oac_23_1_data.groupby('Location')['Yield'].mean().reset_index()\n"
        "            oac_market_class = oac_23_1_data['Market Class'].iloc[0]\n"
        "            \n"
        "            fig.add_trace(go.Scatter(\n"
        "                x=oac_avg['Location'],\n"
        "                y=oac_avg['Yield'],\n"
        "                mode='markers+text+lines',\n"
        "                name=f'OAC 23-1 ({oac_market_class})',\n"
        "                text=[f'{y:.0f}' for y in oac_avg['Yield']],\n"
        "                textposition='top center',\n"
        "                marker=dict(size=12, color='red', line=dict(color='black', width=2)),\n"
        "                line=dict(color='red', width=2, dash='dash')\n"
        "            ))\n"
        "        \n"
        "        # Add kidney beans data (BLUE) - by location average\n"
        "        if not kidney_data.empty:\n"
        "            kidney_avg = kidney_data.groupby('Location')['Yield'].mean().reset_index()\n"
        "            \n"
        "            fig.add_trace(go.Scatter(\n"
        "                x=kidney_avg['Location'],\n"
        "                y=kidney_avg['Yield'],\n"
        "                mode='markers+text+lines',\n"
        "                name='Kidney Beans (Average)',\n"
        "                text=[f'{y:.0f}' for y in kidney_avg['Yield']],\n"
        "                textposition='bottom center',\n"
        "                marker=dict(size=12, color='blue', line=dict(color='black', width=2)),\n"
        "                line=dict(color='blue', width=2)\n"
        "            ))\n"
        "            \n"
        "            # Add regression line for kidney beans\n"
        "            if len(kidney_avg) > 1:\n"
        "                x_vals = np.arange(len(kidney_avg))\n"
        "                y_vals = kidney_avg['Yield'].values\n"
        "                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)\n"
        "                line_x = np.linspace(x_vals.min(), x_vals.max(), 100)\n"
        "                line_y = slope * line_x + intercept\n"
        "                \n"
        "                fig.add_trace(go.Scatter(\n"
        "                    x=kidney_avg['Location'],\n"
        "                    y=line_y,\n"
        "                    mode='lines',\n"
        "                    name=f'Kidney Regression (R² = {r_value**2:.3f})',\n"
        "                    line=dict(color='darkblue', width=3, dash='dot')\n"
        "                ))\n"
        "        \n"
        "        # Update layout\n"
        "        fig.update_layout(\n"
        "            title='OAC 23-1 (White Navy) vs Kidney Beans - 2024<br><sub>Cross-Market Class Comparison</sub>',\n"
        "            xaxis_title='Location',\n"
        "            yaxis_title='Average Yield (kg/ha)',\n"
        "            height=450, width=900,\n"
        "            showlegend=True,\n"
        "            hovermode='x unified'\n"
        "        )\n"
        "    else:\n"
        "        fig = None\n"
        "else:\n"
        "    fig = None\n\n"
        
        "EXAMPLE - Dynamic table generation with dark mode support:\n"
        "# Create a summary table with available data\n"
        "if len(df.columns) >= 2:\n"
        "    group_col = df.columns[0]  # Use first column for grouping\n"
        "    value_col = df.select_dtypes(include=['number']).columns[0] if df.select_dtypes(include=['number']).columns.any() else df.columns[1]\n"
        "    \n"
        "    table_data = df.groupby(group_col)[value_col].mean().reset_index()\n"
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
        
        "EXAMPLE - Linear regression with statistics:\n"
        "# Create scatter plot with regression line and show statistics\n"
        "import numpy as np\n"
        "from scipy import stats\n"
        "import plotly.graph_objects as go\n"
        "\n"
        "# Smart column selection based on user request\n"
        "x_col = y_col = None\n"
        "user_request_lower = str(prompt).lower()\n"
        "\n"
        "# Detect specific relationships from user request\n"
        "if 'yield' in user_request_lower and 'maturity' in user_request_lower:\n"
        "    x_col, y_col = 'Maturity', 'Yield'\n"
        "elif 'maturity' in user_request_lower and 'yield' in user_request_lower:\n"
        "    x_col, y_col = 'Maturity', 'Yield'\n"
        "elif 'seed size' in user_request_lower and 'yield' in user_request_lower:\n"
        "    x_col, y_col = 'Seed Size', 'Yield'\n"
        "else:\n"
        "    # Fallback to first two numeric columns\n"
        "    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n"
        "    if len(numeric_cols) >= 2:\n"
        "        x_col, y_col = numeric_cols[0], numeric_cols[1]\n"
        "\n"
        "if x_col and y_col and x_col in df.columns and y_col in df.columns:\n"
        "    # Remove NaN values\n"
        "    clean_df = df[[x_col, y_col]].dropna()\n"
        "    \n"
        "    if not clean_df.empty and len(clean_df) > 1:\n"
        "        x_vals, y_vals = clean_df[x_col], clean_df[y_col]\n"
        "        \n"
        "        # Calculate regression statistics\n"
        "        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)\n"
        "        \n"
        "        # Create regression line\n"
        "        line_x = np.linspace(x_vals.min(), x_vals.max(), 100)\n"
        "        line_y = slope * line_x + intercept\n"
        "        \n"
        "        fig = go.Figure()\n"
        "        \n"
        "        # Add scatter plot\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=x_vals, y=y_vals,\n"
        "            mode='markers',\n"
        "            name='Data Points',\n"
        "            marker=dict(size=6, opacity=0.6, color='blue')\n"
        "        ))\n"
        "        \n"
        "        # Add regression line\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=line_x, y=line_y,\n"
        "            mode='lines',\n"
        "            name=f'Regression Line (R² = {r_value**2:.3f})',\n"
        "            line=dict(color='red', width=3)\n"
        "        ))\n"
        "        \n"
        "        # Create detailed title with statistics\n"
        "        title = f'Linear Regression: {y_col} vs {x_col}<br>'\n"
        "        title += f'R² = {r_value**2:.3f}, Slope = {slope:.2f}, p-value = {p_value:.3e}'\n"
        "        \n"
        "        fig.update_layout(\n"
        "            title=title,\n"
        "            xaxis_title=x_col,\n"
        "            yaxis_title=y_col,\n"
        "            height=450, width=900,\n"
        "            showlegend=True\n"
        "        )\n"
        "    else:\n"
        "        fig = go.Figure()\n"
        "        fig.add_annotation(text='Insufficient data for regression analysis', x=0.5, y=0.5)\n"
        "        fig.update_layout(title='Insufficient Data', height=450, width=900)\n"
        "else:\n"
        "    fig = go.Figure()\n"
        "    fig.add_annotation(text='Required columns not found for regression', x=0.5, y=0.5)\n"
        "    fig.update_layout(title='Missing Required Columns', height=450, width=900)\n\n"
        
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
        {"role": "user", "content": "CRITICAL MARKET CLASS FILTERING: Use the 'Market Class' column for proper bean type filtering. Kidney beans = anything with 'kidney' in Market Class (case-insensitive: 'kidney', 'Kidney', 'white kidney', 'dark red kidney', 'light red kidney', 'dark red kidney bean', 'light red kidney bean'). Navy beans = 'White Navy' in Market Class. IMPORTANT: When user asks for kidney beans, filter by Market Class containing 'kidney', not by bean_type column. CROSS-MARKET COMPARISONS: When comparing different market classes, you MUST show BOTH the specific cultivar (RED) AND the market class data (BLUE) on the same chart. Do NOT show only one or the other."},
        {"role": "user", "content": "🎯 PERFORMANCE PLOT RULE: When user asks about 'performance' or 'comparing performance' of cultivars, create a SCATTER PLOT with: X-axis = Days to Maturity (Maturity column), Y-axis = Yield (kg/ha). Each cultivar should be ONE POINT (average across all locations). Label each point with cultivar name. Add regression line if multiple cultivars. DO NOT create line charts by location for performance comparisons. REQUIRED CODE PATTERN: ```python\n# Filter data\nfiltered_data = df[(df['Year'] == 2024) & (df['Market Class'].str.contains('cranberry', case=False, na=False))]\n# Group by cultivar and calculate averages\ncultivars_avg = filtered_data.groupby('Cultivar Name')[['Yield', 'Maturity']].mean().reset_index()\n# Create scatter plot\nfig.add_trace(go.Scatter(x=cultivars_avg['Maturity'], y=cultivars_avg['Yield'], mode='markers+text', text=cultivars_avg['Cultivar Name'], textposition='top center'))\n```"},
        {"role": "user", "content": "ENHANCED DATA CONTEXT: This dataset includes enriched breeding information - Market Class, Pedigree, Released Year, Disease Resistance markers (Common Mosaic Virus R1/R15, Anthracnose R17/R23/R73, Common Blight). IMPORTANT: Historical weather data is available in a separate dataset that can be accessed via db_manager.historical_data - it contains 15+ weather variables by location and year that can be linked to bean performance. CRITICAL LOCATION AGGREGATION: When showing performance by location, always group by location and calculate averages - don't show multiple data points per location unless explicitly requested."},
        {"role": "user", "content": f"User request: {prompt}"},
        {"role": "user", "content": "CRITICAL: Extract any cultivar names mentioned in the user request and use them in your analysis"},
        {"role": "user", "content": "🚨 CROSS-MARKET COMPARISON REQUIREMENT: If user asks to compare 'OAC 23-1' with 'Kidney beans', you MUST show BOTH on the same chart: OAC 23-1 data (RED) AND Kidney beans data (BLUE). Do NOT show only kidney beans - that's incomplete!"},
        {"role": "user", "content": "🚨 ABSOLUTE RULE: If your chart would only show 1 data point (1 bar, 1 value, etc.), set fig = None instead. Single-value charts are USELESS and FORBIDDEN."},
        {"role": "user", "content": "🔢 COMPLETE DATA RULE: ALWAYS show ALL available data in charts and text responses. NEVER limit, sample, or truncate data. For market class queries (cranberry beans, kidney beans, etc.), show EVERY cultivar in that market class. For 'list all' queries, show the complete list. NO exceptions - use the full dataset without any .head(), .sample(), or limiting operations."},
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
    # Import database manager for historical data access
    from database.manager import db_manager
    
    # Create safe execution environment for research use
    safe_globals = {
        "df": df, 
        "go": go, 
        "px": px,
        "pd": pd,
        "np": __import__("numpy"),
        "plotly": __import__("plotly"),
        "db_manager": db_manager,  # Add access to historical data
        "stats": __import__("scipy.stats"),  # Add scipy.stats for regression
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
    # Create local namespace with common variables that might be used in generated code
    local_ns: Dict = {
        'target_cultivar': None,
        'market_class_filter': None,
        'colors': None,
        'fig': None
    }

    code = strip_md_fences(code)
    code = re.sub(r"\bfig\.show\(\s*\)", "", code)  # remove any fig.show()

    print("─" * 60, "\nLLM-generated Plotly code:\n", code, "\n", "─" * 60)

    try:
        # Simple execution for research use
        exec(code, safe_globals, local_ns)
        
        # Check if any filtered dataframes in the code resulted in empty data
        for var_name, var_value in local_ns.items():
            if isinstance(var_value, pd.DataFrame) and var_name.endswith('_data') and var_value.empty:
                print(f"⚠️ Warning: {var_name} is empty after filtering")
                
                # Provide debugging information
                print("🔍 Available data for debugging:")
                if 'bean_type' in df.columns:
                    bean_types = df['bean_type'].unique()
                    print(f"  Bean types: {bean_types}")
                    # Check for black bean entries specifically
                    black_related = [bt for bt in bean_types if bt and ('black' in str(bt).lower() or 'colour' in str(bt).lower())]
                    if black_related:
                        print(f"  Black/Coloured bean types found: {black_related}")
                if 'Cultivar Name' in df.columns:
                    all_cultivars = df['Cultivar Name'].unique()
                    print(f"  Total cultivars: {len(all_cultivars)}")
                    # Check for black-related cultivars
                    black_cultivars = [c for c in all_cultivars if c and 'black' in str(c).lower()]
                    if black_cultivars:
                        print(f"  Black cultivars found: {black_cultivars[:10]}")
                if 'Year' in df.columns:
                    print(f"  Years: {sorted(df['Year'].unique())}")
                
                # Don't create error chart - return None to gracefully degrade to text-only
                print("⚠️ Chart generation failed: No data found after filtering")
                print("💡 Returning None - will show text analysis only")
                return None
        
        # Preferred: a variable named `fig`
        fig = local_ns.get("fig") or safe_globals.get("fig")
        
        # Check if LLM deliberately set fig = None (smart chart prevention)
        if fig is None and "fig" in local_ns:
            print("💡 LLM deliberately set fig = None - preventing useless chart")
            return None

        # Fallback: first Figure object we can find
        if fig is None:
            for v in list(local_ns.values()) + list(safe_globals.values()):
                if isinstance(v, go.Figure):
                    fig = v
                    break

        if not isinstance(fig, go.Figure):
            print("⚠️ Chart generation failed: No valid Plotly Figure produced")
            print("💡 Returning None - will show text analysis only")
            return None
        
        # Check if the figure has any data traces (handle all chart types)
        has_data = False
        if fig.data:
            for trace in fig.data:
                # Check for scatter/line plots with x/y data
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                    has_data = True
                    break
                # Check for pie charts with labels/values data
                elif hasattr(trace, 'labels') and hasattr(trace, 'values'):
                    if (trace.labels is not None and len(trace.labels) > 0 and 
                        trace.values is not None and len(trace.values) > 0):
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
            print("⚠️ Chart generation failed: No data found after filtering")
            print("💡 Returning None - will show text analysis only")
            return None
        
        # Check for single-value charts (useless visualizations)
        single_value_chart = False
        total_data_points = 0
        
        for trace in fig.data:
            trace_points = 0
            # Count data points in different chart types
            if hasattr(trace, 'x') and trace.x is not None:
                trace_points = len(trace.x)
            elif hasattr(trace, 'y') and trace.y is not None:
                trace_points = len(trace.y)
            elif hasattr(trace, 'values') and trace.values is not None:
                trace_points = len(trace.values)
            
            total_data_points += trace_points
        
        # If chart only shows 1 data point total, it's useless
        if total_data_points <= 1:
            print("⚠️ Chart generation failed: Single-value chart detected (useless)")
            print("💡 Returning None - will show text analysis only")
            return None
        
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
        
        # Handle chart generation failure gracefully
        if fig is None:
            print("📊 Chart generation failed - returning None for graceful degradation")
            return None
        
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
                if fig is None:
                    raise Exception("Chart generation returned None after bar chart fix attempt")
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
                if fig is None:
                    raise Exception("Chart generation returned None after pandas fix attempt")
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
                if fig is None:
                    raise Exception("Chart generation returned None after scoping fix attempt")
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
        
        print(f"📊 Chart generation failed: {error_msg}")
        print("💡 Returning None - will show text analysis only")
        return None 