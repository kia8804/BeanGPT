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
from database.manager import db_manager

def answer_bean_query(args: Dict) -> Tuple[str, str, Dict]:
    """
    SIMPLIFIED VERSION: Analyze bean data with optional chart generation.
    Only creates charts when explicitly requested.
    """
    
    # Use database manager to get bean data
    df_trials = db_manager.bean_data
    
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
    
    # CRITICAL FIX: Override function call parameters with correctly detected cultivars
    if mentioned_cultivars:
        # Update the cultivar parameter with the first detected cultivar
        args['cultivar'] = str(mentioned_cultivars[0])
        print(f"🔧 Fixed cultivar parameter: '{args.get('cultivar', 'None')}' -> '{mentioned_cultivars[0]}'")
    
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
        clarification += f"- **Cultivars:** {', '.join([str(c) for c in df['Cultivar Name'].dropna().unique()[:8]])}...\n"
        clarification += f"- **Locations:** {', '.join(df['Location'].dropna().unique()[:5])}\n"
        clarification += f"- **Years:** {min(df['Year'].dropna())}-{max(df['Year'].dropna())}\n"
        
        return {}, True, clarification
    
    # Apply ambiguity detection
    resolved_entities, needs_clarification, clarification_message = detect_and_resolve_ambiguity(original_question, args, df)
    
    if needs_clarification:
        # Return the clarification message without chart
        return clarification_message, clarification_message, {}
    
    # Check if charts are requested
    chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show me', 'display', 'table', 'create']
    chart_requested = any(keyword in original_question.lower() for keyword in chart_keywords)
    
    if chart_requested and api_key:
        # Generate chart and description - pass cultivar context
        cultivar_context = f"Focus on these cultivars: {', '.join([str(c) for c in mentioned_cultivars])}" if mentioned_cultivars else ""
        chart_data = create_smart_chart(df, original_question, api_key, cultivar_context)
        
        # Create a data-rich response with actual insights
        response = f"## 📊 **Bean Data Analysis**\n\n"
        
        # Add cultivar context if any were mentioned
        if mentioned_cultivars:
            response += f"**🌱 Cultivars analyzed:** {', '.join([str(c) for c in mentioned_cultivars])}\n\n"
            
            # Add specific data insights for mentioned cultivars
            for cultivar in mentioned_cultivars:
                cultivar_data = df[df['Cultivar Name'] == cultivar]
                if not cultivar_data.empty:
                    response += f"**{cultivar} Performance:**\n"
                    response += f"- **Records:** {len(cultivar_data)} trials\n"
                    if 'Yield' in cultivar_data.columns:
                        avg_yield = cultivar_data['Yield'].mean()
                        response += f"- **Average yield:** {avg_yield:.2f} kg/ha\n"
                    if 'Maturity' in cultivar_data.columns:
                        avg_maturity = cultivar_data['Maturity'].mean()
                        response += f"- **Average maturity:** {avg_maturity:.1f} days\n"
                    if 'Year' in cultivar_data.columns:
                        years = f"{cultivar_data['Year'].min()}-{cultivar_data['Year'].max()}"
                        response += f"- **Years tested:** {years}\n"
                    response += f"- **Locations:** {', '.join(cultivar_data['Location'].unique())}\n\n"
        
        # Add overall dataset context
        response += f"**📊 Dataset context:** {len(df)} total records, {df['Year'].min()}-{df['Year'].max()}\n"
        
        # Add comparison insights if multiple cultivars or filtering
        if len(mentioned_cultivars) > 1:
            response += f"**🔍 Comparison available** between {len(mentioned_cultivars)} cultivars\n"
        elif 'white bean' in original_question.lower() or 'coloured bean' in original_question.lower():
            bean_type = 'white bean' if 'white bean' in original_question.lower() else 'coloured bean'
            bean_data = df[df['bean_type'] == bean_type] if 'bean_type' in df.columns else df
            if not bean_data.empty:
                response += f"**🫘 {bean_type.title()} analysis:** {len(bean_data)} records, avg yield {bean_data['Yield'].mean():.2f} kg/ha\n"
        
        return response, response, chart_data
    
    else:
        # No chart requested, provide text-based analysis
        response = f"## 📊 **Bean Data Overview**\n\n"
        
        # Add cultivar context if any were mentioned
        if mentioned_cultivars:
            response += f"**🌱 Cultivars mentioned:** {', '.join([str(c) for c in mentioned_cultivars])}\n\n"
        
        response += f"**📊 Dataset:** {len(df)} records from Ontario bean trials\n"
        response += f"**📅 Years:** {df['Year'].min()}-{df['Year'].max()}\n"
        response += f"**📍 Locations:** {', '.join(df['Location'].dropna().unique())}\n\n"
        
        # Add summary statistics
        if 'Cultivar Name' in df.columns:
            unique_cultivars = df['Cultivar Name'].dropna().nunique()
            response += f"**🌱 Unique cultivars:** {unique_cultivars}\n"
        
        if 'Yield' in df.columns and not df['Yield'].isna().all():
            avg_yield = df['Yield'].mean()
            min_yield = df['Yield'].min()
            max_yield = df['Yield'].max()
            response += f"**🌾 Yield range:** {min_yield:.1f} - {max_yield:.1f} kg/ha (avg: {avg_yield:.1f})\n"
        
        if 'Maturity' in df.columns and not df['Maturity'].isna().all():
            avg_maturity = df['Maturity'].mean()
            min_maturity = df['Maturity'].min()
            max_maturity = df['Maturity'].max()
            response += f"**⏰ Maturity range:** {min_maturity:.0f} - {max_maturity:.0f} days (avg: {avg_maturity:.1f})\n"
        
        response += f"\n**💡 Tip:** Ask for a chart or visualization to see the data graphically!\n"
        
        return response, response, {}

# Function schema for OpenAI function calling
function_schema = {
    "name": "query_bean_data",
    "description": "Query the Ontario bean trial dataset for cultivar performance, yield data, maturity information, and location-specific results. Use this when users ask about specific bean varieties, yields, growing conditions, or want to compare cultivars.",
    "parameters": {
        "type": "object",
        "properties": {
            "original_question": {
                "type": "string",
                "description": "The original user question for context"
            },
            "cultivar": {
                "type": "string",
                "description": "Specific cultivar name to query (optional)"
            },
            "location": {
                "type": "string", 
                "description": "Research station location (e.g., WOOD, WINC, STHM, AUBN) (optional)"
            },
            "year": {
                "type": "integer",
                "description": "Specific year to query (optional)"
            },
            "trait": {
                "type": "string",
                "description": "Specific trait to analyze (e.g., 'yield', 'maturity', 'lodging') (optional)"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis requested (e.g., 'comparison', 'summary', 'chart', 'trend') (optional)"
            }
        },
        "required": ["original_question"]
    }
} 