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

def answer_bean_query(args: Dict) -> Tuple[str, str, Dict, str]:
    """
    ENHANCED VERSION: Analyze enriched bean data with historical context and optional chart generation.
    Now includes pedigree, market class, disease resistance, and environmental data.
    Also performs web search for questions about regions outside Ontario.
    """
    
    # Use database manager to get both bean and historical data
    df_trials = db_manager.bean_data
    
    # Check if data was loaded successfully
    if df_trials.empty:
        return "Bean trial data could not be loaded.", "", {}
    
    # Get historical data for environmental context (loaded lazily)
    historical_data_available = True
    try:
        hist_data = db_manager.historical_data
        if hist_data.empty:
            historical_data_available = False
    except Exception as e:
        print(f"⚠️ Historical data not available: {e}")
        historical_data_available = False

    # Get climate data for future projections (loaded lazily)
    climate_data_available = True
    try:
        climate_data = db_manager.climate_data
        if climate_data.empty:
            climate_data_available = False
    except Exception as e:
        print(f"⚠️ Climate data not available: {e}")
        climate_data_available = False

    # Extract API key for chart generation
    api_key = args.get('api_key')
    if not api_key:
        print("⚠️ No API key provided for chart generation")
    
    # Initialize chart_data to avoid UnboundLocalError
    chart_data = {}

    # Debug: Print the arguments received
    print(f"🔍 Bean query args received: {args}")
    
    # Apply market class filtering if specified
    df = df_trials.copy()
    
    # Filter by market class if provided in args
    market_class_input = args.get('market_class')
    if market_class_input:
        print(f"🔍 Filtering by market class: {market_class_input}")
        
        # Handle common market class variations
        if market_class_input.lower() in ['dark red kidney', 'dark red kidney bean', 'red kidney']:
            df = df[df['Market Class'].str.contains('dark red kidney', case=False, na=False)]
        elif market_class_input.lower() in ['kidney', 'kidney bean']:
            df = df[df['Market Class'].str.contains('kidney', case=False, na=False)]
        elif market_class_input.lower() in ['navy', 'white navy', 'navy bean']:
            df = df[df['Market Class'].str.contains('navy', case=False, na=False)]
        elif market_class_input.lower() in ['black', 'black bean']:
            df = df[df['Market Class'].str.contains('black', case=False, na=False)]
        elif market_class_input.lower() in ['cranberry', 'cranberry bean']:
            df = df[df['Market Class'].str.contains('cranberry', case=False, na=False)]
        else:
            # Generic filtering for other market classes
            df = df[df['Market Class'].str.contains(market_class_input, case=False, na=False)]
        
        print(f"✅ Filtered dataset: {len(df)} records for {market_class_input}")
        
        if df.empty:
            return f"No data found for market class '{market_class_input}' in the dataset.", "", {}
    print(f"📊 Passing FULL dataset to GPT: {len(df)} rows")

    # Get the original question for analysis
    original_question = args.get("original_question", "")
    
    # EARLY CHECK: Handle climate prediction queries BEFORE bean data validation
    # This prevents climate queries from being caught by bean data validation
    future_keywords = ['2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', 
                      '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049',
                      '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059',
                      '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069',
                      '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079',
                      '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089',
                      '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099',
                      'future', 'predict', 'prediction', 'projection', 'will be', 'climate change', 'scenario']
    is_early_climate_query = any(keyword in original_question.lower() for keyword in future_keywords)
    
    if is_early_climate_query and climate_data_available:
        print(f"🌡️ Early climate detection triggered for: {original_question}")
        
        # Extract decade and scenario information from the question
        import re
        climate_decade = None
        climate_scenario = 'RCP 4.5'  # Default to normal scenario
        
        # Extract decade from question
        decade_match = re.search(r'(20[3-9][0-9])', original_question)
        if decade_match:
            year = int(decade_match.group(1))
            # Round to nearest decade
            climate_decade = (year // 10) * 10
        
        # Extract scenario from question
        if 'best' in original_question.lower() or '2.5' in original_question:
            climate_scenario = 'RCP 2.5'
        elif 'worst' in original_question.lower() or 'worse' in original_question.lower() or '8.5' in original_question:
            climate_scenario = 'RCP 8.5'
        elif 'normal' in original_question.lower() or '4.5' in original_question:
            climate_scenario = 'RCP 4.5'
        
        # Extract location from question or args
        location_input = args.get('location') or 'Elora'  # Default to Elora if not specified
        
        # Convert location codes to full names
        location_code_mapping = {
            'ELOR': 'Elora', 'WOOD': 'Woodstock', 'STHM': 'St. Thomas', 'THOR': 'Thorndale',
            'AUBN': 'Auburn', 'WINC': 'Winchester', 'KEMPT': 'Kempton', 'FERG': 'Fergus'
        }
        
        if location_input in location_code_mapping:
            location_input = location_code_mapping[location_input]
        
        # Handle location extraction from question text if not in args
        if 'elora' in original_question.lower():
            location_input = 'Elora'
        elif 'woodstock' in original_question.lower():
            location_input = 'Woodstock'
        elif 'st. thomas' in original_question.lower() or 'st thomas' in original_question.lower():
            location_input = 'St. Thomas'
        elif 'thorndale' in original_question.lower():
            location_input = 'Thorndale'
        elif 'fergus' in original_question.lower():
            location_input = 'Fergus'
        
        # Get climate data
        if climate_decade:
            climate_info = db_manager.get_climate_data_for_location_decade(location_input, climate_decade, climate_scenario)
            
            if not climate_info.empty:
                climate_row = climate_info.iloc[0]
                
                response = f"## 🌡️ **Climate Projection for {location_input} in {climate_decade}s**\n\n"
                response += f"**📊 Climate Scenario**: {climate_row['Scenario_Description']}\n\n"
                
                response += f"**🌡️ Temperature Projections:**\n"
                response += f"- **Minimum Temperature**: {climate_row['Tmin']:.1f}°C\n"
                response += f"- **Maximum Temperature**: {climate_row['Tmax']:.1f}°C\n"
                response += f"- **Temperature Range**: {climate_row['Tmax'] - climate_row['Tmin']:.1f}°C\n\n"
                
                response += f"**🌧️ Precipitation Projection:**\n"
                response += f"- **Annual Precipitation**: {climate_row['Precipitation']:.1f} mm\n\n"
                
                # Add comparison with current conditions (2020s)
                current_climate = db_manager.get_climate_data_for_location_decade(location_input, 2020, climate_scenario)
                if not current_climate.empty:
                    current_row = current_climate.iloc[0]
                    
                    temp_change = climate_row['Tmax'] - current_row['Tmax']
                    precip_change = climate_row['Precipitation'] - current_row['Precipitation']
                    
                    response += f"**📈 Change from 2020s:**\n"
                    response += f"- **Temperature Change**: {temp_change:+.1f}°C {'🔥' if temp_change > 0 else '❄️' if temp_change < 0 else '🟡'}\n"
                    response += f"- **Precipitation Change**: {precip_change:+.1f} mm {'🌧️' if precip_change > 0 else '☀️' if precip_change < 0 else '🟡'}\n\n"
                
                # Add scenario comparison
                response += f"**🎯 Climate Scenario Information:**\n"
                response += f"- **RCP 2.5 (Best Case)**: Strong mitigation, global warming limited to ~1.5°C\n"
                response += f"- **RCP 4.5 (Normal Case)**: Moderate mitigation, global warming ~2-3°C\n"
                response += f"- **RCP 8.5 (Worst Case)**: High emissions, global warming >4°C\n\n"
                
                response += f"*Climate projections are based on Representative Concentration Pathways (RCP) scenarios from IPCC climate models.*\n"
                
                return response, response, {}, ""
            else:
                return f"**⚠️ No climate data available for {location_input} in {climate_decade}s under {climate_scenario} scenario**", "", {}
        else:
            # General climate query without specific decade
            response = f"## 🌡️ **Climate Information for {location_input}**\n\n"
            
            all_scenarios = ['RCP 2.5', 'RCP 4.5', 'RCP 8.5']
            scenario_names = ['Best Case', 'Normal Case', 'Worst Case']
            
            response += f"**🎯 Available Climate Scenarios:**\n"
            for scenario, name in zip(all_scenarios, scenario_names):
                future_data = db_manager.get_climate_data_for_location_decade(location_input, 2050, scenario)
                if not future_data.empty:
                    temp_2050 = future_data.iloc[0]['Tmax']
                    precip_2050 = future_data.iloc[0]['Precipitation']
                    response += f"- **{scenario} ({name})**: {temp_2050:.1f}°C max temp, {precip_2050:.0f}mm precipitation by 2050s\n"
            
            response += f"\n**📅 Available Decades**: 2030s, 2040s, 2050s, 2060s, 2070s, 2080s, 2090s\n"
            response += f"**💡 Try asking**: 'How will the climate be in {location_input} in 2045?' or 'Compare {location_input} climate in 2030 vs 2060'\n"
            
            return response, response, {}, ""
    
    # Check if question mentions regions/locations not in the Ontario dataset
    # Get all unique locations in the dataset for comparison
    ontario_locations = set(df['Location'].dropna().unique())
    ontario_location_names = {loc.lower() for loc in ontario_locations}
    
    # Add common location variations/mappings from the dataset
    location_mapping = {
        'aubn': 'auburn', 'sthm': 'st. thomas', 'st thomas': 'st. thomas',
        'elor': 'elora', 'wood': 'woodstock', 'winc': 'winchester',
        'blyth': 'blyth', 'harrow-blyth': 'harrow', 'harrow': 'harrow'
    }
    
    # Add mapped location names to the set
    for code, name in location_mapping.items():
        ontario_location_names.add(code.lower())
        ontario_location_names.add(name.lower())
    
    question_lower = original_question.lower()
    
    # Look for potential geographic terms that might indicate external regions
    # These are common geographic indicators that suggest the user is asking about areas beyond Ontario
    geographic_terms = [
        'production in', 'production from', 'grown in', 'cultivated in', 'farming in',
        'yields in', 'performance in', 'grown at', 'production across', 'nationwide',
        'globally', 'worldwide', 'international', 'country', 'countries', 'region', 'regions'
    ]
    
    has_geographic_context = any(term in question_lower for term in geographic_terms)
    
    # Also check for broader scope indicators
    broader_scope_terms = ['canada', 'canadian', 'nationwide', 'national', 'global', 'world', 'international']
    has_broader_scope = any(term in question_lower for term in broader_scope_terms)
    
    # If the question has geographic context or broader scope, and it's not clearly about Ontario locations only,
    # then we should supplement with web search
    is_external_region_query = has_geographic_context or has_broader_scope
    
    # Debug logging for region detection
    print(f"🔍 Question: '{original_question}'")
    print(f"🔍 Should supplement with web search: {is_external_region_query}")
    
    web_context = ""
    web_sources = []
    
    # If question has external geographic context, perform web search to supplement Ontario data
    if is_external_region_query:
        print(f"🌐 Detected non-Ontario region query - performing web search")
        api_key = args.get('api_key')
        if api_key:
            try:
                from .web_search import perform_web_search
                web_results, sources = perform_web_search(original_question, api_key)
                
                # If we got web results, use them directly
                if web_results:  # Only check for web_results, not sources (URLs may not be extracted)
                    web_context = web_results  # Use web results directly
                    web_sources = sources if sources else []  # Use sources if available, empty list if not
                    print(f"🌐 Web search completed - found {len(sources)} sources")
                    print(f"📝 Web context length: {len(web_context)} characters")
                    
                    # For non-Ontario queries with web results, return immediately with web info
                    response = f"## 🌐 **Navy Bean Cultivars in Manitoba**\n\n"
                    response += web_context
                    if sources:
                        response += "\n\n**Sources:**\n"
                        for i, source in enumerate(sources, 1):
                            response += f"{i}. {source}\n"
                    
                    # Add Ontario context as supplementary
                    response += f"\n---\n\n## 📊 **Supplementary: Ontario Bean Data Context**\n\n"
                    # Filter out NaN values and convert to strings
                    valid_locations = [str(loc) for loc in df['Location'].dropna().unique() if str(loc) != 'nan']
                    # Get year range safely
                    year_min = df['Year'].min()
                    year_max = df['Year'].max()
                    year_range = f"{year_min:.0f}-{year_max:.0f}" if not (pd.isna(year_min) or pd.isna(year_max)) else "various years"
                    response += f"For comparison, the Ontario bean trial dataset contains {len(df)} records from {year_range} covering {', '.join(valid_locations)}.\n"
                    
                    return response, response, {}, ""
                else:
                    web_context = ""
                    web_sources = []
                    print("⚠️ No web results found")
            except Exception as e:
                print(f"⚠️ Web search failed: {e}")
                web_context = ""
                web_sources = []
        else:
            print("⚠️ No API key available for web search")
            web_context = ""
            web_sources = []
    
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
    
    # CRITICAL FIX: Validate cultivar parameter from function call
    function_call_cultivar = args.get('cultivar')
    if function_call_cultivar and function_call_cultivar not in df['Cultivar Name'].values:
        print(f"🚨 WARNING: Function call suggested cultivar '{function_call_cultivar}' does not exist in dataset!")
        # Check if it's similar to any real cultivar (handle OAC 23-1D -> OAC 23-1 case)
        all_cultivars = df['Cultivar Name'].dropna().astype(str)
        
        # First try exact partial match (e.g., "OAC 23-1D" should find "OAC 23-1")
        partial_match = None
        for cultivar in all_cultivars.unique():
            cultivar_str = str(cultivar)
            # Check if the function call cultivar is a superset of an actual cultivar
            if cultivar_str in function_call_cultivar or function_call_cultivar.replace('-D', '') == cultivar_str:
                partial_match = cultivar_str
                break
        
        if partial_match:
            print(f"🔧 Fixed cultivar parameter: '{function_call_cultivar}' -> '{partial_match}'")
            args['cultivar'] = partial_match
            # Update mentioned_cultivars with corrected name
            mentioned_cultivars = [partial_match]
        else:
            # Try fuzzy matching
            similar_cultivars = all_cultivars[all_cultivars.str.contains(function_call_cultivar.split()[0] if ' ' in function_call_cultivar else function_call_cultivar[:5], case=False, na=False)]
            if not similar_cultivars.empty:
                print(f"🔍 Similar cultivars found: {list(similar_cultivars.unique())}")
                # Use the first similar cultivar
                args['cultivar'] = similar_cultivars.iloc[0]
                print(f"🔧 Fixed cultivar parameter: '{function_call_cultivar}' -> '{args['cultivar']}'")
                # Update mentioned_cultivars with corrected name
                mentioned_cultivars = [args['cultivar']]
            else:
                print(f"❌ No similar cultivars found. Removing invalid cultivar parameter.")
                args.pop('cultivar', None)  # Remove the invalid parameter

    # Track if we removed an invalid cultivar for user notification
    # Only consider it invalid if we couldn't find a correction
    invalid_cultivar_mentioned = function_call_cultivar and function_call_cultivar not in df['Cultivar Name'].values and not mentioned_cultivars
    invalid_cultivar_name = function_call_cultivar if invalid_cultivar_mentioned else None

    # After cultivar correction, update the original cultivar name tracking
    if function_call_cultivar and function_call_cultivar != args.get('cultivar'):
        print(f"📝 Original cultivar name was '{function_call_cultivar}', corrected to '{args.get('cultivar')}'")
    
    # Override function call parameters with correctly detected cultivars
    if mentioned_cultivars:
        # Update the cultivar parameter with the first detected cultivar
        corrected_cultivar = str(mentioned_cultivars[0])
        args['cultivar'] = corrected_cultivar
        print(f"🔧 Fixed cultivar parameter: '{args.get('cultivar', 'None')}' -> '{corrected_cultivar}'")
        # Update mentioned_cultivars with the corrected name to ensure consistency
        mentioned_cultivars = [corrected_cultivar]
    
    # Check for cross-market class comparison issues
    cross_market_issue = None
    if args.get('cultivar') and args.get('market_class'):
        cultivar_name = args.get('cultivar')
        requested_market_class = args.get('market_class').lower()
        
        # Get the actual market class of the requested cultivar
        cultivar_data = df[df['Cultivar Name'] == cultivar_name]
        if not cultivar_data.empty:
            actual_market_class = cultivar_data['Market Class'].iloc[0]
            actual_market_class_lower = str(actual_market_class).lower()
            
            # Check if there's a mismatch
            if requested_market_class in ['kidney'] and 'kidney' not in actual_market_class_lower:
                cross_market_issue = {
                    'cultivar': cultivar_name,
                    'actual_market_class': actual_market_class,
                    'requested_market_class': args.get('market_class')
                }
                print(f"🚨 CROSS-MARKET COMPARISON DETECTED: {cultivar_name} is {actual_market_class}, not {args.get('market_class')}")
            elif requested_market_class in ['navy', 'white navy'] and 'navy' not in actual_market_class_lower:
                cross_market_issue = {
                    'cultivar': cultivar_name,
                    'actual_market_class': actual_market_class,
                    'requested_market_class': args.get('market_class')
                }
                print(f"🚨 CROSS-MARKET COMPARISON DETECTED: {cultivar_name} is {actual_market_class}, not {args.get('market_class')}")
    
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
        # Filter out NaN values and convert to strings, limit to first 5
        valid_locations = [str(loc) for loc in df['Location'].dropna().unique() if str(loc) != 'nan'][:5]
        clarification += f"- **Locations:** {', '.join(valid_locations)}\n"
        clarification += f"- **Years:** {min(df['Year'].dropna())}-{max(df['Year'].dropna())}\n"
        
        return {}, True, clarification
    
    # Apply ambiguity detection
    resolved_entities, needs_clarification, clarification_message = detect_and_resolve_ambiguity(original_question, args, df)
    
    if needs_clarification:
        # Return the clarification message without chart
        return clarification_message, clarification_message, {}
    
    # Check if charts are requested
    chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show me', 'display', 'table', 'create', 'regression', 'linear regression', 'correlation', 'scatter', 'trend', 'relationship']
    chart_requested = any(keyword in original_question.lower() for keyword in chart_keywords)
    
    # Check if this is primarily a weather/environmental query
    weather_keywords = ['temperature', 'weather', 'precipitation', 'humidity', 'climate', 'rainfall', 'conditions']
    is_weather_query = any(keyword in original_question.lower() for keyword in weather_keywords)
    
    # Check if this is a climate prediction/future query
    future_keywords = ['2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', 
                      '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049',
                      '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059',
                      '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069',
                      '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079',
                      '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089',
                      '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099',
                      'future', 'predict', 'prediction', 'projection', 'will be', 'climate change', 'scenario']
    is_climate_prediction_query = any(keyword in original_question.lower() for keyword in future_keywords)
    
    # Extract decade and scenario information from the question
    climate_decade = None
    climate_scenario = 'RCP 4.5'  # Default to normal scenario
    
    if is_climate_prediction_query:
        import re
        # Extract decade from question
        decade_match = re.search(r'(20[3-9][0-9])', original_question)
        if decade_match:
            year = int(decade_match.group(1))
            # Round to nearest decade
            climate_decade = (year // 10) * 10
        
        # Extract scenario from question
        if 'best' in original_question.lower() or '2.5' in original_question:
            climate_scenario = 'RCP 2.5'
        elif 'worst' in original_question.lower() or 'worse' in original_question.lower() or '8.5' in original_question:
            climate_scenario = 'RCP 8.5'
        elif 'normal' in original_question.lower() or '4.5' in original_question:
            climate_scenario = 'RCP 4.5'
    
    # Check if this is a cross-analysis query (cultivars + locations + environmental factors)
    cross_analysis_keywords = ['highest temperature', 'warmest location', 'hottest location', 'highest average temperature', 
                              'location with highest', 'cultivar had the location', 'location with the most']
    is_cross_analysis = any(phrase in original_question.lower() for phrase in cross_analysis_keywords)
    
    # Handle cross-analysis queries (cultivars + locations + environmental factors)
    if is_cross_analysis and historical_data_available:
        try:
            # Location mapping for cross-analysis
            location_mapping = {
                'Auburn': 'Auburn', 'Blyth': 'Blyth', 'Elora': 'Elora', 'Granton': 'Granton',
                'Kippen': 'Kippen', 'Monkton': 'Monkton', 'St. Thomas': 'St. Thomas',
                'Thorndale': 'Thorndale', 'Winchester': 'Winchester', 'Woodstock': 'Woodstock',
                'Brussels': None, 'Brusselssels': None, 'Kempton': None, 'Kemptonton': None,
                'Harrow-Blyth': 'Harrow', 'Exeter': None,
                # Handle variations
                'AUBN': 'Auburn', 'WOOD': 'Woodstock', 'WINC': 'Winchester', 'STHM': 'St. Thomas'
            }
            
            # Get historical weather data
            hist_data = db_manager.historical_data
            
            # Calculate average temperature by location (growing season: May-September)
            location_temps = {}
            for bean_location in df['Location'].dropna().unique():
                hist_location = location_mapping.get(bean_location, bean_location)
                if hist_location and hist_location in hist_data['Location'].values:
                    location_weather = hist_data[
                        (hist_data['Location'] == hist_location) & 
                        (hist_data['Month'] >= 5) & (hist_data['Month'] <= 9)  # Growing season
                    ]
                    if not location_weather.empty:
                        avg_temp = location_weather['Temperature'].mean()
                        location_temps[bean_location] = {
                            'hist_location': hist_location,
                            'avg_temp': avg_temp,
                            'bean_location': bean_location
                        }
            
            if location_temps:
                # Find location with highest average temperature
                hottest_location = max(location_temps.keys(), key=lambda loc: location_temps[loc]['avg_temp'])
                hottest_temp = location_temps[hottest_location]['avg_temp']
                hottest_hist_location = location_temps[hottest_location]['hist_location']
                
                # Find cultivars grown at the hottest location
                hottest_location_cultivars = df[df['Location'] == hottest_location]
                
                response = f"## 🌡️ **Location Temperature Analysis**\n\n"
                response += f"**🔥 Hottest Location**: {hottest_location}"
                if hottest_location != hottest_hist_location:
                    response += f" ({hottest_hist_location})"
                response += f"\n**📊 Average Growing Season Temperature**: {hottest_temp:.1f}°C\n\n"
                
                if not hottest_location_cultivars.empty:
                    response += f"**🌱 Cultivars Grown at {hottest_location}:**\n"
                    cultivar_summary = hottest_location_cultivars.groupby('Cultivar Name').agg({
                        'Yield': 'mean',
                        'Year': ['min', 'max', 'count']
                    }).round(1)
                    
                    for cultivar in cultivar_summary.index:
                        avg_yield = cultivar_summary.loc[cultivar, ('Yield', 'mean')]
                        trial_count = cultivar_summary.loc[cultivar, ('Year', 'count')]
                        response += f"- **{cultivar}**: {avg_yield:.1f} kg/ha average ({trial_count} trials)\n"
                    
                    response += f"\n**📈 Temperature Comparison with Other Locations:**\n"
                    # Show top 5 hottest locations
                    sorted_locations = sorted(location_temps.items(), key=lambda x: x[1]['avg_temp'], reverse=True)[:5]
                    for i, (loc, data) in enumerate(sorted_locations):
                        status = "🔥" if i == 0 else f"{i+1}."
                        response += f"{status} **{loc}**: {data['avg_temp']:.1f}°C\n"
                    
                    response += f"\n*Analysis based on {len(location_temps)} locations with weather data.*"
                    
                    # Add web search context if available
                    if web_context and web_sources:
                        response += f"\n---\n\n## 🌐 **Global Context & Current Information**\n\n"
                        web_response = web_context
                        for i, source in enumerate(web_sources, 1):
                            web_citation = f"[Web-{i}]"
                            clickable_citation = f"[Web-{i}]({source})"
                            web_response = web_response.replace(web_citation, clickable_citation)
                        response += web_response
                        response += f"\n\n*🔗 Web sources are linked above for verification*\n"
                    
                    return response, response, {}, ""
                else:
                    return f"**⚠️ No cultivar data found for {hottest_location}**", "", {}
            else:
                return "**⚠️ Unable to calculate location temperatures - insufficient weather data linkage**", "", {}
                
        except Exception as e:
            print(f"⚠️ Error processing cross-analysis query: {e}")
            # Fall through to normal processing
    
    # Handle climate prediction queries (future climate scenarios)
    if is_climate_prediction_query and climate_data_available:
        try:
            print(f"🌡️ Processing climate prediction query for decade: {climate_decade}, scenario: {climate_scenario}")
            
            # Extract location from question or args
            location_input = args.get('location') or 'Elora'  # Default to Elora if not specified
            
            # Handle location extraction from question text if not in args
            if 'elora' in original_question.lower():
                location_input = 'Elora'
            elif 'woodstock' in original_question.lower():
                location_input = 'Woodstock'
            elif 'st. thomas' in original_question.lower() or 'st thomas' in original_question.lower():
                location_input = 'St. Thomas'
            
            # Get climate data
            if climate_decade:
                climate_info = db_manager.get_climate_data_for_location_decade(location_input, climate_decade, climate_scenario)
                
                if not climate_info.empty:
                    climate_row = climate_info.iloc[0]
                    
                    response = f"## 🌡️ **Climate Projection for {location_input} in {climate_decade}s**\n\n"
                    response += f"**📊 Climate Scenario**: {climate_row['Scenario_Description']}\n\n"
                    
                    response += f"**🌡️ Temperature Projections:**\n"
                    response += f"- **Minimum Temperature**: {climate_row['Tmin']:.1f}°C\n"
                    response += f"- **Maximum Temperature**: {climate_row['Tmax']:.1f}°C\n"
                    response += f"- **Temperature Range**: {climate_row['Tmax'] - climate_row['Tmin']:.1f}°C\n\n"
                    
                    response += f"**🌧️ Precipitation Projection:**\n"
                    response += f"- **Annual Precipitation**: {climate_row['Precipitation']:.1f} mm\n\n"
                    
                    # Add comparison with current conditions (2020s)
                    current_climate = db_manager.get_climate_data_for_location_decade(location_input, 2020, climate_scenario)
                    if not current_climate.empty:
                        current_row = current_climate.iloc[0]
                        
                        temp_change = climate_row['Tmax'] - current_row['Tmax']
                        precip_change = climate_row['Precipitation'] - current_row['Precipitation']
                        
                        response += f"**📈 Change from 2020s:**\n"
                        response += f"- **Temperature Change**: {temp_change:+.1f}°C {'🔥' if temp_change > 0 else '❄️' if temp_change < 0 else '🟡'}\n"
                        response += f"- **Precipitation Change**: {precip_change:+.1f} mm {'🌧️' if precip_change > 0 else '☀️' if precip_change < 0 else '🟡'}\n\n"
                    
                    # Add cultivar performance implications
                    cultivar_input = args.get('cultivar')
                    if cultivar_input and cultivar_input in df['Cultivar Name'].values:
                        cultivar_data = df[df['Cultivar Name'] == cultivar_input]
                        location_cultivar_data = cultivar_data[cultivar_data['Location'] == location_input]
                        
                        if not location_cultivar_data.empty:
                            avg_yield = location_cultivar_data['Yield'].mean()
                            avg_maturity = location_cultivar_data['Maturity'].mean()
                            
                            response += f"**🌱 {cultivar_input} Performance Context:**\n"
                            response += f"- **Historical Average Yield**: {avg_yield:.1f} kg/ha\n"
                            if not pd.isna(avg_maturity):
                                response += f"- **Average Maturity**: {avg_maturity:.0f} days\n"
                            
                            # Climate impact assessment
                            if temp_change > 2:
                                response += f"- **⚠️ Impact Assessment**: Higher temperatures may affect maturity timing and require heat-tolerant varieties\n"
                            elif temp_change > 0:
                                response += f"- **✅ Impact Assessment**: Moderate temperature increase may extend growing season\n"
                            
                            if precip_change < -50:
                                response += f"- **⚠️ Drought Risk**: Reduced precipitation may require irrigation or drought-resistant varieties\n"
                            elif precip_change > 100:
                                response += f"- **⚠️ Excess Water Risk**: Increased precipitation may require improved drainage\n"
                        
                        response += f"\n"
                    
                    # Add scenario comparison
                    response += f"**🎯 Climate Scenario Information:**\n"
                    response += f"- **RCP 2.5 (Best Case)**: Strong mitigation, global warming limited to ~1.5°C\n"
                    response += f"- **RCP 4.5 (Normal Case)**: Moderate mitigation, global warming ~2-3°C\n"
                    response += f"- **RCP 8.5 (Worst Case)**: High emissions, global warming >4°C\n\n"
                    
                    response += f"*Climate projections are based on Representative Concentration Pathways (RCP) scenarios from IPCC climate models.*\n"
                    
                    # Add web search context if available
                    if web_context and web_sources:
                        response += f"\n---\n\n## 🌐 **Global Context & Current Information**\n\n"
                        web_response = web_context
                        for i, source in enumerate(web_sources, 1):
                            web_citation = f"[Web-{i}]"
                            clickable_citation = f"[Web-{i}]({source})"
                            web_response = web_response.replace(web_citation, clickable_citation)
                        response += web_response
                        response += f"\n\n*🔗 Web sources are linked above for verification*\n"
                    
                    return response, response, {}, ""
                else:
                    return f"**⚠️ No climate data available for {location_input} in {climate_decade}s under {climate_scenario} scenario**", "", {}
            else:
                # General climate query without specific decade
                response = f"## 🌡️ **Climate Information for {location_input}**\n\n"
                
                all_scenarios = ['RCP 2.5', 'RCP 4.5', 'RCP 8.5']
                scenario_names = ['Best Case', 'Normal Case', 'Worst Case']
                
                response += f"**🎯 Available Climate Scenarios:**\n"
                for scenario, name in zip(all_scenarios, scenario_names):
                    future_data = db_manager.get_climate_data_for_location_decade(location_input, 2050, scenario)
                    if not future_data.empty:
                        temp_2050 = future_data.iloc[0]['Tmax']
                        precip_2050 = future_data.iloc[0]['Precipitation']
                        response += f"- **{scenario} ({name})**: {temp_2050:.1f}°C max temp, {precip_2050:.0f}mm precipitation by 2050s\n"
                
                response += f"\n**📅 Available Decades**: 2030s, 2040s, 2050s, 2060s, 2070s, 2080s, 2090s\n"
                response += f"**💡 Try asking**: 'How will the climate be in {location_input} in 2045?' or 'Compare {location_input} climate in 2030 vs 2060'\n"
                
                return response, response, {}, ""
                
        except Exception as e:
            print(f"⚠️ Error processing climate prediction query: {e}")
            # Fall through to normal processing
    
    # Handle pure weather queries for trial locations (including multi-location comparisons)
    if is_weather_query and args.get('location') and historical_data_available:
        try:
            # Location mapping for weather queries
            location_mapping = {
                'Auburn': 'Auburn', 'Blyth': 'Blyth', 'Elora': 'Elora', 'Granton': 'Granton',
                'Kippen': 'Kippen', 'Monkton': 'Monkton', 'St. Thomas': 'St. Thomas',
                'Thorndale': 'Thorndale', 'Winchester': 'Winchester', 'Woodstock': 'Woodstock',
                'Brussels': None, 'Brusselssels': None, 'Kempton': None, 'Kemptonton': None,
                'Harrow-Blyth': 'Harrow', 'Exeter': None,
                # Handle potential variations
                'AUBN': 'Auburn', 'WOOD': 'Woodstock', 'WINC': 'Winchester', 'STHM': 'St. Thomas',
                'ELOR': 'Elora'
            }
            
            location_input = args.get('location')
            print(f"🌍 Processing location input: {location_input}")
            
            # Handle multiple locations (comma-separated)
            locations = [loc.strip() for loc in location_input.split(',')]
            hist_data = db_manager.historical_data
            year_filter = args.get('year')
            
            weather_response = f"## 🌤️ **Weather Comparison for {len(locations)} Locations**\n\n"
            
            location_results = []
            
            for location in locations:
                hist_location = location_mapping.get(location, location)
                print(f"🔍 Mapping {location} -> {hist_location}")
                
                if hist_location:
                    location_data = hist_data[hist_data['Location'] == hist_location]
                    
                    # Apply year filter if specified
                    if year_filter:
                        location_data = location_data[location_data['Year'] == year_filter]
                        data_period = f"{year_filter}"
                    else:
                        # Get recent years data (last 5 years)
                        year_max = location_data['Year'].max()
                        if not pd.isna(year_max):
                            location_data = location_data[location_data['Year'] >= (year_max - 4)]
                            year_min_filtered = location_data['Year'].min()
                            year_max_filtered = location_data['Year'].max()
                            data_period = f"{year_min_filtered:.0f}-{year_max_filtered:.0f}" if not location_data.empty else "No data"
                        else:
                            data_period = "No year data available"
                    
                    if not location_data.empty:
                        # Calculate average conditions
                        avg_temp = location_data['Temperature'].mean()
                        max_temp = location_data['Max_Temperature'].mean()
                        min_temp = location_data['Min_Temperature'].mean()
                        avg_precip = location_data['Total_Precipitation_mm'].mean() * 365  # Annual estimate
                        avg_humidity = location_data['Relative_Humidity_2m_percent'].mean()
                        
                        location_results.append({
                            'original': location,
                            'name': hist_location,
                            'avg_temp': avg_temp,
                            'max_temp': max_temp,
                            'min_temp': min_temp,
                            'precip': avg_precip,
                            'humidity': avg_humidity,
                            'period': data_period,
                            'records': len(location_data)
                        })
                        
                        weather_response += f"### 📍 **{hist_location} Research Station**\n"
                        weather_response += f"**📊 Data Period**: {data_period}\n"
                        weather_response += f"**🌡️ Temperature**: Avg {avg_temp:.1f}°C (Range: {min_temp:.1f}°C to {max_temp:.1f}°C)\n"
                        weather_response += f"**💧 Precipitation**: ~{avg_precip:.0f}mm annually\n"
                        weather_response += f"**💨 Humidity**: {avg_humidity:.1f}%\n"
                        weather_response += f"*Based on {len(location_data):,} weather records*\n\n"
                    else:
                        weather_response += f"### ❌ **{location} ({hist_location})**\n"
                        weather_response += f"**No weather data available for {year_filter if year_filter else 'recent years'}**\n\n"
                else:
                    weather_response += f"### ❌ **{location}**\n"
                    weather_response += f"**Location mapping not found**\n\n"
            
            # Add comparison summary if multiple locations with data
            if len(location_results) > 1:
                weather_response += f"## 🏆 **Comparison Summary**\n\n"
                
                # Find hottest and coolest
                hottest = max(location_results, key=lambda x: x['max_temp'])
                coolest = min(location_results, key=lambda x: x['max_temp'])
                wettest = max(location_results, key=lambda x: x['precip'])
                driest = min(location_results, key=lambda x: x['precip'])
                
                weather_response += f"**🔥 Highest Max Temperature**: {hottest['name']} ({hottest['max_temp']:.1f}°C)\n"
                weather_response += f"**❄️ Lowest Max Temperature**: {coolest['name']} ({coolest['max_temp']:.1f}°C)\n"
                weather_response += f"**💧 Highest Precipitation**: {wettest['name']} ({wettest['precip']:.0f}mm)\n"
                weather_response += f"**🏜️ Lowest Precipitation**: {driest['name']} ({driest['precip']:.0f}mm)\n\n"
            
            # Add bean performance analysis if this is a performance comparison
            if 'performance' in original_question.lower() or 'bean' in original_question.lower():
                weather_response += f"## 🫘 **Bean Performance Analysis**\n\n"
                
                for result in location_results:
                    # Get bean trial data for this location
                    location_bean_data = df[df['Location'].str.contains(result['original'], case=False, na=False)]
                    
                    if year_filter:
                        location_bean_data = location_bean_data[location_bean_data['Year'] == year_filter]
                    
                    if not location_bean_data.empty:
                        avg_yield = location_bean_data['Yield_kg_ha'].mean()
                        trial_count = len(location_bean_data)
                        cultivar_count = location_bean_data['Cultivar Name'].nunique()
                        
                        weather_response += f"### 📈 **{result['name']} Bean Performance**\n"
                        weather_response += f"**Average Yield**: {avg_yield:.0f} kg/ha\n"
                        weather_response += f"**Trials**: {trial_count} trials, {cultivar_count} cultivars\n"
                        weather_response += f"**Environment**: {result['max_temp']:.1f}°C max temp, {result['precip']:.0f}mm precip\n\n"
                    else:
                        weather_response += f"### ❌ **{result['name']}**\n"
                        weather_response += f"**No bean trial data available for {year_filter if year_filter else 'this location'}**\n\n"
            
            weather_response += f"*Historical weather data provided by Environment and Climate Change Canada*\n"
            
            # Add web search context if available
            if web_context and web_sources:
                weather_response += f"\n---\n\n## 🌐 **Global Context & Current Information**\n\n"
                web_response = web_context
                for i, source in enumerate(web_sources, 1):
                    web_citation = f"[Web-{i}]"
                    clickable_citation = f"[Web-{i}]({source})"
                    web_response = web_response.replace(web_citation, clickable_citation)
                weather_response += web_response
                weather_response += f"\n\n*🔗 Web sources are linked above for verification*\n"
            
            return weather_response, weather_response, {}, ""
            
        except Exception as e:
            print(f"⚠️ Error processing weather query: {e}")
            import traceback
            traceback.print_exc()

    # Initialize cultivar_context to avoid UnboundLocalError
    cultivar_context = ""

    if chart_requested and api_key:
        # Generate chart and description - pass cultivar context with environmental info
        if cross_market_issue:
            cultivar_context = f"CROSS-MARKET COMPARISON: {cross_market_issue['cultivar']} is a {cross_market_issue['actual_market_class']} bean, while user requested {cross_market_issue['requested_market_class']} beans. Create a chart showing {cross_market_issue['cultivar']} performance compared to {cross_market_issue['requested_market_class']} beans. Use DIFFERENT COLORS for different market classes - highlight {cross_market_issue['cultivar']} ({cross_market_issue['actual_market_class']}) in RED and {cross_market_issue['requested_market_class']} beans in BLUE. Include both market classes in the title and legend for clarity."
        elif mentioned_cultivars:
            # Always use the corrected cultivar names for highlighting, even if original was misspelled
            cultivar_context = f"HIGHLIGHT_CULTIVAR: {', '.join([str(c) for c in mentioned_cultivars])}"
        elif invalid_cultivar_mentioned:
            # Only use this if no valid cultivars were found after correction
            cultivar_context = f"IMPORTANT: The cultivar '{invalid_cultivar_name}' mentioned in the request does not exist in the dataset. Do not highlight or reference it in the chart. Show only valid cultivars from the dataset."
        else:
            cultivar_context = ""
            
        # Add environmental context for chart generation
        if historical_data_available and 'navy' in original_question.lower():
            cultivar_context += f" ADDITIONAL CONTEXT: Historical weather data is available by location and year. The dataset includes comprehensive environmental variables (temperature, precipitation, humidity, etc.) that can be linked to bean performance by matching location names between the main dataset and historical dataset."
        chart_data = create_smart_chart(df, original_question, api_key, cultivar_context)
        
        # Handle chart generation failure gracefully
        if chart_data is None:
            print("📊 Chart generation failed - showing text analysis only")
            chart_data = {}
        
            # SPECIAL HANDLING: List all cultivars for market class queries OR latest releases
    list_keywords = ['list all', 'show all', 'all the', 'what are all', 'list every', 'all available']
    latest_keywords = ['latest', 'newest', 'most recent', 'recently released', 'new releases']
    is_list_all_query = any(keyword in original_question.lower() for keyword in list_keywords)
    is_latest_query = any(keyword in original_question.lower() for keyword in latest_keywords)
    
    # Check if this is a market class listing query
    market_class_filter = args.get('market_class')
    if is_list_all_query and market_class_filter:
        # Filter by market class and get unique cultivars
        market_class_data = df[df['Market Class'].str.contains(market_class_filter, case=False, na=False)]
        if not market_class_data.empty:
            unique_cultivars = sorted(market_class_data['Cultivar Name'].unique())
            
            response = f"## 📊 **All {market_class_filter} Bean Cultivars in Ontario**\n\n"
            response += f"Based on the Ontario bean trial dataset, here are **all {len(unique_cultivars)} {market_class_filter.lower()} bean cultivars** tested:\n\n"
            
            # Add each cultivar with key details
            for i, cultivar in enumerate(unique_cultivars, 1):
                cultivar_data = market_class_data[market_class_data['Cultivar Name'] == cultivar]
                avg_yield = cultivar_data['Yield'].mean() if 'Yield' in cultivar_data.columns else None
                avg_maturity = cultivar_data['Maturity'].mean() if 'Maturity' in cultivar_data.columns else None
                trial_count = len(cultivar_data)
                
                response += f"**{i}. {cultivar}**\n"
                if avg_yield:
                    response += f"   - Average yield: {avg_yield:.1f} kg/ha\n"
                if avg_maturity:
                    response += f"   - Average maturity: {avg_maturity:.1f} days\n"
                response += f"   - Trial records: {trial_count}\n\n"
            
            # Add summary statistics
            if avg_yield:
                total_avg_yield = market_class_data['Yield'].mean()
                response += f"**📈 {market_class_filter} Class Summary:**\n"
                response += f"- Total cultivars: {len(unique_cultivars)}\n"
                response += f"- Average yield across all cultivars: {total_avg_yield:.1f} kg/ha\n"
                if avg_maturity:
                    total_avg_maturity = market_class_data['Maturity'].mean()
                    response += f"- Average maturity: {total_avg_maturity:.1f} days\n"
                response += f"- Total trial records: {len(market_class_data)}\n\n"
            
            return response, response, {}, ""
    
    # SPECIAL HANDLING: Latest releases for market class queries
    if is_latest_query and market_class_filter:
        # Filter by market class
        market_class_data = df[df['Market Class'].str.contains(market_class_filter, case=False, na=False)]
        if not market_class_data.empty:
            # Find the most recent year with data for this market class
            if 'Released Year' in market_class_data.columns:
                # Use Released Year if available
                market_class_data_clean = market_class_data.dropna(subset=['Released Year'])
                if not market_class_data_clean.empty:
                    latest_year = int(market_class_data_clean['Released Year'].max())
                    latest_cultivars_data = market_class_data_clean[market_class_data_clean['Released Year'] == latest_year]
                    unique_latest_cultivars = sorted(latest_cultivars_data['Cultivar Name'].unique())
                    
                    response = f"## 📊 **Latest {market_class_filter} Bean Cultivars Released in Ontario**\n\n"
                    response += f"The most recent {market_class_filter.lower()} bean cultivars released in Ontario (as of {latest_year}) are:\n\n"
                    
                    # List all cultivars from the latest year
                    for i, cultivar in enumerate(unique_latest_cultivars, 1):
                        cultivar_data = latest_cultivars_data[latest_cultivars_data['Cultivar Name'] == cultivar]
                        avg_yield = cultivar_data['Yield'].mean() if 'Yield' in cultivar_data.columns else None
                        avg_maturity = cultivar_data['Maturity'].mean() if 'Maturity' in cultivar_data.columns else None
                        trial_count = len(cultivar_data)
                        
                        response += f"**{i}. {cultivar}** (Released: {latest_year})\n"
                        if avg_yield:
                            response += f"   - Average yield: {avg_yield:.1f} kg/ha\n"
                        if avg_maturity:
                            response += f"   - Average maturity: {avg_maturity:.1f} days\n"
                        response += f"   - Trial records: {trial_count}\n\n"
                    
                    # Add summary
                    response += f"**📈 Latest {market_class_filter} Releases Summary:**\n"
                    response += f"- Number of cultivars released in {latest_year}: {len(unique_latest_cultivars)}\n"
                    if avg_yield:
                        total_avg_yield = latest_cultivars_data['Yield'].mean()
                        response += f"- Average yield of latest releases: {total_avg_yield:.1f} kg/ha\n"
                    if avg_maturity:
                        total_avg_maturity = latest_cultivars_data['Maturity'].mean()
                        response += f"- Average maturity: {total_avg_maturity:.1f} days\n"
                    response += f"- Total trial records: {len(latest_cultivars_data)}\n\n"
                    
                    return response, response, {}, ""
            
            # Fallback: Use Year column if Released Year not available
            if 'Year' in market_class_data.columns:
                year_max = market_class_data['Year'].max()
                if not pd.isna(year_max):
                    latest_trial_year = int(year_max)
                else:
                    # If all years are NaN, skip this processing
                    return response, response, {}, ""
                latest_year_data = market_class_data[market_class_data['Year'] == latest_trial_year]
                unique_latest_cultivars = sorted(latest_year_data['Cultivar Name'].unique())
                
                response = f"## 📊 **Latest {market_class_filter} Bean Cultivars in Ontario Trials**\n\n"
                response += f"Based on the most recent trial data ({latest_trial_year}), here are **all {len(unique_latest_cultivars)} {market_class_filter.lower()} bean cultivars** tested:\n\n"
                
                # List all cultivars from the latest trial year
                for i, cultivar in enumerate(unique_latest_cultivars, 1):
                    cultivar_data = latest_year_data[latest_year_data['Cultivar Name'] == cultivar]
                    avg_yield = cultivar_data['Yield'].mean() if 'Yield' in cultivar_data.columns else None
                    avg_maturity = cultivar_data['Maturity'].mean() if 'Maturity' in cultivar_data.columns else None
                    trial_count = len(cultivar_data)
                    
                    response += f"**{i}. {cultivar}**\n"
                    if avg_yield:
                        response += f"   - Average yield: {avg_yield:.1f} kg/ha\n"
                    if avg_maturity:
                        response += f"   - Average maturity: {avg_maturity:.1f} days\n"
                    response += f"   - Trial records: {trial_count}\n\n"
                
                response += f"**📈 {latest_trial_year} {market_class_filter} Trial Summary:**\n"
                response += f"- Total cultivars tested: {len(unique_latest_cultivars)}\n"
                if avg_yield:
                    total_avg_yield = latest_year_data['Yield'].mean()
                    response += f"- Average yield: {total_avg_yield:.1f} kg/ha\n"
                if avg_maturity:
                    total_avg_maturity = latest_year_data['Maturity'].mean()
                    response += f"- Average maturity: {total_avg_maturity:.1f} days\n"
                response += f"- Total trial records: {len(latest_year_data)}\n\n"
                
                return response, response, {}, ""
    
    # Create a data-rich response with actual insights
    response = f"## 📊 **Bean Data Analysis**\n\n"
    
    # CRITICAL: Notify user if invalid cultivar was mentioned and no valid ones found
    if invalid_cultivar_mentioned and not mentioned_cultivars:
        response += f"⚠️ **Note:** The cultivar '{invalid_cultivar_name}' was not found in the Ontario bean trial dataset. The analysis below shows navy bean performance patterns without highlighting this specific cultivar.\n\n"
    
    # CRITICAL: Notify user about cross-market class comparison issues
    if cross_market_issue:
        response += f"📊 **Cross-Market Class Comparison:** {cross_market_issue['cultivar']} is a **{cross_market_issue['actual_market_class']}** bean, while you requested comparison with {cross_market_issue['requested_market_class']} beans. "
        response += f"The chart below shows both market classes with different colors for clear distinction.\n\n"
    
    # Add cultivar context if any were mentioned
    if mentioned_cultivars:
        response += f"**🌱 Cultivars analyzed:** {', '.join([str(c) for c in mentioned_cultivars])}\n\n"
        
        # Add specific data insights for mentioned cultivars with enriched information
        for cultivar in mentioned_cultivars:
            # Reset limited data flag for each cultivar
            has_limited_data = False
            cultivar_data = df[df['Cultivar Name'] == cultivar]
            if not cultivar_data.empty:
                    response += f"**{cultivar} Performance:**\n"
                    response += f"- **Records:** {len(cultivar_data)} trials\n"

                    # Check if cultivar has limited data (NaN values for key metrics)
                    has_limited_data = False
                    if 'Yield' in cultivar_data.columns:
                        avg_yield = cultivar_data['Yield'].mean()
                        if pd.isna(avg_yield):
                            has_limited_data = True
                            response += f"- **Average yield:** Data not available\n"
                        else:
                            response += f"- **Average yield:** {avg_yield:.2f} kg/ha\n"
                    else:
                        has_limited_data = True
                        response += f"- **Average yield:** Column not available\n"

                    if 'Maturity' in cultivar_data.columns:
                        avg_maturity = cultivar_data['Maturity'].mean()
                        if pd.isna(avg_maturity):
                            if not has_limited_data:  # Only set to True if not already True
                                has_limited_data = True
                            response += f"- **Average maturity:** Data not available\n"
                        else:
                            response += f"- **Average maturity:** {avg_maturity:.1f} days\n"
                    else:
                        if not has_limited_data:  # Only set to True if not already True
                            has_limited_data = True
                        response += f"- **Average maturity:** Column not available\n"
                    
                    # Enriched breeding information
                    if 'Market Class' in cultivar_data.columns:
                        market_class = cultivar_data['Market Class'].dropna().iloc[0] if not cultivar_data['Market Class'].dropna().empty else None
                        if market_class:
                            response += f"- **Market class:** {market_class}\n"
                    
                    if 'Released Year' in cultivar_data.columns:
                        released_year = cultivar_data['Released Year'].dropna().iloc[0] if not cultivar_data['Released Year'].dropna().empty else None
                        if released_year and not pd.isna(released_year):
                            response += f"- **Released:** {int(released_year)}\n"
                    
                    if 'Pedigree' in cultivar_data.columns:
                        pedigree = cultivar_data['Pedigree'].dropna().iloc[0] if not cultivar_data['Pedigree'].dropna().empty else None
                        if pedigree:
                            response += f"- **Pedigree:** {pedigree}\n"
                    
                    # Disease resistance information
                    resistance_traits = []
                    for col in ['Common Mosaic Virus R1', 'Common Mosaic Virus R15', 'Anthracnose R17', 'Anthracnose R23', 'Anthracnose R73', 'Common Blight']:
                        if col in cultivar_data.columns:
                            resistance = cultivar_data[col].dropna().iloc[0] if not cultivar_data[col].dropna().empty else None
                            if resistance and str(resistance).upper() == 'R':
                                trait_name = col.replace('Common Mosaic Virus R1', 'CMV R1').replace('Common Mosaic Virus R15', 'CMV R15').replace('Anthracnose R17', 'Anth R17').replace('Anthracnose R23', 'Anth R23').replace('Anthracnose R73', 'Anth R73').replace('Common Blight', 'CB')
                                resistance_traits.append(trait_name)
                    
                    if resistance_traits:
                        response += f"- **Disease resistance:** {', '.join(resistance_traits)}\n"

                    # If cultivar has limited data, perform web search to supplement information
                    if has_limited_data and api_key:
                        try:
                            print(f"🌐 Detected limited data for {cultivar}, performing web search...")
                            from .web_search import perform_web_search

                            # Create focused search query for cultivar information
                            search_query = f"{cultivar} dry bean cultivar performance yield maturity disease resistance"

                            web_results, sources = perform_web_search(search_query, api_key)

                            if web_results and web_results.strip():
                                response += f"\n**🌐 Supplementary Web Information for {cultivar}:**\n"
                                response += f"*Note: Limited performance data available locally. The following information is sourced from web search to provide more comprehensive details.*\n\n"
                                response += f"{web_results}\n"

                                if sources:
                                    response += f"**Sources:** "
                                    for i, source in enumerate(sources, 1):
                                        response += f"[Web-{i}]({source}) "
                                    response += "\n"
                                else:
                                    response += f"*Additional information sourced from web search*\n"
                            else:
                                print(f"⚠️ No web results found for {cultivar}")
                        except Exception as e:
                            print(f"⚠️ Web search failed for {cultivar}: {e}")

                    # Trial context
                    if 'Year' in cultivar_data.columns:
                        year_min = cultivar_data['Year'].min()
                        year_max = cultivar_data['Year'].max()
                        if not pd.isna(year_min) and not pd.isna(year_max):
                            years = f"{year_min:.0f}-{year_max:.0f}"
                            response += f"- **Years tested:** {years}\n"
                    # Filter out NaN values and convert to strings
                    valid_locations = [str(loc) for loc in cultivar_data['Location'].dropna().unique() if str(loc) != 'nan']
                    response += f"- **Locations:** {', '.join(valid_locations)}\n"
                    
                    # Add environmental context if available
                    if historical_data_available and 'Year' in cultivar_data.columns and 'Location' in cultivar_data.columns:
                        sample_location = cultivar_data['Location'].iloc[0]
                        year_value = cultivar_data['Year'].iloc[0]

                        # Check if both location and year values are valid (not NaN) before proceeding
                        if not pd.isna(sample_location) and not pd.isna(year_value):
                            sample_year = int(year_value)
                            env_data = db_manager.get_historical_data_for_location_year(sample_location, sample_year, 'growing_season')
                        else:
                            # Skip environmental data if location or year is invalid
                            env_data = pd.DataFrame()  # Empty DataFrame to skip processing
                        if not env_data.empty:
                            if 'Temperature' in env_data.columns:
                                avg_temp = env_data['Temperature'].iloc[0]
                                response += f"- **Growing season temp:** {avg_temp:.1f}°C (sample year/location)\n"
                            if 'Total_Precipitation_mm' in env_data.columns:
                                total_precip = env_data['Total_Precipitation_mm'].iloc[0] * 153  # Approximate growing season days
                                response += f"- **Growing season precip:** {total_precip:.0f}mm (sample year/location)\n"
                    
                    response += "\n"

        # Add market class comparison context if user asked for comparison
        comparison_keywords = ['compar', 'versus', 'vs', 'other', 'against', 'with other']
        if mentioned_cultivars and any(keyword in original_question.lower() for keyword in comparison_keywords) and market_class_input:
            # Get other cultivars in the same market class
            market_class_cultivars = df[df['Market Class'].str.contains(market_class_input, case=False, na=False)]['Cultivar Name'].unique()
            other_cultivars = [c for c in market_class_cultivars if c not in mentioned_cultivars]

            if other_cultivars:
                response += f"\n**📊 {market_class_input} Market Class Comparison:**\n"
                response += f"- **Other cultivars in this market class:** {', '.join(other_cultivars[:5])}"
                if len(other_cultivars) > 5:
                    response += f" (+{len(other_cultivars)-5} more)"
                response += "\n"

                # Show performance data for top other cultivars
                market_class_data = df[df['Market Class'].str.contains(market_class_input, case=False, na=False)]
                other_performers = market_class_data[~market_class_data['Cultivar Name'].isin(mentioned_cultivars)]
                if not other_performers.empty and 'Yield' in other_performers.columns:
                    top_others = other_performers.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False).head(3)
                    response += f"- **Top performers in {market_class_input}:**\n"
                    for cultivar, avg_yield in top_others.items():
                        trial_count = len(other_performers[other_performers['Cultivar Name'] == cultivar])
                        response += f"  - {cultivar}: {avg_yield:.1f} kg/ha ({trial_count} trials)\n"

        # Add year-specific context if specified
        year_filter = args.get('year')
        if year_filter:
            year_data = df[df['Year'] == year_filter]
            if not year_data.empty:
                response += f"\n**📅 {year_filter} Data Summary:**\n"
                response += f"- **Records in {year_filter}:** {len(year_data)}\n"
                if 'Cultivar Name' in year_data.columns:
                    unique_cultivars = year_data['Cultivar Name'].nunique()
                    response += f"- **Cultivars tested in {year_filter}:** {unique_cultivars}\n"
                    if market_class_input:
                        market_year_data = year_data[year_data['Market Class'].str.contains(market_class_input, case=False, na=False)]
                        if not market_year_data.empty:
                            market_cultivars = market_year_data['Cultivar Name'].nunique()
                            response += f"- **{market_class_input} cultivars in {year_filter}:** {market_cultivars} ({', '.join(market_year_data['Cultivar Name'].unique()[:3])}"
                            if market_cultivars > 3:
                                response += f" +{market_cultivars-3} more"
                            response += ")\n"

        # Add overall dataset context
        # Get year range safely for dataset context
        year_min = df['Year'].min()
        year_max = df['Year'].max()
        year_range = f"{year_min:.0f}-{year_max:.0f}" if not (pd.isna(year_min) or pd.isna(year_max)) else "various years"
        response += f"**📊 Dataset context:** {len(df)} total records, {year_range}\n"
        
        # CRITICAL: If no specific cultivars mentioned but question asks about performance, show top performers
        performance_keywords = ['perform', 'best', 'top', 'highest', 'yield', 'productive', 'leading']
        if not mentioned_cultivars and any(keyword in original_question.lower() for keyword in performance_keywords):
            if 'Cultivar Name' in df.columns and 'Yield' in df.columns:
                # Get top 5 performing cultivars by average yield
                top_performers = df.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False)
                response += f"\n**🏆 Top Performing Cultivars:**\n"
                for cultivar, avg_yield in top_performers.items():
                    cultivar_data = df[df['Cultivar Name'] == cultivar]
                    trial_count = len(cultivar_data)
                    response += f"- **{cultivar}**: {avg_yield:.1f} kg/ha average ({trial_count} trials)\n"
                    
                    # Add market class if available
                    if 'Market Class' in cultivar_data.columns:
                        market_class = cultivar_data['Market Class'].dropna().iloc[0] if not cultivar_data['Market Class'].dropna().empty else None
                        if market_class:
                            response += f"  - Market class: {market_class}\n"
                    
                    # Add disease resistance if available
                    resistance_traits = []
                    for col in ['Common Mosaic Virus R1', 'Common Mosaic Virus R15', 'Anthracnose R17', 'Anthracnose R23', 'Anthracnose R73', 'Common Blight']:
                        if col in cultivar_data.columns:
                            resistance = cultivar_data[col].dropna().iloc[0] if not cultivar_data[col].dropna().empty else None
                            if resistance and str(resistance).upper() == 'R':
                                trait_name = col.replace('Common Mosaic Virus R1', 'CMV R1').replace('Common Mosaic Virus R15', 'CMV R15').replace('Anthracnose R17', 'Anth R17').replace('Anthracnose R23', 'Anth R23').replace('Anthracnose R73', 'Anth R73').replace('Common Blight', 'CB')
                                resistance_traits.append(trait_name)
                    
                    if resistance_traits:
                        response += f"  - Disease resistance: {', '.join(resistance_traits)}\n"
                
                response += "\n"
        
        # Add environmental context for navy beans or specific bean types  
        bean_type_check = 'white bean' if 'white bean' in original_question.lower() else 'coloured bean' if 'coloured bean' in original_question.lower() else None
        if historical_data_available and ('navy' in original_question.lower() or 'white bean' in original_question.lower() or bean_type_check == 'white bean'):
            try:
                # Location mapping between bean dataset and historical dataset
                # Most locations now match directly thanks to your fixes!
                location_mapping = {
                    # Perfect matches (10/16 locations) - these work automatically
                    # 'Auburn', 'Blyth', 'Elora', 'Granton', 'Kippen', 'Monkton', 
                    # 'St. Thomas', 'Thorndale', 'Winchester', 'Woodstock'
                    
                    # Manual mappings for remaining 6 locations
                    'Brussels': None,  # No Brussels in historical data
                    'Brusselssels': 'Brussels',  # Assume typo → Brussels (but Brussels has no weather data)
                    'Kempton': None,  # No Kempton in historical data  
                    'Kemptonton': 'Kempton',  # Assume typo → Kempton (but Kempton has no weather data)
                    'Harrow-Blyth': 'Harrow',  # Map compound location to Harrow ✅
                    'Exeter': None,  # No Exeter in historical data
                }
                
                # Get navy bean data
                navy_bean_data = df[df['bean_type'] == 'white bean'] if 'bean_type' in df.columns else df
                if not navy_bean_data.empty:
                    # Get unique locations and years for navy beans
                    navy_locations = navy_bean_data['Location'].dropna().astype(str).unique()
                    navy_years = navy_bean_data['Year'].dropna().astype(int).unique()
                    
                    # Calculate environmental averages for navy bean growing locations
                    env_summaries = []
                    no_weather_locations = []
                    hist_data = db_manager.historical_data
                    
                    for bean_location in navy_locations[:10]:  # Check up to 10 locations
                        # Map bean location to historical location
                        hist_location = location_mapping.get(bean_location)
                        
                        if hist_location is None:
                            no_weather_locations.append(bean_location)
                            continue
                            
                        # Find matching weather data
                        location_env_data = hist_data[
                            (hist_data['Location'] == hist_location) &
                            (hist_data['Year'].isin(navy_years))
                        ]
                        
                        if not location_env_data.empty:
                            # Calculate growing season averages (May-September)
                            growing_season = location_env_data[
                                (location_env_data['Month'] >= 5) & (location_env_data['Month'] <= 9)
                            ]
                            
                            if not growing_season.empty:
                                avg_temp = growing_season['Temperature'].mean()
                                total_precip = growing_season['Total_Precipitation_mm'].sum()
                                avg_humidity = growing_season['Relative_Humidity_2m_percent'].mean()
                                
                                # Get yield for this location
                                location_yield = navy_bean_data[navy_bean_data['Location'] == bean_location]['Yield'].mean()
                                
                                env_summaries.append({
                                    'bean_location': bean_location,
                                    'hist_location': hist_location,
                                    'temp': avg_temp,
                                    'precip': total_precip,
                                    'humidity': avg_humidity,
                                    'yield': location_yield
                                })
                        else:
                            no_weather_locations.append(bean_location)
                    
                    if env_summaries:
                        response += f"**🌤️ Environmental Context for Navy Bean Locations:**\n"
                        for env in env_summaries:
                            display_name = env['bean_location'] if env['bean_location'] == env['hist_location'] else f"{env['bean_location']} ({env['hist_location']})"
                            response += f"- **{display_name}**: {env['temp']:.1f}°C, {env['precip']:.0f}mm precip, {env['humidity']:.0f}% humidity → {env['yield']:.0f} kg/ha avg yield\n"
                        response += "\n"
                        
                        # Add environmental insights
                        avg_temp_all = sum(e['temp'] for e in env_summaries) / len(env_summaries)
                        avg_precip_all = sum(e['precip'] for e in env_summaries) / len(env_summaries)
                        response += f"**🔬 Growing Season Averages**: {avg_temp_all:.1f}°C temperature, {avg_precip_all:.0f}mm precipitation\n\n"
                    
                    # Note locations without weather data
                    if no_weather_locations:
                        response += f"**📍 Note**: Weather data not available for {len(no_weather_locations)} locations: {', '.join(no_weather_locations)}\n\n"
            
            except Exception as e:
                print(f"⚠️ Error generating environmental context: {e}")
                response += f"**⚠️ Environmental data processing error** - historical weather integration needs refinement\n\n"
        
        # Add comparison insights if multiple cultivars or filtering
        elif len(mentioned_cultivars) > 1:
            response += f"**🔍 Comparison available** between {len(mentioned_cultivars)} cultivars\n"
        elif 'white bean' in original_question.lower() or 'coloured bean' in original_question.lower():
            bean_type = 'white bean' if 'white bean' in original_question.lower() else 'coloured bean'
            bean_data = df[df['bean_type'] == bean_type] if 'bean_type' in df.columns else df
            if not bean_data.empty:
                response += f"**🫘 {bean_type.title()} analysis:** {len(bean_data)} records, avg yield {bean_data['Yield'].mean():.2f} kg/ha\n"
        
        return response, response, chart_data, cultivar_context
    
    else:
        # PRIORITY: If we have web search results for non-Ontario queries, lead with that
        if web_context and is_external_region_query:
            response = f"## 🌐 **Global Bean Information**\n\n"
            
            # Convert web sources to clickable inline citations
            web_response = web_context
            for i, source in enumerate(web_sources, 1):
                web_citation = f"[Web-{i}]"
                clickable_citation = f"[Web-{i}]({source})"
                web_response = web_response.replace(web_citation, clickable_citation)
            
            response += web_response
            response += f"\n\n*🔗 Web sources are linked above for verification*\n"
            
            # Add Ontario context as supplementary information
            response += f"\n---\n\n## 📊 **Supplementary: Ontario Bean Data Context**\n\n"
            # Filter out NaN values and convert to strings
            valid_locations = [str(loc) for loc in df['Location'].dropna().unique() if str(loc) != 'nan']
            # Get year range safely for comparison
            year_min = df['Year'].min()
            year_max = df['Year'].max()
            year_range = f"{year_min:.0f}-{year_max:.0f}" if not (pd.isna(year_min) or pd.isna(year_max)) else "various years"
            response += f"For comparison, the Ontario bean trial dataset contains {len(df)} records from {year_range} covering {', '.join(valid_locations)}.\n"
            
            return response, response, {}, ""
        
        # No chart requested, provide text-based analysis
        response = f"## 📊 **Bean Data Overview**\n\n"
        
        # CRITICAL: Notify user if invalid cultivar was mentioned and no valid ones found
        if invalid_cultivar_mentioned and not mentioned_cultivars:
            response += f"⚠️ **Note:** The cultivar '{invalid_cultivar_name}' was not found in the Ontario bean trial dataset. The analysis below shows general bean performance data.\n\n"
        
        # CRITICAL: Notify user about cross-market class comparison issues
        if cross_market_issue:
            response += f"📊 **Cross-Market Class Comparison:** {cross_market_issue['cultivar']} is a **{cross_market_issue['actual_market_class']}** bean, while you requested comparison with {cross_market_issue['requested_market_class']} beans. "
            response += f"The analysis below shows both market classes for educational comparison.\n\n"
        
        # Add cultivar context if any were mentioned
        if mentioned_cultivars:
            response += f"**🌱 Cultivars mentioned:** {', '.join([str(c) for c in mentioned_cultivars])}\n\n"
        
        response += f"**📊 Dataset:** {len(df)} records from Ontario bean trials\n"
        # Get year range safely for years display
        year_min = df['Year'].min()
        year_max = df['Year'].max()
        year_range = f"{year_min:.0f}-{year_max:.0f}" if not (pd.isna(year_min) or pd.isna(year_max)) else "various years"
        response += f"**📅 Years:** {year_range}\n"
        # Filter out NaN values and convert to strings
        valid_locations = [str(loc) for loc in df['Location'].dropna().unique() if str(loc) != 'nan']
        response += f"**📍 Locations:** {', '.join(valid_locations)}\n\n"
        
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
        
        # CRITICAL: If no specific cultivars mentioned but question asks about performance, show top performers
        performance_keywords = ['perform', 'best', 'top', 'highest', 'yield', 'productive', 'leading']
        if not mentioned_cultivars and any(keyword in original_question.lower() for keyword in performance_keywords):
            if 'Cultivar Name' in df.columns and 'Yield' in df.columns:
                # Get top 5 performing cultivars by average yield
                top_performers = df.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False)
                response += f"\n**🏆 Top Performing Cultivars:**\n"
                for cultivar, avg_yield in top_performers.items():
                    cultivar_data = df[df['Cultivar Name'] == cultivar]
                    trial_count = len(cultivar_data)
                    response += f"- **{cultivar}**: {avg_yield:.1f} kg/ha average ({trial_count} trials)\n"
                response += "\n"
        
        response += f"**💡 Tip:** Ask for a chart or visualization to see the data graphically!\n"
        
        return response, response, {}, ""

# Enhanced function schema for OpenAI function calling with new data capabilities
function_schema = {
    "name": "query_bean_data",
    "description": "Query the enhanced Ontario bean trial dataset, historical weather data, AND future climate projections for comprehensive analysis including performance metrics, breeding characteristics, disease resistance, environmental context, climate predictions, and visualizations. ALSO use this for weather/climate queries about trial locations (Auburn, Blyth, Elora, etc.) as it has access to 15+ weather variables including temperature, precipitation, and humidity. NOW INCLUDES future climate data with RCP scenarios (2.5, 4.5, 8.5) for decades 2030s-2090s to predict how climate change will affect bean production. HANDLES 'list all' queries to show complete cultivar lists for market classes (e.g., 'list all cranberry beans', 'show all kidney beans') AND 'latest' queries to show ALL cultivars from the most recent release year (e.g., 'latest kidney beans', 'newest releases'). For questions about bean production/performance in regions OUTSIDE Ontario (e.g., USA, Europe, Brazil, China), this function will automatically supplement Ontario data with current global web search information. Use this when users ask about bean varieties, breeding information, disease resistance, environmental factors, weather data, climate predictions, future scenarios, global bean production, or want comparisons and charts.",
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
                "description": "Research station location codes. Single location (e.g., WOOD, WINC, STHM, AUBN) or multiple locations separated by commas for comparisons (e.g., 'WOOD, ELOR' for Woodstock vs Elora comparison) (optional)"
            },
            "year": {
                "type": "integer",
                "description": "Specific year to query (optional)"
            },
            "trait": {
                "type": "string",
                "description": "Specific trait to analyze (e.g., 'yield', 'maturity', 'harvestability', 'disease_resistance') (optional)"
            },
            "market_class": {
                "type": "string",
                "description": "Market class filter (e.g., 'White Navy', 'Black', 'Kidney', 'Pinto') (optional)"
            },
            "disease_resistance": {
                "type": "string",
                "description": "Disease resistance trait (e.g., 'CMV', 'Anthracnose', 'Common Blight') (optional)"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis requested (e.g., 'comparison', 'summary', 'chart', 'trend', 'breeding_analysis', 'environmental_context') (optional)"
            },
            "include_environmental": {
                "type": "boolean",
                "description": "Whether to include environmental/weather context in the analysis (optional)"
            }
        },
        "required": ["original_question"]
    }
} 