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
                web_context = web_results
                web_sources = sources
                print(f"🌐 Web search completed - found {len(sources)} sources")
            except Exception as e:
                print(f"⚠️ Web search failed: {e}")
        else:
            print("⚠️ No API key available for web search")
    
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
        else:
            # Try fuzzy matching
            similar_cultivars = all_cultivars[all_cultivars.str.contains(function_call_cultivar.split()[0] if ' ' in function_call_cultivar else function_call_cultivar[:5], case=False, na=False)]
            if not similar_cultivars.empty:
                print(f"🔍 Similar cultivars found: {list(similar_cultivars.unique())}")
                # Use the first similar cultivar
                args['cultivar'] = similar_cultivars.iloc[0]
                print(f"🔧 Fixed cultivar parameter: '{function_call_cultivar}' -> '{args['cultivar']}'")
            else:
                print(f"❌ No similar cultivars found. Removing invalid cultivar parameter.")
                args.pop('cultivar', None)  # Remove the invalid parameter
    
    # Track if we removed an invalid cultivar for user notification
    invalid_cultivar_mentioned = function_call_cultivar and function_call_cultivar not in df['Cultivar Name'].values
    invalid_cultivar_name = function_call_cultivar if invalid_cultivar_mentioned else None
    
    # Override function call parameters with correctly detected cultivars
    if mentioned_cultivars:
        # Update the cultivar parameter with the first detected cultivar
        args['cultivar'] = str(mentioned_cultivars[0])
        print(f"🔧 Fixed cultivar parameter: '{args.get('cultivar', 'None')}' -> '{mentioned_cultivars[0]}'")
    
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
        clarification += f"- **Locations:** {', '.join(df['Location'].dropna().unique()[:5])}\n"
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
                    
                    return response, response, {}
                else:
                    return f"**⚠️ No cultivar data found for {hottest_location}**", "", {}
            else:
                return "**⚠️ Unable to calculate location temperatures - insufficient weather data linkage**", "", {}
                
        except Exception as e:
            print(f"⚠️ Error processing cross-analysis query: {e}")
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
                        location_data = location_data[location_data['Year'] >= (location_data['Year'].max() - 4)]
                        data_period = f"{location_data['Year'].min()}-{location_data['Year'].max()}"
                    
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
            
            return weather_response, weather_response, {}
            
        except Exception as e:
            print(f"⚠️ Error processing weather query: {e}")
            import traceback
            traceback.print_exc()
    
    if chart_requested and api_key:
        # Generate chart and description - pass cultivar context with environmental info
        if invalid_cultivar_mentioned:
            cultivar_context = f"IMPORTANT: The cultivar '{invalid_cultivar_name}' mentioned in the request does not exist in the dataset. Do not highlight or reference it in the chart. Show only valid cultivars from the dataset."
        elif cross_market_issue:
            cultivar_context = f"CROSS-MARKET COMPARISON: {cross_market_issue['cultivar']} is a {cross_market_issue['actual_market_class']} bean, while user requested {cross_market_issue['requested_market_class']} beans. Create a chart showing {cross_market_issue['cultivar']} performance compared to {cross_market_issue['requested_market_class']} beans. Use DIFFERENT COLORS for different market classes - highlight {cross_market_issue['cultivar']} ({cross_market_issue['actual_market_class']}) in RED and {cross_market_issue['requested_market_class']} beans in BLUE. Include both market classes in the title and legend for clarity."
        elif mentioned_cultivars:
            cultivar_context = f"Focus on these cultivars: {', '.join([str(c) for c in mentioned_cultivars])}"
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
        
        # Create a data-rich response with actual insights
        response = f"## 📊 **Bean Data Analysis**\n\n"
        
        # CRITICAL: Notify user if invalid cultivar was mentioned
        if invalid_cultivar_mentioned:
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
                cultivar_data = df[df['Cultivar Name'] == cultivar]
                if not cultivar_data.empty:
                    response += f"**{cultivar} Performance:**\n"
                    response += f"- **Records:** {len(cultivar_data)} trials\n"
                    
                    # Core performance metrics
                    if 'Yield' in cultivar_data.columns:
                        avg_yield = cultivar_data['Yield'].mean()
                        response += f"- **Average yield:** {avg_yield:.2f} kg/ha\n"
                    if 'Maturity' in cultivar_data.columns:
                        avg_maturity = cultivar_data['Maturity'].mean()
                        response += f"- **Average maturity:** {avg_maturity:.1f} days\n"
                    
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
                    
                    # Trial context
                    if 'Year' in cultivar_data.columns:
                        years = f"{cultivar_data['Year'].min():.0f}-{cultivar_data['Year'].max():.0f}"
                        response += f"- **Years tested:** {years}\n"
                    response += f"- **Locations:** {', '.join(cultivar_data['Location'].unique())}\n"
                    
                    # Add environmental context if available
                    if historical_data_available and 'Year' in cultivar_data.columns and 'Location' in cultivar_data.columns:
                        sample_location = cultivar_data['Location'].iloc[0]
                        sample_year = int(cultivar_data['Year'].iloc[0])
                        env_data = db_manager.get_historical_data_for_location_year(sample_location, sample_year, 'growing_season')
                        if not env_data.empty:
                            if 'Temperature' in env_data.columns:
                                avg_temp = env_data['Temperature'].iloc[0]
                                response += f"- **Growing season temp:** {avg_temp:.1f}°C (sample year/location)\n"
                            if 'Total_Precipitation_mm' in env_data.columns:
                                total_precip = env_data['Total_Precipitation_mm'].iloc[0] * 153  # Approximate growing season days
                                response += f"- **Growing season precip:** {total_precip:.0f}mm (sample year/location)\n"
                    
                    response += "\n"
        
        # Add overall dataset context
        response += f"**📊 Dataset context:** {len(df)} total records, {df['Year'].min()}-{df['Year'].max()}\n"
        
        # CRITICAL: If no specific cultivars mentioned but question asks about performance, show top performers
        performance_keywords = ['perform', 'best', 'top', 'highest', 'yield', 'productive', 'leading']
        if not mentioned_cultivars and any(keyword in original_question.lower() for keyword in performance_keywords):
            if 'Cultivar Name' in df.columns and 'Yield' in df.columns:
                # Get top 5 performing cultivars by average yield
                top_performers = df.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False).head(5)
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
        
        return response, response, chart_data
    
    else:
        # No chart requested, provide text-based analysis
        response = f"## 📊 **Bean Data Overview**\n\n"
        
        # CRITICAL: Notify user if invalid cultivar was mentioned
        if invalid_cultivar_mentioned:
            response += f"⚠️ **Note:** The cultivar '{invalid_cultivar_name}' was not found in the Ontario bean trial dataset. The analysis below shows general bean performance data.\n\n"
        
        # CRITICAL: Notify user about cross-market class comparison issues
        if cross_market_issue:
            response += f"📊 **Cross-Market Class Comparison:** {cross_market_issue['cultivar']} is a **{cross_market_issue['actual_market_class']}** bean, while you requested comparison with {cross_market_issue['requested_market_class']} beans. "
            response += f"The analysis below shows both market classes for educational comparison.\n\n"
        
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
        
        # CRITICAL: If no specific cultivars mentioned but question asks about performance, show top performers
        performance_keywords = ['perform', 'best', 'top', 'highest', 'yield', 'productive', 'leading']
        if not mentioned_cultivars and any(keyword in original_question.lower() for keyword in performance_keywords):
            if 'Cultivar Name' in df.columns and 'Yield' in df.columns:
                # Get top 5 performing cultivars by average yield
                top_performers = df.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False).head(5)
                response += f"\n**🏆 Top Performing Cultivars:**\n"
                for cultivar, avg_yield in top_performers.items():
                    cultivar_data = df[df['Cultivar Name'] == cultivar]
                    trial_count = len(cultivar_data)
                    response += f"- **{cultivar}**: {avg_yield:.1f} kg/ha average ({trial_count} trials)\n"
                response += "\n"
        
        response += f"**💡 Tip:** Ask for a chart or visualization to see the data graphically!\n"
        
        # Add web search context if available (for non-Ontario region queries)
        if web_context and web_sources:
            response += f"\n---\n\n## 🌐 **Global Context & Current Information**\n\n"
            
            # Convert web sources to clickable inline citations
            web_response = web_context
            for i, source in enumerate(web_sources, 1):
                web_citation = f"[Web-{i}]"
                clickable_citation = f"[Web-{i}]({source})"
                web_response = web_response.replace(web_citation, clickable_citation)
            
            response += web_response
            response += f"\n\n*🔗 Web sources are linked above for verification*\n"
        
        return response, response, {}

# Enhanced function schema for OpenAI function calling with new data capabilities
function_schema = {
    "name": "query_bean_data",
    "description": "Query the enhanced Ontario bean trial dataset AND historical weather data for comprehensive analysis including performance metrics, breeding characteristics, disease resistance, environmental context, and visualizations. ALSO use this for weather/climate queries about trial locations (Auburn, Blyth, Elora, etc.) as it has access to 15+ weather variables including temperature, precipitation, and humidity. For questions about bean production/performance in regions OUTSIDE Ontario (e.g., USA, Europe, Brazil, China), this function will automatically supplement Ontario data with current global web search information. Use this when users ask about bean varieties, breeding information, disease resistance, environmental factors, weather data, global bean production, or want comparisons and charts.",
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