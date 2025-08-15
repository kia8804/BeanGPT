"""
Web search functionality using OpenAI's web search API for up-to-date information.
"""

import json
from typing import List, Dict, Tuple, Optional
from utils.openai_client import create_openai_client


def needs_current_info(question: str) -> bool:
    """
    Determine if a question requires current/recent information that might not be in OpenAI's training data.
    
    Args:
        question: The user's question
        
    Returns:
        bool: True if the question likely needs current information
    """
    # Keywords that indicate need for current information
    current_info_keywords = [
        # Time-related
        "current", "latest", "recent", "now", "today", "this year", "2024", "2023", 
        "updated", "new", "modern", "contemporary",
        
        # Statistics and rankings
        "largest", "biggest", "top producer", "leading producer", "first producer",
        "highest production", "most", "leader", "ranking", "statistics", "data",
        
        # Market and economic terms
        "market", "price", "cost", "economy", "trade", "export", "import",
        "production statistics", "global production", "world production",
        
        # Geographic/country-specific current info
        "which country", "what country", "where is", "who produces",
        
        # Bean-specific current info that might change
        "dry bean production", "bean producer", "bean production by country",
        "legume production", "pulse production"
    ]
    
    # Check if question contains current info indicators
    question_lower = question.lower()
    
    # Direct current info keywords
    if any(keyword in question_lower for keyword in current_info_keywords):
        return True
    
    # Questions about world/global facts that change over time
    global_patterns = [
        "world's", "global", "worldwide", "internationally", "by country",
        "which countries", "top countries", "leading countries"
    ]
    
    if any(pattern in question_lower for pattern in global_patterns):
        return True
    
    # Superlative questions (biggest, largest, etc.) often need current data
    superlative_patterns = [
        "biggest", "largest", "highest", "most", "top", "best", "greatest",
        "leading", "primary", "main", "major", "first", "number one"
    ]
    
    if any(pattern in question_lower for pattern in superlative_patterns):
        return True
    
    return False


def perform_web_search(question: str, api_key: str) -> Tuple[str, List[str]]:
    """
    Perform web search using OpenAI's Responses API with web_search_preview tool.
    
    Args:
        question: The search query
        api_key: OpenAI API key
        
    Returns:
        Tuple of (search_results_text, source_urls)
    """
    try:
        client = create_openai_client(api_key)
        
        # Use OpenAI's Responses API with web search tool
        response = client.responses.create(
            model="gpt-4o",  # or gpt-4o-mini for faster responses
            tools=[{
                "type": "web_search_preview",  # Using preview version as shown in docs
                "search_context_size": "medium"  # balanced context and latency
            }],
            input=f"Search for current information about: {question}. Focus on recent data, statistics, and authoritative sources."
        )
        
        # Extract the response content
        search_content = response.output_text or ""
        
        # Extract source URLs from annotations if available
        source_urls = []
        try:
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    # Access object attributes directly, not as dict
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    for annotation in content_item.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                                            if hasattr(annotation, 'url') and annotation.url:
                                                url = annotation.url
                                                # Clean up URL by removing OpenAI tracking parameters
                                                clean_url = url.split('?utm_source=openai')[0] if '?utm_source=openai' in url else url
                                                if clean_url not in source_urls:
                                                    source_urls.append(clean_url)
        except Exception as e:
            print(f"⚠️ Error extracting URLs: {e}")
        
        print(f"🌐 Web search completed for: {question}")
        print(f"📝 Search result length: {len(search_content)} characters")
        print(f"🔗 Found {len(source_urls)} source URLs")
        
        return search_content, source_urls
        
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return "", []


def combine_web_and_rag_context(
    web_results: str, 
    rag_context: str, 
    question: str
) -> str:
    """
    Combine web search results with RAG context, prioritizing web search for current information.
    
    Args:
        web_results: Results from web search
        rag_context: Context from RAG/literature search
        question: Original question
        
    Returns:
        Combined context with web results prioritized and proper citation instructions
    """
    if not web_results and not rag_context:
        return ""
    
    if not web_results:
        return rag_context
    
    if not rag_context:
        return f"""## 🌐 Current Web Information

{web_results}

**Citation Instructions**: Use [Web-1], [Web-2], etc. to cite web sources in your response."""
    
    # Combine with web results prioritized and citation instructions
    combined_context = f"""## 🌐 Current Web Information (PRIORITIZED)

{web_results}

**Web Citation Instructions**: Use [Web-1], [Web-2], etc. to cite web sources above.

## 📚 Scientific Literature Context

{rag_context}

**Literature Citation Instructions**: Use [1], [2], [3], etc. to cite research papers above.

---

**IMPORTANT RESPONSE FORMATTING REQUIREMENTS**:
1. **Prioritize web information** - Use current data from web sources as primary information
2. **Include in-text citations** - Cite web sources as [Web-1], [Web-2] and literature as [1], [2]
3. **Structure your response** with clear sections and professional formatting
4. **Combine insights** from both current web data and scientific literature
5. **Note data currency** - Clearly indicate when using current vs. historical information"""
    
    return combined_context


def create_web_enhanced_response(
    question: str, 
    web_results: str, 
    conversation_history: List[Dict], 
    api_key: str
) -> str:
    """
    Create a response that prioritizes web search results over OpenAI's training data.
    
    Args:
        question: User's question
        web_results: Results from web search
        conversation_history: Previous conversation
        api_key: OpenAI API key
        
    Returns:
        Enhanced response prioritizing web information
    """
    try:
        client = create_openai_client(api_key)
        
        # Create enhanced system prompt that prioritizes web information
        system_prompt = (
            "You are a helpful dry bean research assistant with access to current web information. "
            "IMPORTANT: When web search results are provided, prioritize that information over your training data "
            "as it contains the most current and accurate information available. "
            "If web results contradict your training data, trust the web results. "
            "Always cite when you're using current web information vs. your general knowledge. "
            "For research questions, mention that you can also help with bean breeding data and genetics literature. "
            "Format your response clearly with sections and use the web information as the primary source.\n\n"
            "**Important:** If asked about who developed or created this system, say it was developed by "
            "the Dry Bean Breeding & Computational Biology Program at the University of Guelph in 2025. "
            "If asked when you were made or created, say 2025. Do not mention OpenAI or any other AI companies."
        )
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add web context if available
        if web_results:
            enhanced_question = f"""Question: {question}

🌐 **Current Web Information (PRIORITIZE THIS):**
{web_results}

**CITATION INSTRUCTIONS**: Use [Web-1], [Web-2], etc. to cite web sources in your response.

Please answer the question above using the current web information provided as your PRIMARY source. Structure your response clearly and include in-text citations [Web-1], [Web-2] when referencing the web sources."""
        else:
            enhanced_question = question
        
        messages.append({"role": "user", "content": enhanced_question})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,  # Slightly higher than web search but still factual
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Enhanced response generation failed: {e}")
        # Fallback to simple response with better formatting
        if web_results:
            return f"""## 🌐 Current Information

{web_results}

*The above information was obtained from current web sources and provides the most up-to-date data available.*"""
        else:
            return "I'm sorry, I couldn't retrieve current information at this time."


async def web_enhanced_answer_stream(
    question: str, 
    conversation_history: List[Dict] = None, 
    api_key: str = None
):
    """
    Stream a web-enhanced answer that checks for current information needs.
    """
    if conversation_history is None:
        conversation_history = []
    
    # Check if question needs current information
    needs_web_search = needs_current_info(question)
    
    if needs_web_search:
        yield {"type": "progress", "data": {"step": "web_search", "detail": "Searching web for current information..."}}
        
        # Perform web search
        web_results, source_urls = perform_web_search(question, api_key)
        
        if web_results:
            yield {"type": "progress", "data": {"step": "synthesis", "detail": "Combining web results with knowledge base..."}}
            
            # Create enhanced response
            response = create_web_enhanced_response(question, web_results, conversation_history, api_key)
            
            # Stream the response
            for char in response:
                yield {"type": "content", "data": char}
            
            # Add sources if available
            if source_urls:
                yield {"type": "sources", "data": source_urls}
            
            return
    
    # Fallback to regular response if no web search needed or web search failed
    yield {"type": "progress", "data": {"step": "thinking", "detail": "Processing question..."}}
    
    from utils.openai_client import create_openai_client
    client = create_openai_client(api_key)
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful dry bean research assistant. Provide informative responses to questions. For research questions, mention that you can help with bean breeding data and genetics literature."
        }
    ]
    
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )
    
    result = response.choices[0].message.content.strip()
    
    for char in result:
        yield {"type": "content", "data": char}
