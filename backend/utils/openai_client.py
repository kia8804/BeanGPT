"""
OpenAI client wrapper using the 0.28.1 API (no proxy issues).
For research use without proxy configuration.
"""

import os
from typing import Optional
import openai


def create_openai_client(api_key: Optional[str] = None):
    """
    Create an OpenAI client using the 0.28.1 API.
    This version doesn't have proxy-related issues.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    # Set the API key for the openai module (0.28.1 style)
    openai.api_key = api_key
    
    # Create a wrapper client that matches the 1.x API interface
    return OpenAIWrapper()


class OpenAIWrapper:
    """
    Wrapper to make 0.28.1 OpenAI API compatible with 1.x style calls.
    """
    
    def __init__(self):
        self.chat = ChatWrapper()


class ChatWrapper:
    """
    Wrapper for chat completions.
    """
    
    def __init__(self):
        self.completions = CompletionsWrapper()


class CompletionsWrapper:
    """
    Wrapper for completions that converts 1.x style calls to 0.28.1 style.
    """
    
    def create(self, **kwargs):
        """
        Create a chat completion using the 0.28.1 API.
        """
        # Convert 1.x style parameters to 0.28.1 style
        model = kwargs.get("model", "gpt-3.5-turbo")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", None)
        stream = kwargs.get("stream", False)
        
        # Handle function calling parameters (added for bean data support)
        functions = kwargs.get("functions", None)
        function_call = kwargs.get("function_call", None)
        
        # Build the API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        
        # Add optional parameters
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        if functions is not None:
            api_params["functions"] = functions
        if function_call is not None:
            api_params["function_call"] = function_call
        
        # Call the 0.28.1 API
        response = openai.ChatCompletion.create(**api_params)
        
        # Handle streaming vs non-streaming
        if stream:
            return StreamingResponseWrapper(response)
        else:
            # Convert the response to match 1.x style
            return OpenAIResponse(response)


class OpenAIResponse:
    """
    Wrapper to make 0.28.1 response compatible with 1.x style access.
    """
    
    def __init__(self, response):
        self.choices = [OpenAIChoice(choice) for choice in response.choices]


class OpenAIChoice:
    """
    Wrapper for choice objects.
    """
    
    def __init__(self, choice):
        self.message = OpenAIMessage(choice.message)
        self.finish_reason = getattr(choice, 'finish_reason', None)


class OpenAIMessage:
    """
    Wrapper for message objects.
    """
    
    def __init__(self, message):
        self.content = message.content
        self.function_call = getattr(message, 'function_call', None)


class StreamingResponseWrapper:
    """
    Wrapper for streaming responses to make them compatible with 1.x style.
    """
    
    def __init__(self, response):
        self.response = response
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            chunk = next(self.response)
            return StreamingChunk(chunk)
        except StopIteration:
            raise


class StreamingChunk:
    """
    Wrapper for individual streaming chunks.
    """
    
    def __init__(self, chunk):
        self.choices = [StreamingChoice(choice) for choice in chunk.choices]


class StreamingChoice:
    """
    Wrapper for streaming choice objects.
    """
    
    def __init__(self, choice):
        self.delta = StreamingDelta(choice.delta)


class StreamingDelta:
    """
    Wrapper for streaming delta objects.
    """
    
    def __init__(self, delta):
        self.content = getattr(delta, 'content', None)


# Test function
def test_client():
    """Test the OpenAI client creation."""
    try:
        client = create_openai_client("test-key")
        print("✅ OpenAI client created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_client() 