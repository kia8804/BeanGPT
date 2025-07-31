"""
OpenAI client wrapper using the modern OpenAI API (v1.x).
Supports both old sk- and new sk-proj- API key formats.
"""

import os
from typing import Optional
from openai import OpenAI


def create_openai_client(api_key: Optional[str] = None):
    """
    Create an OpenAI client using the modern v1.x API.
    Supports project-based API keys (sk-proj-...).
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    # Create the modern OpenAI client
    return OpenAI(api_key=api_key)


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