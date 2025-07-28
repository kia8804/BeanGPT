from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from services.pipeline import answer_question_stream, generate_suggested_questions, continue_with_research_stream

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    genes: List[Dict[str, Any]]
    full_markdown_table: Optional[str] = None
    suggested_questions: List[str] | None = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    def generate():
        # Require user-provided API key - no fallback to environment
        api_key = request.api_key
        if not api_key:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Please enter your OpenAI API key in the interface above to continue.'})}\n\n"
            return
        
        # Validate API key format
        if not api_key.startswith('sk-') or len(api_key) < 20:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Invalid API key format. Please enter a valid OpenAI API key.'})}\n\n"
            return
        
        # Stream the answer
        full_answer = ""
        for chunk in answer_question_stream(request.question, request.conversation_history, api_key):
            if chunk["type"] == "content":
                full_answer += chunk["data"]
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "bean_complete":
                # Bean data analysis is complete, send special message with toggle
                yield f"data: {json.dumps(chunk)}\n\n"
                return  # Stop here, wait for user to decide if they want research
            elif chunk["type"] == "metadata":
                # Send final metadata (sources, genes, etc.)
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.post("/continue-research")
async def continue_research_endpoint(request: ChatRequest):
    def generate():
        # Require user-provided API key - no fallback to environment
        api_key = request.api_key
        if not api_key:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Please enter your OpenAI API key in the interface above to continue.'})}\n\n"
            return
        
        # Validate API key format
        if not api_key.startswith('sk-') or len(api_key) < 20:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Invalid API key format. Please enter a valid OpenAI API key.'})}\n\n"
            return
        
        # Stream the research continuation
        for chunk in continue_with_research_stream(request.question, request.conversation_history, api_key):
            if chunk["type"] == "content":
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "metadata":
                # Send final metadata (sources, genes, etc.)
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    ) 