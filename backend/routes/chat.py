from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from services.pipeline import answer_question_stream, generate_suggested_questions

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    genes: List[Dict[str, Any]]
    full_markdown_table: Optional[str] = None
    suggested_questions: List[str] | None = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    def generate():
        # Stream the answer
        full_answer = ""
        for chunk in answer_question_stream(request.question, request.conversation_history):
            if chunk["type"] == "content":
                full_answer += chunk["data"]
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