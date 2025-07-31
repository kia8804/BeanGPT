"""
Health check routes for monitoring application status.
"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns 200 OK if the service is running.
    """
    return {
        "status": "healthy",
        "message": "BeanGPT backend is running",
        "models_loaded": False  # Models are loaded lazily when first requested
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifies core dependencies are available.
    """
    try:
        # Check if we can connect to Pinecone
        from services.pipeline import pc
        # Simple connection test
        pc.list_indexes()
        
        return {
            "status": "ready",
            "message": "All systems operational",
            "pinecone": "connected"
        }
    except Exception as e:
        return {
            "status": "not_ready", 
            "message": f"Service dependency error: {str(e)}",
            "pinecone": "error"
        }