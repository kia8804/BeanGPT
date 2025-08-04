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
        # Check if we can connect to Zilliz
        from pymilvus import MilvusClient
        from config import settings
        
        client = MilvusClient(
            uri=settings.zilliz_uri,
            token=settings.zilliz_token
        )
        # Simple connection test
        collections = client.list_collections()
        
        return {
            "status": "ready",
            "message": "All systems operational",
            "zilliz": "connected",
            "collections": len(collections)
        }
    except Exception as e:
        return {
            "status": "not_ready", 
            "message": f"Service dependency error: {str(e)}",
            "zilliz": "error"
        }