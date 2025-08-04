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
        # Check if we can connect to Zilliz Cloud
        import requests
        from config import settings
        
        # Test connection with list collections endpoint
        api_url = f"{settings.zilliz_uri.rstrip('/')}/v1/vector/collections"
        headers = {
            "Authorization": f"Bearer {settings.zilliz_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            collections = result.get("data", [])
            return {
                "status": "ready",
                "message": "All systems operational",
                "zilliz": "connected",
                "collections": len(collections)
            }
        else:
            return {
                "status": "not_ready",
                "message": f"Zilliz API error: {response.status_code}",
                "zilliz": "error"
            }
        
    except Exception as e:
        return {
            "status": "not_ready", 
            "message": f"Service dependency error: {str(e)}",
            "zilliz": "error"
        }