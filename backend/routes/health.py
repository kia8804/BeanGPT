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
        # Check if we can connect to Zilliz using REST API
        import requests
        from config import settings
        
        # Extract cluster ID for REST API
        cluster_id = settings.zilliz_uri.split("//")[1].split(".")[0]
        api_url = f"https://{cluster_id}.api.gcp-us-west1.zillizcloud.com/v1/vector/collections"
        
        headers = {
            "Authorization": f"Bearer {settings.zilliz_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {
                "status": "ready",
                "message": "All systems operational",
                "zilliz": "connected"
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