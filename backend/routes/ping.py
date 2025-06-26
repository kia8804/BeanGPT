from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
 
@router.get("/ping")
async def ping():
    return JSONResponse(content={"status": "ok"}) 