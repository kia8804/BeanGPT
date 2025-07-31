from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes import chat, ping
import os

app = FastAPI(
    title="BeanGPT Main Platform API",
    description="API for dry bean genetics research chatbot",
    version="1.0.0"
)

# Debug CORS configuration
print(f"🔧 CORS_ORIGINS env var: {os.getenv('CORS_ORIGINS', 'NOT_SET')}")
print(f"🔧 Parsed CORS origins: {settings.cors_origins}")

# Configure CORS with explicit origins
cors_origins = [
    "https://kia8804.github.io",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173"
]

# Add environment variable origins if available
if hasattr(settings, 'cors_origins') and settings.cors_origins:
    cors_origins.extend(settings.cors_origins)

# Remove duplicates
cors_origins = list(set(cors_origins))
print(f"🔧 Final CORS origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix=settings.api_prefix, tags=["chat"])
app.include_router(ping.router, prefix=settings.api_prefix, tags=["health"])

# Add health checks
from routes import health
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])

# Add debug endpoint
@app.get("/debug/cors")
async def debug_cors():
    return {
        "cors_origins_env": os.getenv('CORS_ORIGINS', 'NOT_SET'),
        "parsed_cors_origins": settings.cors_origins,
        "final_cors_origins": cors_origins
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 