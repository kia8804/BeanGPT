import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes import chat, ping

app = FastAPI(
    title="BeanGPT Main Platform API",
    description="API for dry bean genetics research chatbot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix=settings.api_prefix, tags=["chat"])
app.include_router(ping.router, prefix=settings.api_prefix, tags=["health"])

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for production (Render sets this)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=not settings.is_production) 