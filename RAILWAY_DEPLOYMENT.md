# 🚂 Railway Deployment Guide

## Step 1: Install Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login
```

## Step 2: Initialize and Deploy

```bash
# Navigate to your backend directory
cd backend

# Initialize Railway project
railway init

# Add environment variables
railway variables set PINECONE_API_KEY="your-pinecone-api-key"
# Note: OpenAI API key is NOT needed - users enter it in the UI

# Deploy
railway up
```

## Step 3: Configure Environment Variables

In Railway dashboard:

### Required Variables:
- `PINECONE_API_KEY` - Your Pinecone API key

### Not Required:
- `OPENAI_API_KEY` - Users enter this in the UI

### Optional Variables (already set in railway.toml):
- `BGE_MODEL` = `BAAI/bge-base-en-v1.5`
- `PUBMEDBERT_MODEL` = `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- `BGE_INDEX_NAME` = `bge-production`
- `PUBMEDBERT_INDEX_NAME` = `pubmed-production`
- `TOP_K` = `8`
- `ALPHA` = `0.6`
- `API_PREFIX` = `/api`

## Step 4: Update Frontend API URL

After deployment, update your frontend's `VITE_API_BASE_URL` to point to your Railway URL:

```javascript
// In frontend/src/config.js
export const API_BASE_URL = "https://your-app-name.up.railway.app";
```

## Step 5: Update CORS Origins

Update the `CORS_ORIGINS` variable in Railway dashboard:
```
CORS_ORIGINS=https://your-frontend-domain.github.io,http://localhost:5173
```

## 🎯 Benefits of Railway:

✅ **1GB RAM** on free tier (vs 512MB on Render)  
✅ **Fast builds** with better caching  
✅ **Auto-scaling** and hibernation  
✅ **Simple CLI** deployment  
✅ **Built-in monitoring**  

## 📊 Expected Performance:

- **Startup**: ~60-90 seconds (models loading)
- **Memory usage**: ~800MB (within 1GB limit)
- **Response time**: Fast after startup
- **Auto-sleep**: After 30min inactivity (free tier)

## 🔧 Troubleshooting:

- **Build fails**: Check Dockerfile and requirements.txt
- **Memory issues**: Monitor with `railway logs`
- **API connection**: Verify CORS_ORIGINS includes your frontend domain
- **Models not loading**: Check environment variables are set

## 📱 Testing Your Deployment:

1. **Health Check**: `GET https://your-app.up.railway.app/api/health`
2. **Readiness**: `GET https://your-app.up.railway.app/api/ready`
3. **Chat**: Test with your frontend