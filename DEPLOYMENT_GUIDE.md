# 🚀 Deployment Guide: BeanGPT to Render + GitHub Pages

This guide will help you deploy your BeanGPT application with the backend on Render and frontend on GitHub Pages.

## 📋 Prerequisites

- GitHub account
- Render account (free tier available)
- OpenAI API key
- Pinecone API key
- Your data files uploaded to your Pinecone indexes

## 🔧 Backend Deployment (Render)

### Step 1: Prepare Your Repository
1. Push your code to GitHub (if not already done)
2. Ensure your `backend/` folder contains:
   - ✅ `requirements.txt`
   - ✅ `Dockerfile`
   - ✅ `render.yaml`
   - ✅ Your data files in `data/` folder

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `beangpt-backend` (or your preferred name)
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 3: Set Environment Variables
In Render dashboard, add these environment variables:

**Required:**
- `OPENAI_API_KEY` = `your_openai_api_key`
- `PINECONE_API_KEY` = `your_pinecone_api_key`

**Model Configuration:**
- `BGE_MODEL` = `BAAI/bge-small-en-v1.5`
- `PUBMEDBERT_MODEL` = `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- `BGE_INDEX_NAME` = `dry-bean-bge-abstract`
- `PUBMEDBERT_INDEX_NAME` = `dry-bean-pubmedbert-abstract`

**Other Settings:**
- `TOP_K` = `8`
- `ALPHA` = `0.6`
- `GENE_DB_PATH` = `./data/NCBI_Filtered_Data_Enriched.xlsx`
- `UNIPROT_DB_PATH` = `./data/uniprotkb_Phaseolus_vulgaris.xlsx`
- `MERGED_DATA_PATH` = `./data/Merged_Bean_Dataset.xlsx`
- `CORS_ORIGINS` = `https://yourusername.github.io,http://localhost:5173`
- `API_PREFIX` = `/api`

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for deployment (first deploy takes ~10-15 minutes)
3. Note your Render URL: `https://your-app-name.onrender.com`

## 🌐 Frontend Deployment (GitHub Pages)

### Step 1: Update Configuration
1. Update `frontend/src/config.js` with your Render URL:
   ```javascript
   export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://your-app-name.onrender.com';
   ```

### Step 2: Update CORS Settings
1. In Render dashboard, update `CORS_ORIGINS` to include your GitHub Pages URL:
   ```
   https://yourusername.github.io,http://localhost:5173
   ```

### Step 3: Configure GitHub Repository
1. Go to your GitHub repository
2. Click "Settings" → "Pages"
3. Under "Source", select "GitHub Actions"
4. Go to "Settings" → "Environments"
5. Create environment named `github-pages`
6. Add environment variable:
   - `VITE_API_BASE_URL` = `https://your-app-name.onrender.com`

### Step 4: Deploy
1. Push your code to the `main` branch
2. GitHub Actions will automatically build and deploy
3. Your site will be available at: `https://yourusername.github.io/your-repo-name`

## 🔍 Testing Your Deployment

### Backend Testing
```bash
curl https://your-app-name.onrender.com/api/ping
```
Should return: `{"message": "pong"}`

### Frontend Testing
1. Visit your GitHub Pages URL
2. Enter your OpenAI API key
3. Ask a test question about bean genetics
4. Verify you get responses with proper citations

## 🛠 Troubleshooting

### Common Issues

**Backend Won't Start:**
- Check Render logs for errors
- Verify all environment variables are set
- Ensure data files are in correct paths

**Frontend Can't Connect:**
- Verify `VITE_API_BASE_URL` points to your Render URL
- Check CORS settings include your GitHub Pages domain
- Open browser dev tools to see network errors

**CORS Errors:**
- Update `CORS_ORIGINS` in Render to include your frontend domain
- Restart your Render service after updating CORS

**Model Loading Errors:**
- First request may take 30-60 seconds as models download
- Check Render logs for model loading progress

### Free Tier Limitations

**Render Free Tier:**
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- 750 hours/month limit

**GitHub Pages:**
- 100GB bandwidth/month
- 1GB storage limit

## 🎉 Success!

Your BeanGPT application should now be fully deployed and accessible worldwide!

**URLs:**
- Backend API: `https://your-app-name.onrender.com`
- Frontend: `https://yourusername.github.io/your-repo-name`

Remember to replace the placeholder URLs with your actual deployment URLs.