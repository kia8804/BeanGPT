# Deployment Guide: BeanGPT Platform

This guide will walk you through deploying your BeanGPT application with the backend on Render and frontend on GitHub Pages.

## 🏗️ Architecture Overview

- **Backend**: FastAPI application deployed on Render
- **Frontend**: React/Vite application deployed on GitHub Pages
- **Database**: Your data files will be uploaded to Render
- **APIs**: OpenAI and Pinecone (configured via environment variables)

## 📋 Prerequisites

1. **GitHub Account** with your repository
2. **Render Account** (free tier available)
3. **API Keys**: OpenAI API key and Pinecone API key
4. **Domain/URL Planning**: Know your GitHub username for the frontend URL

## 🚀 Backend Deployment (Render)

### Step 1: Prepare Your Data Files

1. Ensure your data files are in the repository under the `data/` directory:
   - `NCBI_Filtered_Data_Enriched.xlsx`
   - `uniprotkb_Phaseolus_vulgaris.xlsx`
   - `summaries.jsonl`
   - `Merged_Bean_Dataset.xlsx`

### Step 2: Update Render Configuration

1. Open `render.yaml` and update the CORS_ORIGINS:
   ```yaml
   - key: CORS_ORIGINS
     value: "https://yourusername.github.io"
   ```
   Replace `yourusername` with your actual GitHub username.

### Step 3: Deploy to Render

1. **Connect Repository**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select this repository

2. **Configure Environment Variables**:
   Render will automatically read from `render.yaml`, but you need to set these sensitive variables manually:
   
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   
   **Important**: Set these as "secret" variables in Render dashboard for security.

3. **Deploy**:
   - Click "Create" to start deployment
   - Wait for build to complete (may take 10-15 minutes)
   - Note your Render URL (e.g., `https://beangpt-backend.onrender.com`)

### Step 4: Test Backend

Once deployed, test your backend:
```bash
curl https://your-render-url.onrender.com/api/ping
```

You should get a response indicating the service is running.

## 🌐 Frontend Deployment (GitHub Pages)

### Step 1: Update Frontend Configuration

1. **Update production environment**:
   Edit `frontend/.env.production` and replace the URL:
   ```env
   VITE_API_URL=https://your-actual-render-url.onrender.com
   ```

2. **Update package.json homepage**:
   Edit `frontend/package.json`:
   ```json
   "homepage": "https://yourusername.github.io/GuelphResearchCursor"
   ```
   Replace `yourusername` with your GitHub username.

3. **Update Vite config base path**:
   The `vite.config.js` is already configured, but verify the base path matches your repository name.

### Step 2: Enable GitHub Pages

1. **Go to Repository Settings**:
   - Navigate to your GitHub repository
   - Click "Settings" tab
   - Scroll to "Pages" section

2. **Configure GitHub Pages**:
   - Source: "GitHub Actions"
   - This will use the workflow file we created

### Step 3: Update Backend CORS

Update your backend configuration to allow your frontend domain:

1. **Update Render Environment Variables**:
   - In Render dashboard, find your service
   - Go to "Environment" tab
   - Update `CORS_ORIGINS` to include your GitHub Pages URL:
     ```
     https://yourusername.github.io
     ```

### Step 4: Deploy Frontend

1. **Install gh-pages package**:
   ```bash
   cd frontend
   npm install
   ```

2. **Deploy using GitHub Actions**:
   - Push your changes to the main/master branch
   - GitHub Actions will automatically build and deploy
   - Check the "Actions" tab in your repository for progress

3. **Alternative manual deployment**:
   ```bash
   cd frontend
   npm run deploy
   ```

## 🔧 Configuration Summary

### Key Files Created/Modified:

1. **`render.yaml`** - Render deployment configuration
2. **`frontend/.env.production`** - Production environment variables
3. **`frontend/src/config.js`** - Frontend configuration management
4. **`.github/workflows/deploy-frontend.yml`** - GitHub Actions workflow
5. **Updated `backend/config.py`** - Enhanced configuration with environment support
6. **Updated `backend/main.py`** - Production-ready server configuration
7. **Updated `frontend/vite.config.js`** - GitHub Pages deployment support

### Environment Variables:

**Backend (Render)**:
- `OPENAI_API_KEY` (secret)
- `PINECONE_API_KEY` (secret)
- `CORS_ORIGINS` (your frontend URL)
- Other model and database configurations (set in render.yaml)

**Frontend**:
- `VITE_API_URL` (your backend URL)
- `VITE_ENVIRONMENT=production`

## 🎯 Final URLs

After successful deployment:

- **Backend**: `https://your-service-name.onrender.com`
- **Frontend**: `https://yourusername.github.io/GuelphResearchCursor`

## 🔄 Development Workflow

### For Backend Updates:

1. Make changes to backend code
2. Commit and push to GitHub
3. Render automatically rebuilds and deploys
4. Test the changes

### For Frontend Updates:

1. Make changes to frontend code
2. Test locally: `cd frontend && npm run dev`
3. Commit and push to GitHub
4. GitHub Actions automatically builds and deploys to GitHub Pages

## 🛠️ Troubleshooting

### Common Issues:

1. **CORS Errors**:
   - Verify CORS_ORIGINS includes your frontend URL
   - Check both development and production URLs

2. **Backend Not Loading Data**:
   - Ensure data files are in the repository
   - Check file paths in environment variables
   - Verify file permissions

3. **Frontend Build Errors**:
   - Check that all environment variables are set
   - Verify API URL is correct
   - Check console for specific error messages

4. **404 Errors on GitHub Pages**:
   - Verify base path in vite.config.js
   - Check that GitHub Pages is enabled
   - Ensure workflow completed successfully

### Monitoring:

- **Backend**: Check Render dashboard logs
- **Frontend**: Check GitHub Actions logs
- **Runtime**: Use browser developer tools

## 💡 Production Tips

1. **Environment Management**:
   - Keep sensitive keys secure in Render
   - Use different Pinecone indexes for production
   - Monitor API usage and costs

2. **Performance**:
   - Render free tier has limitations (sleeps after 15 min)
   - Consider upgrading for production use
   - Monitor response times

3. **Updates**:
   - Test changes locally first
   - Use feature branches for major changes
   - Monitor deployments after pushing

4. **Backup**:
   - Keep backups of your data files
   - Export environment variables
   - Document your configuration

## 🎉 Next Steps

Once deployed successfully:

1. **Test thoroughly** with real queries
2. **Monitor performance** and error rates
3. **Share your application** with users
4. **Iterate and improve** based on feedback

Your BeanGPT platform is now live and accessible to researchers worldwide! 🌱🧬 