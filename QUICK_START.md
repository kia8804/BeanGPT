# Quick Start: Deploy BeanGPT in 15 Minutes

This guide gets your BeanGPT platform live as quickly as possible.

## ⚡ Quick Steps

### 1. Backend Setup (5 minutes)

1. **Go to [Render](https://dashboard.render.com/)**
2. **Create new Blueprint**
   - Connect your GitHub repo
   - Select this repository
   - Render will read `render.yaml` automatically

3. **Set Environment Variables** (in Render dashboard):
   ```
   OPENAI_API_KEY=your_openai_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   ```

4. **Update CORS Origins**:
   - Find your GitHub username
   - In Render, set `CORS_ORIGINS` to: `https://yourusername.github.io`

5. **Deploy** - Click create and wait ~10 minutes

### 2. Frontend Setup (5 minutes)

1. **Update URLs**:
   - Copy your Render URL (e.g., `https://beangpt-backend.onrender.com`)
   - Edit `frontend/.env.production`:
     ```
     VITE_API_URL=https://your-render-url.onrender.com
     ```
   - Edit `frontend/package.json` homepage:
     ```json
     "homepage": "https://yourusername.github.io/GuelphResearchCursor"
     ```

2. **Enable GitHub Pages**:
   - Go to your repo → Settings → Pages
   - Source: "GitHub Actions"

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Setup production deployment"
   git push
   ```

### 3. Test (5 minutes)

- Backend: `https://your-render-url.onrender.com/api/ping`
- Frontend: `https://yourusername.github.io/GuelphResearchCursor`

## 🔧 Replace These Values

Before deploying, replace:
- `yourusername` → Your GitHub username
- `your-render-url` → Your actual Render URL
- `your_openai_key_here` → Your OpenAI API key
- `your_pinecone_key_here` → Your Pinecone API key

## ✅ You're Live!

Your research platform is now accessible worldwide. See `DEPLOYMENT_GUIDE.md` for detailed configuration and troubleshooting. 