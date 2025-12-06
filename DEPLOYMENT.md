# TRAF-GNN Deployment Guide

## Architecture
```
┌─────────────────┐      ┌──────────────────┐
│   Vercel CDN    │ ──→  │   Railway API    │
│   (Frontend)    │      │   (Flask+PyTorch)│
│   index.html    │      │   /api/*         │
└─────────────────┘      └──────────────────┘
```

---

## Step 1: Deploy Backend to Railway

### 1.1 Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

### 1.2 Deploy from GitHub
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `Transport_Systems` repository
4. Railway will auto-detect Python and deploy

### 1.3 Configure Environment
Railway will automatically:
- Install dependencies from `api/requirements.txt`
- Run the start command from `Procfile`

### 1.4 Get Your Railway URL
After deployment, go to **Settings > Domains** and copy your URL:
```
https://your-app-name.railway.app
```

### 1.5 Important: Data Files
The large data files (`data/processed/*.npy`) may exceed GitHub limits.
Options:
- Use Git LFS for large files
- Upload directly to Railway volume
- Host data on external storage (S3)

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Create Vercel Account
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub

### 2.2 Deploy Frontend
1. Click **"Add New Project"**
2. Import your GitHub repository
3. **IMPORTANT**: Set the Root Directory to `frontend`
4. Click **Deploy**

### 2.3 Update API URL
After Railway is deployed, update `frontend/index.html`:
```javascript
const API = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api' 
    : 'https://YOUR-RAILWAY-URL.railway.app/api';  // ← Update this
```

Then redeploy to Vercel.

---

## Step 3: Verify Deployment

### Test Backend
```bash
curl https://YOUR-RAILWAY-URL.railway.app/api/health
```

### Test Frontend
Open your Vercel URL in browser and check:
- Landing page loads
- Click "Launch Application"
- Predictions work

---

## Troubleshooting

### CORS Issues
If you see CORS errors, ensure Railway allows requests from Vercel:
- The Flask app already has `CORS(app)` enabled

### Large Files
If deployment fails due to file size:
1. Add large files to `.gitignore`
2. Use Git LFS: `git lfs track "*.npy" "*.pth"`
3. Or download at runtime

### Memory Issues
Free tier has memory limits. If model fails to load:
- Consider model quantization
- Use smaller batch sizes
- Upgrade to paid tier

---

## File Structure for Deployment

```
Transport_Systems/
├── api/
│   ├── app.py              # Flask API
│   └── requirements.txt    # API dependencies
├── frontend/
│   ├── index.html          # Main frontend
│   └── vercel.json         # Vercel config
├── checkpoints/
│   ├── best_model.pth      # Trained model
│   └── config.json         # Model config
├── data/processed/
│   ├── metr-la_stats.json  # Scaler
│   ├── metr-la_adj_mx.npy  # Graph
│   └── metr-la_X_test.npy  # Historical data
├── src/
│   └── *.py                # Model code
├── Procfile                # Railway start command
└── railway.json            # Railway config
```

---

## Estimated Costs

| Service | Free Tier | Paid |
|---------|-----------|------|
| Vercel | 100GB bandwidth/month | $20/month |
| Railway | $5 credit/month | $5+/month |

Both services have generous free tiers for small projects.
