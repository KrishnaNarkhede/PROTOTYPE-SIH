# 🚀 Vercel Deployment Guide for BioMapper AI

## Step-by-Step Deployment Instructions

### 1. Prerequisites
- Install Vercel CLI: `npm install -g vercel`
- Create a Vercel account at https://vercel.com
- Install Git if not already installed

### 2. Prepare Your Repository
```bash
# Navigate to your project directory
cd "d:\sih\S42\BioMapper AI - Deep-Sea eDNA Analysis"

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit for Vercel deployment"
```

### 3. Login to Vercel
```bash
vercel login
```

### 4. Deploy to Vercel
```bash
# Deploy with automatic configuration
vercel

# Follow the prompts:
# - Set up and deploy? Y
# - Which scope? (select your account)
# - Link to existing project? N
# - Project name: biomapper-ai (or your preferred name)
# - Directory: ./
# - Override settings? N
```

### 5. Production Deployment
```bash
# Deploy to production
vercel --prod
```

## 📋 Configuration Files Created

### `vercel.json`
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ],
  "functions": {
    "api/index.py": {
      "maxDuration": 30
    }
  },
  "env": {
    "PYTHONPATH": "."
  }
}
```

### `requirements-vercel.txt`
Minimal dependencies for Vercel:
- Flask==2.3.3
- Werkzeug==2.3.7
- numpy==1.24.4
- pandas==2.0.3
- biopython==1.81

### `api/index.py`
Serverless function handler that adapts your Flask app for Vercel.

## 🔧 Environment Variables (Optional)
If you need environment variables:
```bash
# Set environment variables
vercel env add NCBI_API_KEY
vercel env add DATABASE_URL
```

## 🌐 Access Your Deployed App
After deployment, Vercel will provide URLs:
- **Preview URL**: `https://biomapper-ai-xxx.vercel.app`
- **Production URL**: `https://biomapper-ai.vercel.app` (if you have a custom domain)

## 📊 Testing Your Deployment
1. Visit your Vercel URL
2. Upload a FASTA file
3. Check the analysis results
4. Verify all features work correctly

## 🔄 Updating Your Deployment
```bash
# Make changes to your code
git add .
git commit -m "Update analysis features"

# Deploy updates
vercel --prod
```

## 🛠️ Troubleshooting

### Common Issues:
1. **Build Errors**: Check `requirements-vercel.txt` for dependency conflicts
2. **Timeout Issues**: Reduce file size limits or optimize processing
3. **Memory Issues**: Use smaller batch sizes for large datasets

### Debug Commands:
```bash
# Check deployment logs
vercel logs

# Local development
vercel dev
```

## 📈 Performance Optimization
- File size limit: 50MB (reduced from 100MB for Vercel)
- Processing timeout: 30 seconds
- Batch processing for large datasets
- Minimal dependencies for faster cold starts

## 🔒 Security Notes
- No sensitive data stored in the deployment
- File uploads are processed in memory
- No persistent file storage on Vercel

## 📞 Support
If you encounter issues:
1. Check Vercel documentation: https://vercel.com/docs
2. Review deployment logs
3. Test locally with `vercel dev`