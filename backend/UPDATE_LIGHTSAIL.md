# 🚀 How to Update Your AWS Lightsail Backend

This guide will help you update your existing AWS Lightsail instance with the latest changes to your BeanGPT backend.

## 📋 Prerequisites

- AWS CLI installed and configured
- Access to your existing Lightsail instance
- Your instance name (likely `beangpt-backend`)

## 🔍 Step 1: Check Your Current Instance

First, let's see your current Lightsail instances:

```bash
aws lightsail get-instances
```

This will show you:
- Instance name
- Public IP address  
- Current status
- Region

## 📦 Step 2: Prepare Your Code for Upload

### Option A: Create a Deployment Package (Recommended)

1. **Create a deployment archive** (run this from your project root):
```bash
# Navigate to your project root
cd /path/to/GuelphResearchCursor

# Create a deployment package excluding unnecessary files
tar -czf beangpt-backend-update.tar.gz \
  --exclude='backend/venv' \
  --exclude='backend/__pycache__' \
  --exclude='backend/*.pyc' \
  --exclude='backend/.git' \
  --exclude='backend/data' \
  --exclude='frontend/node_modules' \
  backend/
```

### Option B: Use Git (If your code is in a repository)

```bash
# If your code is in a Git repository, you can clone it directly on the server
git add .
git commit -m "Update attribution and remove OpenAI branding"
git push origin main
```

## 🔗 Step 3: Connect to Your Lightsail Instance

### Option A: AWS CLI SSH
```bash
# Get SSH connection details
aws lightsail get-instance-access-details --instance-name beangpt-backend

# Or use the browser-based SSH (easier)
```

### Option B: Browser SSH (Recommended)
1. Go to https://lightsail.aws.amazon.com/
2. Click on your `beangpt-backend` instance
3. Click "Connect using SSH"

## 📤 Step 4: Upload Your Updated Code

### If using the deployment package:

1. **Upload the file** to your Lightsail instance:
```bash
# From your local machine, upload to the instance
scp -i /path/to/your/key.pem beangpt-backend-update.tar.gz ubuntu@YOUR-INSTANCE-IP:/tmp/
```

2. **On your Lightsail instance** (via SSH):
```bash
# Stop the current service
sudo systemctl stop beangpt

# Backup current installation
sudo cp -r /opt/beangpt /opt/beangpt-backup-$(date +%Y%m%d)

# Extract new code
cd /opt
sudo tar -xzf /tmp/beangpt-backend-update.tar.gz
sudo chown -R ubuntu:ubuntu /opt/beangpt

# Copy over any existing environment files or data
sudo cp /opt/beangpt-backup-*/data/* /opt/beangpt/data/ 2>/dev/null || true
```

### If using Git:

```bash
# On your Lightsail instance
cd /opt/beangpt
git pull origin main
```

## 🔄 Step 5: Update Dependencies and Restart

```bash
# Navigate to the application directory
cd /opt/beangpt

# Activate virtual environment
source venv/bin/activate

# Update dependencies (in case requirements.txt changed)
pip install --upgrade pip
pip install -r requirements.txt

# Set proper permissions
sudo chown -R ubuntu:ubuntu /opt/beangpt

# Restart the service
sudo systemctl daemon-reload
sudo systemctl restart beangpt

# Check if it's running
sudo systemctl status beangpt
```

## ✅ Step 6: Verify the Update

1. **Check service status**:
```bash
sudo systemctl status beangpt
```

2. **View recent logs**:
```bash
sudo journalctl -u beangpt -f --lines=50
```

3. **Test the API**:
```bash
# Get your public IP
curl -s http://169.254.169.254/latest/meta-data/public-ipv4

# Test the health endpoint
curl http://YOUR-INSTANCE-IP:8000/health
```

4. **Test the attribution update**:
   - Visit your frontend
   - Ask "Who developed you?" 
   - Should now say "Dry Bean Breeding & Computational Biology Program at the University of Guelph in 2025"

## 🚨 Troubleshooting

### If the service won't start:
```bash
# Check detailed logs
sudo journalctl -u beangpt -n 100

# Check if port is in use
sudo netstat -tlnp | grep :8000

# Manually test the application
cd /opt/beangpt
source venv/bin/activate
python main.py
```

### If you get permission errors:
```bash
sudo chown -R ubuntu:ubuntu /opt/beangpt
sudo chmod +x /opt/beangpt/main.py
```

### If dependencies fail to install:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Reinstall Python packages
cd /opt/beangpt
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔄 Quick Update Script

For future updates, you can create this script on your Lightsail instance:

```bash
# Create update script
cat > /opt/beangpt/quick-update.sh << 'EOF'
#!/bin/bash
echo "🔄 Quick updating BeanGPT backend..."

# Stop service
sudo systemctl stop beangpt

# Pull latest changes (if using git)
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl start beangpt

echo "✅ Update complete!"
sudo systemctl status beangpt
EOF

chmod +x /opt/beangpt/quick-update.sh
```

## 📊 Monitoring Your Service

- **Status**: `sudo systemctl status beangpt`
- **Logs**: `sudo journalctl -u beangpt -f`
- **Restart**: `sudo systemctl restart beangpt`
- **Stop**: `sudo systemctl stop beangpt`
- **Start**: `sudo systemctl start beangpt`

## 💰 Cost Management

Your Lightsail instance continues to charge monthly. If you need to:
- **Stop temporarily**: `aws lightsail stop-instance --instance-name beangpt-backend`
- **Start again**: `aws lightsail start-instance --instance-name beangpt-backend`
- **Delete permanently**: `aws lightsail delete-instance --instance-name beangpt-backend`

---

## 🎉 Success!

After following these steps, your AWS Lightsail backend should be updated with:
- ✅ New attribution (University of Guelph program instead of individual name)
- ✅ Removed OpenAI branding from user messages
- ✅ Updated creation date (2025)
- ✅ All your latest improvements to kidney bean data recognition

Your backend API will be available at: `http://YOUR-INSTANCE-IP:8000`


