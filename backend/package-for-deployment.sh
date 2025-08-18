#!/bin/bash

# Package BeanGPT Backend for AWS Lightsail Deployment
# This script creates a deployment package with your latest changes

echo "📦 Packaging BeanGPT Backend for Deployment"
echo "==========================================="

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Backend directory: $SCRIPT_DIR"

# Create deployment package name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="beangpt-backend-${TIMESTAMP}.tar.gz"

echo "📦 Creating package: $PACKAGE_NAME"

# Navigate to project root
cd "$PROJECT_ROOT"

# Create the deployment package
tar -czf "$PACKAGE_NAME" \
  --exclude='backend/venv' \
  --exclude='backend/__pycache__' \
  --exclude='backend/**/__pycache__' \
  --exclude='backend/*.pyc' \
  --exclude='backend/**/*.pyc' \
  --exclude='backend/.git' \
  --exclude='backend/data' \
  --exclude='frontend/node_modules' \
  --exclude='frontend/dist' \
  --exclude='frontend/.next' \
  --exclude='.git' \
  --exclude='node_modules' \
  backend/

if [ $? -eq 0 ]; then
    echo "✅ Package created successfully: $PACKAGE_NAME"
    echo ""
    echo "📊 Package size:"
    ls -lh "$PACKAGE_NAME"
    echo ""
    echo "📋 Next steps:"
    echo "1. Upload this package to your Lightsail instance:"
    echo "   scp $PACKAGE_NAME ubuntu@YOUR-INSTANCE-IP:/tmp/"
    echo ""
    echo "2. SSH into your instance and run:"
    echo "   sudo systemctl stop beangpt"
    echo "   cd /opt"
    echo "   sudo tar -xzf /tmp/$PACKAGE_NAME"
    echo "   sudo chown -R ubuntu:ubuntu /opt/beangpt"
    echo "   cd /opt/beangpt"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo "   sudo systemctl restart beangpt"
    echo ""
    echo "3. Or use the detailed guide in UPDATE_LIGHTSAIL.md"
    echo ""
    echo "🌐 Your updated backend will be available at: http://YOUR-INSTANCE-IP:8000"
else
    echo "❌ Failed to create package"
    exit 1
fi


