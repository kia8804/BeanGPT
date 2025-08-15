#!/bin/bash

# AWS Lightsail Startup Script for BeanGPT Backend
# This script runs when the instance first boots up

echo "🚀 Starting BeanGPT Backend Setup on AWS Lightsail"
echo "=================================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install Python 3.11 and pip
echo "🐍 Installing Python 3.11..."
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y git curl wget unzip build-essential

# Create application directory
echo "📁 Creating application directory..."
mkdir -p /opt/beangpt
cd /opt/beangpt

# Clone or prepare for code deployment
echo "📋 Preparing for code deployment..."
echo "Code will be deployed manually after instance creation"

# Create systemd service file for auto-start
echo "⚙️ Creating systemd service..."
cat > /etc/systemd/system/beangpt.service << 'EOF'
[Unit]
Description=BeanGPT Backend API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/beangpt
Environment=PATH=/opt/beangpt/venv/bin
ExecStart=/opt/beangpt/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set up firewall
echo "🔥 Configuring firewall..."
ufw allow 22/tcp   # SSH
ufw allow 8000/tcp # FastAPI
ufw --force enable

# Create deployment script for later use
cat > /opt/beangpt/deploy.sh << 'EOF'
#!/bin/bash
echo "🚀 Deploying BeanGPT Backend..."

# Stop service if running
sudo systemctl stop beangpt 2>/dev/null || true

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Set permissions
sudo chown -R ubuntu:ubuntu /opt/beangpt

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable beangpt
sudo systemctl start beangpt

echo "✅ BeanGPT Backend deployed successfully!"
echo "🌐 Service should be available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "📊 Check status: sudo systemctl status beangpt"
echo "📋 View logs: sudo journalctl -u beangpt -f"
EOF

chmod +x /opt/beangpt/deploy.sh

# Set ownership
chown -R ubuntu:ubuntu /opt/beangpt

echo "✅ Lightsail instance setup complete!"
echo ""
echo "📝 Next steps after connecting to your instance:"
echo "1. Upload your code to /opt/beangpt/"
echo "2. Run: cd /opt/beangpt && ./deploy.sh"
echo "3. Your API will be available at http://[YOUR-IP]:8000"
echo ""
echo "💡 Useful commands:"
echo "   - Check service: sudo systemctl status beangpt"
echo "   - View logs: sudo journalctl -u beangpt -f"
echo "   - Restart service: sudo systemctl restart beangpt"
