#!/bin/bash

# Interactive AWS Lightsail Deployment Script for BeanGPT Backend
# This script lets you choose your preferred configuration

echo "🚀 Interactive AWS Lightsail Deployment for BeanGPT Backend"
echo "=========================================================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if Lightsail CLI plugin is available
if ! aws lightsail help &> /dev/null; then
    echo "❌ AWS Lightsail CLI is not available. Please ensure AWS CLI is properly configured."
    exit 1
fi

echo "✅ AWS CLI is available"
echo ""

# Step 1: Choose Operating System
echo "🖥️  Step 1: Choose Operating System"
echo "=================================="
echo "Available blueprints:"
echo "1) ubuntu_22_04     - Ubuntu 22.04 LTS (Recommended for Python)"
echo "2) ubuntu_20_04     - Ubuntu 20.04 LTS"
echo "3) amazon_linux_2   - Amazon Linux 2"
echo "4) centos_7         - CentOS 7"
echo "5) debian_11        - Debian 11"
echo ""
read -p "Choose blueprint (1-5) [default: 1]: " blueprint_choice
blueprint_choice=${blueprint_choice:-1}

case $blueprint_choice in
    1) BLUEPRINT_ID="ubuntu_22_04"; OS_NAME="Ubuntu 22.04" ;;
    2) BLUEPRINT_ID="ubuntu_20_04"; OS_NAME="Ubuntu 20.04" ;;
    3) BLUEPRINT_ID="amazon_linux_2"; OS_NAME="Amazon Linux 2" ;;
    4) BLUEPRINT_ID="centos_7"; OS_NAME="CentOS 7" ;;
    5) BLUEPRINT_ID="debian_11"; OS_NAME="Debian 11" ;;
    *) BLUEPRINT_ID="ubuntu_22_04"; OS_NAME="Ubuntu 22.04" ;;
esac

echo "✅ Selected: $OS_NAME"
echo ""

# Step 2: Choose Instance Size
echo "💰 Step 2: Choose Instance Size & Pricing"
echo "========================================"
echo "Available bundles:"
echo "1) nano_2_0    - $3.50/month  - 512MB RAM, 1 vCPU, 20GB SSD   (Recommended for small apps)"
echo "2) micro_2_0   - $5.00/month  - 1GB RAM,   1 vCPU, 40GB SSD   (Better performance)"
echo "3) small_2_0   - $10.00/month - 2GB RAM,   1 vCPU, 60GB SSD   (High performance)"
echo "4) medium_2_0  - $20.00/month - 4GB RAM,   2 vCPU, 80GB SSD   (Production ready)"
echo "5) large_2_0   - $40.00/month - 8GB RAM,   2 vCPU, 160GB SSD  (Heavy workloads)"
echo ""
echo "💡 For BeanGPT backend with AI models, we recommend at least micro_2_0 (1GB RAM)"
echo ""
read -p "Choose bundle (1-5) [default: 2]: " bundle_choice
bundle_choice=${bundle_choice:-2}

case $bundle_choice in
    1) BUNDLE_ID="nano_2_0"; PRICE="$3.50/month"; SPECS="512MB RAM, 1 vCPU, 20GB SSD" ;;
    2) BUNDLE_ID="micro_2_0"; PRICE="$5.00/month"; SPECS="1GB RAM, 1 vCPU, 40GB SSD" ;;
    3) BUNDLE_ID="small_2_0"; PRICE="$10.00/month"; SPECS="2GB RAM, 1 vCPU, 60GB SSD" ;;
    4) BUNDLE_ID="medium_2_0"; PRICE="$20.00/month"; SPECS="4GB RAM, 2 vCPU, 80GB SSD" ;;
    5) BUNDLE_ID="large_2_0"; PRICE="$40.00/month"; SPECS="8GB RAM, 2 vCPU, 160GB SSD" ;;
    *) BUNDLE_ID="micro_2_0"; PRICE="$5.00/month"; SPECS="1GB RAM, 1 vCPU, 40GB SSD" ;;
esac

echo "✅ Selected: $BUNDLE_ID ($PRICE) - $SPECS"
echo ""

# Step 3: Choose Region
echo "🌍 Step 3: Choose Region"
echo "======================="
echo "Available regions (choose closest to your users):"
echo "1) us-east-1      - US East (N. Virginia)     - Lowest latency to East Coast US"
echo "2) us-west-2      - US West (Oregon)          - Lowest latency to West Coast US"
echo "3) eu-west-1      - Europe (Ireland)          - Lowest latency to Europe"
echo "4) ap-southeast-1 - Asia Pacific (Singapore)  - Lowest latency to Asia"
echo "5) ap-northeast-1 - Asia Pacific (Tokyo)      - Lowest latency to Japan"
echo "6) eu-central-1   - Europe (Frankfurt)        - Lowest latency to Central Europe"
echo ""
read -p "Choose region (1-6) [default: 1]: " region_choice
region_choice=${region_choice:-1}

case $region_choice in
    1) REGION="us-east-1"; AVAILABILITY_ZONE="us-east-1a"; REGION_NAME="US East (N. Virginia)" ;;
    2) REGION="us-west-2"; AVAILABILITY_ZONE="us-west-2a"; REGION_NAME="US West (Oregon)" ;;
    3) REGION="eu-west-1"; AVAILABILITY_ZONE="eu-west-1a"; REGION_NAME="Europe (Ireland)" ;;
    4) REGION="ap-southeast-1"; AVAILABILITY_ZONE="ap-southeast-1a"; REGION_NAME="Asia Pacific (Singapore)" ;;
    5) REGION="ap-northeast-1"; AVAILABILITY_ZONE="ap-northeast-1a"; REGION_NAME="Asia Pacific (Tokyo)" ;;
    6) REGION="eu-central-1"; AVAILABILITY_ZONE="eu-central-1a"; REGION_NAME="Europe (Frankfurt)" ;;
    *) REGION="us-east-1"; AVAILABILITY_ZONE="us-east-1a"; REGION_NAME="US East (N. Virginia)" ;;
esac

echo "✅ Selected: $REGION_NAME"
echo ""

# Step 4: Choose Instance Name
echo "🏷️  Step 4: Instance Name"
echo "======================="
read -p "Enter instance name [default: beangpt-backend]: " instance_name
INSTANCE_NAME=${instance_name:-beangpt-backend}
echo "✅ Instance name: $INSTANCE_NAME"
echo ""

# Summary
echo "📋 Deployment Summary"
echo "===================="
echo "   Instance Name: $INSTANCE_NAME"
echo "   Operating System: $OS_NAME ($BLUEPRINT_ID)"
echo "   Instance Size: $BUNDLE_ID ($PRICE)"
echo "   Specifications: $SPECS"
echo "   Region: $REGION_NAME ($REGION)"
echo "   Availability Zone: $AVAILABILITY_ZONE"
echo ""
echo "💰 Monthly Cost: $PRICE"
echo ""

# Confirmation
read -p "Do you want to proceed with creating this Lightsail instance? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled."
    exit 1
fi

echo "🔧 Creating Lightsail instance..."

# Create the Lightsail instance
aws lightsail create-instances \
    --instance-names "$INSTANCE_NAME" \
    --availability-zone "$AVAILABILITY_ZONE" \
    --blueprint-id "$BLUEPRINT_ID" \
    --bundle-id "$BUNDLE_ID" \
    --user-data file://lightsail-startup.sh \
    --region "$REGION"

if [ $? -eq 0 ]; then
    echo "✅ Lightsail instance '$INSTANCE_NAME' created successfully!"
    echo ""
    echo "📝 Next Steps:"
    echo "1. Wait 3-5 minutes for the instance to fully initialize"
    echo "2. Get your instance's public IP:"
    echo "   aws lightsail get-instance --instance-name $INSTANCE_NAME --region $REGION"
    echo "3. Open port 8000 for your FastAPI app:"
    echo "   aws lightsail open-instance-public-ports --instance-name $INSTANCE_NAME --port-info fromPort=8000,toPort=8000,protocol=TCP --region $REGION"
    echo "4. Update your frontend config with the new IP address"
    echo "5. Set up your environment variables in the instance"
    echo ""
    echo "🌐 Your backend will be available at: http://[INSTANCE-IP]:8000"
    echo ""
    echo "💡 To access your instance:"
    echo "   - SSH: aws lightsail get-instance-access-details --instance-name $INSTANCE_NAME --region $REGION"
    echo "   - Browser: https://lightsail.aws.amazon.com/ → Click your instance → 'Connect using SSH'"
    echo ""
    echo "📋 Save these details:"
    echo "   Instance Name: $INSTANCE_NAME"
    echo "   Region: $REGION"
    echo "   Monthly Cost: $PRICE"
else
    echo "❌ Failed to create Lightsail instance. Please check your AWS credentials and try again."
    exit 1
fi
