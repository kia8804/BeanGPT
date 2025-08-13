#!/bin/bash

# AWS Lightsail Deployment Script for BeanGPT Backend
# This script helps deploy your FastAPI backend to AWS Lightsail

echo "🚀 AWS Lightsail Deployment Script for BeanGPT Backend"
echo "=================================================="

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

# Configuration variables
INSTANCE_NAME="beangpt-backend"
BLUEPRINT_ID="ubuntu_22_04"
BUNDLE_ID="nano_2_0"  # $3.50/month - 512MB RAM, 1 vCPU, 20GB SSD
REGION="us-east-1"
AVAILABILITY_ZONE="us-east-1a"

echo ""
echo "📋 Deployment Configuration:"
echo "   Instance Name: $INSTANCE_NAME"
echo "   Blueprint: $BLUEPRINT_ID"
echo "   Bundle: $BUNDLE_ID (512MB RAM, 1 vCPU, 20GB SSD - $3.50/month)"
echo "   Region: $REGION"
echo "   Availability Zone: $AVAILABILITY_ZONE"
echo ""

read -p "Do you want to proceed with creating the Lightsail instance? (y/n): " -n 1 -r
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
    --user-data file://lightsail-startup.sh

if [ $? -eq 0 ]; then
    echo "✅ Lightsail instance '$INSTANCE_NAME' created successfully!"
    echo ""
    echo "📝 Next Steps:"
    echo "1. Wait 3-5 minutes for the instance to fully initialize"
    echo "2. Get your instance's public IP:"
    echo "   aws lightsail get-instance --instance-name $INSTANCE_NAME"
    echo "3. Open port 8000 for your FastAPI app:"
    echo "   aws lightsail open-instance-public-ports --instance-name $INSTANCE_NAME --port-info fromPort=8000,toPort=8000,protocol=TCP"
    echo "4. Update your frontend config with the new IP address"
    echo "5. Set up your environment variables in the instance"
    echo ""
    echo "🌐 Your backend will be available at: http://[INSTANCE-IP]:8000"
    echo ""
    echo "💡 To SSH into your instance:"
    echo "   aws lightsail get-instance-access-details --instance-name $INSTANCE_NAME"
else
    echo "❌ Failed to create Lightsail instance. Please check your AWS credentials and try again."
    exit 1
fi
