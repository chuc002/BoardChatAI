#!/bin/bash

# BoardContinuity MVP Production Deployment Script

set -e

echo "🚀 Starting BoardContinuity MVP deployment..."

# Check if environment file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please copy .env.example to .env and configure your environment variables."
    exit 1
fi

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p uploads logs ssl static

# Set proper permissions
chmod 755 uploads logs static
chmod 700 ssl

# Check if SSL certificates exist
if [ ! -f "ssl/boardcontinuity.crt" ] || [ ! -f "ssl/private/boardcontinuity.key" ]; then
    echo "🔒 SSL certificates not found. Generating self-signed certificates for development..."
    mkdir -p ssl/private
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/private/boardcontinuity.key \
        -out ssl/certs/boardcontinuity.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=boardcontinuity.local"
    echo "⚠️  Warning: Using self-signed certificates. For production, use proper SSL certificates."
fi

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Build the application
echo "🔨 Building BoardContinuity application..."
docker-compose -f docker-compose.prod.yml build --no-cache

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start the services
echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost/api/health; then
    echo "✅ BoardContinuity MVP is healthy and running!"
    echo ""
    echo "🎉 Deployment successful!"
    echo "📱 Access your application at: https://localhost"
    echo "🔧 API health check: https://localhost/api/health"
    echo ""
    echo "📊 Container status:"
    docker-compose -f docker-compose.prod.yml ps
else
    echo "❌ Health check failed. Check logs:"
    docker-compose -f docker-compose.prod.yml logs --tail=50
    exit 1
fi

echo ""
echo "🔧 Useful commands:"
echo "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop services: docker-compose -f docker-compose.prod.yml down"
echo "  Restart: docker-compose -f docker-compose.prod.yml restart"
echo "  Scale app: docker-compose -f docker-compose.prod.yml up -d --scale boardcontinuity=3"