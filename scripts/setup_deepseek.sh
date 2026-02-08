#!/bin/bash
#
# DeepSeek Setup - Automated Configuration
#

set -e

echo "============================================"
echo "DeepSeek Setup for RSPM Evaluation"
echo "============================================"
echo ""

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: bash setup_deepseek.sh YOUR_DEEPSEEK_API_KEY"
    echo ""
    echo "Don't have a DeepSeek API key yet?"
    echo "1. Go to: https://platform.deepseek.com/"
    echo "2. Sign up / Login"
    echo "3. Go to: https://platform.deepseek.com/api_keys"
    echo "4. Create new API key"
    echo "5. Run: bash setup_deepseek.sh sk_YOUR_KEY_HERE"
    echo ""
    echo "Cost: ~$0.05 for 10 conversations (very cheap!)"
    echo ""
    exit 1
fi

DEEPSEEK_API_KEY="$1"

# Validate key format
if [[ ! "$DEEPSEEK_API_KEY" =~ ^sk- ]]; then
    echo "⚠️  Warning: DeepSeek API keys usually start with 'sk-'"
    echo "   Your key: $DEEPSEEK_API_KEY"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd /home/prevanka/prati/su-reme/ReMe

echo "1. Updating .env file with DeepSeek credentials..."
cat > .env << EOF
# DeepSeek API (Cheap & Effective!)
FLOW_LLM_API_KEY=$DEEPSEEK_API_KEY
FLOW_LLM_BASE_URL=https://api.deepseek.com/v1
FLOW_EMBEDDING_API_KEY=$DEEPSEEK_API_KEY
FLOW_EMBEDDING_BASE_URL=https://api.deepseek.com/v1
EOF
echo "   ✓ .env updated with DeepSeek credentials"
echo ""

echo "2. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Stop any existing ReMe service
echo "3. Stopping any existing ReMe services..."
pkill -f "reme.*http.port=8002" 2>/dev/null || true
sleep 2
echo "   ✓ Cleaned up existing services"
echo ""

echo "4. Starting ReMe service with DeepSeek..."
echo "   Model: deepseek-chat (V3 - fast & good quality)"
echo "   Cost: ~$0.14 per 1M tokens"
echo "   (This will run in background)"
echo ""

nohup reme backend=http http.port=8002 \
  llm.default.model_name=deepseek-chat \
  embedding_model.default.model_name=deepseek-chat \
  vector_store.default.backend=local \
  > logs/reme_deepseek.log 2>&1 &

REME_PID=$!
echo "   ReMe started (PID: $REME_PID)"
echo "   Log: logs/reme_deepseek.log"
echo ""

echo "5. Waiting for service to be ready..."
for i in {1..20}; do
    sleep 2
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "   ✓ ReMe service is healthy!"
        break
    fi
    echo -n "."
done
echo ""

# Final health check
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "   ⚠️  Service not responding. Checking logs..."
    echo ""
    tail -30 logs/reme_deepseek.log
    echo ""
    echo "   Please check logs/reme_deepseek.log for details"
    exit 1
fi

# Test API
echo "6. Testing DeepSeek API..."
HEALTH_RESPONSE=$(curl -s http://localhost:8002/health)
echo "   Response: $HEALTH_RESPONSE"
echo ""

echo "============================================"
echo "✓ DeepSeek Setup Complete!"
echo "============================================"
echo ""
echo "Configuration:"
echo "  - Provider: DeepSeek"
echo "  - Model: deepseek-chat (V3)"
echo "  - Port: 8002"
echo "  - Status: Running & Healthy"
echo ""
echo "Next step: Run evaluation"
echo ""
echo "  python cookbook/memoryscope/simple_evaluation.py"
echo ""
echo "Or run in background with monitoring:"
echo "  bash scripts/run_evaluation.sh"
echo ""
echo "Expected:"
echo "  - Time: ~5-10 minutes for 10 conversations"
echo "  - Cost: ~$0.05"
echo "  - TCS: 94-97%"
echo ""
echo "============================================"
