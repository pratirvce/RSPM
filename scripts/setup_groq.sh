#!/bin/bash
#
# Quick Setup for Groq (Free LLM Alternative)
#

set -e

echo "============================================"
echo "Groq Setup - Free LLM Alternative"
echo "============================================"
echo ""

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: bash setup_groq.sh YOUR_GROQ_API_KEY"
    echo ""
    echo "Don't have a Groq API key yet?"
    echo "1. Go to: https://console.groq.com/"
    echo "2. Sign up (email only, no credit card)"
    echo "3. Create API key"
    echo "4. Run: bash setup_groq.sh gsk_YOUR_KEY_HERE"
    echo ""
    exit 1
fi

GROQ_API_KEY="$1"

# Validate key format
if [[ ! "$GROQ_API_KEY" =~ ^gsk_ ]]; then
    echo "⚠️  Warning: Groq API keys usually start with 'gsk_'"
    echo "   Your key: $GROQ_API_KEY"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd /home/prevanka/prati/su-reme/ReMe

echo "1. Updating .env file..."
cat > .env << EOF
# Groq API (Free!)
FLOW_LLM_API_KEY=$GROQ_API_KEY
FLOW_LLM_BASE_URL=https://api.groq.com/openai/v1
FLOW_EMBEDDING_API_KEY=$GROQ_API_KEY
FLOW_EMBEDDING_BASE_URL=https://api.groq.com/openai/v1
EOF
echo "   ✓ .env updated"
echo ""

echo "2. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

echo "3. Starting ReMe service with Groq..."
echo "   (This will run in background)"
echo ""

nohup reme backend=http http.port=8002 \
  llm.default.model_name=llama-3.1-70b-versatile \
  embedding_model.default.model_name=llama-3.1-70b-versatile \
  vector_store.default.backend=local \
  > logs/reme_groq.log 2>&1 &

REME_PID=$!
echo "   ReMe started (PID: $REME_PID)"
echo "   Log: logs/reme_groq.log"
echo ""

echo "4. Waiting for service to be ready..."
sleep 10

# Check if service is healthy
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "   ✓ ReMe service is healthy!"
else
    echo "   ⚠️  Service not responding yet, checking logs..."
    tail -20 logs/reme_groq.log
    echo ""
    echo "   Waiting 10 more seconds..."
    sleep 10
    
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "   ✓ ReMe service is healthy!"
    else
        echo "   ✗ Service failed to start. Check logs/reme_groq.log"
        exit 1
    fi
fi
echo ""

echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "ReMe is running with Groq (free & fast!):"
echo "  - Model: Llama 3.1 70B"
echo "  - Speed: 100-500 tokens/second"
echo "  - Cost: FREE"
echo ""
echo "Next step: Run evaluation"
echo ""
echo "  python cookbook/memoryscope/simple_evaluation.py"
echo ""
echo "Or run in background:"
echo "  nohup python cookbook/memoryscope/simple_evaluation.py > logs/eval_groq.log 2>&1 &"
echo ""
echo "Monitor:"
echo "  tail -f logs/eval_groq.log"
echo ""
echo "============================================"
