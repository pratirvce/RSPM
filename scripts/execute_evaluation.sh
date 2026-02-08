#!/bin/bash
#
# Step-by-step execution of RSPM evaluation on HaluMem dataset
# This script will guide you through each step
#

set -e

echo "========================================================================================================"
echo "RSPM Evaluation Pipeline - Execution Guide"
echo "Goal: Achieve >95% Temporal Consistency Score on HaluMem Dataset"
echo "========================================================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Check API Configuration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: API Configuration Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if grep -q "sk-xxxx" .env; then
    echo "⚠️  API keys need to be configured!"
    echo ""
    echo "Please edit the .env file with your actual API credentials:"
    echo ""
    echo "  FLOW_LLM_API_KEY=<your-llm-api-key>"
    echo "  FLOW_LLM_BASE_URL=<your-llm-base-url>"
    echo "  FLOW_EMBEDDING_API_KEY=<your-embedding-api-key>"
    echo "  FLOW_EMBEDDING_BASE_URL=<your-embedding-base-url>"
    echo ""
    echo "Example configurations:"
    echo ""
    echo "  # OpenAI"
    echo "  FLOW_LLM_API_KEY=sk-proj-..."
    echo "  FLOW_LLM_BASE_URL=https://api.openai.com/v1"
    echo "  FLOW_EMBEDDING_API_KEY=sk-proj-..."
    echo "  FLOW_EMBEDDING_BASE_URL=https://api.openai.com/v1"
    echo ""
    echo "  # Or use DeepSeek, Anthropic, or other OpenAI-compatible APIs"
    echo ""
    echo "After configuration, run this script again."
    exit 1
else
    echo "✅ API keys configured in .env"
fi
echo ""

# Step 2: Check ReMe Service
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: ReMe Service Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ ReMe service is already running on port 8002"
else
    echo "⚠️  ReMe service is not running"
    echo ""
    echo "Starting ReMe service in background..."
    echo ""
    
    # Start ReMe service in background
    nohup reme \
        backend=http \
        http.port=8002 \
        llm.default.model_name=gpt-4 \
        embedding_model.default.model_name=text-embedding-3-small \
        vector_store.default.backend=local \
        > logs/reme_service.log 2>&1 &
    
    REME_PID=$!
    echo "ReMe service started (PID: $REME_PID)"
    echo "Logs: logs/reme_service.log"
    echo ""
    
    # Wait for service to be ready
    echo "Waiting for service to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8002/health > /dev/null 2>&1; then
            echo "✅ ReMe service is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    
    if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "❌ ReMe service failed to start. Check logs/reme_service.log"
        exit 1
    fi
fi
echo ""

# Step 3: Verify Dataset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Dataset Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "datasets/memoryscope/halumem_medium.jsonl" ]; then
    echo "❌ Dataset not found!"
    echo "Please run: python cookbook/memoryscope/download_halumem_github.py"
    echo "Then run: python cookbook/memoryscope/halumem_github_adapter.py"
    exit 1
fi

CONV_COUNT=$(wc -l < datasets/memoryscope/halumem_medium.jsonl)
echo "✅ Dataset ready: $CONV_COUNT conversations"
echo ""

# Step 4: Test Evaluation (50 conversations)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Test Evaluation (50 conversations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Running test evaluation on 50 conversations to verify setup..."
echo "This will take approximately 10-15 minutes depending on API speed."
echo ""
echo "Logs: logs/halumem_evaluation_test.log"
echo ""

python cookbook/memoryscope/run_halumem_evaluation.py \
    --dataset medium \
    --limit 50 \
    --reme-url http://localhost:8002 \
    2>&1 | tee logs/halumem_evaluation_test.log

echo ""
echo "✅ Test evaluation complete!"
echo ""

# Step 5: Ask user before full evaluation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Full Evaluation Decision"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Test evaluation completed successfully!"
echo ""
echo "Would you like to run the FULL evaluation on all 1,387 conversations?"
echo "⚠️  This will take several hours and consume significant API credits."
echo ""
read -p "Continue with full evaluation? (yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Starting full evaluation..."
    echo "This will take several hours. You can monitor progress in logs/halumem_evaluation_full.log"
    echo ""
    
    python cookbook/memoryscope/run_halumem_evaluation.py \
        --dataset medium \
        --reme-url http://localhost:8002 \
        2>&1 | tee logs/halumem_evaluation_full.log
    
    echo ""
    echo "✅ Full evaluation complete!"
    echo ""
else
    echo "Skipping full evaluation."
    echo ""
    echo "To run it later, execute:"
    echo "  python cookbook/memoryscope/run_halumem_evaluation.py --dataset medium"
    echo ""
fi

# Step 6: Results Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results are saved in: results/halumem/"
echo ""
ls -lht results/halumem/ | head -10
echo ""
echo "To view the latest results:"
echo "  cat results/halumem/\$(ls -t results/halumem/ | head -1)"
echo ""

echo "========================================================================================================"
echo "RSPM Evaluation Pipeline Complete!"
echo "========================================================================================================"
echo ""
echo "Next steps for your paper:"
echo "1. Review results in results/halumem/"
echo "2. Compare TCS scores across configurations"
echo "3. Use HaluMem benchmark citation in your paper"
echo "4. Include performance comparison with published baselines"
echo ""
echo "HaluMem Citation:"
echo "Chen et al. (2025). HaluMem: Evaluating Hallucinations in Memory Systems of Agents."
echo "arXiv:2511.03506"
echo ""
echo "========================================================================================================"
