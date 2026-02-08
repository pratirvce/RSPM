#!/bin/bash
#
# Automated RSPM Evaluation Runner (Background Execution)
# Runs test evaluation automatically without user interaction
#

set -e

cd /home/prevanka/prati/su-reme/ReMe
source venv/bin/activate

echo "=========================================="
echo "RSPM Auto-Evaluation Starting"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Create directories
mkdir -p logs results/halumem

# Check dataset
if [ ! -f "datasets/memoryscope/halumem_medium.jsonl" ]; then
    echo "ERROR: Dataset not found!"
    exit 1
fi

CONV_COUNT=$(wc -l < datasets/memoryscope/halumem_medium.jsonl)
echo "Dataset: $CONV_COUNT conversations"
echo ""

# Check if ReMe is running
echo "Checking ReMe service..."
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "WARNING: ReMe service not running on port 8002"
    echo "Please start ReMe service manually:"
    echo "  reme backend=http http.port=8002 \\"
    echo "    llm.default.model_name=gpt-4 \\"
    echo "    embedding_model.default.model_name=text-embedding-3-small \\"
    echo "    vector_store.default.backend=local"
    echo ""
    echo "Exiting..."
    exit 1
fi

echo "✓ ReMe service is running"
echo ""

# Run test evaluation (limit 20 for quick test)
echo "Starting TEST evaluation (20 conversations)..."
echo "Log: logs/halumem_auto_test.log"
echo ""

python cookbook/memoryscope/run_halumem_evaluation.py \
    --dataset medium \
    --limit 20 \
    --reme-url http://localhost:8002 \
    > logs/halumem_auto_test.log 2>&1

echo ""
echo "✓ Test evaluation complete!"
echo ""

# Show summary
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
tail -50 logs/halumem_auto_test.log | grep -A 20 "FINAL EVALUATION SUMMARY" || echo "Check logs/halumem_auto_test.log for details"
echo ""

echo "=========================================="
echo "Auto-Evaluation Complete"
echo "=========================================="
echo "Time: $(date)"
echo ""
echo "Results saved to: results/halumem/"
echo "Full log: logs/halumem_auto_test.log"
echo ""
echo "To run full evaluation (1,387 conversations):"
echo "  python cookbook/memoryscope/run_halumem_evaluation.py --dataset medium"
echo ""
