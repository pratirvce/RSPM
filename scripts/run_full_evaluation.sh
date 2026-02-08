#!/bin/bash
#
# Master script to download, convert, and evaluate RSPM on HaluMem dataset
# Goal: Achieve >95% Temporal Consistency Score
#

set -e  # Exit on error

echo "========================================================================================================"
echo "RSPM Evaluation on HaluMem Dataset"
echo "Goal: Temporal Consistency Score > 95%"
echo "========================================================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p logs results/halumem datasets/memoryscope

# Step 1: Download Dataset (if not already done)
if [ ! -f "datasets/halumem/stage5_1_dialogue_generation.jsonl" ]; then
    echo "[Step 1/4] Downloading HaluMem dataset from GitHub..."
    python cookbook/memoryscope/download_halumem_github.py 2>&1 | tee logs/halumem_download.log
    echo "✓ Download complete"
else
    echo "[Step 1/4] Dataset already downloaded ✓"
fi
echo ""

# Step 2: Convert Dataset
if [ ! -f "datasets/memoryscope/halumem_medium.jsonl" ]; then
    echo "[Step 2/4] Converting HaluMem to MemoryScope format..."
    python cookbook/memoryscope/halumem_github_adapter.py 2>&1 | tee logs/halumem_conversion.log
    echo "✓ Conversion complete"
else
    echo "[Step 2/4] Dataset already converted ✓"
fi
echo ""

# Step 3: Start ReMe Service (if not running)
echo "[Step 3/4] Checking ReMe service..."
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "ReMe service not running. Please start it with:"
    echo "  reme backend=http http.port=8002 llm.default.model_name=<your_model> embedding_model.default.model_name=<your_embedding>"
    echo ""
    echo "For now, continuing with evaluation setup (will fail if service not started)..."
else
    echo "✓ ReMe service is running"
fi
echo ""

# Step 4: Run Evaluation
echo "[Step 4/4] Running RSPM evaluation on HaluMem-Medium (with conversation limit for testing)..."
echo "Starting evaluation... This will take a while."
echo "Logs will be saved to logs/halumem_evaluation.log"
echo ""

# Run with limit of 50 conversations first for testing
python cookbook/memoryscope/run_halumem_evaluation.py \
    --dataset medium \
    --limit 50 \
    --reme-url http://localhost:8002 \
    2>&1 | tee logs/halumem_evaluation_test.log

echo ""
echo "========================================================================================================"
echo "Test Evaluation Complete!"
echo "========================================================================================================"
echo ""
echo "Results saved to: results/halumem/"
echo "Logs saved to: logs/"
echo ""
echo "Next steps:"
echo "1. Review test results in results/halumem/"
echo "2. If successful, run full evaluation:"
echo "   python cookbook/memoryscope/run_halumem_evaluation.py --dataset medium"
echo "3. Then run on Long dataset:"
echo "   python cookbook/memoryscope/run_halumem_evaluation.py --dataset long --limit 100"
echo ""
echo "========================================================================================================"
