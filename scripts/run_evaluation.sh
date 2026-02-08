#!/bin/bash
#
# Run RSPM Evaluation with Live Progress
#

cd /home/prevanka/prati/su-reme/ReMe
source venv/bin/activate

echo "============================================"
echo "RSPM Evaluation Starting"
echo "============================================"
echo "Time: $(date)"
echo ""

# Check if ReMe is running
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "⚠️  ReMe service not running!"
    echo ""
    echo "Please start ReMe first:"
    echo "  bash scripts/setup_groq.sh YOUR_GROQ_API_KEY"
    echo ""
    exit 1
fi

echo "✓ ReMe service is running"
echo ""
echo "Starting evaluation..."
echo "  - Dataset: HaluMem Medium (1,387 conversations)"
echo "  - Testing: 10 conversations"
echo "  - Configurations: RSPM-Basic & RSPM-Advanced"
echo "  - Expected time: 3-5 minutes with Groq"
echo ""
echo "Log file: logs/eval_live.log"
echo ""

# Run evaluation with live output
python cookbook/memoryscope/simple_evaluation.py 2>&1 | tee logs/eval_live.log

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Time: $(date)"
echo ""

# Show results if available
if ls results/halumem/simplified_eval_*.json > /dev/null 2>&1; then
    LATEST=$(ls -t results/halumem/simplified_eval_*.json | head -1)
    echo "Results saved to: $LATEST"
    echo ""
    echo "Quick summary:"
    cat "$LATEST" | python3 -m json.tool | grep -A 3 "tcs"
else
    echo "No results found. Check logs/eval_live.log for details."
fi

echo ""
echo "============================================"
