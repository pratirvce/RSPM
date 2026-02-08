"""
PARALLEL RSPM Evaluation on HaluMem
Processes multiple conversations concurrently for 5-10x speedup
"""
import sys
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')
os.chdir('/home/prevanka/prati/su-reme/ReMe')

from cookbook.memoryscope.data_loader import MemoryScopeDataset
from cookbook.memoryscope.rspm_agent import RSPMAgent
from cookbook.memoryscope.metrics import MemoryScopeMetrics

print("="*80)
print("PARALLEL RSPM EVALUATION ON HALUMEM DATASET")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
LIMIT = 20  # Number of conversations to test for robust validation
REME_URL = "http://localhost:8002"
MAX_WORKERS = 5  # Process 5 conversations in parallel

# Load dataset
print(f"📁 Loading HaluMem dataset...")
dataset = MemoryScopeDataset("datasets/memoryscope/halumem_medium.jsonl")
print(f"   ✓ Loaded {len(dataset)} conversations\n")

# Configurations to test - ONLY RSPM-Basic for robust validation
configs = [
    {
        "name": "RSPM-Basic",
        "params": {
            "workspace_id": "halumem_basic_parallel",
            "sleep_frequency": 10,
            "enable_hierarchical": False,
            "enable_reranking": False,
            "reme_url": REME_URL
        }
    }
]

def process_single_conversation(agent, conv, conv_idx, metrics, metrics_lock):
    """Process a single conversation and update metrics thread-safely"""
    try:
        # Process conversation
        response, conflicts = agent.process_conversation(conv)
        
        # Evaluate
        ground_truth = conv.get('ground_truth', {})
        result = agent.evaluate_response(response, ground_truth)
        
        # Update metrics (thread-safe)
        with metrics_lock:
            metrics.update(result)
        
        return {
            'idx': conv_idx,
            'status': 'success',
            'conv_id': conv.get('conversation_id', f'conv_{conv_idx}')
        }
        
    except Exception as e:
        return {
            'idx': conv_idx,
            'status': 'error',
            'error': str(e)[:100],
            'conv_id': conv.get('conversation_id', f'conv_{conv_idx}')
        }

results = {}

# Run each configuration
for config in configs:
    print(f"\n{'#'*80}")
    print(f"# Configuration: {config['name']}")
    print(f"{'#'*80}\n")
    
    # Create agent
    print(f"🤖 Creating agent...")
    agent = RSPMAgent(**config['params'])
    print(f"   ✓ Agent created\n")
    
    # Clear workspace
    print(f"🧹 Clearing workspace...")
    agent.clear_workspace()
    print(f"   ✓ Workspace cleared\n")
    
    # Initialize metrics with thread-safe lock
    metrics = MemoryScopeMetrics()
    metrics_lock = Lock()
    
    # Process conversations in parallel
    print(f"🚀 Processing {LIMIT} conversations with {MAX_WORKERS} parallel workers...")
    success_count = 0
    error_count = 0
    
    # Create list of conversations to process
    conversations = [(dataset[i], i) for i in range(min(LIMIT, len(dataset)))]
    
    # Process in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_conv = {
            executor.submit(process_single_conversation, agent, conv, idx, metrics, metrics_lock): (conv, idx)
            for conv, idx in conversations
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_conv):
            result = future.result()
            completed += 1
            
            if result['status'] == 'success':
                success_count += 1
                print(f"   ✓ [{completed}/{LIMIT}] Completed: {result['conv_id']}")
            else:
                error_count += 1
                print(f"   ✗ [{completed}/{LIMIT}] Error in {result['conv_id']}: {result['error']}")
    
    print(f"\n   ✓ All conversations completed: {success_count} successful, {error_count} errors\n")
    
    # Compute metrics
    print(f"📈 Computing metrics...")
    final_metrics = metrics.compute()
    
    print(f"\n   Results for {config['name']}:")
    print(f"   ─────────────────────────────────────")
    print(f"   Temporal Consistency Score:  {final_metrics['tcs']:.1%}")
    print(f"   Overall Accuracy:            {final_metrics['overall_accuracy']:.1%}")
    print(f"   Total Processed:             {final_metrics['total']}")
    print(f"   Correct:                     {final_metrics['correct']}")
    print(f"   Incorrect:                   {final_metrics['incorrect']}")
    
    # Check goal
    if final_metrics['tcs'] >= 0.95:
        print(f"\n   🎉 GOAL ACHIEVED! TCS >= 95%")
    else:
        gap = (0.95 - final_metrics['tcs']) * 100
        print(f"\n   ⚠️  Goal not reached. Gap: {gap:.1f}%")
    
    results[config['name']] = final_metrics

# Final Summary
print(f"\n\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}\n")

print(f"{'Configuration':<25} {'TCS':<12} {'Accuracy':<12} {'Goal':<10}")
print(f"─"*80)

for config_name, metrics in results.items():
    goal = "✓ YES" if metrics['tcs'] >= 0.95 else "✗ NO"
    print(f"{config_name:<25} {metrics['tcs']:<12.1%} {metrics['overall_accuracy']:<12.1%} {goal}")

# Best configuration
if results:
    best = max(results.items(), key=lambda x: x[1]['tcs'])
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION: {best[0]}")
    print(f"  TCS: {best[1]['tcs']:.1%}")
    print(f"  Accuracy: {best[1]['overall_accuracy']:.1%}")
    print(f"  Goal (>95%): {'✓ YES' if best[1]['tcs'] >= 0.95 else '✗ NO'}")
    print(f"{'='*80}")

# Save results
os.makedirs("results/halumem", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/halumem/parallel_eval_{timestamp}.json"

with open(filename, 'w') as f:
    json.dump({
        "timestamp": timestamp,
        "dataset": "halumem_medium",
        "conversations_tested": LIMIT,
        "max_workers": MAX_WORKERS,
        "results": results
    }, f, indent=2)

print(f"\n✓ Results saved to: {filename}")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
