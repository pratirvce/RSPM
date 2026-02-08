"""
Simplified RSPM Evaluation on HaluMem
Works with current agent implementation
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')
os.chdir('/home/prevanka/prati/su-reme/ReMe')

from cookbook.memoryscope.data_loader import MemoryScopeDataset
from cookbook.memoryscope.rspm_agent import RSPMAgent
from cookbook.memoryscope.metrics import MemoryScopeMetrics

print("="*80)
print("RSPM EVALUATION ON HALUMEM DATASET")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
LIMIT = 10  # Number of conversations to test
REME_URL = "http://localhost:8002"

# Load dataset
print(f"📁 Loading HaluMem dataset...")
dataset = MemoryScopeDataset("datasets/memoryscope/halumem_medium.jsonl")
print(f"   ✓ Loaded {len(dataset)} conversations\n")

# Configurations to test
configs = [
    {
        "name": "RSPM-Basic",
        "params": {
            "workspace_id": "halumem_basic",
            "sleep_frequency": 10,
            "enable_hierarchical": False,
            "enable_reranking": False,
            "reme_url": REME_URL
        }
    },
    {
        "name": "RSPM-Advanced",
        "params": {
            "workspace_id": "halumem_advanced",
            "sleep_frequency": 10,
            "enable_hierarchical": True,
            "enable_reranking": True,
            "reme_url": REME_URL
        }
    }
]

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
    
    # Initialize metrics
    metrics = MemoryScopeMetrics()
    
    # Process conversations
    print(f"📊 Processing {LIMIT} conversations...")
    success_count = 0
    error_count = 0
    
    for i in range(min(LIMIT, len(dataset))):
        conv = dataset[i]
        conv_id = conv.get('conversation_id', f'conv_{i}')
        
        try:
            # Process conversation
            response, conflicts = agent.process_conversation(conv)
            
            # Evaluate
            ground_truth = conv.get('ground_truth', {})
            result = agent.evaluate_response(response, ground_truth)
            
            # Update metrics
            metrics.update(result)
            success_count += 1
            
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{LIMIT} conversations processed")
                
        except Exception as e:
            error_count += 1
            print(f"   ✗ Error in conversation {i}: {str(e)[:100]}")
            continue
    
    print(f"\n   ✓ Completed: {success_count} successful, {error_count} errors\n")
    
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
filename = f"results/halumem/simplified_eval_{timestamp}.json"

with open(filename, 'w') as f:
    json.dump({
        "timestamp": timestamp,
        "dataset": "halumem_medium",
        "conversations_tested": LIMIT,
        "results": results
    }, f, indent=2)

print(f"\n✓ Results saved to: {filename}")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
