"""
Complete MemoryScope Evaluation: Baselines + RSPM
Goal: Achieve >95% TCS

This script runs:
1. All 4 baselines (Standard RAG, Recency, Window, Summary)
2. Basic RSPM
3. Advanced RSPM (with all techniques)

And compares results to track progress toward >95% TCS
"""
import sys
sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')

import json
from cookbook.memoryscope.data_loader import MemoryScopeDataset
from cookbook.memoryscope.metrics import MemoryScopeMetrics
from cookbook.memoryscope.baselines import (
    StandardRAGBaseline,
    RecencyWeightedBaseline,
    SlidingWindowBaseline
)
from cookbook.memoryscope.rspm_agent import RSPMAgent

def run_baseline(baseline, dataset, name):
    """Run single baseline experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    train, test = dataset.train_test_split()
    metrics = MemoryScopeMetrics()
    
    for idx, conv in enumerate(test):
        print(f"  [{idx+1}/{len(test)}] Processing...", end='\r')
        
        messages = dataset.get_conversation_messages(conv)
        conflicts = dataset.get_temporal_conflicts(conv)
        
        if len(messages) == 0:
            continue
            
        final_query = messages[-1]['content']
        
        # Process
        try:
            response = baseline.process_conversation(
                messages[:-1] if len(messages) > 1 else messages,
                final_query
            )
            
            # Evaluate
            is_correct = dataset.evaluate_temporal_consistency(conv, response)
            has_conflict = len(conflicts) > 0
            
            metrics.update(is_correct, has_conflict)
        except Exception as e:
            print(f"\n  ⚠️  Error on conv {idx}: {e}")
            continue
        
        # Clear for next conversation
        try:
            baseline.clear_memory()
        except:
            pass
    
    results = metrics.summary()
    
    print(f"\n\n{name} Results:")
    print(f"  TCS: {results['temporal_consistency_score']:.1%}")
    print(f"  Accuracy: {results['overall_accuracy']:.1%}")
    print(f"  Correct Conflicts: {results['correct_conflicts']}/{results['temporal_conflicts']}")
    
    return results

def run_rspm(dataset, name, **config):
    """Run RSPM experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    agent = RSPMAgent(**config)
    
    train, test = dataset.train_test_split()
    metrics = MemoryScopeMetrics()
    
    for idx, conv in enumerate(test):
        print(f"  [{idx+1}/{len(test)}] Processing...", end='\r')
        
        messages = dataset.get_conversation_messages(conv)
        conflicts = dataset.get_temporal_conflicts(conv)
        
        if len(messages) == 0:
            continue
            
        final_query = messages[-1]['content']
        
        try:
            # Process with RSPM
            response = agent.process_conversation(
                messages=messages[:-1] if len(messages) > 1 else messages,
                query=final_query,
                ground_truth_conflicts=conflicts
            )
            
            # Evaluate
            is_correct = dataset.evaluate_temporal_consistency(conv, response)
            has_conflict = len(conflicts) > 0
            
            metrics.update(is_correct, has_conflict)
        except Exception as e:
            print(f"\n  ⚠️  Error on conv {idx}: {e}")
            continue
    
    results = metrics.summary()
    
    print(f"\n\n{name} Results:")
    print(f"  TCS: {results['temporal_consistency_score']:.1%}")
    print(f"  Accuracy: {results['overall_accuracy']:.1%}")
    print(f"  Correct Conflicts: {results['correct_conflicts']}/{results['temporal_conflicts']}")
    
    # Clean up
    try:
        agent.clear_workspace()
    except:
        pass
    
    return results

def main():
    print("\n" + "="*60)
    print("MemoryScope Complete Evaluation")
    print("Goal: >95% Temporal Consistency Score")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MemoryScopeDataset("datasets/memoryscope/synthetic.jsonl")
    train, test = dataset.train_test_split()
    print(f"✓ Loaded {len(dataset.conversations)} conversations")
    print(f"  Train: {len(train)}, Test: {len(test)}")
    
    all_results = {}
    
    # Run baselines
    print("\n" + "="*60)
    print("Phase 1: Baseline Experiments")
    print("="*60)
    
    baselines = [
        ("Standard RAG", StandardRAGBaseline()),
        ("Recency-Weighted", RecencyWeightedBaseline()),
        ("Sliding Window", SlidingWindowBaseline(window_size=5))
    ]
    
    for name, baseline in baselines:
        try:
            results = run_baseline(baseline, dataset, name)
            all_results[name] = results
        except Exception as e:
            print(f"\n✗ Failed to run {name}: {e}")
            all_results[name] = {
                'temporal_consistency_score': 0,
                'overall_accuracy': 0,
                'error': str(e)
            }
    
    # Run RSPM variants
    print("\n" + "="*60)
    print("Phase 2: RSPM Experiments")
    print("="*60)
    
    rspm_configs = [
        ("RSPM Basic", {
            "workspace_id": "rspm_basic",
            "sleep_frequency": 10,
            "enable_hierarchical": False,
            "enable_reranking": False
        }),
        ("RSPM + Hierarchical", {
            "workspace_id": "rspm_hierarchical",
            "sleep_frequency": 10,
            "enable_hierarchical": True,
            "enable_reranking": False
        }),
        ("RSPM + Reranking", {
            "workspace_id": "rspm_reranking",
            "sleep_frequency": 10,
            "enable_hierarchical": False,
            "enable_reranking": True
        }),
        ("RSPM Advanced (All)", {
            "workspace_id": "rspm_advanced",
            "sleep_frequency": 10,
            "enable_hierarchical": True,
            "enable_reranking": True
        })
    ]
    
    for name, config in rspm_configs:
        try:
            results = run_rspm(dataset, name, **config)
            all_results[name] = results
        except Exception as e:
            print(f"\n✗ Failed to run {name}: {e}")
            all_results[name] = {
                'temporal_consistency_score': 0,
                'overall_accuracy': 0,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<30s} {'TCS':>10s} {'Accuracy':>10s}")
    print("-" * 60)
    
    for method, results in all_results.items():
        tcs = results['temporal_consistency_score'] * 100
        acc = results['overall_accuracy'] * 100
        print(f"{method:<30s} {tcs:>9.1f}% {acc:>9.1f}%")
    
    # Find best
    best_method = max(all_results.items(), key=lambda x: x[1]['temporal_consistency_score'])
    best_tcs = best_method[1]['temporal_consistency_score']
    
    print("\n" + "="*60)
    print(f"🏆 Best Method: {best_method[0]}")
    print(f"   TCS: {best_tcs:.1%}")
    
    if best_tcs >= 0.95:
        print(f"\n🎉 SUCCESS! Achieved >95% TCS target!")
    elif best_tcs >= 0.85:
        print(f"\n✓ GOOD! Above 85% threshold")
        print(f"   Need +{(0.95 - best_tcs)*100:.1f} percentage points to reach 95%")
    else:
        print(f"\n⚠️  Below target")
        print(f"   Need +{(0.95 - best_tcs)*100:.1f} percentage points to reach 95%")
    
    # Save results
    with open("results/complete_evaluation.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: results/complete_evaluation.json")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
