"""
Multi-Dataset Parallel RSPM Evaluation
=======================================
Evaluates RSPM across all 4 datasets (dev, validation, test splits)
using parallel processing. Records all scores to a unified results file.

Datasets:
  1. HaluMem     - Temporal consistency in multi-turn conversations
  2. LoCoMo      - Long-term conversational memory QA
  3. TimeBench   - Temporal reasoning tasks
  4. TemporalMem - Temporal memory questions

Usage:
  python -u cookbook/memoryscope/multi_dataset_evaluation.py
"""
import sys
import os
import json
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')
os.chdir('/home/prevanka/prati/su-reme/ReMe')

from cookbook.memoryscope.rspm_agent import RSPMAgent
from cookbook.memoryscope.metrics import MemoryScopeMetrics

# ============================================================
# CONFIGURATION
# ============================================================
REME_URL = "http://localhost:8002"
MAX_WORKERS = 5          # Parallel workers per dataset
SPLITS_DIR = Path("datasets/splits")
RESULTS_DIR = Path("results/multi_dataset")

# Limits per split (for time management with DeepSeek-R1)
LIMITS = {
    "dev": 20,          # 20 per dev split
    "validation": 20,   # 20 per validation split
    "test": 20          # 20 per test split
}

# Which splits to evaluate
EVAL_SPLITS = ["dev", "validation", "test"]

# Datasets to evaluate
DATASETS = ["halumem", "locomo", "timebench", "temporal_memory"]

# RSPM Configurations
RSPM_CONFIGS = [
    {
        "name": "RSPM-Basic",
        "params": {
            "sleep_frequency": 10,
            "enable_hierarchical": False,
            "enable_reranking": False,
            "reme_url": REME_URL
        }
    },
    {
        "name": "RSPM-Advanced",
        "params": {
            "sleep_frequency": 10,
            "enable_hierarchical": True,
            "enable_reranking": True,
            "reme_url": REME_URL
        }
    }
]


# ============================================================
# DATA LOADING
# ============================================================
def load_split(dataset_name: str, split: str):
    """Load a dataset split from JSONL"""
    filepath = SPLITS_DIR / dataset_name / f"{split}.jsonl"
    if not filepath.exists():
        print(f"    [WARN] {filepath} not found")
        return []
    
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_locomo_conversation(conv_ref: str):
    """Load a LoCoMo conversation by reference"""
    conv_path = SPLITS_DIR / "locomo" / "conversations" / f"{conv_ref}.json"
    if conv_path.exists():
        with open(conv_path) as f:
            return json.load(f)
    return None


# ============================================================
# DATASET-SPECIFIC EVALUATORS
# ============================================================
def evaluate_halumem_item(agent: RSPMAgent, item: dict) -> dict:
    """
    Evaluate a single HaluMem conversation.
    Uses RSPM agent's process_conversation + evaluate_response.
    """
    response, conflicts = agent.process_conversation(item)
    ground_truth = item.get('ground_truth', {})
    result = agent.evaluate_response(response, ground_truth)
    return result


def evaluate_locomo_item(agent: RSPMAgent, item: dict) -> dict:
    """
    Evaluate a single LoCoMo QA pair.
    Loads conversation context, processes it, checks answer.
    """
    # Load conversation if referenced
    conv_ref = item.get('conversation_ref', '')
    if conv_ref and not item.get('messages'):
        conv_data = load_locomo_conversation(conv_ref)
        if conv_data:
            item['messages'] = conv_data.get('messages', [])
    
    query = item.get('query', '')
    gt_answer = str(item.get('ground_truth', {}).get('answer', '')).lower()
    category = item.get('ground_truth', {}).get('category', '')
    is_temporal = item.get('metadata', {}).get('is_temporal', False)
    
    # Process with agent
    response, conflicts = agent.process_conversation(item, query=query)
    response_lower = response.lower() if response else ""
    
    # Evaluate: check if ground truth answer appears in response
    correct = False
    if gt_answer:
        # For LoCoMo, check if the answer is contained in the response
        if gt_answer in response_lower:
            correct = True
        else:
            # Fuzzy: check if key words from answer are in response
            answer_words = set(gt_answer.split()) - {'a', 'the', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'and', 'or'}
            if answer_words:
                match_ratio = sum(1 for w in answer_words if w in response_lower) / len(answer_words)
                correct = match_ratio >= 0.6
    
    return {
        'correct': correct,
        'has_conflict': is_temporal,
        'category': category
    }


def evaluate_timebench_item(agent: RSPMAgent, item: dict) -> dict:
    """
    Evaluate a single TimeBench temporal reasoning item.
    """
    query = item.get('query', '')
    context = item.get('ground_truth', {}).get('context', '')
    gt_answer = item.get('ground_truth', {}).get('answer', '')
    
    # Build a message from context + query
    messages = []
    if context:
        messages.append({"role": "user", "content": context})
    if query:
        messages.append({"role": "user", "content": query})
    
    item_with_messages = item.copy()
    item_with_messages['messages'] = messages
    
    response, _ = agent.process_conversation(item_with_messages, query=query)
    response_lower = response.lower() if response else ""
    
    # Evaluate
    correct = False
    if isinstance(gt_answer, list):
        # Multiple acceptable answers
        correct = any(str(a).lower() in response_lower for a in gt_answer)
    elif isinstance(gt_answer, str) and gt_answer:
        correct = gt_answer.lower() in response_lower
    
    return {
        'correct': correct,
        'has_conflict': True,  # All TimeBench items are temporal
        'category': item.get('metadata', {}).get('category', '')
    }


def evaluate_temporal_memory_item(agent: RSPMAgent, item: dict) -> dict:
    """
    Evaluate a single TemporalMemoryDataset item.
    """
    query = item.get('query', '')
    gt_answer = str(item.get('ground_truth', {}).get('answer', '')).lower()
    
    response, _ = agent.process_conversation(item, query=query)
    response_lower = response.lower() if response else ""
    
    correct = False
    if gt_answer:
        if gt_answer in response_lower:
            correct = True
        else:
            answer_words = set(gt_answer.split()) - {'a', 'the', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'and', 'or'}
            if answer_words:
                match_ratio = sum(1 for w in answer_words if w in response_lower) / len(answer_words)
                correct = match_ratio >= 0.5
    
    return {
        'correct': correct,
        'has_conflict': True,
        'category': item.get('metadata', {}).get('question_type', '')
    }


# Dispatcher
EVALUATORS = {
    "halumem": evaluate_halumem_item,
    "locomo": evaluate_locomo_item,
    "timebench": evaluate_timebench_item,
    "temporal_memory": evaluate_temporal_memory_item,
}


# ============================================================
# PARALLEL EVALUATION ENGINE
# ============================================================
def evaluate_dataset_split(
    dataset_name: str,
    split: str,
    config: dict,
    limit: int
) -> dict:
    """
    Evaluate a single dataset split with a given RSPM configuration.
    Returns metrics dictionary.
    """
    config_name = config['name']
    workspace_id = f"{dataset_name}_{split}_{config_name.lower().replace('-','_')}"
    
    # Load data
    data = load_split(dataset_name, split)
    if not data:
        return {"error": "no data", "total": 0}
    
    actual_limit = min(limit, len(data))
    data = data[:actual_limit]
    
    print(f"  [{dataset_name}/{split}/{config_name}] Starting: {actual_limit} items, {MAX_WORKERS} workers")
    
    # Create agent
    agent = RSPMAgent(
        workspace_id=workspace_id,
        **config['params']
    )
    agent.clear_workspace()
    
    # Get evaluator
    evaluator = EVALUATORS.get(dataset_name)
    if not evaluator:
        return {"error": f"no evaluator for {dataset_name}", "total": 0}
    
    # Metrics with thread safety
    metrics = MemoryScopeMetrics()
    metrics_lock = Lock()
    success_count = 0
    error_count = 0
    errors = []
    start_time = time.time()
    
    def process_one(item, idx):
        try:
            result = evaluator(agent, item)
            with metrics_lock:
                metrics.update(result)
            return {"status": "success", "idx": idx}
        except Exception as e:
            return {"status": "error", "idx": idx, "error": str(e)[:150]}
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, item, idx): idx
            for idx, item in enumerate(data)
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result['status'] == 'success':
                success_count += 1
            else:
                error_count += 1
                errors.append(result.get('error', 'unknown'))
            
            # Progress every 10 items
            if completed % 10 == 0 or completed == actual_limit:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{dataset_name}/{split}/{config_name}] Progress: {completed}/{actual_limit} ({rate:.1f} items/s)")
    
    elapsed = time.time() - start_time
    final = metrics.compute()
    
    result = {
        "dataset": dataset_name,
        "split": split,
        "config": config_name,
        "total_items": actual_limit,
        "success": success_count,
        "errors": error_count,
        "error_samples": errors[:5],
        "elapsed_seconds": round(elapsed, 1),
        "items_per_second": round(actual_limit / elapsed, 2) if elapsed > 0 else 0,
        "tcs": final.get('tcs', 0),
        "overall_accuracy": final.get('overall_accuracy', 0),
        "correct": final.get('correct', 0),
        "incorrect": final.get('incorrect', 0),
        "temporal_conflicts": final.get('temporal_conflicts', 0),
        "correct_conflicts": final.get('correct_conflicts', 0),
    }
    
    tcs_str = f"{result['tcs']:.1%}"
    acc_str = f"{result['overall_accuracy']:.1%}"
    goal = "PASS" if result['tcs'] >= 0.95 else "FAIL"
    print(f"  [{dataset_name}/{split}/{config_name}] DONE: TCS={tcs_str} Acc={acc_str} [{goal}] ({elapsed:.1f}s)")
    
    return result


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    print("=" * 90)
    print("MULTI-DATASET PARALLEL RSPM EVALUATION")
    print("=" * 90)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ReMe URL:   {REME_URL}")
    print(f"Workers:    {MAX_WORKERS}")
    print(f"Datasets:   {', '.join(DATASETS)}")
    print(f"Splits:     {', '.join(EVAL_SPLITS)}")
    print(f"Configs:    {', '.join(c['name'] for c in RSPM_CONFIGS)}")
    print(f"Limits:     {LIMITS}")
    print()
    
    all_results = []
    global_start = time.time()
    
    # Evaluate each dataset x split x config combination
    # We process datasets sequentially but items within each dataset in parallel
    for dataset_name in DATASETS:
        print(f"\n{'#' * 90}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#' * 90}")
        
        for split in EVAL_SPLITS:
            for config in RSPM_CONFIGS:
                try:
                    result = evaluate_dataset_split(
                        dataset_name=dataset_name,
                        split=split,
                        config=config,
                        limit=LIMITS.get(split, 50)
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  [{dataset_name}/{split}/{config['name']}] FATAL ERROR: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "dataset": dataset_name,
                        "split": split,
                        "config": config['name'],
                        "error": str(e)[:200],
                        "tcs": 0,
                        "overall_accuracy": 0,
                        "total_items": 0
                    })
    
    global_elapsed = time.time() - global_start
    
    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print(f"\n\n{'=' * 90}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"Total Time: {global_elapsed:.0f}s ({global_elapsed/60:.1f} min)\n")
    
    # Table header
    header = f"{'Dataset':<18} {'Split':<12} {'Config':<16} {'Items':>6} {'TCS':>8} {'Accuracy':>10} {'Correct':>8} {'Errors':>7} {'Time':>8} {'Goal':>6}"
    print(header)
    print("-" * 115)
    
    for r in all_results:
        if r.get('error') and r.get('total_items', 0) == 0:
            print(f"{r['dataset']:<18} {r['split']:<12} {r['config']:<16} {'ERROR':>6}")
            continue
        
        tcs = r.get('tcs', 0)
        goal = " PASS" if tcs >= 0.95 else " FAIL"
        print(
            f"{r['dataset']:<18} {r['split']:<12} {r['config']:<16} "
            f"{r.get('total_items',0):>6} "
            f"{tcs:>7.1%} "
            f"{r.get('overall_accuracy',0):>9.1%} "
            f"{r.get('correct',0):>8} "
            f"{r.get('errors',0):>7} "
            f"{r.get('elapsed_seconds',0):>7.1f}s"
            f"{goal:>6}"
        )
    
    # Per-dataset summary
    print(f"\n{'=' * 90}")
    print("PER-DATASET AVERAGES (across splits)")
    print(f"{'=' * 90}")
    
    dataset_summaries = {}
    for ds in DATASETS:
        ds_results = [r for r in all_results if r['dataset'] == ds and r.get('total_items', 0) > 0]
        if ds_results:
            avg_tcs = sum(r['tcs'] for r in ds_results) / len(ds_results)
            avg_acc = sum(r['overall_accuracy'] for r in ds_results) / len(ds_results)
            total_items = sum(r.get('total_items', 0) for r in ds_results)
            total_correct = sum(r.get('correct', 0) for r in ds_results)
            
            dataset_summaries[ds] = {
                "avg_tcs": avg_tcs,
                "avg_accuracy": avg_acc,
                "total_items": total_items,
                "total_correct": total_correct,
                "num_evals": len(ds_results)
            }
            
            goal = "PASS" if avg_tcs >= 0.95 else "FAIL"
            print(f"  {ds:<20} Avg TCS: {avg_tcs:>7.1%}  Avg Acc: {avg_acc:>7.1%}  Items: {total_items:>6}  [{goal}]")
    
    # Per-config summary
    print(f"\n{'=' * 90}")
    print("PER-CONFIG AVERAGES (across datasets)")
    print(f"{'=' * 90}")
    
    config_summaries = {}
    for config in RSPM_CONFIGS:
        cfg_name = config['name']
        cfg_results = [r for r in all_results if r.get('config') == cfg_name and r.get('total_items', 0) > 0]
        if cfg_results:
            avg_tcs = sum(r['tcs'] for r in cfg_results) / len(cfg_results)
            avg_acc = sum(r['overall_accuracy'] for r in cfg_results) / len(cfg_results)
            total_items = sum(r.get('total_items', 0) for r in cfg_results)
            
            config_summaries[cfg_name] = {
                "avg_tcs": avg_tcs,
                "avg_accuracy": avg_acc,
                "total_items": total_items,
                "num_evals": len(cfg_results)
            }
            
            goal = "PASS" if avg_tcs >= 0.95 else "FAIL"
            print(f"  {cfg_name:<20} Avg TCS: {avg_tcs:>7.1%}  Avg Acc: {avg_acc:>7.1%}  Items: {total_items:>6}  [{goal}]")
    
    # Overall
    valid_results = [r for r in all_results if r.get('total_items', 0) > 0]
    if valid_results:
        overall_tcs = sum(r['tcs'] for r in valid_results) / len(valid_results)
        overall_acc = sum(r['overall_accuracy'] for r in valid_results) / len(valid_results)
        total_processed = sum(r.get('total_items', 0) for r in valid_results)
        total_correct = sum(r.get('correct', 0) for r in valid_results)
        
        print(f"\n{'=' * 90}")
        print(f"OVERALL: Avg TCS = {overall_tcs:.1%} | Avg Accuracy = {overall_acc:.1%}")
        print(f"         Total Items = {total_processed} | Total Correct = {total_correct}")
        print(f"         Goal (>95% TCS): {'ACHIEVED' if overall_tcs >= 0.95 else 'NOT YET'}")
        print(f"{'=' * 90}")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_payload = {
        "timestamp": timestamp,
        "start_time": datetime.now().isoformat(),
        "total_elapsed_seconds": round(global_elapsed, 1),
        "configuration": {
            "reme_url": REME_URL,
            "max_workers": MAX_WORKERS,
            "limits": LIMITS,
            "splits": EVAL_SPLITS,
            "datasets": DATASETS,
            "rspm_configs": [c['name'] for c in RSPM_CONFIGS]
        },
        "detailed_results": all_results,
        "dataset_summaries": dataset_summaries,
        "config_summaries": config_summaries,
        "overall": {
            "avg_tcs": overall_tcs if valid_results else 0,
            "avg_accuracy": overall_acc if valid_results else 0,
            "total_items": total_processed if valid_results else 0,
            "total_correct": total_correct if valid_results else 0,
            "goal_achieved": overall_tcs >= 0.95 if valid_results else False
        }
    }
    
    results_file = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_payload, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Also save a human-readable scoreboard
    scoreboard_file = RESULTS_DIR / f"scoreboard_{timestamp}.txt"
    with open(scoreboard_file, 'w') as f:
        f.write("RSPM Multi-Dataset Evaluation Scoreboard\n")
        f.write(f"{'=' * 90}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {global_elapsed:.0f}s\n\n")
        
        f.write(f"{'Dataset':<18} {'Split':<12} {'Config':<16} {'Items':>6} {'TCS':>8} {'Accuracy':>10} {'Goal':>6}\n")
        f.write(f"{'-' * 80}\n")
        
        for r in all_results:
            if r.get('total_items', 0) > 0:
                tcs = r.get('tcs', 0)
                goal = "PASS" if tcs >= 0.95 else "FAIL"
                f.write(
                    f"{r['dataset']:<18} {r['split']:<12} {r['config']:<16} "
                    f"{r.get('total_items',0):>6} "
                    f"{tcs:>7.1%} "
                    f"{r.get('overall_accuracy',0):>9.1%} "
                    f"{goal:>6}\n"
                )
        
        f.write(f"\n{'=' * 80}\n")
        if valid_results:
            f.write(f"OVERALL: Avg TCS = {overall_tcs:.1%} | Avg Accuracy = {overall_acc:.1%}\n")
            f.write(f"Goal (>95%): {'ACHIEVED' if overall_tcs >= 0.95 else 'NOT YET'}\n")
    
    print(f"Scoreboard saved to: {scoreboard_file}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == '__main__':
    main()
