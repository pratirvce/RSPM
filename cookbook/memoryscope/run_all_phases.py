#!/usr/bin/env python3
"""
Master RSPM Evaluation Pipeline
================================
Fully automated pipeline implementing the 5-day research plan:

  Phase 1: Baseline comparison  (NaiveRAG  vs  RSPM-Auto  vs  RSPM-Oracle)
  Phase 2: LLM-based detection  (RSPM-LLMDetect)
  Phase 3: Proactive reconciliation (RSPM-Proactive)
  Phase 4: Full 12-dataset evaluation with all configs
  Phase 5: Analysis & report generation

Features:
  - Budget monitoring (checks DeepSeek balance before each phase)
  - Error handling with retries and exponential backoff
  - Per-phase checkpoints (skips completed phases on re-run)
  - Comprehensive logging to file and stdout
  - Final comparison report with paper-ready tables

Usage:
  python -u cookbook/memoryscope/run_all_phases.py          # run all phases
  python -u cookbook/memoryscope/run_all_phases.py --phase 3 # start from phase 3
"""
import sys
import os
import json
import time
import math
import traceback
import logging
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/prevanka/prati/su-reme/ReMe")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cookbook" / "memoryscope"))
os.chdir(str(PROJECT_ROOT))

from rspm_agent import RSPMAgent
from metrics import MemoryScopeMetrics

# Import evaluators and utilities from the existing evaluation module
from multi_dataset_evaluation import (
    load_split, EVALUATORS, STOP_WORDS,
    fuzzy_match, multi_answer_match, normalize_text, extract_content_words,
    log_diagnostic, diagnostic_logs, diagnostic_lock,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESULTS_BASE = PROJECT_ROOT / "results" / "pipeline_run"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = RESULTS_BASE / "checkpoint.json"

log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file)),
    ],
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("FLOW_LLM_API_KEY", "sk-1bdb6f5137694533a8084459f655e28a")
)
REME_URL = "http://localhost:8002"
MAX_WORKERS = 5
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

ALL_DATASETS = [
    "halumem", "locomo", "timebench", "temporal_memory",
    "longmemeval", "personamem", "memoryagentbench",
    "atoke", "memtrack", "dynaquest", "fifa_synth", "reviseqa_synth",
]
CORE_DATASETS = ["halumem", "atoke", "locomo", "personamem"]

_LLM = dict(
    llm_model="deepseek-chat",
    llm_api_key=API_KEY,
    llm_base_url="https://api.deepseek.com",
    reme_url=REME_URL,
)

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
PHASES = [
    {
        "id": 1,
        "name": "Phase 1: Baseline Comparison",
        "datasets": CORE_DATASETS,
        "splits": ["dev", "test"],
        "limits": {"dev": 10, "test": 10},
        "configs": [
            {"name": "NaiveRAG", "params": dict(
                sleep_frequency=99999, enable_hierarchical=False,
                enable_reranking=False, enable_llm_generation=True,
                conflict_mode="none", proactive=False, **_LLM)},
            {"name": "RSPM-Auto", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=True, enable_llm_generation=True,
                conflict_mode="auto", proactive=False, **_LLM)},
            {"name": "RSPM-Oracle", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=True, enable_llm_generation=True,
                conflict_mode="oracle", proactive=False, **_LLM)},
        ],
    },
    {
        "id": 2,
        "name": "Phase 2: LLM-Based Detection",
        "datasets": CORE_DATASETS,
        "splits": ["dev", "test"],
        "limits": {"dev": 10, "test": 10},
        "configs": [
            {"name": "RSPM-LLMDetect", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=True, enable_llm_generation=True,
                conflict_mode="llm", proactive=False, **_LLM)},
        ],
    },
    {
        "id": 3,
        "name": "Phase 3: Proactive Reconciliation",
        "datasets": CORE_DATASETS,
        "splits": ["dev", "test"],
        "limits": {"dev": 10, "test": 10},
        "configs": [
            {"name": "RSPM-Proactive", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=False, enable_llm_generation=True,
                conflict_mode="llm", proactive=True, **_LLM)},
        ],
    },
    {
        "id": 4,
        "name": "Phase 4: Full 12-Dataset Evaluation",
        "datasets": ALL_DATASETS,
        "splits": ["dev", "test"],
        "limits": {"dev": 15, "test": 15},
        "configs": [
            {"name": "NaiveRAG", "params": dict(
                sleep_frequency=99999, enable_hierarchical=False,
                enable_reranking=False, enable_llm_generation=True,
                conflict_mode="none", proactive=False, **_LLM)},
            {"name": "RSPM-Auto", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=True, enable_llm_generation=True,
                conflict_mode="auto", proactive=False, **_LLM)},
            {"name": "RSPM-Proactive", "params": dict(
                sleep_frequency=99999, enable_hierarchical=True,
                enable_reranking=False, enable_llm_generation=True,
                conflict_mode="llm", proactive=True, **_LLM)},
        ],
    },
]

# ---------------------------------------------------------------------------
# Budget monitoring
# ---------------------------------------------------------------------------
def check_balance() -> float:
    """Return remaining DeepSeek balance in USD, or -1 on failure."""
    try:
        import requests as _req
        resp = _req.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        data = resp.json()
        for info in data.get("balance_infos", []):
            if info.get("currency") == "USD":
                return float(info["total_balance"])
    except Exception:
        pass
    return -1.0


def estimate_phase_cost(phase: dict) -> float:
    """Rough estimate of API cost for a phase in USD."""
    n_items = 0
    for ds in phase["datasets"]:
        for sp in phase["splits"]:
            n_items += phase["limits"].get(sp, 10)
    n_configs = len(phase["configs"])
    # ~1500 input + 200 output tokens per item => ~$0.00027 per item
    base_cost = n_items * n_configs * 0.0003
    # LLM detection / proactive adds ~1 extra call per item
    has_llm_detect = any(
        c["params"].get("conflict_mode") == "llm" or c["params"].get("proactive")
        for c in phase["configs"]
    )
    if has_llm_detect:
        base_cost *= 2.0
    return round(base_cost, 3)


# ---------------------------------------------------------------------------
# Evaluation engine (with retries)
# ---------------------------------------------------------------------------
def evaluate_single_item(evaluator, agent, item, idx, max_retries=3):
    """Evaluate one item with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            result = evaluator(agent, item, idx)
            return {"status": "ok", "result": result, "idx": idx}
        except Exception as e:
            err = str(e)
            is_transient = any(k in err.lower() for k in ["rate", "429", "timeout", "connection"])
            if is_transient and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            tb = traceback.format_exc()
            return {"status": "error", "idx": idx, "error": f"{err[:200]}\n{tb[-300:]}"}
    return {"status": "error", "idx": idx, "error": "max retries exceeded"}


def evaluate_dataset_split(dataset_name, split, config, limit):
    """Evaluate one dataset/split/config. Returns metrics dict."""
    config_name = config["name"]
    workspace_id = f"{dataset_name}_{split}_{config_name.lower().replace('-', '_')}"

    data = load_split(dataset_name, split)
    if not data:
        log.warning(f"  [{dataset_name}/{split}] No data found")
        return {"error": "no data", "total_items": 0}

    actual = min(limit, len(data))
    data = data[:actual]
    log.info(f"  [{dataset_name}/{split}/{config_name}] {actual} items, {MAX_WORKERS} workers")

    agent = RSPMAgent(workspace_id=workspace_id, **config["params"])
    agent.clear_workspace()

    evaluator = EVALUATORS.get(dataset_name)
    if not evaluator:
        log.error(f"  No evaluator for {dataset_name}")
        return {"error": f"no evaluator", "total_items": 0}

    metrics = MemoryScopeMetrics()
    metrics_lock = Lock()
    success_count = 0
    error_count = 0
    errors = []
    consecutive_errors = 0
    start = time.time()

    def process_one(item, idx):
        return evaluate_single_item(evaluator, agent, item, idx)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, item, idx): idx for idx, item in enumerate(data)}
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            done_count += 1
            if res["status"] == "ok":
                with metrics_lock:
                    metrics.update(res["result"])
                success_count += 1
                consecutive_errors = 0
            else:
                error_count += 1
                errors.append(res.get("error", "unknown"))
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    log.error(f"  [{dataset_name}/{split}/{config_name}] 10 consecutive errors — aborting split")
                    break

            if done_count % 5 == 0 or done_count == actual:
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                log.info(f"  [{dataset_name}/{split}/{config_name}] {done_count}/{actual} ({rate:.1f}/s)")

    elapsed = time.time() - start
    final = metrics.compute()
    tcs = final.get("tcs", 0)
    acc = final.get("overall_accuracy", 0)
    goal = "PASS" if tcs >= 0.95 else "FAIL"
    log.info(
        f"  [{dataset_name}/{split}/{config_name}] DONE: "
        f"TCS={tcs:.1%} Acc={acc:.1%} Correct={final.get('correct',0)}/{actual} "
        f"Errors={error_count} [{goal}] ({elapsed:.1f}s)"
    )
    return {
        "dataset": dataset_name, "split": split, "config": config_name,
        "total_items": actual, "success": success_count, "errors": error_count,
        "error_samples": errors[:3], "elapsed_seconds": round(elapsed, 1),
        "tcs": tcs, "overall_accuracy": acc,
        "correct": final.get("correct", 0), "incorrect": final.get("incorrect", 0),
        "temporal_conflicts": final.get("temporal_conflicts", 0),
        "correct_conflicts": final.get("correct_conflicts", 0),
    }


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_phases": [], "results": {}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------
def run_phase(phase, ckpt):
    pid = phase["id"]
    name = phase["name"]

    if pid in ckpt["completed_phases"]:
        log.info(f"\n{'='*80}\n SKIPPING {name} (already completed)\n{'='*80}")
        return ckpt["results"].get(str(pid), [])

    log.info(f"\n{'='*80}")
    log.info(f" {name}")
    log.info(f"{'='*80}")

    # Budget check
    balance = check_balance()
    est_cost = estimate_phase_cost(phase)
    log.info(f"  Balance: ${balance:.2f}  |  Estimated cost: ${est_cost:.3f}")
    if 0 < balance < est_cost * 0.8:
        log.warning(f"  LOW BUDGET — reducing limits by 50%")
        phase = {**phase, "limits": {k: max(5, v // 2) for k, v in phase["limits"].items()}}

    if 0 < balance < 0.05:
        log.error(f"  INSUFFICIENT BUDGET (${balance:.2f}) — skipping phase")
        return []

    phase_results = []
    phase_start = time.time()

    for ds in phase["datasets"]:
        for sp in phase["splits"]:
            for cfg in phase["configs"]:
                limit = phase["limits"].get(sp, 10)
                try:
                    result = evaluate_dataset_split(ds, sp, cfg, limit)
                    phase_results.append(result)
                except Exception as e:
                    log.error(f"  FATAL: {ds}/{sp}/{cfg['name']}: {e}")
                    traceback.print_exc()
                    phase_results.append({
                        "dataset": ds, "split": sp, "config": cfg["name"],
                        "error": str(e)[:200], "tcs": 0, "overall_accuracy": 0,
                        "total_items": 0,
                    })

    elapsed = time.time() - phase_start
    log.info(f"  {name} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save phase results
    phase_file = RESULTS_BASE / f"phase{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(phase_file, "w") as f:
        json.dump({"phase": name, "elapsed_seconds": elapsed, "results": phase_results}, f, indent=2)
    log.info(f"  Results saved to {phase_file}")

    # Update checkpoint
    ckpt["completed_phases"].append(pid)
    ckpt["results"][str(pid)] = phase_results
    save_checkpoint(ckpt)

    return phase_results


# ---------------------------------------------------------------------------
# Report generation (Phase 5)
# ---------------------------------------------------------------------------
def generate_report(all_results):
    """Generate final comparison report with paper-ready tables."""
    log.info(f"\n{'='*80}")
    log.info(" Phase 5: Generating Analysis Report")
    log.info(f"{'='*80}")

    # Flatten all results
    flat = []
    for phase_results in all_results.values():
        if isinstance(phase_results, list):
            flat.extend([r for r in phase_results if r.get("total_items", 0) > 0])

    if not flat:
        log.warning("  No results to report")
        return

    # --- Per-config averages ---
    configs = sorted(set(r["config"] for r in flat))
    datasets = sorted(set(r["dataset"] for r in flat))

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("RSPM PIPELINE — FINAL COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 90)

    # Config comparison table
    report_lines.append("\n## Per-Configuration Averages (across all datasets & splits)\n")
    report_lines.append(f"{'Config':<20} {'Avg TCS':>10} {'Avg Acc':>10} {'Items':>8} {'Correct':>8}")
    report_lines.append("-" * 60)
    for cfg in configs:
        cfg_results = [r for r in flat if r["config"] == cfg]
        avg_tcs = sum(r["tcs"] for r in cfg_results) / len(cfg_results) if cfg_results else 0
        avg_acc = sum(r["overall_accuracy"] for r in cfg_results) / len(cfg_results) if cfg_results else 0
        total = sum(r.get("total_items", 0) for r in cfg_results)
        correct = sum(r.get("correct", 0) for r in cfg_results)
        report_lines.append(f"{cfg:<20} {avg_tcs:>9.1%} {avg_acc:>9.1%} {total:>8} {correct:>8}")

    # Dataset comparison table
    report_lines.append("\n## Per-Dataset Averages (across all configs & splits)\n")
    report_lines.append(f"{'Dataset':<20} {'Avg TCS':>10} {'Avg Acc':>10} {'Items':>8}")
    report_lines.append("-" * 52)
    for ds in datasets:
        ds_results = [r for r in flat if r["dataset"] == ds]
        avg_tcs = sum(r["tcs"] for r in ds_results) / len(ds_results) if ds_results else 0
        avg_acc = sum(r["overall_accuracy"] for r in ds_results) / len(ds_results) if ds_results else 0
        total = sum(r.get("total_items", 0) for r in ds_results)
        report_lines.append(f"{ds:<20} {avg_tcs:>9.1%} {avg_acc:>9.1%} {total:>8}")

    # Cross-table: Dataset × Config
    report_lines.append("\n## TCS Matrix: Dataset × Configuration\n")
    header = f"{'Dataset':<20}" + "".join(f"{c:>18}" for c in configs)
    report_lines.append(header)
    report_lines.append("-" * (20 + 18 * len(configs)))
    for ds in datasets:
        row = f"{ds:<20}"
        for cfg in configs:
            matches = [r for r in flat if r["dataset"] == ds and r["config"] == cfg]
            if matches:
                avg = sum(r["tcs"] for r in matches) / len(matches)
                row += f"{avg:>17.1%}"
            else:
                row += f"{'—':>18}"
        report_lines.append(row)

    # TCS guarantee analysis
    report_lines.append("\n## Formal TCS Guarantee Analysis")
    report_lines.append("-" * 60)
    for cfg in configs:
        cfg_results = [r for r in flat if r["config"] == cfg]
        if not cfg_results:
            continue
        tcs_values = [r["tcs"] for r in cfg_results]
        avg = sum(tcs_values) / len(tcs_values)
        std = (sum((t - avg) ** 2 for t in tcs_values) / len(tcs_values)) ** 0.5
        ci95 = 1.96 * std / len(tcs_values) ** 0.5
        report_lines.append(
            f"  {cfg:<20} mean={avg:.3f}  std={std:.3f}  95%CI=[{avg-ci95:.3f}, {avg+ci95:.3f}]"
        )

    # Proactive TCS bound
    proactive_results = [r for r in flat if r["config"] == "RSPM-Proactive"]
    if proactive_results:
        conflict_items = [r for r in proactive_results if r.get("temporal_conflicts", 0) > 0]
        if conflict_items:
            total_conflicts = sum(r["temporal_conflicts"] for r in conflict_items)
            correct_conflicts = sum(r["correct_conflicts"] for r in conflict_items)
            precision = correct_conflicts / total_conflicts if total_conflicts > 0 else 0
            total_items = sum(r["total_items"] for r in proactive_results)
            conflict_rate = sum(r["temporal_conflicts"] for r in proactive_results) / total_items if total_items > 0 else 0
            tcs_bound = 1 - (1 - precision) * conflict_rate
            report_lines.append(f"\n  Proactive TCS Bound: TCS >= 1 - (1 - precision) * conflict_rate")
            report_lines.append(f"    precision = {precision:.3f}")
            report_lines.append(f"    conflict_rate = {conflict_rate:.3f}")
            report_lines.append(f"    Theoretical TCS >= {tcs_bound:.3f}")

    # Overall summary
    report_lines.append(f"\n{'='*90}")
    overall_tcs = sum(r["tcs"] for r in flat) / len(flat) if flat else 0
    overall_acc = sum(r["overall_accuracy"] for r in flat) / len(flat) if flat else 0
    report_lines.append(f"OVERALL: Avg TCS = {overall_tcs:.1%} | Avg Accuracy = {overall_acc:.1%}")
    report_lines.append(f"Total evaluations: {len(flat)} | Total items: {sum(r.get('total_items',0) for r in flat)}")
    report_lines.append(f"Goal (>95% TCS): {'ACHIEVED' if overall_tcs >= 0.95 else 'NOT YET'}")
    report_lines.append("=" * 90)

    report_text = "\n".join(report_lines)
    log.info("\n" + report_text)

    # Save report
    report_file = RESULTS_BASE / f"FINAL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    log.info(f"\n  Report saved to {report_file}")

    # Save raw results JSON
    raw_file = RESULTS_BASE / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "all_results": {k: v for k, v in all_results.items()},
            "flat_results": flat,
            "configs_tested": configs,
            "datasets_tested": datasets,
        }, f, indent=2)
    log.info(f"  Raw results saved to {raw_file}")

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RSPM Master Evaluation Pipeline")
    parser.add_argument("--phase", type=int, default=1, help="Start from this phase (1-4)")
    parser.add_argument("--reset", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("RSPM MASTER EVALUATION PIPELINE")
    log.info("=" * 80)
    log.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"API key:    {API_KEY[:8]}...{API_KEY[-4:]}")
    log.info(f"Results:    {RESULTS_BASE}")
    log.info(f"Log file:   {log_file}")

    balance = check_balance()
    log.info(f"Balance:    ${balance:.2f}")

    if balance >= 0 and balance < 0.10:
        log.error("INSUFFICIENT BUDGET — need at least $0.10 to proceed")
        log.error("Top up at https://platform.deepseek.com/top_up")
        sys.exit(1)

    # Load or reset checkpoint
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    ckpt = load_checkpoint()

    # Run phases
    pipeline_start = time.time()
    for phase in PHASES:
        if phase["id"] < args.phase:
            continue
        try:
            run_phase(phase, ckpt)
        except KeyboardInterrupt:
            log.warning("\n  Interrupted by user — saving checkpoint")
            save_checkpoint(ckpt)
            sys.exit(130)
        except Exception as e:
            log.error(f"  Phase {phase['id']} failed: {e}")
            traceback.print_exc()
            continue

    pipeline_elapsed = time.time() - pipeline_start

    # Phase 5: Generate report
    generate_report(ckpt["results"])

    # Final budget
    final_balance = check_balance()
    cost = balance - final_balance if balance >= 0 and final_balance >= 0 else -1
    log.info(f"\nPipeline completed in {pipeline_elapsed:.0f}s ({pipeline_elapsed/60:.1f} min)")
    log.info(f"Starting balance: ${balance:.2f}")
    log.info(f"Final balance:    ${final_balance:.2f}")
    if cost >= 0:
        log.info(f"Total API cost:   ${cost:.3f}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
