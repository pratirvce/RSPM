"""
Dataset Splitter & Unified Adapter for RSPM Evaluation

Splits all datasets into dev/test/validation sets and converts
them into a unified format for RSPM evaluation.

Supported datasets:
1. HaluMem (temporal consistency in multi-turn conversations)
2. LoCoMo (very long-term conversational memory)
3. TimeBench (temporal reasoning)
4. TemporalMemoryDataset (temporal memory questions)

Split ratios: dev=10%, validation=20%, test=70%
"""
import os
import sys
import json
import random
import hashlib
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Ensure proper imports
sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')
os.chdir('/home/prevanka/prati/su-reme/ReMe')

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Split ratios
DEV_RATIO = 0.10      # 10% for development/debugging
VAL_RATIO = 0.20      # 20% for validation/hyperparameter tuning
TEST_RATIO = 0.70     # 70% for final testing/reporting

# Base directories
BASE_DIR = Path('/home/prevanka/prati/su-reme/ReMe')
DATASETS_DIR = BASE_DIR / 'datasets'
SPLITS_DIR = DATASETS_DIR / 'splits'


def split_data(data: List, ratios: Tuple[float, float, float] = (DEV_RATIO, VAL_RATIO, TEST_RATIO)) -> Tuple[List, List, List]:
    """
    Split data into dev, validation, test sets.
    
    Args:
        data: List of data items
        ratios: (dev_ratio, val_ratio, test_ratio) - must sum to 1.0
    
    Returns:
        (dev_set, val_set, test_set)
    """
    shuffled = list(data)
    random.shuffle(shuffled)
    
    n = len(shuffled)
    dev_end = max(1, int(n * ratios[0]))
    val_end = max(dev_end + 1, int(n * (ratios[0] + ratios[1])))
    
    dev_set = shuffled[:dev_end]
    val_set = shuffled[dev_end:val_end]
    test_set = shuffled[val_end:]
    
    return dev_set, val_set, test_set


def save_split(data: List, filepath: Path, format: str = 'jsonl'):
    """Save split data to file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'jsonl':
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"   Saved {len(data)} items to {filepath}")


def save_stats(stats: Dict, filepath: Path):
    """Save dataset statistics"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


# ============================================================
# 1. HaluMem Dataset Splitter
# ============================================================
def split_halumem():
    """
    Split HaluMem dataset into dev/val/test.
    HaluMem has conversations with temporal updates.
    """
    print("\n" + "="*70)
    print("1. SPLITTING HALUMEM DATASET")
    print("="*70)
    
    source = DATASETS_DIR / 'memoryscope' / 'halumem_medium.jsonl'
    out_dir = SPLITS_DIR / 'halumem'
    
    if not source.exists():
        print(f"   ✗ Source not found: {source}")
        return None
    
    # Load data
    data = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"   Loaded {len(data)} conversations")
    
    # Count conversations with temporal updates
    with_updates = [d for d in data if d.get('ground_truth', {}).get('updates', [])]
    without_updates = [d for d in data if not d.get('ground_truth', {}).get('updates', [])]
    
    print(f"   With temporal updates: {len(with_updates)}")
    print(f"   Without updates: {len(without_updates)}")
    
    # Stratified split: maintain ratio of updates vs no-updates
    dev_u, val_u, test_u = split_data(with_updates)
    dev_n, val_n, test_n = split_data(without_updates)
    
    dev = dev_u + dev_n
    val = val_u + val_n
    test = test_u + test_n
    
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    # Save splits
    save_split(dev, out_dir / 'dev.jsonl')
    save_split(val, out_dir / 'validation.jsonl')
    save_split(test, out_dir / 'test.jsonl')
    
    stats = {
        "dataset": "HaluMem",
        "source": str(source),
        "total": len(data),
        "splits": {
            "dev": {"total": len(dev), "with_updates": len(dev_u), "without_updates": len(dev_n)},
            "validation": {"total": len(val), "with_updates": len(val_u), "without_updates": len(val_n)},
            "test": {"total": len(test), "with_updates": len(test_u), "without_updates": len(test_n)}
        },
        "split_ratios": {"dev": DEV_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "created": datetime.now().isoformat()
    }
    save_stats(stats, out_dir / 'split_stats.json')
    
    print(f"\n   ✓ HaluMem split complete:")
    print(f"     Dev:        {len(dev):>5} ({len(dev_u)} with updates)")
    print(f"     Validation: {len(val):>5} ({len(val_u)} with updates)")
    print(f"     Test:       {len(test):>5} ({len(test_u)} with updates)")
    
    return stats


# ============================================================
# 2. LoCoMo Dataset Splitter
# ============================================================
def split_locomo():
    """
    Split LoCoMo dataset into dev/val/test.
    LoCoMo has 10 long conversations with 1986 QA pairs.
    We split at the QA pair level within each conversation.
    """
    print("\n" + "="*70)
    print("2. SPLITTING LOCOMO DATASET")
    print("="*70)
    
    source = DATASETS_DIR / 'locomo_repo' / 'data' / 'locomo10.json'
    out_dir = SPLITS_DIR / 'locomo'
    
    if not source.exists():
        print(f"   ✗ Source not found: {source}")
        return None
    
    with open(source) as f:
        conversations = json.load(f)
    
    print(f"   Loaded {len(conversations)} conversations")
    
    # First, save conversations separately (they're large and shared across QA pairs)
    # Then reference them by conv_id in each QA item
    conv_out_dir = SPLITS_DIR / 'locomo' / 'conversations'
    conv_out_dir.mkdir(parents=True, exist_ok=True)
    
    all_items = []
    qa_categories = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain", 5: "unanswerable"}
    
    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get('sample_id', f'locomo_{conv_idx}')
        qa_pairs = conv.get('qa', [])
        event_summary = conv.get('event_summary', [])
        
        # Build conversation messages from LoCoMo's session format
        conv_data = conv.get('conversation', {})
        speaker_a = conv_data.get('speaker_a', 'user')
        speaker_b = conv_data.get('speaker_b', 'assistant')
        
        session_keys = sorted(
            [k for k in conv_data.keys() if k.startswith('session_') and not k.endswith('date_time')],
            key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0
        )
        
        messages = []
        for session_key in session_keys:
            session = conv_data.get(session_key, [])
            session_date = conv_data.get(f'{session_key}_date_time', '')
            
            if not isinstance(session, list):
                continue
            
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                
                speaker = turn.get('speaker', '')
                role = 'user' if speaker == speaker_a else 'assistant'
                
                messages.append({
                    "role": role,
                    "content": turn.get('text', ''),
                    "timestamp": session_date,
                    "turn": len(messages),
                    "dia_id": turn.get('dia_id', '')
                })
        
        # Save conversation separately
        with open(conv_out_dir / f'{conv_id}.json', 'w') as f:
            json.dump({
                "conversation_id": conv_id,
                "messages": messages,
                "event_summary": event_summary,
                "num_sessions": len(session_keys),
                "num_turns": len(messages),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b
            }, f)
        
        # Create lightweight QA items (reference conversation, don't embed it)
        for qa_idx, qa in enumerate(qa_pairs):
            category_id = qa.get('category', 0)
            category_name = qa_categories.get(category_id, f"category_{category_id}")
            
            item = {
                "conversation_id": f"{conv_id}_qa_{qa_idx}",
                "dataset": "locomo",
                "conversation_ref": conv_id,  # Reference to conversation file
                "messages": [],  # Loaded at evaluation time from conversation file
                "query": qa.get('question', ''),
                "ground_truth": {
                    "answer": qa.get('answer', ''),
                    "evidence": qa.get('evidence', []),
                    "category": category_name,
                    "category_id": category_id,
                    "updates": []
                },
                "metadata": {
                    "num_sessions": len(session_keys),
                    "num_turns": len(messages),
                    "event_summary_count": len(event_summary),
                    "is_temporal": category_id == 2
                }
            }
            all_items.append(item)
    
    print(f"   Total QA items: {len(all_items)}")
    
    # Count by category
    temporal_items = [x for x in all_items if x['metadata']['is_temporal']]
    non_temporal = [x for x in all_items if not x['metadata']['is_temporal']]
    print(f"   Temporal QA: {len(temporal_items)}")
    print(f"   Non-temporal QA: {len(non_temporal)}")
    
    # Stratified split by temporal vs non-temporal
    dev_t, val_t, test_t = split_data(temporal_items)
    dev_n, val_n, test_n = split_data(non_temporal)
    
    dev = dev_t + dev_n
    val = val_t + val_n
    test = test_t + test_n
    
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    save_split(dev, out_dir / 'dev.jsonl')
    save_split(val, out_dir / 'validation.jsonl')
    save_split(test, out_dir / 'test.jsonl')
    
    stats = {
        "dataset": "LoCoMo",
        "source": str(source),
        "total": len(all_items),
        "conversations": len(conversations),
        "splits": {
            "dev": {"total": len(dev), "temporal": len(dev_t), "non_temporal": len(dev_n)},
            "validation": {"total": len(val), "temporal": len(val_t), "non_temporal": len(val_n)},
            "test": {"total": len(test), "temporal": len(test_t), "non_temporal": len(test_n)}
        },
        "split_ratios": {"dev": DEV_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "created": datetime.now().isoformat()
    }
    save_stats(stats, out_dir / 'split_stats.json')
    
    print(f"\n   ✓ LoCoMo split complete:")
    print(f"     Dev:        {len(dev):>5} ({len(dev_t)} temporal)")
    print(f"     Validation: {len(val):>5} ({len(val_t)} temporal)")
    print(f"     Test:       {len(test):>5} ({len(test_t)} temporal)")
    
    return stats


# ============================================================
# 3. TimeBench Dataset Splitter
# ============================================================
def split_timebench():
    """
    Split TimeBench dataset into dev/val/test.
    TimeBench has 7,553 examples across multiple temporal reasoning categories.
    """
    print("\n" + "="*70)
    print("3. SPLITTING TIMEBENCH DATASET")
    print("="*70)
    
    source_dir = DATASETS_DIR / 'timebench_repo' / 'TimeBench-subset-7553'
    out_dir = SPLITS_DIR / 'timebench'
    
    if not source_dir.exists():
        print(f"   ✗ Source not found: {source_dir}")
        return None
    
    # Load all categories
    all_items = []
    category_counts = {}
    
    for category_dir in sorted(source_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        
        for jsonl_file in sorted(category_dir.glob('*.jsonl')):
            subcategory = jsonl_file.stem
            count = 0
            
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    item_data = json.loads(line)
                    
                    # Convert to unified format
                    item = {
                        "conversation_id": f"timebench_{category}_{subcategory}_{count}",
                        "dataset": "timebench",
                        "messages": [],  # TimeBench doesn't have conversation context
                        "query": item_data.get('question', ''),
                        "ground_truth": {
                            "answer": item_data.get('answer', ''),
                            "context": item_data.get('context', ''),
                            "updates": []
                        },
                        "metadata": {
                            "category": category,
                            "subcategory": subcategory,
                            "is_temporal": True  # All TimeBench items are temporal
                        }
                    }
                    all_items.append(item)
                    count += 1
            
            key = f"{category}/{subcategory}"
            category_counts[key] = count
    
    print(f"   Total items: {len(all_items)}")
    print(f"   Categories: {len(category_counts)}")
    for cat, cnt in sorted(category_counts.items()):
        print(f"     {cat}: {cnt}")
    
    # Split maintaining category distribution
    # Group by category
    by_category = {}
    for item in all_items:
        cat = item['metadata']['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    dev, val, test = [], [], []
    dev_cats, val_cats, test_cats = {}, {}, {}
    
    for cat, items in by_category.items():
        d, v, t = split_data(items)
        dev.extend(d)
        val.extend(v)
        test.extend(t)
        dev_cats[cat] = len(d)
        val_cats[cat] = len(v)
        test_cats[cat] = len(t)
    
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    save_split(dev, out_dir / 'dev.jsonl')
    save_split(val, out_dir / 'validation.jsonl')
    save_split(test, out_dir / 'test.jsonl')
    
    stats = {
        "dataset": "TimeBench",
        "source": str(source_dir),
        "total": len(all_items),
        "categories": category_counts,
        "splits": {
            "dev": {"total": len(dev), "by_category": dev_cats},
            "validation": {"total": len(val), "by_category": val_cats},
            "test": {"total": len(test), "by_category": test_cats}
        },
        "split_ratios": {"dev": DEV_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "created": datetime.now().isoformat()
    }
    save_stats(stats, out_dir / 'split_stats.json')
    
    print(f"\n   ✓ TimeBench split complete:")
    print(f"     Dev:        {len(dev):>5}")
    print(f"     Validation: {len(val):>5}")
    print(f"     Test:       {len(test):>5}")
    
    return stats


# ============================================================
# 4. TemporalMemoryDataset Splitter
# ============================================================
def split_temporal_memory():
    """
    Split TemporalMemoryDataset into dev/val/test.
    Contains conversations + temporal questions about them.
    """
    print("\n" + "="*70)
    print("4. SPLITTING TEMPORAL MEMORY DATASET")
    print("="*70)
    
    repo_dir = DATASETS_DIR / 'temporal_memory_repo'
    out_dir = SPLITS_DIR / 'temporal_memory'
    
    if not repo_dir.exists():
        print(f"   ✗ Source not found: {repo_dir}")
        return None
    
    # Load test data (questions)
    all_items = []
    
    # Content-time questions
    content_file = repo_dir / 'TestData' / 'content_time_qs' / 'content_time_qs.json'
    if content_file.exists():
        with open(content_file) as f:
            content_data = json.load(f)
        
        # Load conversations for context
        conv_dir = repo_dir / 'ConversationData'
        conversations = {}
        if conv_dir.exists():
            for conv_file in conv_dir.glob('*.json'):
                file_id = conv_file.stem
                with open(conv_file) as f:
                    conversations[file_id] = json.load(f)
        
        file_indexes = content_data.get('file_indexes', [])
        
        for file_id in file_indexes:
            key = f'file_{file_id}'
            if key not in content_data:
                continue
            
            questions = content_data[key]
            conv_data = conversations.get(str(file_id), {})
            
            # Build messages from conversation
            messages = []
            if isinstance(conv_data, dict):
                for session_key in sorted(conv_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                    session = conv_data[session_key]
                    if isinstance(session, list):
                        for turn in session:
                            if isinstance(turn, dict):
                                messages.append({
                                    "role": turn.get("role", "user"),
                                    "content": turn.get("content", str(turn)),
                                    "turn": len(messages)
                                })
            
            for q_idx, question in enumerate(questions):
                if isinstance(question, dict):
                    q_text = question.get('question', question.get('q', ''))
                    a_text = question.get('answer', question.get('a', ''))
                else:
                    continue
                
                item = {
                    "conversation_id": f"temporal_memory_{file_id}_q{q_idx}",
                    "dataset": "temporal_memory",
                    "messages": messages,
                    "query": q_text,
                    "ground_truth": {
                        "answer": a_text,
                        "updates": []
                    },
                    "metadata": {
                        "file_id": file_id,
                        "question_type": "content_time",
                        "is_temporal": True,
                        "num_turns": len(messages)
                    }
                }
                all_items.append(item)
    
    # Ambiguous time questions
    ambiguous_dir = repo_dir / 'TestData' / 'ambiguous_time_qs'
    if ambiguous_dir.exists():
        for test_file in ambiguous_dir.glob('*.json'):
            q_type = test_file.stem.replace('test_', '')
            
            with open(test_file) as f:
                questions = json.load(f)
            
            if isinstance(questions, list):
                for q_idx, question in enumerate(questions):
                    if isinstance(question, dict):
                        q_text = question.get('question', question.get('q', ''))
                        a_text = question.get('answer', question.get('a', ''))
                    else:
                        continue
                    
                    item = {
                        "conversation_id": f"temporal_memory_ambig_{q_type}_{q_idx}",
                        "dataset": "temporal_memory",
                        "messages": [],
                        "query": q_text,
                        "ground_truth": {
                            "answer": a_text,
                            "updates": []
                        },
                        "metadata": {
                            "question_type": f"ambiguous_{q_type}",
                            "is_temporal": True
                        }
                    }
                    all_items.append(item)
            elif isinstance(questions, dict):
                for key, val in questions.items():
                    if isinstance(val, list):
                        for q_idx, question in enumerate(val):
                            if isinstance(question, dict):
                                q_text = question.get('question', question.get('q', ''))
                                a_text = question.get('answer', question.get('a', ''))
                            else:
                                continue
                            
                            item = {
                                "conversation_id": f"temporal_memory_ambig_{q_type}_{key}_{q_idx}",
                                "dataset": "temporal_memory",
                                "messages": [],
                                "query": q_text,
                                "ground_truth": {
                                    "answer": a_text,
                                    "updates": []
                                },
                                "metadata": {
                                    "question_type": f"ambiguous_{q_type}",
                                    "is_temporal": True
                                }
                            }
                            all_items.append(item)
    
    print(f"   Total items: {len(all_items)}")
    
    if len(all_items) == 0:
        print("   ✗ No items loaded")
        return None
    
    # Count by type
    by_type = {}
    for item in all_items:
        t = item['metadata']['question_type']
        by_type[t] = by_type.get(t, 0) + 1
    
    for t, c in sorted(by_type.items()):
        print(f"     {t}: {c}")
    
    dev, val, test = split_data(all_items)
    
    save_split(dev, out_dir / 'dev.jsonl')
    save_split(val, out_dir / 'validation.jsonl')
    save_split(test, out_dir / 'test.jsonl')
    
    stats = {
        "dataset": "TemporalMemoryDataset",
        "source": str(repo_dir),
        "total": len(all_items),
        "question_types": by_type,
        "splits": {
            "dev": {"total": len(dev)},
            "validation": {"total": len(val)},
            "test": {"total": len(test)}
        },
        "split_ratios": {"dev": DEV_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "created": datetime.now().isoformat()
    }
    save_stats(stats, out_dir / 'split_stats.json')
    
    print(f"\n   ✓ TemporalMemory split complete:")
    print(f"     Dev:        {len(dev):>5}")
    print(f"     Validation: {len(val):>5}")
    print(f"     Test:       {len(test):>5}")
    
    return stats


# ============================================================
# MAIN: Run all splits
# ============================================================
def main():
    print("="*70)
    print("RSPM DATASET SPLITTER - Dev / Validation / Test")
    print("="*70)
    print(f"Split Ratios: Dev={DEV_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Output Directory: {SPLITS_DIR}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    all_stats = {}
    
    # 1. HaluMem
    stats = split_halumem()
    if stats:
        all_stats['halumem'] = stats
    
    # 2. LoCoMo
    stats = split_locomo()
    if stats:
        all_stats['locomo'] = stats
    
    # 3. TimeBench
    stats = split_timebench()
    if stats:
        all_stats['timebench'] = stats
    
    # 4. TemporalMemoryDataset
    stats = split_temporal_memory()
    if stats:
        all_stats['temporal_memory'] = stats
    
    # Save combined stats
    save_stats(all_stats, SPLITS_DIR / 'all_splits_stats.json')
    
    # Print final summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<25} {'Total':>8} {'Dev':>8} {'Val':>8} {'Test':>8}")
    print("-"*70)
    
    grand_total = 0
    grand_dev = 0
    grand_val = 0
    grand_test = 0
    
    for name, stats in all_stats.items():
        total = stats['total']
        dev = stats['splits']['dev']['total']
        val = stats['splits']['validation']['total']
        test = stats['splits']['test']['total']
        
        print(f"{name:<25} {total:>8} {dev:>8} {val:>8} {test:>8}")
        grand_total += total
        grand_dev += dev
        grand_val += val
        grand_test += test
    
    print("-"*70)
    print(f"{'TOTAL':<25} {grand_total:>8} {grand_dev:>8} {grand_val:>8} {grand_test:>8}")
    print(f"\nOutput directory: {SPLITS_DIR}")
    print(f"\nStructure:")
    print(f"  datasets/splits/")
    print(f"  ├── halumem/")
    print(f"  │   ├── dev.jsonl")
    print(f"  │   ├── validation.jsonl")
    print(f"  │   ├── test.jsonl")
    print(f"  │   └── split_stats.json")
    print(f"  ├── locomo/")
    print(f"  │   ├── dev.jsonl")
    print(f"  │   ├── validation.jsonl")
    print(f"  │   ├── test.jsonl")
    print(f"  │   └── split_stats.json")
    print(f"  ├── timebench/")
    print(f"  │   ├── dev.jsonl")
    print(f"  │   ├── validation.jsonl")
    print(f"  │   ├── test.jsonl")
    print(f"  │   └── split_stats.json")
    print(f"  ├── temporal_memory/")
    print(f"  │   ├── dev.jsonl")
    print(f"  │   ├── validation.jsonl")
    print(f"  │   ├── test.jsonl")
    print(f"  │   └── split_stats.json")
    print(f"  └── all_splits_stats.json")
    
    print(f"\n✅ All datasets split successfully!")
    print(f"   Total data points: {grand_total}")
    print(f"   Dev: {grand_dev} | Validation: {grand_val} | Test: {grand_test}")


if __name__ == '__main__':
    main()
