"""
New Dataset Adapter & Splitter for RSPM Evaluation
===================================================
Converts LongMemEval, PersonaMem-v2, and MemoryAgentBench
into the unified RSPM evaluation format and splits into
dev/validation/test sets.

Datasets:
  1. LongMemEval (ICLR 2025) - knowledge-update + temporal-reasoning
  2. PersonaMem-v2 (COLM 2025) - dynamic user profiling with evolving prefs
  3. MemoryAgentBench (ICLR 2026) - conflict resolution in memory

Split ratios: dev=10%, validation=20%, test=70%
"""
import os
import sys
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')
os.chdir('/home/prevanka/prati/su-reme/ReMe')

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Split ratios
DEV_RATIO = 0.10
VAL_RATIO = 0.20
TEST_RATIO = 0.70

# Directories
BASE_DIR = Path('/home/prevanka/prati/su-reme/ReMe')
DATASETS_DIR = BASE_DIR / 'datasets'
SPLITS_DIR = DATASETS_DIR / 'splits'


def split_data(data: List, ratios: Tuple[float, float, float] = (DEV_RATIO, VAL_RATIO, TEST_RATIO)) -> Tuple[List, List, List]:
    """Split data into dev, validation, test sets."""
    random.shuffle(data)
    n = len(data)
    dev_end = int(n * ratios[0])
    val_end = dev_end + int(n * ratios[1])
    return data[:dev_end], data[dev_end:val_end], data[val_end:]


def save_split(items: List[Dict], output_dir: Path, split_name: str):
    """Save a split as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{split_name}.jsonl"
    with open(filepath, 'w') as f:
        for item in items:
            f.write(json.dumps(item, default=str) + '\n')
    print(f"    Saved {len(items)} items to {filepath}")


# ============================================================
# 1. LongMemEval Adapter
# ============================================================
def adapt_longmemeval():
    """
    Convert LongMemEval to unified RSPM format.
    Focus on knowledge-update and temporal-reasoning types.
    
    LongMemEval format:
      - question_id, question_type, question, answer, question_date
      - haystack_sessions: list of sessions, each session = list of {role, content, has_answer?}
      - answer_session_ids: evidence session IDs
    
    Unified format:
      - conversation_id, dataset, messages, query, ground_truth, metadata
    """
    print("\n=== Adapting LongMemEval ===")
    
    oracle_path = DATASETS_DIR / 'longmemeval' / 'longmemeval_oracle.json'
    if not oracle_path.exists():
        print(f"  [ERROR] {oracle_path} not found")
        return
    
    with open(oracle_path) as f:
        data = json.load(f)
    
    print(f"  Total items in oracle: {len(data)}")
    
    # Count types
    types = Counter(item['question_type'] for item in data)
    print(f"  Question types: {dict(types)}")
    
    # Focus on knowledge-update and temporal-reasoning (most relevant to TCS)
    relevant_types = {'knowledge-update', 'temporal-reasoning'}
    relevant_items = [d for d in data if d['question_type'] in relevant_types]
    print(f"  Relevant items (knowledge-update + temporal-reasoning): {len(relevant_items)}")
    
    # Convert to unified format
    all_items = []
    for item in relevant_items:
        # Flatten sessions into messages
        messages = []
        evidence_turns = []
        
        for s_idx, session in enumerate(item.get('haystack_sessions', [])):
            session_date = ''
            if item.get('haystack_dates') and s_idx < len(item['haystack_dates']):
                session_date = item['haystack_dates'][s_idx]
            
            for turn in session:
                msg = {
                    "role": turn.get('role', 'user'),
                    "content": turn.get('content', ''),
                    "timestamp": session_date,
                    "turn": len(messages)
                }
                messages.append(msg)
                
                if turn.get('has_answer'):
                    evidence_turns.append(turn.get('content', ''))
        
        is_knowledge_update = item['question_type'] == 'knowledge-update'
        
        # For knowledge-update, build updates from evidence turns
        updates = []
        if is_knowledge_update and len(evidence_turns) >= 2:
            # Evidence turns show the progression: old value -> new value
            updates.append({
                "from": evidence_turns[0][:500],  # Older evidence
                "to": evidence_turns[-1][:500],    # Latest evidence
                "attribute": "knowledge"
            })
        
        unified_item = {
            "conversation_id": item['question_id'],
            "dataset": "longmemeval",
            "messages": messages,
            "query": item['question'],
            "ground_truth": {
                "answer": item['answer'],
                "updates": updates,
                "evidence": evidence_turns,
                "question_type": item['question_type']
            },
            "metadata": {
                "question_date": item.get('question_date', ''),
                "num_sessions": len(item.get('haystack_sessions', [])),
                "num_turns": len(messages),
                "is_temporal": True,
                "has_conflict": is_knowledge_update,
                "answer_session_ids": item.get('answer_session_ids', [])
            }
        }
        all_items.append(unified_item)
    
    # Split
    dev, val, test = split_data(all_items)
    output_dir = SPLITS_DIR / 'longmemeval'
    save_split(dev, output_dir, 'dev')
    save_split(val, output_dir, 'validation')
    save_split(test, output_dir, 'test')
    
    # Stats
    stats = {
        "dataset": "longmemeval",
        "source": "xiaowu0162/longmemeval-cleaned (ICLR 2025)",
        "total_items": len(all_items),
        "dev": len(dev),
        "validation": len(val),
        "test": len(test),
        "question_types": dict(Counter(i['ground_truth']['question_type'] for i in all_items)),
        "knowledge_update_count": sum(1 for i in all_items if i['metadata']['has_conflict']),
        "avg_turns": sum(i['metadata']['num_turns'] for i in all_items) / len(all_items) if all_items else 0
    }
    with open(output_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Total adapted: {len(all_items)} (dev={len(dev)}, val={len(val)}, test={len(test)})")
    return stats


# ============================================================
# 2. PersonaMem-v2 Adapter
# ============================================================
def adapt_personamem():
    """
    Convert PersonaMem-v2 to unified RSPM format.
    Focus on 'updated' items that test temporal consistency.
    
    PersonaMem format:
      - persona_id, user_query, correct_answer, incorrect_answers
      - updated (bool), prev_pref (old preference), preference (current)
      - pref_type, related_conversation_snippet
    
    For temporal consistency, we use:
      - prev_pref -> from (old value)
      - preference -> to (new value / "forget this")
    """
    print("\n=== Adapting PersonaMem-v2 ===")
    
    benchmark_path = DATASETS_DIR / 'personamem' / 'benchmark_text.jsonl'
    if not benchmark_path.exists():
        print(f"  [ERROR] {benchmark_path} not found")
        return
    
    with open(benchmark_path) as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"  Total items in benchmark_text: {len(all_data)}")
    
    # Focus on updated items (temporal consistency relevant)
    updated_items = [i for i in all_data if i.get('updated')]
    print(f"  Updated items (temporal consistency): {len(updated_items)}")
    
    # Also include some non-updated items as baselines/negatives
    non_updated = [i for i in all_data if not i.get('updated')]
    # Sample proportionally - keep ~200 non-updated for balance
    random.shuffle(non_updated)
    non_updated_sample = non_updated[:200]
    print(f"  Non-updated sample (baseline): {len(non_updated_sample)}")
    
    combined = updated_items + non_updated_sample
    
    # Convert to unified format
    all_items = []
    for item in combined:
        is_updated = item.get('updated', False)
        prev_pref = item.get('prev_pref', '')
        current_pref = item.get('preference', '')
        
        # Build messages from related_conversation_snippet
        messages = []
        snippet = item.get('related_conversation_snippet', '')
        if isinstance(snippet, str) and snippet:
            try:
                snippet_data = json.loads(snippet)
                if isinstance(snippet_data, list):
                    for s_idx, turn in enumerate(snippet_data):
                        messages.append({
                            "role": turn.get('role', 'user'),
                            "content": turn.get('content', ''),
                            "turn": s_idx
                        })
            except (json.JSONDecodeError, TypeError):
                messages.append({"role": "user", "content": str(snippet), "turn": 0})
        elif isinstance(snippet, list):
            for s_idx, turn in enumerate(snippet):
                if isinstance(turn, dict):
                    messages.append({
                        "role": turn.get('role', 'user'),
                        "content": turn.get('content', ''),
                        "turn": s_idx
                    })
        
        # Build the query from user_query
        query_text = ''
        user_query = item.get('user_query', '')
        if isinstance(user_query, dict):
            query_text = user_query.get('content', '')
        elif isinstance(user_query, str):
            query_text = user_query
        
        # Build updates for temporal consistency
        updates = []
        if is_updated and prev_pref:
            updates.append({
                "from": str(prev_pref),
                "to": str(current_pref),
                "attribute": item.get('topic_preference', 'preference')
            })
        
        # Correct answer
        correct_answer = item.get('correct_answer', '')
        if isinstance(correct_answer, dict):
            correct_answer = correct_answer.get('content', str(correct_answer))
        
        # Incorrect answers for MCQ evaluation
        incorrect_answers = item.get('incorrect_answers', [])
        if isinstance(incorrect_answers, str):
            try:
                incorrect_answers = json.loads(incorrect_answers)
            except (json.JSONDecodeError, TypeError):
                incorrect_answers = [incorrect_answers]
        
        unified_item = {
            "conversation_id": f"personamem_{item.get('persona_id', 0)}_{len(all_items)}",
            "dataset": "personamem",
            "messages": messages,
            "query": query_text,
            "ground_truth": {
                "answer": str(correct_answer),
                "updates": updates,
                "incorrect_answers": incorrect_answers,
                "question_type": item.get('pref_type', 'unknown')
            },
            "metadata": {
                "persona_id": item.get('persona_id', ''),
                "is_temporal": is_updated,
                "has_conflict": is_updated,
                "updated": is_updated,
                "prev_pref": str(prev_pref) if prev_pref else '',
                "current_pref": str(current_pref),
                "pref_type": item.get('pref_type', ''),
                "topic_preference": item.get('topic_preference', ''),
                "num_turns": len(messages)
            }
        }
        all_items.append(unified_item)
    
    # Split (stratified by updated/not)
    updated_adapted = [i for i in all_items if i['metadata']['has_conflict']]
    non_updated_adapted = [i for i in all_items if not i['metadata']['has_conflict']]
    
    u_dev, u_val, u_test = split_data(updated_adapted)
    n_dev, n_val, n_test = split_data(non_updated_adapted)
    
    dev = u_dev + n_dev
    val = u_val + n_val
    test = u_test + n_test
    
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    output_dir = SPLITS_DIR / 'personamem'
    save_split(dev, output_dir, 'dev')
    save_split(val, output_dir, 'validation')
    save_split(test, output_dir, 'test')
    
    stats = {
        "dataset": "personamem",
        "source": "bowen-upenn/PersonaMem-v2 (COLM 2025)",
        "total_items": len(all_items),
        "dev": len(dev),
        "validation": len(val),
        "test": len(test),
        "updated_count": len(updated_adapted),
        "non_updated_count": len(non_updated_adapted),
        "pref_types": dict(Counter(i['metadata']['pref_type'] for i in all_items)),
        "avg_turns": sum(i['metadata']['num_turns'] for i in all_items) / len(all_items) if all_items else 0
    }
    with open(output_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Total adapted: {len(all_items)} (dev={len(dev)}, val={len(val)}, test={len(test)})")
    return stats


# ============================================================
# 3. MemoryAgentBench Adapter
# ============================================================
def adapt_memoryagentbench():
    """
    Convert MemoryAgentBench Conflict_Resolution to unified RSPM format.
    
    MemoryAgentBench format:
      - context: string of facts
      - questions: list of question strings  
      - answers: list of answer lists
      - metadata: {source, qa_pair_ids, ...}
    
    Each item has ~100 Q/A pairs about conflicting facts.
    We expand each Q/A pair into a separate evaluation item.
    """
    print("\n=== Adapting MemoryAgentBench ===")
    
    cr_path = DATASETS_DIR / 'memoryagentbench' / 'Conflict_Resolution.jsonl'
    if not cr_path.exists():
        print(f"  [ERROR] {cr_path} not found")
        return
    
    with open(cr_path) as f:
        raw_items = [json.loads(line) for line in f]
    
    print(f"  Raw items (contexts): {len(raw_items)}")
    total_qa = sum(len(item['questions']) for item in raw_items)
    print(f"  Total Q/A pairs: {total_qa}")
    
    # Also load Accurate_Retrieval for non-conflict baseline
    ar_path = DATASETS_DIR / 'memoryagentbench' / 'Accurate_Retrieval.jsonl'
    ar_items = []
    if ar_path.exists():
        with open(ar_path) as f:
            ar_items = [json.loads(line) for line in f]
        print(f"  Accurate_Retrieval items: {len(ar_items)}")
    
    # Convert each Q/A pair to unified format
    all_items = []
    
    # Conflict Resolution items
    for ctx_idx, raw in enumerate(raw_items):
        context = raw.get('context', '')
        questions = raw.get('questions', [])
        answers = raw.get('answers', [])
        metadata = raw.get('metadata', {})
        source = metadata.get('source', f'conflict_{ctx_idx}')
        qa_pair_ids = metadata.get('qa_pair_ids', []) or []
        
        # Parse facts from context
        facts = []
        for line in context.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                facts.append(line.split('. ', 1)[1])
        
        # Build messages from facts (simulate conversation where facts are shared)
        messages = []
        # Group facts into batches for realistic conversation
        batch_size = 10
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i+batch_size]
            content = "Here are some facts I'd like you to remember:\n" + \
                      "\n".join(f"- {fact}" for fact in batch)
            messages.append({
                "role": "user",
                "content": content,
                "turn": len(messages)
            })
            messages.append({
                "role": "assistant",
                "content": f"I've noted those {len(batch)} facts.",
                "turn": len(messages)
            })
        
        for q_idx, (question, answer_list) in enumerate(zip(questions, answers)):
            qa_id = qa_pair_ids[q_idx] if q_idx < len(qa_pair_ids) else f"{source}_q{q_idx}"
            
            unified_item = {
                "conversation_id": f"mab_{source}_{q_idx}",
                "dataset": "memoryagentbench",
                "messages": messages,
                "query": question,
                "ground_truth": {
                    "answer": answer_list[0] if isinstance(answer_list, list) and answer_list else str(answer_list),
                    "all_answers": answer_list if isinstance(answer_list, list) else [str(answer_list)],
                    "updates": [],  # Conflicts are implicit in the facts
                    "question_type": "conflict_resolution"
                },
                "metadata": {
                    "source": source,
                    "qa_pair_id": qa_id,
                    "is_temporal": True,
                    "has_conflict": True,
                    "num_facts": len(facts),
                    "num_turns": len(messages),
                    "context_idx": ctx_idx
                }
            }
            all_items.append(unified_item)
    
    # Add some Accurate_Retrieval items as non-conflict baseline
    ar_adapted = []
    for ctx_idx, raw in enumerate(ar_items[:3]):  # Take 3 contexts
        context = raw.get('context', '')
        questions = raw.get('questions', [])
        answers = raw.get('answers', [])
        
        # Build simple messages from context
        messages = []
        # Split context into chunks
        chunks = context.split('\n\n')
        for i, chunk in enumerate(chunks[:5]):  # First 5 chunks
            if chunk.strip():
                messages.append({
                    "role": "user",
                    "content": chunk.strip()[:500],
                    "turn": len(messages)
                })
        
        for q_idx, (question, answer_list) in enumerate(zip(questions[:20], answers[:20])):
            item = {
                "conversation_id": f"mab_ar_{ctx_idx}_{q_idx}",
                "dataset": "memoryagentbench",
                "messages": messages,
                "query": question,
                "ground_truth": {
                    "answer": answer_list[0] if isinstance(answer_list, list) and answer_list else str(answer_list),
                    "all_answers": answer_list if isinstance(answer_list, list) else [str(answer_list)],
                    "updates": [],
                    "question_type": "accurate_retrieval"
                },
                "metadata": {
                    "source": "accurate_retrieval",
                    "is_temporal": False,
                    "has_conflict": False,
                    "num_turns": len(messages),
                    "context_idx": ctx_idx
                }
            }
            ar_adapted.append(item)
    
    print(f"  Conflict Resolution adapted: {len(all_items)}")
    print(f"  Accurate Retrieval adapted: {len(ar_adapted)}")
    
    combined = all_items + ar_adapted
    
    # Stratified split
    conflict = [i for i in combined if i['metadata']['has_conflict']]
    non_conflict = [i for i in combined if not i['metadata']['has_conflict']]
    
    c_dev, c_val, c_test = split_data(conflict)
    n_dev, n_val, n_test = split_data(non_conflict)
    
    dev = c_dev + n_dev
    val = c_val + n_val
    test = c_test + n_test
    
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    output_dir = SPLITS_DIR / 'memoryagentbench'
    save_split(dev, output_dir, 'dev')
    save_split(val, output_dir, 'validation')
    save_split(test, output_dir, 'test')
    
    stats = {
        "dataset": "memoryagentbench",
        "source": "ai-hyz/MemoryAgentBench (ICLR 2026)",
        "total_items": len(combined),
        "dev": len(dev),
        "validation": len(val),
        "test": len(test),
        "conflict_resolution_count": len(all_items),
        "accurate_retrieval_count": len(ar_adapted),
        "question_types": dict(Counter(i['ground_truth']['question_type'] for i in combined)),
        "avg_turns": sum(i['metadata']['num_turns'] for i in combined) / len(combined) if combined else 0
    }
    with open(output_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Total adapted: {len(combined)} (dev={len(dev)}, val={len(val)}, test={len(test)})")
    return stats


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("NEW DATASET ADAPTER & SPLITTER")
    print("=" * 70)
    
    all_stats = {}
    
    # 1. LongMemEval
    stats = adapt_longmemeval()
    if stats:
        all_stats['longmemeval'] = stats
    
    # 2. PersonaMem-v2
    stats = adapt_personamem()
    if stats:
        all_stats['personamem'] = stats
    
    # 3. MemoryAgentBench
    stats = adapt_memoryagentbench()
    if stats:
        all_stats['memoryagentbench'] = stats
    
    # Save combined stats
    with open(SPLITS_DIR / 'new_datasets_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("ADAPTATION SUMMARY")
    print(f"{'=' * 70}")
    total = 0
    for name, s in all_stats.items():
        t = s.get('total_items', 0)
        total += t
        print(f"  {name:<25} {t:>6} items (dev={s.get('dev',0)}, val={s.get('validation',0)}, test={s.get('test',0)})")
    print(f"  {'TOTAL':<25} {total:>6} items")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
