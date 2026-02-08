"""
Fix All Dataset Splits
======================
Repairs broken/incomplete dataset splits for all 7 datasets.

Issues fixed:
  1. LoCoMo:          Empty messages → load from conversation ref files
  2. TimeBench:       Empty queries/answers → filter valid, fix list answers
  3. TemporalMemory:  Everything empty → re-adapt from raw data
  4. PersonaMem:      Query is dict → extract string
  5. LongMemEval:     OK (just cap messages)
  6. MemoryAgentBench: OK (just cap messages)
  7. HaluMem:         Already working fine
"""
import json
import os
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/home/prevanka/prati/su-reme/ReMe")
SPLITS_DIR = BASE_DIR / "datasets" / "splits"
RAW_DIR = BASE_DIR / "datasets"

MAX_MESSAGES = 40  # Cap messages per item

stats = {}


def save_split(items, output_dir, split_name):
    """Save items to a JSONL split file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{split_name}.jsonl"
    with open(filepath, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
    print(f"    Saved {len(items)} items to {filepath}")


def split_data(items, dev_ratio=0.1, val_ratio=0.2):
    """Split into dev/validation/test"""
    random.shuffle(items)
    n = len(items)
    dev_n = max(1, int(n * dev_ratio))
    val_n = max(1, int(n * val_ratio))
    return items[:dev_n], items[dev_n:dev_n+val_n], items[dev_n+val_n:]


# ============================================================
# 1. FIX LOCOMO
# ============================================================
def fix_locomo():
    """Load conversation messages from ref files and embed them in split items"""
    print("\n[1/6] Fixing LoCoMo...")
    
    # Load raw data to get full conversations with QA pairs
    raw_file = RAW_DIR / "locomo_repo" / "data" / "locomo10.json"
    if not raw_file.exists():
        print("    SKIP: Raw LoCoMo data not found")
        return
    
    with open(raw_file) as f:
        raw_conversations = json.load(f)
    
    print(f"    Found {len(raw_conversations)} raw conversations")
    
    # Build conversation index: extract messages from session-based format
    all_items = []
    
    for conv_idx, conv in enumerate(raw_conversations):
        conversation = conv.get('conversation', {})
        qa_pairs = conv.get('qa', [])
        speaker_a = conversation.get('speaker_a', 'User')
        speaker_b = conversation.get('speaker_b', 'Assistant')
        
        # Extract all messages from sessions
        messages = []
        for i in range(1, 100):
            session_key = f'session_{i}'
            date_key = f'session_{i}_date_time'
            if session_key not in conversation:
                break
            session = conversation[session_key]
            session_date = conversation.get(date_key, '')
            if isinstance(session, list):
                for turn in session:
                    speaker = turn.get('speaker', '')
                    text = turn.get('text', '')
                    dia_id = turn.get('dia_id', '')
                    role = 'user' if speaker == speaker_a else 'assistant'
                    messages.append({
                        'role': role,
                        'content': text,
                        'dia_id': dia_id,
                        'session_date': session_date,
                        'turn': len(messages)
                    })
        
        if not messages:
            continue
        
        # Create QA items - use last N messages as context
        # Category mapping: 1=single_hop, 2=multi_hop, 3=temporal, 4=open_ended, 5=unanswerable
        CATEGORY_MAP = {1: 'single_hop', 2: 'multi_hop', 3: 'temporal', 4: 'open_ended', 5: 'unanswerable'}
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            evidence = qa.get('evidence', [])
            category_id = qa.get('category', 0)
            category = CATEGORY_MAP.get(category_id, 'unknown')
            
            if not question or not answer:
                continue
            
            # For temporal questions (cat 3), mark as conflict
            is_temporal = category_id == 3
            
            # Get relevant messages based on evidence (dia_ids like "D1:3")
            # Also include surrounding context
            relevant_msgs = messages[-MAX_MESSAGES:]  # Default: last N messages
            
            if evidence:
                # Try to find evidence messages and include context around them
                evidence_indices = set()
                for ev in evidence:
                    for msg_idx, msg in enumerate(messages):
                        if msg.get('dia_id') == ev:
                            evidence_indices.add(msg_idx)
                
                if evidence_indices:
                    # Include ±5 messages around each evidence
                    context_indices = set()
                    for idx in evidence_indices:
                        for offset in range(-5, 6):
                            if 0 <= idx + offset < len(messages):
                                context_indices.add(idx + offset)
                    
                    relevant_msgs = [messages[i] for i in sorted(context_indices)]
                    # Cap
                    if len(relevant_msgs) > MAX_MESSAGES:
                        relevant_msgs = relevant_msgs[-MAX_MESSAGES:]
            
            item = {
                "conversation_id": f"locomo_{conv_idx}_{qa_idx}",
                "dataset": "locomo",
                "messages": relevant_msgs,
                "query": question,
                "ground_truth": {
                    "answer": answer,
                    "evidence": evidence,
                    "category": category,
                    "category_id": category_id,
                    "updates": []
                },
                "metadata": {
                    "is_temporal": is_temporal,
                    "has_conflict": is_temporal,
                    "category": category,
                    "conv_idx": conv_idx,
                    "num_messages": len(relevant_msgs)
                }
            }
            all_items.append(item)
    
    print(f"    Created {len(all_items)} QA items")
    
    # Split with stratification (temporal vs non-temporal)
    temporal = [i for i in all_items if i['metadata']['is_temporal']]
    non_temporal = [i for i in all_items if not i['metadata']['is_temporal']]
    
    t_dev, t_val, t_test = split_data(temporal)
    nt_dev, nt_val, nt_test = split_data(non_temporal)
    
    dev = t_dev + nt_dev
    val = t_val + nt_val
    test = t_test + nt_test
    random.shuffle(dev)
    random.shuffle(val)
    random.shuffle(test)
    
    output_dir = SPLITS_DIR / "locomo"
    save_split(dev, output_dir, "dev")
    save_split(val, output_dir, "validation")
    save_split(test, output_dir, "test")
    
    stats['locomo'] = {
        'total': len(all_items),
        'temporal': len(temporal),
        'non_temporal': len(non_temporal),
        'dev': len(dev), 'val': len(val), 'test': len(test)
    }
    print(f"    Done: {len(temporal)} temporal, {len(non_temporal)} non-temporal")


# ============================================================
# 2. FIX TIMEBENCH
# ============================================================
def fix_timebench():
    """Filter valid items, fix answer format, build messages from context"""
    print("\n[2/6] Fixing TimeBench...")
    
    tb_dir = RAW_DIR / "timebench_repo" / "TimeBench-subset-7553"
    if not tb_dir.exists():
        print("    SKIP: TimeBench raw data not found")
        return
    
    all_items = []
    
    # Process each subdataset
    for subdir in sorted(tb_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        sub_name = subdir.name
        
        for jsonl_file in sorted(subdir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    question = raw.get('question', '')
                    answer = raw.get('answer', '')
                    context = raw.get('context', '')
                    
                    # Fix answer format - convert list to string
                    if isinstance(answer, list):
                        answer = ', '.join(str(a) for a in answer)
                    else:
                        answer = str(answer)
                    
                    # Skip empty items
                    if not question or not answer:
                        continue
                    
                    # Build messages from context
                    messages = []
                    if context:
                        messages.append({
                            "role": "user",
                            "content": f"Context: {context}",
                            "turn": 0
                        })
                    
                    item = {
                        "conversation_id": f"timebench_{sub_name}_{line_idx}",
                        "dataset": "timebench",
                        "messages": messages,
                        "query": question,
                        "ground_truth": {
                            "answer": answer,
                            "context": context,
                            "updates": []
                        },
                        "metadata": {
                            "category": sub_name,
                            "subcategory": jsonl_file.stem,
                            "is_temporal": True,
                            "has_conflict": False  # TimeBench tests reasoning, not conflict resolution
                        }
                    }
                    all_items.append(item)
    
    print(f"    Created {len(all_items)} valid items (from raw)")
    
    dev, val, test = split_data(all_items)
    
    output_dir = SPLITS_DIR / "timebench"
    save_split(dev, output_dir, "dev")
    save_split(val, output_dir, "validation")
    save_split(test, output_dir, "test")
    
    stats['timebench'] = {
        'total': len(all_items),
        'dev': len(dev), 'val': len(val), 'test': len(test)
    }
    print(f"    Done: {len(all_items)} items across subdatasets")


# ============================================================
# 3. FIX TEMPORAL MEMORY
# ============================================================
def fix_temporal_memory():
    """Re-adapt from raw data: conversations + content_time_qs"""
    print("\n[3/6] Fixing TemporalMemory...")
    
    raw_dir = RAW_DIR / "temporal_memory_repo"
    conv_dir = raw_dir / "ConversationData"
    test_dir = raw_dir / "TestData"
    
    if not conv_dir.exists():
        print("    SKIP: TemporalMemory raw data not found")
        return
    
    # Load all conversations
    conversations = {}
    for conv_file in sorted(conv_dir.glob("*.json")):
        file_id = conv_file.stem
        try:
            with open(conv_file) as f:
                conv_data = json.load(f)
            
            # Extract messages from sessions
            messages = []
            speaker_a = conv_data.get('speaker_a', 'User')
            speaker_b = conv_data.get('speaker_b', 'Assistant')
            
            for i in range(1, 100):
                session_key = f'session_{i}'
                date_key = f'session_{i}_date_time'
                if session_key not in conv_data:
                    break
                session = conv_data[session_key]
                session_date = conv_data.get(date_key, '')
                if isinstance(session, list):
                    for turn in session:
                        speaker = turn.get('speaker', '')
                        text = turn.get('text', '')
                        role = 'user' if speaker == speaker_a else 'assistant'
                        messages.append({
                            'role': role,
                            'content': text,
                            'session_date': session_date,
                            'session_idx': i,
                            'turn': len(messages)
                        })
            
            conversations[file_id] = {
                'messages': messages,
                'speaker_a': speaker_a,
                'speaker_b': speaker_b
            }
        except Exception as e:
            print(f"    Warning: Failed to load {conv_file}: {e}")
    
    print(f"    Loaded {len(conversations)} conversations")
    
    # Load content-based temporal questions (most testable)
    all_items = []
    
    ctq_file = test_dir / "content_time_qs" / "content_time_qs.json"
    if ctq_file.exists():
        with open(ctq_file) as f:
            ctq_data = json.load(f)
        
        file_indexes = ctq_data.get('file_indexes', [])
        
        for file_id in file_indexes:
            file_key = f'file_{file_id}'
            if file_key not in ctq_data:
                continue
            
            conv_data = conversations.get(str(file_id))
            if not conv_data:
                continue
            
            all_messages = conv_data['messages']
            questions = ctq_data[file_key]
            
            for q_idx, q_group in enumerate(questions):
                q_variants = q_group.get('questions', [])
                relevant_docs = q_group.get('relevant_docs', [])
                
                if not q_variants or not relevant_docs:
                    continue
                
                # Use first question variant
                question = q_variants[0]
                
                # Extract answer from relevant session messages
                # relevant_docs are message indices (0-based into the conversation)
                answer_parts = []
                for doc_idx in relevant_docs:
                    if 0 <= doc_idx < len(all_messages):
                        answer_parts.append(all_messages[doc_idx]['content'])
                
                if not answer_parts:
                    continue
                
                # The answer is the content from the relevant messages
                answer = ' '.join(answer_parts[:3])  # First 3 relevant messages
                
                # Get context messages (last MAX_MESSAGES or around relevant docs)
                context_msgs = all_messages[-MAX_MESSAGES:]
                
                item = {
                    "conversation_id": f"temporal_memory_{file_id}_{q_idx}",
                    "dataset": "temporal_memory",
                    "messages": context_msgs,
                    "query": question,
                    "ground_truth": {
                        "answer": answer,
                        "relevant_docs": relevant_docs,
                        "updates": []
                    },
                    "metadata": {
                        "question_type": "content_time",
                        "is_temporal": True,
                        "has_conflict": False,  # These test temporal retrieval, not conflict
                        "file_id": file_id,
                        "num_messages": len(context_msgs)
                    }
                }
                all_items.append(item)
    
    # Also load time-based questions
    for tq_filename in ['test_dates.json', 'test_session.json', 'test_rel_session.json']:
        tq_file = test_dir / "time_qs" / tq_filename
        if not tq_file.exists():
            continue
        
        with open(tq_file) as f:
            tq_data = json.load(f)
        
        file_indexes = tq_data.get('file_indexes', [])
        
        for file_id in file_indexes:
            file_key = f'file_{file_id}'
            if file_key not in tq_data:
                continue
            
            conv_data = conversations.get(str(file_id))
            if not conv_data:
                continue
            
            all_messages = conv_data['messages']
            questions = tq_data[file_key]
            
            for q_idx, q_group in enumerate(questions):
                q_variants = q_group.get('questions', [])
                relevant_docs = q_group.get('relevant_docs', [])
                
                if not q_variants or not isinstance(q_variants[0], str):
                    continue
                
                question = q_variants[0]
                
                # Extract answer from relevant session messages
                answer_parts = []
                for doc_idx in relevant_docs[:5]:
                    if 0 <= doc_idx < len(all_messages):
                        answer_parts.append(all_messages[doc_idx]['content'])
                
                if not answer_parts:
                    continue
                
                answer = ' '.join(answer_parts[:3])
                context_msgs = all_messages[-MAX_MESSAGES:]
                
                q_type = tq_filename.replace('test_', '').replace('.json', '')
                
                item = {
                    "conversation_id": f"temporal_memory_{file_id}_{q_type}_{q_idx}",
                    "dataset": "temporal_memory",
                    "messages": context_msgs,
                    "query": question,
                    "ground_truth": {
                        "answer": answer,
                        "relevant_docs": relevant_docs,
                        "updates": []
                    },
                    "metadata": {
                        "question_type": q_type,
                        "is_temporal": True,
                        "has_conflict": False,
                        "file_id": file_id,
                        "num_messages": len(context_msgs)
                    }
                }
                all_items.append(item)
    
    print(f"    Created {len(all_items)} items")
    
    if all_items:
        dev, val, test = split_data(all_items)
        
        output_dir = SPLITS_DIR / "temporal_memory"
        save_split(dev, output_dir, "dev")
        save_split(val, output_dir, "validation")
        save_split(test, output_dir, "test")
        
        stats['temporal_memory'] = {
            'total': len(all_items),
            'dev': len(dev), 'val': len(val), 'test': len(test)
        }
    else:
        print("    WARNING: No items created")


# ============================================================
# 4. FIX PERSONAMEM
# ============================================================
def fix_personamem():
    """Fix query format (dict → string) and ensure proper structure"""
    print("\n[4/6] Fixing PersonaMem...")
    
    pm_dir = SPLITS_DIR / "personamem"
    if not pm_dir.exists():
        print("    SKIP: PersonaMem splits not found")
        return
    
    total_fixed = 0
    
    for split_name in ['dev', 'validation', 'test']:
        filepath = pm_dir / f"{split_name}.jsonl"
        if not filepath.exists():
            continue
        
        items = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                
                # Fix query: extract from dict if needed
                query = item.get('query', '')
                if isinstance(query, dict):
                    query = query.get('content', str(query))
                    item['query'] = query
                    total_fixed += 1
                
                # Fix empty query
                if not query and item.get('messages'):
                    # Use last user message as query
                    for msg in reversed(item['messages']):
                        if msg.get('role') == 'user':
                            item['query'] = msg.get('content', '')
                            break
                
                # Cap messages
                if len(item.get('messages', [])) > MAX_MESSAGES:
                    item['messages'] = item['messages'][-MAX_MESSAGES:]
                
                items.append(item)
        
        save_split(items, pm_dir, split_name)
    
    stats['personamem'] = {'fixed_queries': total_fixed}
    print(f"    Fixed {total_fixed} dict queries")


# ============================================================
# 5. FIX LONGMEMEVAL
# ============================================================
def fix_longmemeval():
    """Cap messages to prevent overload"""
    print("\n[5/6] Fixing LongMemEval...")
    
    lme_dir = SPLITS_DIR / "longmemeval"
    if not lme_dir.exists():
        print("    SKIP: LongMemEval splits not found")
        return
    
    total_capped = 0
    
    for split_name in ['dev', 'validation', 'test']:
        filepath = lme_dir / f"{split_name}.jsonl"
        if not filepath.exists():
            continue
        
        items = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                
                # Cap messages
                msgs = item.get('messages', [])
                if len(msgs) > MAX_MESSAGES:
                    item['messages'] = msgs[-MAX_MESSAGES:]
                    total_capped += 1
                
                items.append(item)
        
        save_split(items, lme_dir, split_name)
    
    stats['longmemeval'] = {'capped': total_capped}
    print(f"    Capped {total_capped} items to {MAX_MESSAGES} messages")


# ============================================================
# 6. FIX MEMORYAGENTBENCH
# ============================================================
def fix_memoryagentbench():
    """Cap messages to prevent overload"""
    print("\n[6/6] Fixing MemoryAgentBench...")
    
    mab_dir = SPLITS_DIR / "memoryagentbench"
    if not mab_dir.exists():
        print("    SKIP: MemoryAgentBench splits not found")
        return
    
    total_capped = 0
    
    for split_name in ['dev', 'validation', 'test']:
        filepath = mab_dir / f"{split_name}.jsonl"
        if not filepath.exists():
            continue
        
        items = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                
                # Cap messages - for MemoryAgentBench, the facts are in messages
                # Keep the FIRST batch (most relevant context) and LAST batch
                msgs = item.get('messages', [])
                if len(msgs) > MAX_MESSAGES:
                    # Keep first 20 + last 20
                    half = MAX_MESSAGES // 2
                    item['messages'] = msgs[:half] + msgs[-half:]
                    total_capped += 1
                
                items.append(item)
        
        save_split(items, mab_dir, split_name)
    
    stats['memoryagentbench'] = {'capped': total_capped}
    print(f"    Capped {total_capped} items to {MAX_MESSAGES} messages")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("FIX ALL DATASET SPLITS")
    print("=" * 70)
    
    fix_locomo()
    fix_timebench()
    fix_temporal_memory()
    fix_personamem()
    fix_longmemeval()
    fix_memoryagentbench()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ds, s in stats.items():
        print(f"  {ds}: {s}")
    
    # Save stats
    stats_file = SPLITS_DIR / "fix_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_file}")
    print("Done!")


if __name__ == '__main__':
    main()
