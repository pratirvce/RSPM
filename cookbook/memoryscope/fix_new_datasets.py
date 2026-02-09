"""
Adapt New Datasets for RSPM Evaluation
========================================
Converts 5 new datasets into the standard RSPM evaluation format:
  1. AToKe   - Temporal Knowledge Editing (AAAI 2024)
  2. MEMTRACK - Multi-Platform Memory State Tracking (NeurIPS 2025 SEA)
  3. DynaQuest - Dynamic QA with Knowledge Changes (ACL 2025)
  4. FiFA-Synth - Synthesized Selective Forgetting benchmark
  5. ReviseQA-Synth - Synthesized Belief Revision benchmark

Standard format per item:
  {
    "conversation_id": str,
    "dataset": str,
    "messages": [{"role": str, "content": str, "turn": int}],
    "query": str,
    "ground_truth": {"answer": str, "all_answers": list, "updates": list, ...},
    "metadata": {"is_temporal": bool, "has_conflict": bool, ...}
  }

Usage:
  python -u cookbook/memoryscope/fix_new_datasets.py
"""
import json
import os
import re
import random
import yaml
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
# 1. AToKe - Temporal Knowledge Editing
# ============================================================
def adapt_atoke():
    """
    Adapt AToKe dataset (AAAI 2024) for RSPM evaluation.
    
    AToKe tests temporal knowledge editing: updating facts while preserving history.
    We create two types of QA items per case:
      - CURRENT queries: "What is X now?" -> new_answer (temporal consistency)
      - HISTORICAL queries: "What was X before?" -> old answer (history preservation)
    
    Messages simulate a conversation where facts are introduced then updated.
    """
    print("\n[1/5] Adapting AToKe (Temporal Knowledge Editing)...")
    
    atoke_dir = RAW_DIR / "atoke_repo" / "datasets"
    if not atoke_dir.exists():
        print("    SKIP: AToKe data not found")
        return
    
    all_items = []
    
    # Process AToKe-ME (Multiple Edits) - most complex, most relevant
    me_file = atoke_dir / "AToKe-ME.json"
    with open(me_file) as f:
        me_data = json.load(f)
    
    print(f"    Found {len(me_data)} AToKe-ME items")
    
    # Sample a manageable subset (8820 items is a lot)
    # Take every 4th item for a good spread
    sampled = me_data[::4]
    print(f"    Sampled {len(sampled)} items from AToKe-ME")
    
    for case_idx, case in enumerate(sampled):
        case_id = case.get('case_id', case_idx)
        rewrites = case.get('requested_rewrite', [])
        history_evals = case.get('history_evaluation', [])
        old_answers = case.get('answer', [])
        old_aliases = case.get('answer_alias', [])
        new_answers = case.get('new_answer', [])
        new_aliases = case.get('new_answer_alias', [])
        
        if not rewrites:
            continue
        
        subject = rewrites[0].get('subject', 'Unknown')
        
        # Build messages: simulate a conversation about the subject
        messages = []
        turn = 0
        
        # First message: introduce the subject with original facts
        for i, rewrite in enumerate(rewrites):
            time_true = rewrite.get('time_true', {})
            target_true = rewrite.get('target_true', {}).get('str', '')
            since = time_true.get('since', '')
            until = time_true.get('until', '')
            
            if target_true:
                msg = f"From {since} to {until}, {subject}'s association is {target_true}."
                messages.append({"role": "user", "content": msg, "turn": turn})
                turn += 1
        
        # Update messages: introduce the new facts
        for i, rewrite in enumerate(rewrites):
            time_new = rewrite.get('time_new', {})
            target_new = rewrite.get('target_new', {}).get('str', '')
            since = time_new.get('since', '')
            until = time_new.get('until', '')
            
            if target_new:
                msg = f"UPDATE: From {since} to {until}, {subject}'s association changed to {target_new}."
                messages.append({"role": "user", "content": msg, "turn": turn})
                turn += 1
        
        # Cap messages
        if len(messages) > MAX_MESSAGES:
            messages = messages[:MAX_MESSAGES//2] + messages[-MAX_MESSAGES//2:]
        
        # CURRENT query: ask about the latest state
        last_rewrite = rewrites[-1]
        eval_info = last_rewrite.get('evaluation', {})
        current_query = eval_info.get('questions', eval_info.get('completion', ''))
        
        if new_answers and current_query:
            # Current answer is the last new answer
            last_new = new_answers[-1] if isinstance(new_answers[-1], list) else [new_answers[-1]]
            current_answer = last_new[0] if last_new else ''
            
            # Collect all acceptable answers
            all_current = [str(a) for a in last_new]
            if new_aliases and len(new_aliases) > len(rewrites) - 1:
                last_alias_set = new_aliases[-1]
                if isinstance(last_alias_set, list):
                    all_current.extend([str(a) for a in last_alias_set])
            
            # Build updates list
            updates = []
            for i, rewrite in enumerate(rewrites):
                old_val = rewrite.get('target_true', {}).get('str', '')
                new_val = rewrite.get('target_new', {}).get('str', '')
                if old_val and new_val:
                    updates.append({
                        'from': old_val,
                        'to': new_val,
                        'field': 'association',
                        'subject': subject,
                        'time_from': rewrite.get('time_true', {}),
                        'time_to': rewrite.get('time_new', {})
                    })
            
            item = {
                'conversation_id': f'atoke_me_current_{case_id}',
                'dataset': 'atoke',
                'messages': messages,
                'query': current_query,
                'ground_truth': {
                    'answer': str(current_answer),
                    'all_answers': all_current,
                    'updates': updates,
                    'query_type': 'current'
                },
                'metadata': {
                    'is_temporal': True,
                    'has_conflict': len(updates) > 0,
                    'case_id': case_id,
                    'subject': subject,
                    'num_edits': len(rewrites)
                }
            }
            all_items.append(item)
        
        # HISTORICAL query: ask about previous state (if history_evaluation exists)
        if history_evals and old_answers:
            last_hist = history_evals[-1] if history_evals else {}
            hist_query = last_hist.get('questions', last_hist.get('time_questions', ''))
            
            if hist_query and old_answers:
                # Historical answer
                last_old = old_answers[-1] if isinstance(old_answers[-1], list) else [old_answers[-1]]
                hist_answer = last_old[0] if last_old else ''
                all_hist = [str(a) for a in last_old]
                
                if old_aliases and len(old_aliases) > len(rewrites) - 1:
                    last_old_alias = old_aliases[-1]
                    if isinstance(last_old_alias, list):
                        all_hist.extend([str(a) for a in last_old_alias])
                
                hist_item = {
                    'conversation_id': f'atoke_me_history_{case_id}',
                    'dataset': 'atoke',
                    'messages': messages,
                    'query': hist_query,
                    'ground_truth': {
                        'answer': str(hist_answer),
                        'all_answers': all_hist,
                        'updates': [],
                        'query_type': 'historical'
                    },
                    'metadata': {
                        'is_temporal': True,
                        'has_conflict': False,  # Historical queries don't have temporal conflicts
                        'case_id': case_id,
                        'subject': subject,
                        'num_edits': len(rewrites)
                    }
                }
                all_items.append(hist_item)
    
    # Also process AToKe-SE (Single Edit) for simpler temporal consistency tests
    se_file = atoke_dir / "AToKe-SE.json"
    with open(se_file) as f:
        se_data = json.load(f)
    
    print(f"    Found {len(se_data)} AToKe-SE items")
    sampled_se = se_data[::8]  # Take every 8th item
    print(f"    Sampled {len(sampled_se)} items from AToKe-SE")
    
    for case_idx, case in enumerate(sampled_se):
        case_id = case.get('case_id', case_idx)
        raw_rw = case.get('requested_rewrite', {})
        new_answers = case.get('new_answer', [])
        new_aliases = case.get('new_answer_alias', [])
        
        # SE has a single dict rewrite, not a list
        if isinstance(raw_rw, dict):
            rewrite = raw_rw
        elif isinstance(raw_rw, list) and raw_rw:
            rewrite = raw_rw[0]
        else:
            continue
        
        subject = rewrite.get('subject', 'Unknown')
        old_val = rewrite.get('target_true', {}).get('str', '')
        new_val = rewrite.get('target_new', {}).get('str', '')
        time_true = rewrite.get('time_true', {})
        time_new = rewrite.get('time_new', {})
        eval_info = rewrite.get('evaluation', {})
        
        # Messages
        messages = [
            {"role": "user", "content": f"From {time_true.get('since','')} to {time_true.get('until','')}, {subject}'s association is {old_val}.", "turn": 0},
            {"role": "user", "content": f"UPDATE: From {time_new.get('since','')} to {time_new.get('until','')}, {subject}'s association changed to {new_val}.", "turn": 1}
        ]
        
        query = eval_info.get('questions', eval_info.get('completion', ''))
        if not query:
            continue
        
        answer = new_val
        all_ans = [new_val]
        if new_answers:
            # new_answer can be [[list]] for ME or [list] for SE
            first = new_answers[0] if isinstance(new_answers, list) and new_answers else new_answers
            if isinstance(first, list):
                all_ans = [str(a) for a in first if a]
            elif first:
                all_ans = [str(first)]
            if new_aliases:
                first_alias = new_aliases[0] if isinstance(new_aliases, list) and new_aliases else new_aliases
                if isinstance(first_alias, list):
                    all_ans.extend([str(a) for a in first_alias if a])
                elif first_alias:
                    all_ans.append(str(first_alias))
        
        item = {
            'conversation_id': f'atoke_se_{case_id}',
            'dataset': 'atoke',
            'messages': messages,
            'query': query,
            'ground_truth': {
                'answer': str(answer),
                'all_answers': all_ans,
                'updates': [{'from': old_val, 'to': new_val, 'field': 'association', 'subject': subject}],
                'query_type': 'current_single_edit'
            },
            'metadata': {
                'is_temporal': True,
                'has_conflict': True,
                'case_id': case_id,
                'subject': subject,
                'num_edits': 1
            }
        }
        all_items.append(item)
    
    print(f"    Total AToKe items: {len(all_items)}")
    
    # Split and save
    dev, val, test = split_data(all_items)
    out_dir = SPLITS_DIR / "atoke"
    save_split(dev, out_dir, "dev")
    save_split(val, out_dir, "validation")
    save_split(test, out_dir, "test")
    
    stats['atoke'] = {'total': len(all_items), 'dev': len(dev), 'val': len(val), 'test': len(test)}


# ============================================================
# 2. MEMTRACK - Multi-Platform Memory State Tracking
# ============================================================
def adapt_memtrack():
    """
    Adapt MEMTRACK dataset (Patronus AI, NeurIPS 2025 SEA Workshop).
    
    MEMTRACK tests memory across Slack/Linear/Git event timelines with
    conflicting, cross-referencing information.
    
    Each config has an event_history (events as messages) and questions/answers.
    """
    print("\n[2/5] Adapting MEMTRACK (Multi-Platform State Tracking)...")
    
    memtrack_dir = RAW_DIR / "memtrack" / "Memtrak"
    configs_dir = memtrack_dir / "test_configs"
    events_dir = memtrack_dir / "test_event_histories"
    
    if not configs_dir.exists():
        print("    SKIP: MEMTRACK configs not found")
        return
    
    all_items = []
    
    # Load all configs
    config_files = sorted(configs_dir.glob("*.yaml"))
    print(f"    Found {len(config_files)} MEMTRACK configs")
    
    for cfg_file in config_files:
        with open(cfg_file) as f:
            config = yaml.safe_load(f)
        
        if not config:
            continue
        
        benchmark = config.get('benchmark', {})
        questions = benchmark.get('questions', [])
        answers = benchmark.get('expected_answers', [])
        event_history_name = benchmark.get('event_history', '')
        
        if not questions or not answers:
            continue
        
        # Load event history
        event_file = memtrack_dir / event_history_name
        if not event_file.exists():
            # Try matching by config name
            base_name = cfg_file.stem.replace('config_', 'event_history_')
            event_file = events_dir / f"{base_name}.json"
        
        if not event_file.exists():
            continue
        
        with open(event_file) as f:
            events = json.load(f)
        
        # Convert events to messages
        messages = []
        for i, event in enumerate(events):
            timestamp = event.get('timestamp', '')
            platform = event.get('platform', 'unknown')
            meta = event.get('generation_meta_data', {})
            
            # Build message content from event metadata
            parts = [f"[{platform.upper()}] [{timestamp}]"]
            
            if isinstance(meta, dict):
                title = meta.get('title', '')
                if title:
                    parts.append(f"Title: {title}")
                
                desc = meta.get('description', '')
                if desc:
                    parts.append(desc[:300])
                
                # Include status/assignee changes (often contain temporal conflicts)
                status = meta.get('status', '')
                if status:
                    parts.append(f"Status: {status}")
                
                assignee = meta.get('assignee', meta.get('lead', ''))
                if assignee:
                    parts.append(f"Assigned to: {assignee}")
                
                # Include comments/content
                content = meta.get('content', meta.get('message', meta.get('body', '')))
                if content:
                    parts.append(str(content)[:500])
                
                # Include changes
                changes = meta.get('changes', meta.get('updates', []))
                if changes:
                    if isinstance(changes, list):
                        for change in changes[:3]:
                            if isinstance(change, dict):
                                parts.append(f"Change: {json.dumps(change)[:200]}")
                            else:
                                parts.append(f"Change: {str(change)[:200]}")
                    elif isinstance(changes, dict):
                        parts.append(f"Changes: {json.dumps(changes)[:200]}")
            
            content_str = " | ".join(parts)
            messages.append({
                "role": "user",
                "content": content_str,
                "turn": i
            })
        
        # Cap messages
        if len(messages) > MAX_MESSAGES:
            half = MAX_MESSAGES // 2
            messages = messages[:half] + messages[-half:]
        
        cfg_name = cfg_file.stem
        
        # Create one item per question
        for q_idx, (question, answer) in enumerate(zip(questions, answers)):
            item = {
                'conversation_id': f'memtrack_{cfg_name}_{q_idx}',
                'dataset': 'memtrack',
                'messages': messages,
                'query': question,
                'ground_truth': {
                    'answer': str(answer),
                    'all_answers': [str(answer)],
                    'updates': [],
                    'query_type': 'state_tracking'
                },
                'metadata': {
                    'is_temporal': True,
                    'has_conflict': True,  # MEMTRACK items involve conflicting/evolving state
                    'config': cfg_name,
                    'num_events': len(events),
                    'num_messages': len(messages)
                }
            }
            all_items.append(item)
    
    print(f"    Total MEMTRACK items: {len(all_items)}")
    
    dev, val, test = split_data(all_items)
    out_dir = SPLITS_DIR / "memtrack"
    save_split(dev, out_dir, "dev")
    save_split(val, out_dir, "validation")
    save_split(test, out_dir, "test")
    
    stats['memtrack'] = {'total': len(all_items), 'dev': len(dev), 'val': len(val), 'test': len(test)}


# ============================================================
# 3. DynaQuest - Dynamic QA with Knowledge Changes
# ============================================================
def adapt_dynaquest():
    """
    Adapt DynaQuest dataset (ACL 2025 Findings).
    
    DynaQuest tests time-sensitive QA from Wikipedia changes.
    Each item has: question, answer (current), answer_old (previous),
    and paragraphs (evidence context).
    """
    print("\n[3/5] Adapting DynaQuest (Dynamic QA)...")
    
    sample_file = RAW_DIR / "dynaquest_repo" / "CARL" / "data" / "sample.jsonl"
    if not sample_file.exists():
        print("    SKIP: DynaQuest data not found")
        return
    
    with open(sample_file) as f:
        raw_items = [json.loads(l) for l in f if l.strip()]
    
    print(f"    Found {len(raw_items)} DynaQuest items")
    
    all_items = []
    
    for idx, raw in enumerate(raw_items):
        question = raw.get('question', '')
        answer = raw.get('answer', '')
        answer_old = raw.get('answer_old', '')
        paragraphs = raw.get('paragraphs', [])
        ans_in_context = raw.get('ans_in_context', True)
        
        if not question or not answer:
            continue
        
        # Build messages from paragraphs
        messages = []
        turn = 0
        for para in paragraphs:
            title = para.get('title', '')
            text = para.get('text', '')
            if text:
                content = f"{title}: {text}" if title else text
                messages.append({"role": "user", "content": content[:500], "turn": turn})
                turn += 1
        
        # Cap messages
        if len(messages) > MAX_MESSAGES:
            half = MAX_MESSAGES // 2
            messages = messages[:half] + messages[-half:]
        
        # Determine if there's a temporal conflict
        has_conflict = answer != answer_old and answer_old
        
        updates = []
        if has_conflict:
            updates.append({
                'from': answer_old,
                'to': answer,
                'field': 'factual_knowledge'
            })
        
        item = {
            'conversation_id': f'dynaquest_{idx}',
            'dataset': 'dynaquest',
            'messages': messages,
            'query': question,
            'ground_truth': {
                'answer': str(answer),
                'all_answers': [str(answer)],
                'answer_old': str(answer_old),
                'updates': updates,
                'query_type': 'temporal_qa',
                'ans_in_context': ans_in_context
            },
            'metadata': {
                'is_temporal': True,
                'has_conflict': has_conflict,
                'ans_in_context': ans_in_context,
                'num_paragraphs': len(paragraphs)
            }
        }
        all_items.append(item)
    
    print(f"    Total DynaQuest items: {len(all_items)}")
    
    dev, val, test = split_data(all_items)
    out_dir = SPLITS_DIR / "dynaquest"
    save_split(dev, out_dir, "dev")
    save_split(val, out_dir, "validation")
    save_split(test, out_dir, "test")
    
    stats['dynaquest'] = {'total': len(all_items), 'dev': len(dev), 'val': len(val), 'test': len(test)}


# ============================================================
# 4. FiFA-Synth - Synthesized Selective Forgetting Benchmark
# ============================================================
def synthesize_fifa():
    """
    Synthesize a selective forgetting benchmark inspired by FiFA/MaRS.
    
    Tests whether the memory agent correctly:
      - Remembers important facts (retention test)
      - Forgets irrelevant/expired facts (forgetting test)
      - Handles privacy-sensitive info (privacy test)
      - Resolves conflicts between old and new facts (conflict test)
    
    Each scenario has a sequence of messages, some of which become obsolete.
    """
    print("\n[4/5] Synthesizing FiFA-style (Selective Forgetting)...")
    
    # Scenario templates for forgetting tests
    scenarios = [
        # Retention tests - should remember
        {
            'topic': 'medical',
            'messages': [
                "Patient has a severe allergy to penicillin.",
                "Patient was diagnosed with Type 2 diabetes in 2020.",
                "Patient currently takes metformin 500mg twice daily.",
                "Patient had a routine checkup last week, everything normal.",
                "Patient mentioned they enjoy gardening as a hobby.",
            ],
            'query': "What medication is the patient currently taking?",
            'answer': "metformin 500mg twice daily",
            'forget_indices': [3, 4],  # Routine checkup and hobby can be forgotten
            'has_conflict': False,
            'category': 'retention'
        },
        {
            'topic': 'work',
            'messages': [
                "Alice joined the engineering team in January 2024.",
                "Alice was promoted to Senior Engineer in June 2024.",
                "The team had a pizza party on Friday.",
                "Alice is leading the authentication module redesign.",
                "Bob mentioned he likes Alice's presentation style.",
            ],
            'query': "What is Alice's current role?",
            'answer': "Senior Engineer",
            'forget_indices': [2, 4],
            'has_conflict': False,
            'category': 'retention'
        },
        # Forgetting tests - should NOT recall forgotten info
        {
            'topic': 'preferences',
            'messages': [
                "User prefers dark mode for all applications.",
                "User's favorite restaurant is 'Bella Italia'.",
                "User changed their favorite restaurant to 'Tokyo Garden'.",
                "User mentioned they dislike spam emails.",
                "User prefers notifications via email over SMS.",
            ],
            'query': "What is the user's favorite restaurant?",
            'answer': "Tokyo Garden",
            'forget_indices': [1],  # Old restaurant preference should be forgotten
            'has_conflict': True,
            'category': 'forgetting'
        },
        {
            'topic': 'address',
            'messages': [
                "User lives at 123 Oak Street, Springfield.",
                "User recently moved to 456 Maple Avenue, Portland.",
                "User's phone number is 555-0123.",
                "User mentioned they have two cats.",
                "User works at TechCorp as a data analyst.",
            ],
            'query': "Where does the user currently live?",
            'answer': "456 Maple Avenue, Portland",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'forgetting'
        },
        # Privacy tests - should forget sensitive info when asked
        {
            'topic': 'privacy',
            'messages': [
                "User's credit card number is 4532-XXXX-XXXX-7890.",
                "User requested to delete their payment information.",
                "User prefers to pay with PayPal going forward.",
                "User's email is user@example.com.",
                "User's shipping address is 789 Pine Road.",
            ],
            'query': "What is the user's preferred payment method?",
            'answer': "PayPal",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'privacy'
        },
        {
            'topic': 'access',
            'messages': [
                "User was granted admin access on January 1st.",
                "User's admin access was revoked on March 15th due to role change.",
                "User now has read-only access to the system.",
                "User requested a password reset last week.",
                "User logged in from a new device yesterday.",
            ],
            'query': "What level of access does the user currently have?",
            'answer': "read-only access",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'privacy'
        },
        # Conflict resolution tests
        {
            'topic': 'project',
            'messages': [
                "Project deadline is December 15, 2025.",
                "Team size is 5 engineers.",
                "Project deadline extended to January 30, 2026.",
                "Budget approved at $50,000.",
                "Two new engineers joined, team size is now 7.",
            ],
            'query': "What is the project deadline?",
            'answer': "January 30, 2026",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'conflict_resolution'
        },
        {
            'topic': 'meeting',
            'messages': [
                "Weekly standup is at 9 AM every Monday.",
                "Design review meeting scheduled for Thursday at 2 PM.",
                "Weekly standup moved to 10 AM every Monday starting next week.",
                "Sprint planning is on the first Monday of each month.",
                "The design review for this week is cancelled.",
            ],
            'query': "When is the weekly standup?",
            'answer': "10 AM every Monday",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'conflict_resolution'
        },
        {
            'topic': 'config',
            'messages': [
                "System configuration: max_retries = 3.",
                "Database timeout set to 30 seconds.",
                "System configuration updated: max_retries = 5.",
                "Logging level set to DEBUG for troubleshooting.",
                "Cache TTL is 300 seconds.",
            ],
            'query': "What is the max_retries configuration?",
            'answer': "5",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'conflict_resolution'
        },
        {
            'topic': 'inventory',
            'messages': [
                "Product A has 150 units in stock.",
                "Product B has 200 units in stock.",
                "Product A received a shipment, now has 300 units.",
                "Product C was discontinued last month.",
                "Warehouse B is at 80% capacity.",
            ],
            'query': "How many units of Product A are in stock?",
            'answer': "300",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'conflict_resolution'
        },
        # Multi-hop retention
        {
            'topic': 'travel',
            'messages': [
                "User booked a flight to Paris for December 20.",
                "Hotel reservation at Hotel Lumiere for Dec 20-25.",
                "Flight was cancelled, rebooked for December 22.",
                "Hotel reservation updated to Dec 22-27.",
                "User added a dinner reservation at Le Petit Bistro for Dec 23.",
            ],
            'query': "When does the user's hotel reservation end?",
            'answer': "December 27",
            'forget_indices': [0, 1],
            'has_conflict': True,
            'category': 'conflict_resolution'
        },
        {
            'topic': 'school',
            'messages': [
                "Student enrolled in Math 101, Physics 101, and English 101.",
                "Student dropped Physics 101 and added Chemistry 101.",
                "Student's GPA is 3.5.",
                "Study group meets Tuesdays at 4 PM.",
                "Final exams start on May 15.",
            ],
            'query': "What courses is the student currently enrolled in?",
            'answer': "Math 101, Chemistry 101, and English 101",
            'forget_indices': [0],
            'has_conflict': True,
            'category': 'forgetting'
        },
    ]
    
    # Generate multiple variations per scenario template
    all_items = []
    
    for s_idx, scenario in enumerate(scenarios):
        messages = []
        for t_idx, msg in enumerate(scenario['messages']):
            messages.append({"role": "user", "content": msg, "turn": t_idx})
        
        updates = []
        if scenario['has_conflict']:
            # The forget indices point to outdated messages
            for fi in scenario['forget_indices']:
                if fi < len(scenario['messages']):
                    updates.append({
                        'from': scenario['messages'][fi],
                        'to': 'updated/replaced by later message',
                        'field': scenario['topic']
                    })
        
        item = {
            'conversation_id': f'fifa_synth_{s_idx}',
            'dataset': 'fifa_synth',
            'messages': messages,
            'query': scenario['query'],
            'ground_truth': {
                'answer': scenario['answer'],
                'all_answers': [scenario['answer']],
                'updates': updates,
                'query_type': scenario['category']
            },
            'metadata': {
                'is_temporal': True,
                'has_conflict': scenario['has_conflict'],
                'category': scenario['category'],
                'topic': scenario['topic'],
                'forget_indices': scenario['forget_indices']
            }
        }
        all_items.append(item)
    
    # Generate additional variations through augmentation
    augmented = []
    name_swaps = [
        ('Alice', 'Carol'), ('Bob', 'David'), ('Springfield', 'Oakland'),
        ('Portland', 'Seattle'), ('Paris', 'London'), ('Tokyo', 'Seoul'),
    ]
    
    for item in all_items:
        aug_item = json.loads(json.dumps(item))
        aug_item['conversation_id'] = item['conversation_id'] + '_aug'
        
        # Swap some names
        for old_name, new_name in random.sample(name_swaps, min(2, len(name_swaps))):
            for msg in aug_item['messages']:
                msg['content'] = msg['content'].replace(old_name, new_name)
            aug_item['query'] = aug_item['query'].replace(old_name, new_name)
            aug_item['ground_truth']['answer'] = aug_item['ground_truth']['answer'].replace(old_name, new_name)
            aug_item['ground_truth']['all_answers'] = [
                a.replace(old_name, new_name) for a in aug_item['ground_truth']['all_answers']
            ]
        
        augmented.append(aug_item)
    
    all_items.extend(augmented)
    
    print(f"    Total FiFA-synth items: {len(all_items)}")
    
    dev, val, test = split_data(all_items)
    out_dir = SPLITS_DIR / "fifa_synth"
    save_split(dev, out_dir, "dev")
    save_split(val, out_dir, "validation")
    save_split(test, out_dir, "test")
    
    stats['fifa_synth'] = {'total': len(all_items), 'dev': len(dev), 'val': len(val), 'test': len(test)}


# ============================================================
# 5. ReviseQA-Synth - Synthesized Belief Revision Benchmark
# ============================================================
def synthesize_reviseqa():
    """
    Synthesize a belief revision benchmark inspired by ReviseQA (ICML 2025).
    
    Tests whether the agent can update its beliefs when information changes:
      - Facts are introduced
      - Some facts are retracted or contradicted  
      - Agent must answer based on the CURRENT state of knowledge
    """
    print("\n[5/5] Synthesizing ReviseQA-style (Belief Revision)...")
    
    # Multi-turn belief revision scenarios
    scenarios = [
        # Simple fact revision
        {
            'turns': [
                ("The capital of Country X is Oldtown.", None),
                ("Actually, Country X moved its capital to Newville in 2023.", None),
            ],
            'query': "What is the current capital of Country X?",
            'answer': "Newville",
            'has_conflict': True,
            'category': 'simple_revision'
        },
        {
            'turns': [
                ("The company CEO is John Smith.", None),
                ("John Smith resigned. The new CEO is Sarah Johnson.", None),
            ],
            'query': "Who is the CEO of the company?",
            'answer': "Sarah Johnson",
            'has_conflict': True,
            'category': 'simple_revision'
        },
        {
            'turns': [
                ("The project uses Python 3.8.", None),
                ("We upgraded to Python 3.11 last month.", None),
            ],
            'query': "What version of Python does the project use?",
            'answer': "Python 3.11",
            'has_conflict': True,
            'category': 'simple_revision'
        },
        # Multi-step revision
        {
            'turns': [
                ("The team has 5 members: Alice, Bob, Carol, David, Eve.", None),
                ("David left the team.", None),
                ("Frank joined the team.", None),
            ],
            'query': "Who are the current team members?",
            'answer': "Alice, Bob, Carol, Eve, and Frank",
            'has_conflict': True,
            'category': 'multi_step'
        },
        {
            'turns': [
                ("Budget for Q1 is $10,000.", None),
                ("Budget increased to $15,000 due to new requirements.", None),
                ("After review, budget finalized at $12,000.", None),
            ],
            'query': "What is the finalized Q1 budget?",
            'answer': "$12,000",
            'has_conflict': True,
            'category': 'multi_step'
        },
        {
            'turns': [
                ("Meeting scheduled for Monday at 9 AM.", None),
                ("Meeting moved to Tuesday at 10 AM.", None),
                ("Final update: meeting is on Wednesday at 2 PM.", None),
            ],
            'query': "When is the meeting?",
            'answer': "Wednesday at 2 PM",
            'has_conflict': True,
            'category': 'multi_step'
        },
        # Partial revision (some facts change, others don't)
        {
            'turns': [
                ("The server runs Ubuntu 20.04 with 16GB RAM.", None),
                ("We upgraded the OS to Ubuntu 22.04. RAM remains the same.", None),
            ],
            'query': "What OS is the server running?",
            'answer': "Ubuntu 22.04",
            'has_conflict': True,
            'category': 'partial_revision'
        },
        {
            'turns': [
                ("The server runs Ubuntu 20.04 with 16GB RAM.", None),
                ("We upgraded the OS to Ubuntu 22.04. RAM remains the same.", None),
            ],
            'query': "How much RAM does the server have?",
            'answer': "16GB",
            'has_conflict': False,  # RAM didn't change
            'category': 'partial_revision'
        },
        # Retraction
        {
            'turns': [
                ("Report: The product launch is set for March 15.", None),
                ("The product launch has been postponed indefinitely.", None),
            ],
            'query': "When is the product launch?",
            'answer': "postponed indefinitely",
            'has_conflict': True,
            'category': 'retraction'
        },
        {
            'turns': [
                ("User reported a bug in the login module.", None),
                ("The bug report was a false alarm. The login module is working correctly.", None),
            ],
            'query': "Is there a bug in the login module?",
            'answer': "No, the login module is working correctly",
            'has_conflict': True,
            'category': 'retraction'
        },
        # Conditional revision
        {
            'turns': [
                ("Employee benefits include health insurance and 401k.", None),
                ("Starting January, dental insurance is also included.", None),
                ("The 401k match has been increased from 3% to 5%.", None),
            ],
            'query': "What is the 401k match percentage?",
            'answer': "5%",
            'has_conflict': True,
            'category': 'conditional'
        },
        {
            'turns': [
                ("The API rate limit is 100 requests per minute.", None),
                ("Premium users now have a rate limit of 500 requests per minute.", None),
                ("Free tier rate limit remains at 100 requests per minute.", None),
            ],
            'query': "What is the rate limit for premium users?",
            'answer': "500 requests per minute",
            'has_conflict': True,
            'category': 'conditional'
        },
        # Chain revision
        {
            'turns': [
                ("The primary database is MySQL.", None),
                ("We migrated from MySQL to PostgreSQL.", None),
                ("We added a Redis cache layer in front of PostgreSQL.", None),
            ],
            'query': "What is the primary database?",
            'answer': "PostgreSQL",
            'has_conflict': True,
            'category': 'chain'
        },
        {
            'turns': [
                ("Store hours are 9 AM to 5 PM.", None),
                ("Extended hours: 8 AM to 7 PM on weekdays.", None),
                ("Weekend hours added: 10 AM to 4 PM.", None),
            ],
            'query': "What are the weekday store hours?",
            'answer': "8 AM to 7 PM",
            'has_conflict': True,
            'category': 'chain'
        },
        # Numeric updates
        {
            'turns': [
                ("Current stock price is $45.20.", None),
                ("Stock price dropped to $42.15 after earnings report.", None),
                ("Stock recovered to $47.80 following positive analyst review.", None),
            ],
            'query': "What is the current stock price?",
            'answer': "$47.80",
            'has_conflict': True,
            'category': 'numeric_chain'
        },
        {
            'turns': [
                ("Population of the city: 250,000.", None),
                ("After the 2024 census, the population is 278,000.", None),
            ],
            'query': "What is the city's population?",
            'answer': "278,000",
            'has_conflict': True,
            'category': 'numeric_chain'
        },
    ]
    
    all_items = []
    
    for s_idx, scenario in enumerate(scenarios):
        messages = []
        updates = []
        
        for t_idx, (text, _) in enumerate(scenario['turns']):
            messages.append({"role": "user", "content": text, "turn": t_idx})
        
        # Build updates from consecutive turns
        if scenario['has_conflict']:
            for i in range(1, len(scenario['turns'])):
                updates.append({
                    'from': scenario['turns'][i-1][0],
                    'to': scenario['turns'][i][0],
                    'field': 'belief_revision'
                })
        
        item = {
            'conversation_id': f'reviseqa_synth_{s_idx}',
            'dataset': 'reviseqa_synth',
            'messages': messages,
            'query': scenario['query'],
            'ground_truth': {
                'answer': scenario['answer'],
                'all_answers': [scenario['answer']],
                'updates': updates,
                'query_type': scenario['category']
            },
            'metadata': {
                'is_temporal': True,
                'has_conflict': scenario['has_conflict'],
                'category': scenario['category'],
                'num_turns': len(scenario['turns'])
            }
        }
        all_items.append(item)
    
    # Generate variations
    augmented = []
    for item in all_items:
        # Variation 1: more context noise
        aug1 = json.loads(json.dumps(item))
        aug1['conversation_id'] = item['conversation_id'] + '_noisy'
        noise_msgs = [
            {"role": "user", "content": "The weather today is sunny with a high of 72F.", "turn": 50},
            {"role": "user", "content": "Don't forget the team lunch at noon.", "turn": 51},
            {"role": "user", "content": "New parking policy effective next month.", "turn": 52},
        ]
        aug1['messages'].extend(random.sample(noise_msgs, min(2, len(noise_msgs))))
        augmented.append(aug1)
        
        # Variation 2: different phrasing
        aug2 = json.loads(json.dumps(item))
        aug2['conversation_id'] = item['conversation_id'] + '_rephrase'
        for msg in aug2['messages']:
            msg['content'] = msg['content'].replace("Actually,", "Correction:").replace(
                "We upgraded", "The team upgraded").replace(
                "Starting", "Effective").replace(
                "has been", "was")
        augmented.append(aug2)
    
    all_items.extend(augmented)
    
    print(f"    Total ReviseQA-synth items: {len(all_items)}")
    
    dev, val, test = split_data(all_items)
    out_dir = SPLITS_DIR / "reviseqa_synth"
    save_split(dev, out_dir, "dev")
    save_split(val, out_dir, "validation")
    save_split(test, out_dir, "test")
    
    stats['reviseqa_synth'] = {'total': len(all_items), 'dev': len(dev), 'val': len(val), 'test': len(test)}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ADAPT NEW DATASETS FOR RSPM EVALUATION")
    print("=" * 70)
    
    adapt_atoke()
    adapt_memtrack()
    adapt_dynaquest()
    synthesize_fifa()
    synthesize_reviseqa()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ds, s in stats.items():
        print(f"  {ds}: {s}")
    
    stats_file = SPLITS_DIR / "new_dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_file}")
    print("Done!")


if __name__ == '__main__':
    main()
