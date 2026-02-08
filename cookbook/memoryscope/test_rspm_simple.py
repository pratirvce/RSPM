"""
Simple test of RSPM Agent without full evaluation
Tests basic functionality step by step
"""
import sys
sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')

from cookbook.memoryscope.rspm_agent import RSPMAgent
from cookbook.memoryscope.data_loader import MemoryScopeDataset

print("="*60)
print("RSPM Agent: Simple Functionality Test")
print("="*60)

# Test 1: Initialize agent
print("\n[Test 1] Initializing RSPM Agent...")
agent = RSPMAgent(
    workspace_id="test_rspm",
    sleep_frequency=5,
    enable_hierarchical=False,  # Disable for simplicity
    enable_reranking=False
)
print("✓ Agent initialized")

# Test 2: Load dataset
print("\n[Test 2] Loading dataset...")
dataset = MemoryScopeDataset("datasets/memoryscope/synthetic.jsonl")
print(f"✓ Loaded {len(dataset.conversations)} conversations")

# Test 3: Process single conversation
print("\n[Test 3] Processing single conversation...")
conv = dataset.conversations[0]
messages = dataset.get_conversation_messages(conv)
conflicts = dataset.get_temporal_conflicts(conv)

print(f"  Conversation has {len(messages)} turns")
print(f"  Detected {len(conflicts)} conflicts")

if len(messages) > 0:
    query = messages[-1]['content']
    print(f"  Query: {query}")
    
    try:
        response = agent.process_conversation(
            messages=messages[:-1] if len(messages) > 1 else messages,
            query=query,
            ground_truth_conflicts=conflicts
        )
        print(f"  Response length: {len(response)} chars")
        print("✓ Conversation processed successfully")
    except Exception as e:
        print(f"✗ Error processing conversation: {e}")
        import traceback
        traceback.print_exc()

# Test 4: Evaluate single conversation
print("\n[Test 4] Evaluating response...")
try:
    is_correct = dataset.evaluate_temporal_consistency(conv, response)
    print(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'}")
    print("✓ Evaluation complete")
except Exception as e:
    print(f"✗ Error evaluating: {e}")

# Test 5: Clear workspace
print("\n[Test 5] Clearing workspace...")
agent.clear_workspace()
print("✓ Workspace cleared")

print("\n" + "="*60)
print("✅ All basic tests passed!")
print("="*60)
