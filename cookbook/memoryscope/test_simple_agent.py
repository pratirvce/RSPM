"""
Simple test to verify RSPM agent works with HaluMem data
"""
import sys
sys.path.insert(0, '/home/prevanka/prati/su-reme/ReMe')

from cookbook.memoryscope.data_loader import MemoryScopeDataset
from cookbook.memoryscope.rspm_agent import RSPMAgent

print("=== RSPM Agent Simple Test ===\n")

# Load dataset
print("1. Loading dataset...")
dataset = MemoryScopeDataset("datasets/memoryscope/halumem_medium.jsonl")
print(f"   Loaded {len(dataset)} conversations\n")

# Get first conversation
print("2. Getting first conversation...")
conv = dataset[0]
print(f"   Conversation ID: {conv.get('conversation_id', 'unknown')}")
print(f"   Messages: {len(conv.get('messages', []))}")
print(f"   QA pairs: {len(conv.get('qa_pairs', []))}")
print(f"   Has updates: {len(conv.get('ground_truth', {}).get('updates', []))}\n")

# Create agent
print("3. Creating RSPM agent...")
agent = RSPMAgent(
    workspace_id="test_simple",
    sleep_frequency=10,
    enable_hierarchical=True,
    enable_reranking=True,
    reme_url="http://localhost:8002"
)
print("   Agent created\n")

# Process conversation
print("4. Processing conversation...")
try:
    response, conflicts = agent.process_conversation(conv)
    print(f"   ✓ Response generated (length: {len(response)})")
    print(f"   ✓ Conflicts detected: {len(conflicts)}\n")
    
    # Evaluate
    print("5. Evaluating response...")
    ground_truth = conv.get('ground_truth', {})
    result = agent.evaluate_response(response, ground_truth)
    print(f"   ✓ Correct: {result['correct']}")
    print(f"   ✓ Has conflict: {result['has_conflict']}")
    print(f"   ✓ Used outdated: {result.get('used_outdated', False)}\n")
    
    print("=== SUCCESS! ===")
    print("All components working correctly.")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
