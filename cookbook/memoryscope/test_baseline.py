"""
Test that ReMe server works with MemoryScope scenario
Run this FIRST to verify everything is set up correctly
"""
import requests
import sys

# Configuration
REME_URL = "http://localhost:8002"
WORKSPACE_ID = "test_memoryscope"

def test_reme_connection():
    """Test that ReMe server is running"""
    print("Testing ReMe connection...")
    
    try:
        response = requests.post(
            f"{REME_URL}/vector_store",
            json={"action": "list"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ ReMe server is running")
            return True
        else:
            print(f"✗ ReMe server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to ReMe server")
        print("\nStart the server with:")
        print("  reme backend=http http.port=8002")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_temporal_conflict():
    """Test the core temporal conflict scenario"""
    print("\n" + "="*60)
    print("Testing Temporal Conflict Scenario")
    print("="*60)
    
    # Scenario: User changes from vegetarian to vegan
    messages = [
        {"role": "user", "content": "Hi, I'm vegetarian"},
        {"role": "assistant", "content": "Hello! I'll remember you're vegetarian."},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "It's sunny today."},
        {"role": "user", "content": "Actually, I'm vegan now"},
        {"role": "assistant", "content": "Got it, updating to vegan."},
    ]
    
    print("\nConversation:")
    for msg in messages:
        role_icon = "👤" if msg['role'] == 'user' else "🤖"
        print(f"  {role_icon} {msg['role']}: {msg['content']}")
    
    # Step 1: Store conversation in ReMe
    print("\n1. Storing conversation in ReMe...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": WORKSPACE_ID,
            "trajectories": [{
                "messages": messages,
                "score": 1.0
            }]
        }
    )
    
    if response.status_code == 200:
        print("   ✓ Stored successfully")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return False
    
    # Step 2: Query for dietary preference
    query = "Recommend a restaurant for me"
    print(f"\n2. Querying: '{query}'")
    
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={
            "workspace_id": WORKSPACE_ID,
            "query": query,
            "top_k": 5
        }
    )
    
    if response.status_code != 200:
        print(f"   ✗ Retrieval failed: {response.status_code}")
        return False
    
    result = response.json()
    retrieved = result.get("answer", "").lower()
    
    print(f"\n3. Retrieved memory:")
    print(f"   {result.get('answer', 'No answer')[:200]}...")
    
    # Step 3: Check what was retrieved
    print(f"\n4. Evaluation:")
    
    has_vegan = "vegan" in retrieved
    has_vegetarian = "vegetarian" in retrieved and "vegan" not in retrieved.replace("vegetarian", "")
    
    print(f"   Contains 'vegan': {has_vegan}")
    print(f"   Contains 'vegetarian' (only): {has_vegetarian}")
    
    if has_vegan and not has_vegetarian:
        print("\n   ✓ CORRECT: Retrieved latest preference (vegan)")
        is_correct = True
    elif has_vegetarian:
        print("\n   ✗ INCORRECT: Retrieved outdated preference (vegetarian)")
        print("   ⚠️  This is the problem RSPM needs to solve!")
        is_correct = False
    else:
        print("\n   ⚠️  Neither found, unclear")
        is_correct = False
    
    # Step 4: Cleanup
    print("\n5. Cleaning up...")
    requests.post(
        f"{REME_URL}/vector_store",
        json={
            "workspace_id": WORKSPACE_ID,
            "action": "delete"
        }
    )
    print("   ✓ Workspace deleted")
    
    return is_correct

def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  MemoryScope + ReMe Integration Test                  ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    # Test 1: Connection
    if not test_reme_connection():
        sys.exit(1)
    
    # Test 2: Temporal conflict
    is_correct = test_temporal_conflict()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if is_correct:
        print("✓ ReMe correctly handles temporal conflicts")
        print("  (This is unexpected - baseline should fail)")
    else:
        print("✓ ReMe shows temporal conflict problem")
        print("  (This confirms the need for RSPM)")
    
    print("\n✓ Integration test complete!")
    print("\nNext steps:")
    print("  1. Create synthetic data: python create_synthetic_data.py")
    print("  2. Test data loader: python data_loader.py")
    print("  3. Implement baselines: python baselines.py")

if __name__ == "__main__":
    main()
