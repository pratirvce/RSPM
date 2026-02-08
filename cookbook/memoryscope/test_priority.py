"""
Test 3: Verify if score/priority affects retrieval order in ReMe
Run this to understand if high-priority rules will be retrieved first
"""
import requests
import sys

REME_URL = "http://localhost:8002"
workspace_id = "test_priority"

def test_priority_retrieval():
    """Test if score parameter affects retrieval order"""
    print("\n" + "="*60)
    print("Test 3: Score/Priority in Retrieval")
    print("="*60)
    
    # Step 1: Insert low-priority memory
    print("\n1. Inserting low-priority memory (score=0.3)...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": workspace_id,
            "trajectories": [{
                "messages": [{"role": "user", "content": "Low priority information about topics"}],
                "score": 0.3
            }]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to insert low priority: {response.status_code}")
        return False
    
    print("✓ Inserted low-priority memory")
    
    # Step 2: Insert high-priority rule
    print("\n2. Inserting high-priority rule (score=0.95)...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": workspace_id,
            "trajectories": [{
                "messages": [{"role": "system", "content": "[RULE: High priority rule about topics]"}],
                "score": 0.95
            }]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to insert high priority: {response.status_code}")
        return False
    
    print("✓ Inserted high-priority rule")
    
    # Step 3: Insert medium-priority memory
    print("\n3. Inserting medium-priority memory (score=0.6)...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": workspace_id,
            "trajectories": [{
                "messages": [{"role": "user", "content": "Medium priority data about topics"}],
                "score": 0.6
            }]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to insert medium priority: {response.status_code}")
        return False
    
    print("✓ Inserted medium-priority memory")
    
    # Step 4: Retrieve all with a query
    print("\n4. Retrieving memories with query 'topics'...")
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={"workspace_id": workspace_id, "query": "topics", "top_k": 10}
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to retrieve: {response.status_code}")
        return False
    
    memories = response.json().get('metadata', {}).get('memory_list', [])
    
    # Step 5: Display retrieval order
    print("\n" + "="*60)
    print("RETRIEVAL ORDER:")
    print("="*60)
    
    if len(memories) == 0:
        print("⚠️  No memories retrieved!")
        return False
    
    for idx, mem in enumerate(memories):
        score = mem.get('score', 'N/A')
        content = mem['content'][:60]
        print(f"{idx+1}. Score: {score} | {content}")
    
    # Step 6: Analyze results
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    
    # Check if high-priority rule comes first
    first_content = memories[0]['content']
    is_rule_first = '[RULE:' in first_content
    
    print(f"\nFirst retrieved: {first_content[:50]}")
    print(f"Is high-priority rule first: {is_rule_first}")
    
    if is_rule_first:
        print("\n✅ HIGH-PRIORITY RULES WORK!")
        print("   Score affects retrieval order.")
        print("   Rules with score=0.95 will be retrieved first.")
        print("\n   RECOMMENDATION: Use score=0.95 for RSPM rules")
        return True
    else:
        print("\n⚠️  PRIORITY DOESN'T AFFECT ORDER")
        print("   Score does NOT affect retrieval order.")
        print("   ReMe retrieves by semantic similarity only.")
        print("\n   RECOMMENDATION: Use alternative approach:")
        print("   1. Tag rules in content with [RULE: prefix")
        print("   2. Query for rules separately")
        print("   3. Prepend rules to context manually")
        return False

def cleanup():
    """Cleanup test workspace"""
    requests.post(
        f"{REME_URL}/vector_store",
        json={"workspace_id": workspace_id, "action": "delete"}
    )

if __name__ == "__main__":
    try:
        # Test connection first
        print("Testing ReMe connection...")
        response = requests.post(
            f"{REME_URL}/vector_store",
            json={"action": "list"},
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"✗ ReMe server not responding: {response.status_code}")
            sys.exit(1)
        
        print("✓ ReMe server is running\n")
        
        # Run test
        test_priority_retrieval()
        
        # Cleanup
        print("\nCleaning up test workspace...")
        cleanup()
        print("✓ Cleanup complete")
        
        sys.exit(0)
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to ReMe server")
        print("\nStart the server with:")
        print("  reme backend=http http.port=8002")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
