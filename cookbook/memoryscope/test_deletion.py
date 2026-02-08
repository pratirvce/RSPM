"""
Test 1: Verify ReMe's vector_store deletion works
Run this FIRST to verify basic deletion functionality
"""
import requests
import sys

REME_URL = "http://localhost:8002"
workspace_id = "test_deletion"

def test_workspace_deletion():
    """Test deleting entire workspace"""
    print("\n" + "="*60)
    print("Test 1: Workspace Deletion")
    print("="*60)
    
    # Step 1: Insert memories
    print("\n1. Inserting 2 memories...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": workspace_id,
            "trajectories": [{
                "messages": [
                    {"role": "user", "content": "Memory 1: Test data"},
                    {"role": "user", "content": "Memory 2: More test data"}
                ],
                "score": 1.0
            }]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to insert: {response.status_code}")
        return False
    
    print("✓ Inserted successfully")
    
    # Step 2: Retrieve (should find 2)
    print("\n2. Retrieving memories before deletion...")
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={"workspace_id": workspace_id, "query": "Memory", "top_k": 10}
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to retrieve: {response.status_code}")
        return False
    
    memory_list = response.json().get('metadata', {}).get('memory_list', [])
    count_before = len(memory_list)
    print(f"✓ Found {count_before} memories before deletion")
    
    if count_before == 0:
        print("⚠️  Warning: No memories found! Check if insertion worked.")
    
    # Step 3: Delete workspace using vector_store action
    print("\n3. Deleting workspace...")
    response = requests.post(
        f"{REME_URL}/vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "delete"
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to delete: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    print("✓ Deletion request successful")
    
    # Step 4: Retrieve again (should find 0)
    print("\n4. Retrieving memories after deletion...")
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={"workspace_id": workspace_id, "query": "Memory", "top_k": 10}
    )
    
    if response.status_code != 200:
        # This might be OK if workspace doesn't exist
        print(f"⚠️  Retrieve returned: {response.status_code}")
    
    memory_list = response.json().get('metadata', {}).get('memory_list', [])
    count_after = len(memory_list)
    print(f"✓ Found {count_after} memories after deletion")
    
    # Step 5: Verify
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Before deletion: {count_before} memories")
    print(f"After deletion:  {count_after} memories")
    
    if count_after == 0:
        print("\n✅ TEST PASSED: Workspace deletion works!")
        return True
    else:
        print(f"\n❌ TEST FAILED: Expected 0, found {count_after}")
        return False

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
            print(f"✗ ReMe server not responding properly: {response.status_code}")
            print("\nMake sure ReMe server is running:")
            print("  reme backend=http http.port=8002")
            sys.exit(1)
        
        print("✓ ReMe server is running\n")
        
        # Run test
        success = test_workspace_deletion()
        
        sys.exit(0 if success else 1)
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to ReMe server")
        print("\nStart the server with:")
        print("  reme backend=http http.port=8002")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
