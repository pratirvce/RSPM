"""
Test 2: Verify ReMe's selective deletion by memory ID works
Run this AFTER test_deletion.py passes
"""
import requests
import sys

REME_URL = "http://localhost:8002"
workspace_id = "test_selective"

def test_selective_deletion():
    """Test deleting specific memories by ID"""
    print("\n" + "="*60)
    print("Test 2: Selective Deletion by Memory ID")
    print("="*60)
    
    # Step 1: Insert 3 distinct memories
    print("\n1. Inserting 3 memories...")
    response = requests.post(
        f"{REME_URL}/summary_task_memory",
        json={
            "workspace_id": workspace_id,
            "trajectories": [{
                "messages": [
                    {"role": "user", "content": "I'm vegetarian"},
                    {"role": "user", "content": "I like pizza"},
                    {"role": "user", "content": "I'm vegan now"}
                ],
                "score": 1.0
            }]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to insert: {response.status_code}")
        return False
    
    print("✓ Inserted 3 memories")
    
    # Step 2: Retrieve all memories
    print("\n2. Retrieving all memories...")
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={"workspace_id": workspace_id, "query": "dietary", "top_k": 10}
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to retrieve: {response.status_code}")
        return False
    
    memories = response.json().get('metadata', {}).get('memory_list', [])
    print(f"✓ Found {len(memories)} memories:")
    for idx, mem in enumerate(memories):
        print(f"  {idx+1}. {mem['content'][:50]}")
    
    if len(memories) == 0:
        print("⚠️  No memories found! Check if insertion worked.")
        return False
    
    # Step 3: Find "vegetarian" memory (not "vegan")
    print("\n3. Identifying 'vegetarian' memory to delete...")
    vegetarian_mems = [
        m for m in memories 
        if 'vegetarian' in m['content'].lower() and 'vegan' not in m['content'].lower()
    ]
    
    if not vegetarian_mems:
        print("⚠️  Could not find 'vegetarian' memory")
        print("Memories found:")
        for mem in memories:
            print(f"  - {mem['content']}")
        return False
    
    mem_to_delete = vegetarian_mems[0]
    mem_id = mem_to_delete['memory_id']
    print(f"✓ Found vegetarian memory: {mem_id}")
    print(f"  Content: {mem_to_delete['content']}")
    
    # Step 4: Delete only this memory using vector_store action
    print(f"\n4. Deleting memory {mem_id}...")
    response = requests.post(
        f"{REME_URL}/vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "delete_ids",
            "memory_ids": [mem_id]
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to delete: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    print("✓ Deletion request successful")
    
    # Step 5: Verify deletion
    print("\n5. Verifying deletion...")
    response = requests.post(
        f"{REME_URL}/retrieve_task_memory",
        json={"workspace_id": workspace_id, "query": "dietary", "top_k": 10}
    )
    
    remaining = response.json().get('metadata', {}).get('memory_list', [])
    print(f"✓ Remaining memories: {len(remaining)}")
    for idx, mem in enumerate(remaining):
        print(f"  {idx+1}. {mem['content']}")
    
    # Step 6: Check results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Before deletion: {len(memories)} memories")
    print(f"After deletion:  {len(remaining)} memories")
    
    # Verify "vegetarian" is gone but "pizza" and "vegan" remain
    has_vegetarian = any('vegetarian' in m['content'].lower() for m in remaining)
    has_pizza = any('pizza' in m['content'].lower() for m in remaining)
    has_vegan = any('vegan' in m['content'].lower() for m in remaining)
    
    print(f"\nContains 'vegetarian': {has_vegetarian} (should be False)")
    print(f"Contains 'pizza':      {has_pizza} (should be True)")
    print(f"Contains 'vegan':      {has_vegan} (should be True)")
    
    if not has_vegetarian and has_pizza and has_vegan:
        print("\n✅ TEST PASSED: Selective deletion works correctly!")
        print("   - 'vegetarian' was deleted")
        print("   - 'pizza' and 'vegan' remain")
        return True
    else:
        print("\n❌ TEST FAILED: Deletion was not selective")
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
        success = test_selective_deletion()
        
        # Cleanup
        print("\nCleaning up test workspace...")
        cleanup()
        print("✓ Cleanup complete")
        
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
        cleanup()
        sys.exit(1)
