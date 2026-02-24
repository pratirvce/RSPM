"""
Sleep cycle for RSPM - consolidates memories every N tasks
CORRECTED VERSION - Uses ReMe's actual API
"""
import requests
from typing import List, Dict

class SleepCycle:
    """Manages when and how to consolidate memories"""
    
    def __init__(
        self,
        workspace_id: str,
        sleep_frequency: int = 10,
        conflict_threshold: int = 3
    ):
        self.workspace_id = workspace_id
        self.sleep_frequency = sleep_frequency
        self.conflict_threshold = conflict_threshold
        
        self.task_count = 0
        self.failure_buffer = []
        self.reme_url = "http://localhost:8002"
    
    def record_task(
        self, 
        messages: List[Dict],
        conflicts: List[Dict],
        is_failure: bool
    ):
        """Record task result"""
        self.task_count += 1
        
        if is_failure and conflicts:
            self.failure_buffer.append({
                "messages": messages,
                "conflicts": conflicts,
                "task_id": self.task_count
            })
    
    def should_sleep(self) -> bool:
        """Determine if sleep cycle should trigger"""
        # Condition 1: Every N tasks
        if self.task_count % self.sleep_frequency == 0:
            return True
        
        # Condition 2: Conflict rate exceeds threshold
        recent_failures = len([
            f for f in self.failure_buffer[-5:]
        ])
        if recent_failures >= self.conflict_threshold:
            return True
        
        return False
    
    def execute_sleep_cycle(self):
        """
        Execute sleep cycle:
        1. Extract rules from failures
        2. Prune failed memories
        3. Insert rules
        """
        if not self.failure_buffer:
            print("No failures to consolidate")
            return
        
        print(f"🌙 Sleep cycle triggered: {len(self.failure_buffer)} failures")
        
        # Step 1: Extract rules
        rules = self._extract_rules()
        
        # Step 2: Prune failed memories
        self._prune_failures()
        
        # Step 3: Insert rules
        self._insert_rules(rules)
        
        # Clear buffer
        self.failure_buffer = []
        
        print(f"✅ Sleep cycle complete: {len(rules)} rules extracted")
    
    def _extract_rules(self) -> List[Dict]:
        """Extract negative constraints from failures"""
        rules = []
        
        for failure in self.failure_buffer:
            conflicts = failure['conflicts']
            
            for conflict in conflicts:
                old_stmt = conflict.get('old_statement', conflict.get('old_value', 'unknown'))
                new_stmt = conflict.get('new_statement', conflict.get('new_value', 'unknown'))
                rule_text = f"""[RULE: Do not use {conflict.get('old_value', old_stmt)} when user has updated to {conflict.get('new_value', new_stmt)}. 

Always check for the most recent value in temporal contexts. 
When user says "actually" or "now", they are updating previous information.

Specifically: {old_stmt} was replaced by {new_stmt} at turn {conflict.get('new_turn', 'N/A')}.]"""
                
                rules.append({
                    "type": "negative_constraint",
                    "content": rule_text,
                    "applicable_to": conflict['type'],
                    "priority": 0.95,
                    "source_task": failure['task_id']
                })
        
        return rules
    
    def _prune_failures(self):
        """
        Delete episodic memories from conflict turns
        Uses ReMe's vector_store action with delete_ids
        """
        for failure in self.failure_buffer:
            conflicts = failure['conflicts']
            
            for conflict in conflicts:
                # Step 1: Query for memories containing old value
                old_stmt = conflict.get('old_statement', conflict.get('old_value', ''))
                if not old_stmt:
                    continue
                response = requests.post(
                    f"{self.reme_url}/retrieve_task_memory",
                    json={
                        "workspace_id": self.workspace_id,
                        "query": old_stmt,
                        "top_k": 20
                    }
                )
                
                if response.status_code != 200:
                    print(f"✗ Failed to retrieve memories: {response.status_code}")
                    continue
                
                result = response.json()
                memory_list = result.get('metadata', {}).get('memory_list', [])
                
                # Step 2: Filter for memories with old value but not new value
                memories_to_delete = []
                
                for memory in memory_list:
                    content = memory.get('content', '').lower()
                    old_value = conflict['old_value'].lower()
                    new_value = conflict['new_value'].lower()
                    
                    # Delete if contains old value but NOT new value
                    if old_value in content and new_value not in content:
                        memories_to_delete.append(memory['memory_id'])
                
                # Step 3: Delete identified memories using vector_store action
                if memories_to_delete:
                    response = requests.post(
                        f"{self.reme_url}/vector_store",
                        json={
                            "workspace_id": self.workspace_id,
                            "action": "delete_ids",
                            "memory_ids": memories_to_delete
                        }
                    )
                    
                    if response.status_code == 200:
                        print(f"  ✓ Pruned {len(memories_to_delete)} memories from turn {conflict['old_turn']}")
                    else:
                        print(f"  ✗ Failed to prune memories: {response.status_code}")
    
    def _insert_rules(self, rules: List[Dict]):
        """Insert negative constraints into memory"""
        for rule in rules:
            # Insert as high-priority memory
            response = requests.post(
                f"{self.reme_url}/summary_task_memory",
                json={
                    "workspace_id": self.workspace_id,
                    "trajectories": [{
                        "messages": [{
                            "role": "system",
                            "content": rule['content']
                        }],
                        "score": rule['priority']
                    }]
                }
            )
            
            if response.status_code == 200:
                print(f"  ✓ Inserted rule: {rule['type']}")
            else:
                print(f"  ✗ Failed to insert rule: {response.status_code}")

# Test
if __name__ == "__main__":
    sleep_cycle = SleepCycle(workspace_id="test")
    
    # Simulate 10 tasks with 3 failures
    for i in range(10):
        is_failure = i % 3 == 0
        conflicts = [{
            "old_turn": 1,
            "new_turn": 5,
            "old_value": "vegetarian",
            "new_value": "vegan",
            "old_statement": "I'm vegetarian",
            "new_statement": "I'm vegan now",
            "type": "diet_change"
        }] if is_failure else []
        
        sleep_cycle.record_task(
            messages=[],
            conflicts=conflicts,
            is_failure=is_failure
        )
        
        if sleep_cycle.should_sleep():
            sleep_cycle.execute_sleep_cycle()
