"""
Adapter for HaluMem GitHub dataset format
Converts stage5_1_dialogue_generation.jsonl to MemoryScope evaluation format
"""
import json
import os
from typing import List, Dict, Any

class HaluMemGitHubAdapter:
    """Convert HaluMem GitHub format to MemoryScope format for RSPM evaluation"""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.stats = {
            "total_users": 0,
            "total_events": 0,
            "total_conversations": 0,
            "total_messages": 0,
            "temporal_updates": 0,
            "conflicts_detected": 0,
            "qa_pairs": 0,
            "medium_users": 0,
            "long_users": 0
        }
    
    def load_dataset(self):
        """Load HaluMem dataset from JSONL file"""
        print(f"Loading dataset from {self.input_path}...")
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Dataset not found at {self.input_path}")
        
        users = []
        with open(self.input_path, 'r') as f:
            for line in f:
                if line.strip():
                    users.append(json.loads(line))
        
        print(f"Loaded {len(users)} users from HaluMem")
        return users
    
    def convert_to_memoryscope(self, users: List[Dict]) -> tuple:
        """
        Convert HaluMem format to MemoryScope format
        Returns (medium_conversations, long_conversations)
        """
        print("\nConverting HaluMem to MemoryScope format...")
        
        medium_conversations = []
        long_conversations = []
        
        for user_data in users:
            user_id = user_data['uuid']
            event_list = user_data.get('event_list', [])
            
            self.stats["total_users"] += 1
            self.stats["total_events"] += len(event_list)
            
            print(f"\nProcessing User {user_id[:8]}...")
            print(f"  - Events: {len(event_list)}")
            
            user_conversations = []
            
            # Process each event as a conversation
            for event in event_list:
                dialogue_info = event.get('dialogue_info', {})
                
                if not dialogue_info or 'dialogue' not in dialogue_info:
                    continue
                
                conversation_id = f"{user_id}_event_{event['event_index']}"
                
                # Extract messages
                # Dialogue is a dict with keys like 'dialogue_turn_1', 'dialogue_turn_2', etc.
                # Each value is a list: [user_msg, assistant_msg, timestamp_dict]
                messages = []
                dialogue = dialogue_info.get('dialogue', {})
                
                # Sort dialogue turns by turn number
                turn_keys = sorted(dialogue.keys(), key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
                
                for turn_idx, turn_key in enumerate(turn_keys):
                    turn_data = dialogue[turn_key]
                    
                    if isinstance(turn_data, list) and len(turn_data) >= 2:
                        # Extract user and assistant messages
                        user_msg = turn_data[0] if len(turn_data) > 0 else {}
                        assistant_msg = turn_data[1] if len(turn_data) > 1 else {}
                        timestamp_info = turn_data[2] if len(turn_data) > 2 else {}
                        
                        # Get timestamp
                        if isinstance(timestamp_info, dict) and 'timestamp' in timestamp_info:
                            timestamp = timestamp_info['timestamp']
                        else:
                            timestamp = ''
                        
                        # Add user message
                        if user_msg and 'role' in user_msg:
                            messages.append({
                                "role": user_msg['role'],
                                "content": user_msg.get('content', ''),
                                "timestamp": timestamp,
                                "turn": turn_idx
                            })
                        
                        # Add assistant message
                        if assistant_msg and 'role' in assistant_msg:
                            messages.append({
                                "role": assistant_msg['role'],
                                "content": assistant_msg.get('content', ''),
                                "timestamp": timestamp,
                                "turn": turn_idx
                            })
                
                self.stats["total_messages"] += len(messages)
                
                # Extract ground truth from memory points
                memory_points = dialogue_info.get('memory_points', [])
                current_info = {}
                updates = []
                
                for mem in memory_points:
                    memory_type = mem.get('memory_type', 'Unknown')
                    memory_content = mem.get('memory_content', '')
                    is_update = mem.get('is_update', 'False')
                    
                    # Track current state
                    if memory_type not in current_info:
                        current_info[memory_type] = []
                    current_info[memory_type].append(memory_content)
                    
                    # Track updates for temporal consistency
                    if is_update == 'True' or is_update == True:
                        self.stats["temporal_updates"] += 1
                        original_memories = mem.get('original_memories', [])
                        
                        for orig in original_memories:
                            updates.append({
                                "from": orig,
                                "to": memory_content,
                                "timestamp": mem.get('timestamp', ''),
                                "memory_type": memory_type
                            })
                
                # Extract QA pairs
                qa_pairs = []
                questions = dialogue_info.get('questions', [])
                
                for q in questions:
                    qa_pairs.append({
                        "question": q.get('question', ''),
                        "answer": q.get('answer', ''),
                        "type": q.get('question_type', 'Unknown'),
                        "difficulty": q.get('difficulty', 'medium'),
                        "evidence": q.get('evidence', [])
                    })
                
                self.stats["qa_pairs"] += len(qa_pairs)
                
                # Create MemoryScope conversation entry
                conversation = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "event_info": {
                        "event_index": event.get('event_index', 0),
                        "event_type": event.get('event_type', ''),
                        "event_name": event.get('event_name', ''),
                        "event_time": event.get('event_time', ''),
                        "start_time": dialogue_info.get('start_time_point', ''),
                        "end_time": dialogue_info.get('end_time_point', ''),
                        "dialogue_goal": dialogue_info.get('dialogue_goal', '')
                    },
                    "messages": messages,
                    "ground_truth": {
                        "current_info": current_info,
                        "updates": updates,
                        "memory_points": memory_points,
                        "dialogue_summary": dialogue_info.get('dialogue_summary', '')
                    },
                    "qa_pairs": qa_pairs,
                    "metadata": {
                        "source": "HaluMem_GitHub",
                        "has_temporal_updates": len(updates) > 0,
                        "update_count": len(updates),
                        "message_count": len(messages)
                    }
                }
                
                user_conversations.append(conversation)
                self.stats["total_conversations"] += 1
                
                # Detect conflicts
                for msg in messages:
                    content_lower = msg['content'].lower()
                    if any(word in content_lower for word in ['actually', 'changed', 'update', 'correction', 'instead', 'now', 'currently', 'no longer']):
                        self.stats["conflicts_detected"] += 1
                        break
            
            # Classify user as Medium or Long based on conversation count
            if len(user_conversations) <= 100:
                medium_conversations.extend(user_conversations)
                self.stats["medium_users"] += 1
                print(f"  - Classified as MEDIUM ({len(user_conversations)} conversations)")
            else:
                long_conversations.extend(user_conversations)
                self.stats["long_users"] += 1
                print(f"  - Classified as LONG ({len(user_conversations)} conversations)")
        
        print(f"\n✓ Conversion complete!")
        print(f"  - Medium conversations: {len(medium_conversations)}")
        print(f"  - Long conversations: {len(long_conversations)}")
        print(f"  - Total messages: {self.stats['total_messages']}")
        print(f"  - Temporal updates: {self.stats['temporal_updates']}")
        print(f"  - Conflicts detected: {self.stats['conflicts_detected']}")
        
        return medium_conversations, long_conversations
    
    def save_converted_data(self, medium_convs: List[Dict], long_convs: List[Dict]):
        """Save converted conversations in JSONL format"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\nSaving converted data...")
        
        # Save Medium
        medium_path = os.path.join(self.output_dir, "halumem_medium.jsonl")
        with open(medium_path, 'w') as f:
            for conv in medium_convs:
                f.write(json.dumps(conv) + '\n')
        print(f"  ✓ Medium: {len(medium_convs)} conversations -> {medium_path}")
        
        # Save Long
        long_path = os.path.join(self.output_dir, "halumem_long.jsonl")
        with open(long_path, 'w') as f:
            for conv in long_convs:
                f.write(json.dumps(conv) + '\n')
        print(f"  ✓ Long: {len(long_convs)} conversations -> {long_path}")
        
        # Save stats
        stats_path = os.path.join(self.output_dir, "halumem_conversion_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"  ✓ Stats -> {stats_path}")
    
    def convert_and_save(self):
        """Main conversion pipeline"""
        print("\n" + "=" * 60)
        print("HaluMem GitHub to MemoryScope Adapter")
        print("=" * 60)
        
        # Load
        users = self.load_dataset()
        
        # Convert
        medium_convs, long_convs = self.convert_to_memoryscope(users)
        
        # Save
        self.save_converted_data(medium_convs, long_convs)
        
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        for key, value in self.stats.items():
            print(f"{key}: {value}")
        print("=" * 60)
        
        return medium_convs, long_convs

def main():
    """Convert HaluMem GitHub dataset"""
    
    input_path = "datasets/halumem/stage5_1_dialogue_generation.jsonl"
    output_dir = "datasets/memoryscope"
    
    if not os.path.exists(input_path):
        print(f"✗ Dataset not found at {input_path}")
        print("\nPlease run the following command first:")
        print("  python cookbook/memoryscope/download_halumem_github.py")
        return
    
    adapter = HaluMemGitHubAdapter(
        input_path=input_path,
        output_dir=output_dir
    )
    
    try:
        adapter.convert_and_save()
        print(f"\n✓ HaluMem conversion complete!")
        print(f"\nDatasets ready for evaluation:")
        print(f"  - Medium: {output_dir}/halumem_medium.jsonl")
        print(f"  - Long: {output_dir}/halumem_long.jsonl")
    except Exception as e:
        print(f"\n✗ Error converting HaluMem: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
