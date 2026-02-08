"""
Adapter to convert HaluMem dataset format to MemoryScope evaluation format
"""
import json
import os
from typing import List, Dict, Any
from datasets import load_from_disk
from datetime import datetime

class HaluMemAdapter:
    """Convert HaluMem format to MemoryScope format for RSPM evaluation"""
    
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.stats = {
            "total_users": 0,
            "total_conversations": 0,
            "total_messages": 0,
            "temporal_updates": 0,
            "conflicts_detected": 0,
            "qa_pairs": 0
        }
    
    def load_halumem(self):
        """Load HaluMem dataset from disk"""
        print(f"Loading HaluMem dataset from {self.dataset_path}...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        dataset = load_from_disk(self.dataset_path)
        print(f"Loaded {len(dataset)} users from HaluMem")
        return dataset
    
    def convert_halumem_to_memoryscope(self, halumem_data) -> List[Dict]:
        """
        Convert HaluMem format to MemoryScope format
        
        HaluMem structure:
        {
            "uuid": "user_id",
            "persona_info": {...},
            "sessions": [
                {
                    "dialogue": [{"role": "user", "content": "...", "timestamp": "..."}],
                    "memory_points": [{"memory_content": "...", "is_update": "True", ...}],
                    "questions": [{"question": "...", "answer": "...", "question_type": "..."}]
                }
            ]
        }
        
        MemoryScope format:
        {
            "conversation_id": "conv_xxx",
            "messages": [{"role": "user", "content": "...", "timestamp": "..."}],
            "ground_truth": {
                "current_info": {...},
                "updates": [{"from": "...", "to": "...", "timestamp": "..."}]
            },
            "qa_pairs": [{"question": "...", "answer": "...", "type": "..."}]
        }
        """
        converted_conversations = []
        
        print("\nConverting HaluMem to MemoryScope format...")
        
        # Handle different dataset structures
        if hasattr(halumem_data, 'to_dict'):
            data_dict = halumem_data.to_dict()
        elif isinstance(halumem_data, dict):
            data_dict = halumem_data
        else:
            # Assume it's a dataset with train split
            data_dict = halumem_data['train'].to_dict()
        
        # Process each user
        for idx in range(len(data_dict.get('uuid', []))):
            user_id = data_dict['uuid'][idx]
            sessions = data_dict['sessions'][idx]
            
            self.stats["total_users"] += 1
            
            print(f"\nProcessing User {user_id}:")
            print(f"  - Sessions: {len(sessions)}")
            
            # Process each session as a conversation
            for session_idx, session in enumerate(sessions):
                conversation_id = f"{user_id}_session_{session_idx}"
                
                # Extract messages
                messages = []
                for msg in session.get('dialogue', []):
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content'],
                        "timestamp": msg.get('timestamp', ''),
                        "turn": msg.get('dialogue_turn', 0)
                    })
                
                self.stats["total_messages"] += len(messages)
                
                # Extract ground truth from memory points
                memory_points = session.get('memory_points', [])
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
                questions = session.get('questions', [])
                
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
                    "session_info": {
                        "start_time": session.get('start_time', ''),
                        "end_time": session.get('end_time', ''),
                        "turn_count": session.get('dialogue_turn_num', 0),
                        "token_length": session.get('dialogue_token_length', 0)
                    },
                    "messages": messages,
                    "ground_truth": {
                        "current_info": current_info,
                        "updates": updates,
                        "memory_points": memory_points
                    },
                    "qa_pairs": qa_pairs,
                    "metadata": {
                        "source": "HaluMem",
                        "has_temporal_updates": len(updates) > 0,
                        "update_count": len(updates)
                    }
                }
                
                converted_conversations.append(conversation)
                self.stats["total_conversations"] += 1
                
                # Detect conflicts (messages containing contradiction keywords)
                for msg in messages:
                    content_lower = msg['content'].lower()
                    if any(word in content_lower for word in ['actually', 'changed', 'update', 'correction', 'instead', 'now', 'currently']):
                        self.stats["conflicts_detected"] += 1
                        break
        
        print(f"\n✓ Conversion complete!")
        print(f"  - Total conversations: {len(converted_conversations)}")
        print(f"  - Total messages: {self.stats['total_messages']}")
        print(f"  - Temporal updates: {self.stats['temporal_updates']}")
        print(f"  - Conflicts detected: {self.stats['conflicts_detected']}")
        
        return converted_conversations
    
    def save_converted_data(self, conversations: List[Dict]):
        """Save converted conversations in JSONL format"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        print(f"\nSaving converted data to {self.output_path}...")
        
        with open(self.output_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        # Save stats
        stats_path = self.output_path.replace('.jsonl', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✓ Saved {len(conversations)} conversations")
        print(f"✓ Stats saved to {stats_path}")
    
    def convert_and_save(self):
        """Main conversion pipeline"""
        print("\n" + "=" * 60)
        print("HaluMem to MemoryScope Adapter")
        print("=" * 60)
        
        # Load HaluMem
        halumem_data = self.load_halumem()
        
        # Convert
        conversations = self.convert_halumem_to_memoryscope(halumem_data)
        
        # Save
        self.save_converted_data(conversations)
        
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        for key, value in self.stats.items():
            print(f"{key}: {value}")
        print("=" * 60)
        
        return conversations

def main():
    """Convert HaluMem dataset and separate into Medium and Long versions"""
    
    print(f"\n\n{'='*60}")
    print(f"Processing HaluMem Dataset")
    print('='*60)
    
    input_path = "datasets/halumem/full"
    
    if not os.path.exists(input_path):
        print(f"✗ Dataset not found at {input_path}")
        print("\nPlease run the following command first:")
        print("  python cookbook/memoryscope/download_halumem.py")
        return
    
    # First convert the full dataset
    print("\nConverting full HaluMem dataset...")
    adapter = HaluMemAdapter(
        dataset_path=input_path,
        output_path="datasets/memoryscope/halumem_full.jsonl"
    )
    
    try:
        conversations = adapter.convert_and_save()
        
        # Now separate into Medium and Long based on session counts
        # Medium: users with ~70 sessions (up to 100)
        # Long: users with ~120 sessions (100+)
        
        print("\n\n" + "="*60)
        print("Separating into Medium and Long versions")
        print("="*60)
        
        medium_convs = []
        long_convs = []
        
        # Group by user
        user_sessions = {}
        for conv in conversations:
            user_id = conv['user_id']
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            user_sessions[user_id].append(conv)
        
        print(f"\nTotal users: {len(user_sessions)}")
        
        for user_id, user_convs in user_sessions.items():
            session_count = len(user_convs)
            print(f"  User {user_id}: {session_count} sessions", end="")
            
            if session_count <= 100:
                medium_convs.extend(user_convs)
                print(" -> Medium")
            else:
                long_convs.extend(user_convs)
                print(" -> Long")
        
        # Save separated versions
        print(f"\nSaving separated datasets...")
        print(f"  Medium: {len(medium_convs)} conversations")
        print(f"  Long: {len(long_convs)} conversations")
        
        # Save Medium
        medium_path = "datasets/memoryscope/halumem_medium.jsonl"
        with open(medium_path, 'w') as f:
            for conv in medium_convs:
                f.write(json.dumps(conv) + '\n')
        print(f"  ✓ Medium saved to: {medium_path}")
        
        # Save Long
        long_path = "datasets/memoryscope/halumem_long.jsonl"
        with open(long_path, 'w') as f:
            for conv in long_convs:
                f.write(json.dumps(conv) + '\n')
        print(f"  ✓ Long saved to: {long_path}")
        
        print(f"\n✓ HaluMem conversion complete!")
        print(f"\nDatasets ready for evaluation:")
        print(f"  - Medium: {medium_path}")
        print(f"  - Long: {long_path}")
        
    except Exception as e:
        print(f"\n✗ Error converting HaluMem: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
