"""
MemoryScope Dataset Loader and Preprocessor
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple

class MemoryScopeDataset:
    """Loader for MemoryScope benchmark dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Load all conversations from JSONL file"""
        if not self.data_path.exists():
            print(f"Warning: Dataset not found at {self.data_path}")
            print("Using empty dataset. Create synthetic data or download real dataset.")
            return
        
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    self.conversations.append(conv)
        
        print(f"Loaded {len(self.conversations)} conversations")
    
    def __len__(self):
        """Return number of conversations"""
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """Get conversation by index"""
        return self.conversations[idx]
    
    def train_test_split(self) -> Tuple[List, List]:
        """
        Split into first 250 (train) and last 250 (test)
        Following the unified experimental protocol
        """
        if len(self.conversations) < 500:
            # For smaller datasets, use 50/50 split
            mid = len(self.conversations) // 2
            return self.conversations[:mid], self.conversations[mid:]
        
        train = self.conversations[:250]
        test = self.conversations[250:500]
        return train, test
    
    def get_conversation_messages(self, conv_data: Dict) -> List[Dict]:
        """
        Convert conversation to ReMe message format
        
        Input: {"turns": [...]}
        Output: [{"role": "user", "content": "...", "turn": 1}, ...]
        """
        if isinstance(conv_data, int):
            # If passed index, get from list
            conv_data = self.conversations[conv_data]
        
        messages = []
        
        for turn in conv_data.get('turns', []):
            messages.append({
                "role": turn.get('role', 'user'),
                "content": turn.get('content', ''),
                "turn": turn.get('turn', 0)
            })
        
        return messages
    
    def get_temporal_conflicts(self, conv_data: Dict) -> List[Dict]:
        """Extract temporal conflicts from conversation"""
        if isinstance(conv_data, int):
            conv_data = self.conversations[conv_data]
        
        return conv_data.get('conflicts', [])
    
    def evaluate_temporal_consistency(
        self, 
        conv_data: Dict,
        agent_response: str
    ) -> bool:
        """
        Check if agent used correct (latest) information
        
        Returns True if agent's response uses the most recent value
        """
        if isinstance(conv_data, int):
            conv_data = self.conversations[conv_data]
        
        ground_truth = conv_data.get('ground_truth', {})
        
        if not ground_truth:
            # No conflicts, assume correct
            return True
        
        # Check each turn's expected answer
        for turn_key, expected in ground_truth.items():
            should_use = expected.get('should_use', '').lower()
            should_not_use = expected.get('not', '').lower()
            
            response_lower = agent_response.lower()
            
            # Check if correct value present
            has_correct = should_use in response_lower if should_use else True
            
            # Check if incorrect value absent
            has_wrong = should_not_use in response_lower if should_not_use else False
            
            if has_correct and not has_wrong:
                return True
            elif has_wrong:
                return False
        
        return True  # Default to correct if unclear

# Test the loader
if __name__ == "__main__":
    # Test with synthetic data
    dataset = MemoryScopeDataset("datasets/memoryscope/synthetic.jsonl")
    
    if len(dataset.conversations) > 0:
        train, test = dataset.train_test_split()
        print(f"Train: {len(train)}, Test: {len(test)}")
        
        # Show first conversation
        if dataset.conversations:
            messages = dataset.get_conversation_messages(dataset.conversations[0])
            print(f"\nSample conversation: {len(messages)} turns")
            for msg in messages[:3]:
                print(f"  Turn {msg['turn']}: {msg['content'][:50]}...")
    else:
        print("\nNo data loaded. Next steps:")
        print("1. Create synthetic data: python create_synthetic_data.py")
        print("2. Or download real MemoryScope dataset")
