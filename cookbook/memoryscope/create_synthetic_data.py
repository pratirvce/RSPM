"""
Create synthetic MemoryScope data for testing
Run this first if you don't have the real dataset yet
"""
import json
from pathlib import Path

def create_synthetic_memoryscope(num_conversations: int = 30):
    """
    Create synthetic MemoryScope-style conversations with temporal conflicts
    
    This generates test data following the MemoryScope format:
    - Multi-turn conversations (20-40 turns)
    - Temporal conflicts (preference changes, fact updates)
    - Ground truth for evaluation
    """
    
    conversations = []
    
    # Template 1: Dietary preference change (5 variations)
    dietary_templates = [
        ("vegetarian", "vegan"),
        ("meat-eater", "vegetarian"),
        ("vegan", "flexitarian"),
        ("no restrictions", "gluten-free"),
        ("pescatarian", "vegan")
    ]
    
    for idx, (old_pref, new_pref) in enumerate(dietary_templates):
        conversations.append({
            "conversation_id": f"diet_{idx:03d}",
            "turns": [
                {"turn": 1, "role": "user", "content": f"Hi, I'm {old_pref}"},
                {"turn": 2, "role": "assistant", "content": f"Hello! I'll remember you're {old_pref}."},
                {"turn": 10, "role": "user", "content": "What's the weather like?"},
                {"turn": 11, "role": "assistant", "content": "It's sunny today."},
                {"turn": 20, "role": "user", "content": f"Actually, I'm {new_pref} now"},
                {"turn": 21, "role": "assistant", "content": f"Got it, updating to {new_pref}."},
                {"turn": 30, "role": "user", "content": "Recommend a restaurant for dinner"}
            ],
            "conflicts": [{
                "old_turn": 1,
                "new_turn": 20,
                "old_value": old_pref,
                "new_value": new_pref,
                "old_statement": f"Hi, I'm {old_pref}",
                "new_statement": f"Actually, I'm {new_pref} now",
                "type": "diet_preference_change"
            }],
            "ground_truth": {
                "turn_30": {
                    "should_use": new_pref,
                    "not": old_pref
                }
            }
        })
    
    # Template 2: Location change (5 variations)
    location_templates = [
        ("New York", "San Francisco"),
        ("London", "Paris"),
        ("Tokyo", "Seoul"),
        ("Berlin", "Amsterdam"),
        ("Sydney", "Melbourne")
    ]
    
    for idx, (old_loc, new_loc) in enumerate(location_templates):
        conversations.append({
            "conversation_id": f"location_{idx:03d}",
            "turns": [
                {"turn": 1, "role": "user", "content": f"I live in {old_loc}"},
                {"turn": 2, "role": "assistant", "content": f"Great, {old_loc} is a wonderful city."},
                {"turn": 15, "role": "user", "content": "Tell me a joke"},
                {"turn": 16, "role": "assistant", "content": "Why did the chicken cross the road?"},
                {"turn": 25, "role": "user", "content": f"I moved to {new_loc} last month"},
                {"turn": 26, "role": "assistant", "content": f"Congratulations on moving to {new_loc}!"},
                {"turn": 35, "role": "user", "content": "What's the weather like where I live?"}
            ],
            "conflicts": [{
                "old_turn": 1,
                "new_turn": 25,
                "old_value": old_loc,
                "new_value": new_loc,
                "old_statement": f"I live in {old_loc}",
                "new_statement": f"I moved to {new_loc} last month",
                "type": "location_change"
            }],
            "ground_truth": {
                "turn_35": {
                    "should_use": new_loc,
                    "not": old_loc
                }
            }
        })
    
    # Template 3: Budget change (5 variations)
    budget_templates = [
        ("$500", "$800"),
        ("$1000", "$1500"),
        ("$200", "$500"),
        ("$5000", "$3000"),
        ("$100", "$250")
    ]
    
    for idx, (old_budget, new_budget) in enumerate(budget_templates):
        conversations.append({
            "conversation_id": f"budget_{idx:03d}",
            "turns": [
                {"turn": 1, "role": "user", "content": f"My budget is {old_budget}"},
                {"turn": 2, "role": "assistant", "content": f"Understood, working with {old_budget}."},
                {"turn": 20, "role": "user", "content": "What's 2+2?"},
                {"turn": 21, "role": "assistant", "content": "4"},
                {"turn": 40, "role": "user", "content": f"I can increase budget to {new_budget}"},
                {"turn": 41, "role": "assistant", "content": f"Great! New budget is {new_budget}."},
                {"turn": 50, "role": "user", "content": "Recommend options within my budget"}
            ],
            "conflicts": [{
                "old_turn": 1,
                "new_turn": 40,
                "old_value": old_budget,
                "new_value": new_budget,
                "old_statement": f"My budget is {old_budget}",
                "new_statement": f"I can increase budget to {new_budget}",
                "type": "budget_change"
            }],
            "ground_truth": {
                "turn_50": {
                    "should_use": new_budget,
                    "not": old_budget
                }
            }
        })
    
    # Add more templates as needed to reach num_conversations
    # Template 4: Allergy changes, hobby changes, etc.
    
    # Limit to requested number
    conversations = conversations[:num_conversations]
    
    # Save to file
    output_dir = Path("datasets/memoryscope")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "synthetic.jsonl"
    with open(output_file, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    
    print(f"✓ Created {len(conversations)} synthetic conversations")
    print(f"✓ Saved to: {output_file}")
    
    # Print statistics
    total_conflicts = sum(len(c.get('conflicts', [])) for c in conversations)
    print(f"\nStatistics:")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total conflicts: {total_conflicts}")
    print(f"  Avg turns per conversation: {sum(len(c['turns']) for c in conversations) / len(conversations):.1f}")
    
    return conversations

if __name__ == "__main__":
    import sys
    
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    conversations = create_synthetic_memoryscope(num)
    
    print("\n✓ Ready to test!")
    print("Next: python cookbook/memoryscope/data_loader.py")
