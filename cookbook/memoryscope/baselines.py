"""
Baseline implementations for MemoryScope
CORRECTED VERSION - Uses ReMe's actual API
"""
import requests
from typing import List, Dict

class StandardRAGBaseline:
    """Baseline 1: Standard RAG (No memory management)"""
    
    def __init__(self, workspace_id: str = "memoryscope_baseline_rag"):
        self.workspace_id = workspace_id
        self.reme_url = "http://localhost:8002"
    
    def process_conversation(
        self, 
        messages: List[Dict],
        query: str
    ) -> str:
        """
        Process conversation and answer query
        Uses ReMe's existing task memory retrieval
        """
        # 1. Store all turns in memory
        response = requests.post(
            f"{self.reme_url}/summary_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "trajectories": [{
                    "messages": messages,
                    "score": 1.0  # All messages stored with equal priority
                }]
            }
        )
        
        # 2. Retrieve relevant memories for query
        response = requests.post(
            f"{self.reme_url}/retrieve_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "query": query,
                "top_k": 5
            }
        )
        
        return response.json().get("answer", "")
    
    def clear_memory(self):
        """Clear workspace for next conversation using vector_store action"""
        requests.post(
            f"{self.reme_url}/vector_store",
            json={
                "workspace_id": self.workspace_id,
                "action": "delete"
            }
        )

class RecencyWeightedBaseline:
    """Baseline 2: Recency-Weighted Retrieval with post-processing"""
    
    def __init__(self, workspace_id: str = "memoryscope_baseline_recency"):
        self.workspace_id = workspace_id
        self.reme_url = "http://localhost:8002"
        self.alpha = 0.7  # Similarity weight
    
    def process_conversation(
        self, 
        messages: List[Dict],
        query: str
    ) -> str:
        """
        Retrieval with recency weighting
        score = alpha * similarity + (1-alpha) * recency
        
        Note: ReMe doesn't support custom scoring directly,
        so we retrieve many candidates and post-process
        """
        import time
        current_time = time.time()
        
        # Store messages with timestamp metadata
        for idx, msg in enumerate(messages):
            if 'metadata' not in msg:
                msg['metadata'] = {}
            msg['metadata']['timestamp'] = current_time - (len(messages) - idx) * 60
            msg['metadata']['turn'] = idx
        
        # Store in memory
        requests.post(
            f"{self.reme_url}/summary_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "trajectories": [{
                    "messages": messages,
                    "score": 1.0
                }]
            }
        )
        
        # Retrieve many candidates for reranking
        response = requests.post(
            f"{self.reme_url}/retrieve_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "query": query,
                "top_k": 50  # Get many for post-processing
            }
        )
        
        result = response.json()
        memories = result.get('metadata', {}).get('memory_list', [])
        
        # Post-process: Apply recency weighting
        if memories:
            max_timestamp = max(
                m.get('metadata', {}).get('timestamp', 0) for m in memories
            )
            
            for mem in memories:
                timestamp = mem.get('metadata', {}).get('timestamp', 0)
                similarity_score = mem.get('score', 0)
                
                # Normalize recency (1.0 = most recent)
                recency = timestamp / max_timestamp if max_timestamp > 0 else 0
                
                # Combined score
                mem['combined_score'] = (
                    self.alpha * similarity_score + 
                    (1 - self.alpha) * recency
                )
            
            # Re-sort by combined score
            memories.sort(key=lambda m: m.get('combined_score', 0), reverse=True)
            
            # Take top-5
            top_memories = memories[:5]
            answer = "\n".join([m['content'] for m in top_memories])
        else:
            answer = ""
        
        return answer
    
    def clear_memory(self):
        """Clear workspace using vector_store action"""
        requests.post(
            f"{self.reme_url}/vector_store",
            json={
                "workspace_id": self.workspace_id,
                "action": "delete"
            }
        )

class SlidingWindowBaseline:
    """Baseline 3: Sliding Window (Keep last 100 messages)"""
    
    def __init__(
        self, 
        workspace_id: str = "memoryscope_baseline_window",
        window_size: int = 100
    ):
        self.workspace_id = workspace_id
        self.reme_url = "http://localhost:8002"
        self.window_size = window_size
        self.message_buffer = []
    
    def process_conversation(
        self, 
        messages: List[Dict],
        query: str
    ) -> str:
        """
        Keep only last N messages using clear/re-insert pattern
        
        Note: We manually manage the buffer and clear/re-insert
        into ReMe each time
        """
        # Add new messages to buffer
        self.message_buffer.extend(messages)
        
        # Keep only last window_size
        if len(self.message_buffer) > self.window_size:
            self.message_buffer = self.message_buffer[-self.window_size:]
        
        # Clear workspace using vector_store action
        requests.post(
            f"{self.reme_url}/vector_store",
            json={
                "workspace_id": self.workspace_id,
                "action": "delete"
            }
        )
        
        # Re-insert windowed messages
        requests.post(
            f"{self.reme_url}/summary_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "trajectories": [{
                    "messages": self.message_buffer,
                    "score": 1.0
                }]
            }
        )
        
        # Retrieve
        response = requests.post(
            f"{self.reme_url}/retrieve_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "query": query,
                "top_k": 5
            }
        )
        
        return response.json().get("answer", "")
    
    def clear_memory(self):
        """Clear workspace and buffer"""
        self.message_buffer = []
        requests.post(
            f"{self.reme_url}/vector_store",
            json={
                "workspace_id": self.workspace_id,
                "action": "delete"
            }
        )

class EpisodicSummaryBaseline:
    """Baseline 4: Episodic Summary (GPT-4 summarizes every N turns)"""
    
    def __init__(
        self, 
        workspace_id: str = "memoryscope_baseline_summary",
        summary_interval: int = 50
    ):
        self.workspace_id = workspace_id
        self.reme_url = "http://localhost:8002"
        self.summary_interval = summary_interval
        self.turn_count = 0
        self.accumulated_messages = []
    
    def process_conversation(
        self, 
        messages: List[Dict],
        query: str
    ) -> str:
        """Summarize every N turns"""
        self.accumulated_messages.extend(messages)
        self.turn_count += len(messages)
        
        # Check if summary needed
        if self.turn_count >= self.summary_interval:
            self._create_summary()
            self.turn_count = 0
            self.accumulated_messages = []
        
        # Store messages (original + summaries)
        requests.post(
            f"{self.reme_url}/summary_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "trajectories": [{
                    "messages": self.accumulated_messages,
                    "score": 1.0
                }]
            }
        )
        
        # Retrieve
        response = requests.post(
            f"{self.reme_url}/retrieve_task_memory",
            json={
                "workspace_id": self.workspace_id,
                "query": query,
                "top_k": 5
            }
        )
        
        return response.json().get("answer", "")
    
    def _create_summary(self):
        """Create summary using GPT-4"""
        # TODO: Implement GPT-4 summarization
        # For now, simple concatenation
        summary_text = f"Summary of {len(self.accumulated_messages)} messages"
        self.accumulated_messages = [
            {"role": "system", "content": summary_text}
        ]
    
    def clear_memory(self):
        """Clear workspace and buffers"""
        self.turn_count = 0
        self.accumulated_messages = []
        requests.post(
            f"{self.reme_url}/vector_store",
            json={
                "workspace_id": self.workspace_id,
                "action": "delete"
            }
        )

# Test the baselines
if __name__ == "__main__":
    from data_loader import MemoryScopeDataset
    from metrics import MemoryScopeMetrics
    
    dataset = MemoryScopeDataset("datasets/memoryscope/synthetic.jsonl")
    baseline = StandardRAGBaseline()
    metrics = MemoryScopeMetrics()
    
    # Test on first 5 conversations
    for conv in dataset.conversations[:5]:
        messages = dataset.get_conversation_messages(conv)
        final_query = messages[-1]['content']
        
        response = baseline.process_conversation(messages[:-1], final_query)
        is_correct = dataset.evaluate_temporal_consistency(conv, response)
        conflicts = dataset.get_temporal_conflicts(conv)
        
        metrics.update(is_correct, len(conflicts) > 0)
        baseline.clear_memory()
    
    metrics.print_summary("Standard RAG Baseline (5 convs)")
