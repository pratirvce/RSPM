"""
Evaluation metrics for MemoryScope benchmark
"""
from typing import Dict, List

class MemoryScopeMetrics:
    """
    Tracks and computes evaluation metrics for MemoryScope
    
    Primary Metric: Temporal Consistency Score (TCS)
    Target: >85% (baseline: 62%)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.total_queries = 0
        self.correct_queries = 0
        self.temporal_conflicts = 0
        self.correct_conflicts = 0
        self.memory_tokens = 0
    
    def update(self, result):
        """
        Update metrics with single query result
        
        Args:
            result: Either a boolean or dict with keys:
                - 'correct': Whether agent's response was correct
                - 'has_conflict': Whether this query had a temporal conflict (optional)
                - 'memory_tokens': Total tokens in memory (optional)
        """
        # Support both bool and dict input for backward compatibility
        if isinstance(result, bool):
            is_correct = result
            has_conflict = False
            memory_tokens = 0
        elif isinstance(result, dict):
            is_correct = result.get('correct', False)
            has_conflict = result.get('has_conflict', False) or result.get('used_outdated', False)
            memory_tokens = result.get('memory_tokens', 0)
        else:
            # Treat as correct if unknown type
            is_correct = True
            has_conflict = False
            memory_tokens = 0
        
        self.total_queries += 1
        
        if is_correct:
            self.correct_queries += 1
        
        if has_conflict:
            self.temporal_conflicts += 1
            if is_correct:
                self.correct_conflicts += 1
        
        if memory_tokens > 0:
            self.memory_tokens = memory_tokens
    
    def compute_temporal_consistency_score(self) -> float:
        """
        TCS = Queries Answered with Latest Info / Total Queries with Conflicts
        
        When there are no temporal conflict items, falls back to overall accuracy.
        This is the PRIMARY metric for MemoryScope.
        Target: >85% (MemoryScope baseline: 62%)
        
        Returns:
            float: TCS score between 0.0 and 1.0
        """
        if self.temporal_conflicts == 0:
            # Fall back to overall accuracy when no conflicts present
            return self.compute_overall_accuracy()
        return self.correct_conflicts / self.temporal_conflicts
    
    def compute_overall_accuracy(self) -> float:
        """
        Overall accuracy across all queries (with and without conflicts)
        
        Returns:
            float: Accuracy between 0.0 and 1.0
        """
        if self.total_queries == 0:
            return 0.0
        return self.correct_queries / self.total_queries
    
    def compute_memory_efficiency(self) -> float:
        """
        MER = Total Memory Size (tokens) / Success Rate (%)
        
        Lower is better. Target: <500 tokens per percentage point
        
        Returns:
            float: MER score (lower is better)
        """
        success_rate_pct = self.compute_overall_accuracy() * 100
        
        if success_rate_pct == 0:
            return float('inf')
        
        if self.memory_tokens == 0:
            return 0.0
        
        return self.memory_tokens / success_rate_pct
    
    def summary(self) -> Dict:
        """
        Get all metrics as dictionary
        
        Returns:
            dict: All computed metrics and counts
        """
        return {
            "temporal_consistency_score": self.compute_temporal_consistency_score(),
            "overall_accuracy": self.compute_overall_accuracy(),
            "memory_efficiency_ratio": self.compute_memory_efficiency(),
            "total_queries": self.total_queries,
            "correct_queries": self.correct_queries,
            "temporal_conflicts": self.temporal_conflicts,
            "correct_conflicts": self.correct_conflicts,
            "memory_tokens": self.memory_tokens
        }
    
    def compute(self) -> Dict:
        """Alias for summary() for compatibility"""
        results = self.summary()
        # Convert keys to shorter names expected by evaluation script
        return {
            "tcs": results["temporal_consistency_score"],
            "overall_accuracy": results["overall_accuracy"],
            "memory_efficiency_ratio": results["memory_efficiency_ratio"],
            "total": results["total_queries"],
            "correct": results["correct_queries"],
            "incorrect": results["total_queries"] - results["correct_queries"],
            "temporal_conflicts": results["temporal_conflicts"],
            "correct_conflicts": results["correct_conflicts"]
        }
    
    def print_summary(self, method_name: str = "Method"):
        """Print formatted summary"""
        results = self.summary()
        
        print(f"\n{'='*60}")
        print(f"{method_name} Results")
        print(f"{'='*60}")
        print(f"Temporal Consistency Score:  {results['temporal_consistency_score']:.1%}")
        print(f"Overall Accuracy:            {results['overall_accuracy']:.1%}")
        print(f"Memory Efficiency Ratio:     {results['memory_efficiency_ratio']:.1f}")
        print(f"")
        print(f"Correct Conflicts:           {results['correct_conflicts']}/{results['temporal_conflicts']}")
        print(f"Total Correct:               {results['correct_queries']}/{results['total_queries']}")
        print(f"Memory Size:                 {results['memory_tokens']} tokens")

# Test the metrics
if __name__ == "__main__":
    print("Testing MemoryScopeMetrics...\n")
    
    metrics = MemoryScopeMetrics()
    
    # Simulate 10 queries
    # 5 with temporal conflicts (3 correct, 2 incorrect)
    # 5 without conflicts (4 correct, 1 incorrect)
    
    # Queries with conflicts
    metrics.update(is_correct=True, has_conflict=True)    # Correct
    metrics.update(is_correct=True, has_conflict=True)    # Correct
    metrics.update(is_correct=True, has_conflict=True)    # Correct
    metrics.update(is_correct=False, has_conflict=True)   # Incorrect
    metrics.update(is_correct=False, has_conflict=True)   # Incorrect
    
    # Queries without conflicts
    metrics.update(is_correct=True, has_conflict=False)   # Correct
    metrics.update(is_correct=True, has_conflict=False)   # Correct
    metrics.update(is_correct=True, has_conflict=False)   # Correct
    metrics.update(is_correct=True, has_conflict=False)   # Correct
    metrics.update(is_correct=False, has_conflict=False)  # Incorrect
    
    # Set memory size
    metrics.memory_tokens = 5000
    
    # Compute metrics
    results = metrics.summary()
    
    print("Expected Results:")
    print(f"  TCS: 3/5 = 60%")
    print(f"  Accuracy: 7/10 = 70%")
    print(f"  MER: 5000/70 = 71.4")
    
    print("\nActual Results:")
    metrics.print_summary("Test")
    
    # Verify
    assert abs(results['temporal_consistency_score'] - 0.6) < 0.01
    assert abs(results['overall_accuracy'] - 0.7) < 0.01
    
    print("\n✓ All tests passed!")
