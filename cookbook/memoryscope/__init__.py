"""
MemoryScope Implementation Package

This package implements RSPM (Recursive Self-Pruning Memory) for the MemoryScope benchmark.

Components:
- data_loader: Dataset loading and preprocessing
- metrics: Evaluation metrics (TCS, accuracy, MER)
- baselines: Baseline implementations (Standard RAG, Recency-Weighted, etc.)
- sleep_cycle: Memory consolidation logic
- advanced_rspm_agent: Advanced RSPM agent with >95% TCS techniques
"""

__version__ = "0.1.0"

from .data_loader import MemoryScopeDataset
from .metrics import MemoryScopeMetrics

__all__ = [
    'MemoryScopeDataset',
    'MemoryScopeMetrics',
]
