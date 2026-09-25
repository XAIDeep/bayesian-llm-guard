"""
Unit tests for the UQEngine module.
Ensures the mathematical variance calculation works accurately with tensor batching.
"""

import pytest
import torch
import torch.nn as nn
from bayesian_llm_guard import UQEngine


class MockBayesianModel(nn.Module):
    """A simple neural network with heavy dropout to simulate epistemic uncertainty."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        # High probability dropout to guarantee variance > 0 during tests
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.fc(x))


def test_uq_engine_variance_calculation():
    """Test if the engine correctly calculates variance using batched tensors."""
    model = MockBayesianModel()
    engine = UQEngine(model=model, mc_samples=10)
    
    # Mocking a batch of input features from an LLM (Batch size 1, Feature dim 10)
    input_tensor = torch.randn(1, 10)
    variance = engine.estimate_variance(input_tensor)
    
    # Variance should be a float
    assert isinstance(variance, float)
    
    # Because of the 50% dropout, the variance across 10 passes MUST be strictly positive
    assert variance > 0.0