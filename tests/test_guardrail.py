"""
Unit tests for the Guardrail Decorator.
Verifies the interception logic and exception handling.
"""

import pytest
import torch
import torch.nn as nn
from bayesian_llm_guard import UQEngine, UQGuardConfig, uq_guard, UQGuardException


class MockNoisyModel(nn.Module):
    """Simulates a hallucinating model that outputs pure noise."""
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.9) 

    def forward(self, x):
        return self.dropout(x)


def test_guardrail_raises_exception_on_hallucination():
    """Test if the guardrail correctly blocks high-variance responses."""
    model = MockNoisyModel()
    engine = UQEngine(model=model, mc_samples=20)
    
    # Setting an extremely strict threshold so the guardrail definitely trips
    config = UQGuardConfig(threshold=0.0001) 

    @uq_guard(uq_engine=engine, config=config)
    def mock_rag_pipeline():
        """Simulating a function returning features from a Retrieval-Augmented Generation pipeline."""
        return {
            "tensor_features": torch.randn(1, 10), 
            "response": "This is a hallucinated fake answer."
        }

    # We expect the UQGuardException to be raised and block execution
    with pytest.raises(UQGuardException):
        mock_rag_pipeline()