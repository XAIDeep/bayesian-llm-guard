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
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.fc(x))


class ConstantModel(nn.Module):
    """Model with deterministic output to validate zero-variance behavior."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_uq_engine_variance_calculation():
    """Test if the engine correctly calculates variance using batched tensors."""
    model = MockBayesianModel()
    engine = UQEngine(model=model, mc_samples=10)

    input_tensor = torch.randn(1, 10)
    variance = engine.estimate_variance(input_tensor)

    assert isinstance(variance, float)
    assert variance > 0.0


@pytest.mark.parametrize(
    "mc_samples, expected_error",
    [
        (0, ValueError),
        (1, ValueError),
        (-2, ValueError),
    ],
)
def test_uq_engine_rejects_invalid_mc_samples(mc_samples, expected_error):
    """Reject invalid Monte Carlo sample counts before running inference."""
    model = MockBayesianModel()

    with pytest.raises(expected_error, match="mc_samples"):
        UQEngine(model=model, mc_samples=mc_samples)


@pytest.mark.parametrize(
    "bad_input",
    [
        torch.empty(0),
        torch.tensor(1.0),
    ],
)
def test_uq_engine_rejects_invalid_inputs(bad_input):
    """Reject invalid, empty, or malformed input tensors."""
    engine = UQEngine(model=MockBayesianModel(), mc_samples=10)

    with pytest.raises((TypeError, ValueError)):
        engine.estimate_variance(bad_input)


def test_uq_engine_accepts_vector_inputs():
    """A single feature vector is a valid batch of one sample."""
    engine = UQEngine(model=MockBayesianModel(), mc_samples=10)

    input_tensor = torch.randn(10)
    variance = engine.estimate_variance(input_tensor)

    assert isinstance(variance, float)
    assert variance >= 0.0


def test_uq_engine_handles_zero_variance_model():
    """Deterministic models should produce zero variance, not a crash."""
    model = ConstantModel()
    engine = UQEngine(model=model, mc_samples=10)

    input_tensor = torch.randn(2, 3)
    variance = engine.estimate_variance(input_tensor)

    assert variance == pytest.approx(0.0, abs=1e-8)


def test_uq_engine_rejects_nan_and_inf_values():
    """Reject inputs that would produce unstable numerical outputs."""
    engine = UQEngine(model=MockBayesianModel(), mc_samples=10)

    with pytest.raises(ValueError, match="NaN|infinite"):
        engine.estimate_variance(torch.tensor([[1.0, float("nan")]]))

    with pytest.raises(ValueError, match="infinite"):
        engine.estimate_variance(torch.tensor([[1.0, float("inf")]]))