"""
bayesian-llm-guard
Epistemic Uncertainty Estimation & Guardrails for Agentic LLMs.
"""

from .core import (
    UQGuardConfig,
    UQEngine,
    uq_guard,
    UQGuardException,
    AgentSelfCorrectionLoop
)

__version__ = "0.1.0"
__all__ = [
    "UQGuardConfig",
    "UQEngine",
    "uq_guard",
    "UQGuardException",
    "AgentSelfCorrectionLoop"
]