# Bayesian LLM Guard

**Epistemic Uncertainty Estimation & Guardrails for Agentic LLMs and RAG Systems**

Bayesian LLM Guard is an enterprise-grade middleware designed to detect and prevent Large Language Model (LLM) hallucinations. By leveraging Vectorized Monte Carlo Dropout, it calculates the epistemic variance of neural network representations, allowing you to intercept uncertain or fabricated responses in real-time before they reach the user.

## Features

- **High-Throughput GPU Batching:** Utilizes tensor batching for zero-latency uncertainty quantification.
- **Deterministic Guardrails:** Decorator-based interception (`@uq_guard`) for clean integration into RAG pipelines.
- **Autonomous Self-Correction:** Seamlessly integrates with Agentic loops for query refinement upon hallucination detection.
- **Model Agnostic:** Works with any underlying PyTorch-based neural architecture.

## Installation

Stable release from PyPI:

```bash
pip install bayesian-llm-guard
```

## Quick Start

### 1. Basic Uncertainty Quantification

```python
import torch
import torch.nn as nn
from bayesian_llm_guard import UQEngine

# Initialize with your custom PyTorch model
model = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.5))
engine = UQEngine(model=model, mc_samples=30)

# Input features (e.g., embeddings from an LLM)
features = torch.randn(1, 768)

# Estimate epistemic variance
variance = engine.estimate_variance(features)
print(f"Model Uncertainty (Variance): {variance}")
```

### 2. Using the Guardrail Decorator

Integrate directly into your RAG or LLM generation pipeline to automatically block uncertain responses.

```python
from bayesian_llm_guard import UQGuardConfig, uq_guard, UQGuardException

config = UQGuardConfig(threshold=0.15)

@uq_guard(uq_engine=engine, config=config)
def generate_response(prompt: str):
    # Your RAG/LLM logic here
    # The function must return a dictionary containing 'tensor_features'
    return {
        "response": "The capital of France is Paris.",
        "tensor_features": torch.randn(1, 768)
    }

try:
    result = generate_response("What is the capital of France?")
    print(result["response"])
except UQGuardException as e:
    print(f"Blocked due to high uncertainty: {e}")
```

## Architecture

Developed and maintained by the **XAIDeep Research Team**. The core mathematical engine relies on stochastic forward passes with active dropout layers during inference, capturing the model's internal confidence distribution without requiring external APIs or secondary evaluation models.

## License

Apache License 2.0
