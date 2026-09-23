"""
Uncertainty Quantification (UQ) Engine.
Implements Epistemic Uncertainty estimation using Monte Carlo Dropout.
Optimized for high-throughput GPU inference using Tensor Batching.
"""

import torch
import torch.nn as nn


class UQEngine:
    """
    Executes vectorized stochastic forward passes through a PyTorch model
    to calculate epistemic variance.
    """
    
    def __init__(self, model: nn.Module, mc_samples: int = 30) -> None:
        self.model = model
        self.mc_samples = mc_samples

    def _enable_dropout(self) -> None:
        """Forces dropout layers to remain active during evaluation mode."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def estimate_variance(self, input_tensor: torch.Tensor) -> float:
        """
        Calculates the maximum epistemic variance using vectorized Monte Carlo samples.
        """
        self._enable_dropout()
        
        with torch.no_grad():
            # ENTERPRISE BATCHING: Duplicate the tensor along the batch dimension.
            # If input is (1, 768), repeated_tensor becomes (30, 768).
            # This allows the GPU to process all Monte Carlo samples in parallel.
            repeat_dims = [self.mc_samples] + [1] * (input_tensor.dim() - 1)
            repeated_tensor = input_tensor.repeat(*repeat_dims)
            
            # Single highly-parallel forward pass
            predictions = self.model(repeated_tensor)
            
        # Calculate statistical variance across the batch dimension (dim=0)
        variances = torch.var(predictions, dim=0, unbiased=True)
        
        return float(variances.max().item())