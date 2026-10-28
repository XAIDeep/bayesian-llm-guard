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
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module.")
        if not isinstance(mc_samples, int):
            raise TypeError("mc_samples must be an integer.")
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2 to compute an unbiased variance estimate.")

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
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("input_tensor must be a torch.Tensor.")
        if input_tensor.numel() == 0:
            raise ValueError("input_tensor must not be empty.")
        if input_tensor.dim() == 0:
            raise ValueError("input_tensor must be at least 1-dimensional.")
        if not torch.is_floating_point(input_tensor):
            raise TypeError("input_tensor must have a floating point dtype.")
        if torch.isnan(input_tensor).any():
            raise ValueError("input_tensor contains NaN values.")
        if torch.isinf(input_tensor).any():
            raise ValueError("input_tensor contains infinite values.")

        self._enable_dropout()

        with torch.no_grad():
            repeated_tensor = input_tensor.unsqueeze(0).repeat(
                self.mc_samples,
                *([1] * input_tensor.dim()),
            )
            predictions = self.model(repeated_tensor)

        if not isinstance(predictions, torch.Tensor):
            raise TypeError("model output must be a torch.Tensor.")
        if predictions.numel() == 0:
            raise ValueError("model output must not be empty.")
        if predictions.ndim == 0:
            raise ValueError("model output must have at least one dimension.")

        variances = torch.var(predictions, dim=0, unbiased=True)
        max_variance = variances.max()

        if not torch.isfinite(max_variance):
            raise ValueError("Computed variance is not finite; check model output and input tensor.")

        return float(max_variance.item())