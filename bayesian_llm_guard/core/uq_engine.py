"""
Uncertainty Quantification (UQ) Engine.
Implements Epistemic Uncertainty estimation using Monte Carlo Dropout.
"""

import torch
import torch.nn as nn


class UQEngine:
    """
    Executes multiple stochastic forward passes through a PyTorch model
    to calculate epistemic uncertainty (variance).
    """
    
    def __init__(self, model: nn.Module, mc_samples: int = 30) -> None:
        """
        Args:
            model (nn.Module): The PyTorch neural network model/layer.
            mc_samples (int): Number of stochastic passes to perform.
        """
        self.model = model
        self.mc_samples = mc_samples

    def _enable_dropout(self) -> None:
        """
        Forces dropout layers to remain active even when the model is in evaluation mode.
        This is required for Monte Carlo Dropout.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def estimate_variance(self, input_tensor: torch.Tensor) -> float:
        """
        Calculates the maximum epistemic variance for the given input.
        
        Args:
            input_tensor (torch.Tensor): The input data (e.g., embeddings or logits).
            
        Returns:
            float: The maximum variance score across all dimensions.
        """
        self._enable_dropout()
        
        with torch.no_grad():
            # Perform T stochastic forward passes
            # Using list comprehension for clean and readable code in the public version
            predictions = torch.stack([self.model(input_tensor) for _ in range(self.mc_samples)])
            
        # Calculate variance across the Monte Carlo samples (dim=0)
        variances = torch.var(predictions, dim=0, unbiased=True)
        
        # Return the maximum variance as the primary uncertainty score
        max_variance = variances.max().item()
        
        return float(max_variance)