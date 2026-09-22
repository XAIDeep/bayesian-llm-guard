"""
Configuration Module for bayesian-llm-guard.
Manages type-safe settings and environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class UQGuardConfig(BaseSettings):
    """
    Core configuration for the Bayesian Guardrail.
    Values can be overridden using environment variables with the 'UQ_' prefix.
    Example: export UQ_THRESHOLD=0.03
    """
    
    threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed epistemic variance. Values above this trigger the guardrail."
    )
    
    mc_samples: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of Monte Carlo forward passes for uncertainty estimation."
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of self-correction attempts when high uncertainty is detected."
    )

    class Config:
        env_prefix = "UQ_"
