"""
Guardrail Module for bayesian-llm-guard.
Provides the @uq_guard decorator to intercept LLM/RAG pipelines
and evaluate epistemic uncertainty before returning results.
"""

import logging
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional

from bayesian_llm_guard.core.uq_engine import UQEngine
from bayesian_llm_guard.core.config import UQGuardConfig

# Setup basic logging for the guardrail
logger = logging.getLogger("bayesian_guardrail")
logging.basicConfig(level=logging.INFO)


class UQGuardException(Exception):
    """
    Raised when the model's epistemic uncertainty exceeds the defined safety threshold.
    Indicates a potential hallucination.
    """
    pass


def uq_guard(
    uq_engine: UQEngine,
    config: Optional[UQGuardConfig] = None,
    fallback_action: Optional[Callable[..., Any]] = None,
) -> Callable[..., Callable[..., Any]]:
    """
    A decorator that wraps LLM/RAG execution functions.
    It intercepts the returned tensor features, calculates the variance using the UQEngine,
    and blocks the response if the model is too uncertain. Supports both sync and async functions.

    Args:
        uq_engine (UQEngine): The instantiated uncertainty quantification engine.
        config (UQGuardConfig, optional): Configuration containing the safety threshold.
        fallback_action (Callable, optional): A function to execute if the threshold is breached.
    """
    conf = config or UQGuardConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        
        # ---------------------------------------------------------
        # ASYNC WRAPPER
        # ---------------------------------------------------------
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                # Execute the underlying async LLM or RAG pipeline
                result = await func(*args, **kwargs)
                
                if isinstance(result, dict) and "tensor_features" in result:
                    features = result["tensor_features"]
                    max_variance = uq_engine.estimate_variance(features)

                    logger.info(
                        f"Calculated Epistemic Variance: {max_variance:.4f} "
                        f"(Threshold: {conf.threshold:.4f})"
                    )

                    if max_variance > conf.threshold:
                        logger.warning("Uncertainty threshold exceeded! High risk of hallucination.")
                        
                        if fallback_action:
                            logger.info("Executing defined fallback action.")
                            # Check if the fallback action itself is async
                            if inspect.iscoroutinefunction(fallback_action):
                                return await fallback_action(*args, **kwargs, uncertainty=max_variance)
                            return fallback_action(*args, **kwargs, uncertainty=max_variance)
                        
                        raise UQGuardException(
                            f"Safety block triggered: Uncertainty score {max_variance:.4f} "
                            f"is higher than the allowed threshold of {conf.threshold:.4f}."
                        )
                    
                    result["uq_metrics"] = {"epistemic_variance": max_variance}
                
                return result
            return async_wrapper

        # ---------------------------------------------------------
        # SYNC WRAPPER
        # ---------------------------------------------------------
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                # Execute the underlying sync LLM or RAG pipeline
                result = func(*args, **kwargs)
                
                if isinstance(result, dict) and "tensor_features" in result:
                    features = result["tensor_features"]
                    max_variance = uq_engine.estimate_variance(features)

                    logger.info(
                        f"Calculated Epistemic Variance: {max_variance:.4f} "
                        f"(Threshold: {conf.threshold:.4f})"
                    )

                    if max_variance > conf.threshold:
                        logger.warning("Uncertainty threshold exceeded! High risk of hallucination.")
                        
                        if fallback_action:
                            logger.info("Executing defined fallback action.")
                            return fallback_action(*args, **kwargs, uncertainty=max_variance)
                        
                        raise UQGuardException(
                            f"Safety block triggered: Uncertainty score {max_variance:.4f} "
                            f"is higher than the allowed threshold of {conf.threshold:.4f}."
                        )
                    
                    result["uq_metrics"] = {"epistemic_variance": max_variance}
                
                return result
            return sync_wrapper

    return decorator