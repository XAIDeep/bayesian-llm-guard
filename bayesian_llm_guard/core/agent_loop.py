"""
Agent Loop Module for bayesian-llm-guard.
Implements an autonomous self-correction loop to retry and refine 
queries when high uncertainty (hallucination) is detected.
"""

import logging
from typing import Any, Callable, Dict

from bayesian_llm_guard.core.guardrail import UQGuardException

logger = logging.getLogger("bayesian_agent_loop")


class AgentSelfCorrectionLoop:
    """
    Manages iterative query refinement and re-execution for RAG pipelines.
    Automatically catches uncertainty exceptions and refines the user's prompt.
    """

    def __init__(
        self, 
        max_retries: int = 3, 
        query_refiner: Callable[[str], str] = lambda q: q + " (provide more specific context)"
    ) -> None:
        """
        Args:
            max_retries (int): Maximum number of times to retry the pipeline.
            query_refiner (Callable): A function that modifies the failed query to improve retrieval.
        """
        self.max_retries = max_retries
        self.query_refiner = query_refiner

    def execute_with_fallback(
        self, 
        rag_pipeline_func: Callable[..., Dict[str, Any]], 
        initial_query: str, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes the provided RAG function. If it triggers the UQ Guardrail, 
        the query is refined and retried automatically.

        Args:
            rag_pipeline_func (Callable): The decorated RAG function to execute.
            initial_query (str): The original user prompt.
            
        Returns:
            Dict[str, Any]: The safe, validated result from the pipeline.
            
        Raises:
            RuntimeError: If the maximum number of retries is exhausted.
        """
        current_query = initial_query
        attempt = 0

        while attempt < self.max_retries:
            try:
                logger.info(f"Agent Loop Attempt {attempt + 1}/{self.max_retries}. Query: '{current_query}'")
                
                # Attempt to execute the pipeline
                return rag_pipeline_func(query=current_query, **kwargs)
                
            except UQGuardException as exception:
                attempt += 1
                logger.warning(f"Guardrail intercepted the response: {exception}")
                
                if attempt < self.max_retries:
                    logger.info("Refining query and retrying...")
                    current_query = self.query_refiner(current_query)

        # If the loop finishes without returning, the model could not produce a safe answer
        error_msg = (
            f"Agent loop failed after {self.max_retries} attempts. "
            "The model consistently exhibited high epistemic uncertainty. "
            "Please check your knowledge base or refine the original prompt."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)