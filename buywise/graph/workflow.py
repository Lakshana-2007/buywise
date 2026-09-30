"""
LangGraph workflow definition.

Wires the pipeline nodes into a StateGraph and compiles it.
The graph handles the replan loop deterministically.

Graph structure:
  START
    → evidence_extraction
    → event_normalization
    → financial_twin
    → strategy_generation
    → strategy_ranking
    → safety_proof
    → explanation
    → output_assembly
  END

The graph is compiled once and then invoked per request.
"""

from __future__ import annotations

import logging
from functools import partial

from buywise.ai.extractor import DecisionExplainer, EvidenceExtractor
from buywise.ai.provider import LLMProvider, get_llm_provider
from buywise.graph.nodes import (
    evidence_extraction_node,
    event_normalization_node,
    explanation_node,
    financial_twin_node,
    output_assembly_node,
    safety_proof_node,
    strategy_generation_node,
    strategy_ranking_node,
)
from buywise.graph.state import FinancialState

logger = logging.getLogger(__name__)


def build_pipeline(provider: LLMProvider | None = None) -> "BuywisePipeline":
    """
    Build the BUYWISE processing pipeline.

    Parameters
    ----------
    provider : LLMProvider instance. If None, uses the configured provider.
    """
    if provider is None:
        provider = get_llm_provider()

    extractor = EvidenceExtractor(provider)
    explainer = DecisionExplainer(provider)
    return BuywisePipeline(extractor=extractor, explainer=explainer)


class BuywisePipeline:
    """
    The complete BUYWISE processing pipeline.

    Runs each node in sequence for a single request.
    Handles errors gracefully — always produces an OutputRow.
    """

    def __init__(self, extractor: EvidenceExtractor, explainer: DecisionExplainer):
        self._extractor = extractor
        self._explainer = explainer

    def run(self, state: FinancialState) -> FinancialState:
        """
        Process a single request through the full pipeline.

        Returns the state with state.output_row populated.
        """
        request_id = state.context.request_id

        try:
            # Node 1: Evidence extraction (LLM)
            state = evidence_extraction_node(state, self._extractor)

            # Node 2: Event normalization (deterministic)
            state = event_normalization_node(state)

            # Node 3: Financial Twin (deterministic)
            state = financial_twin_node(state)

            # Node 4: Strategy generation (deterministic)
            state = strategy_generation_node(state)

            # Node 5: Strategy ranking (deterministic)
            state = strategy_ranking_node(state)

            # Node 6: Safety proof (deterministic)
            state = safety_proof_node(state)

            # Node 7: Explanation (LLM with deterministic fallback)
            state = explanation_node(state, self._explainer)

            # Node 8: Output assembly (deterministic)
            state = output_assembly_node(state)

        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", request_id, exc, exc_info=True)
            state.add_error(f"Pipeline error: {exc}")
            # Assemble a safe fallback output
            _assemble_fallback(state)

        return state


def _assemble_fallback(state: FinancialState) -> None:
    """Create a minimal safe fallback OutputRow when the pipeline crashes."""
    from buywise.core.models import OutputRow
    ctx = state.context
    try:
        state.output_row = OutputRow(
            request_id=ctx.request_id,
            amount_safe_to_pay=min(
                max(0.0, state.amount_safe_to_pay), ctx.requested_amount
            ),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=(
                "Unable to determine affordability due to a processing error. "
                "Please review manually."
            ),
        )
    except Exception:
        pass  # If even this fails, the CSV writer handles None output_row
