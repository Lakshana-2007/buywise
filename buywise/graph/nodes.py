"""
LangGraph pipeline nodes.

Each node is a pure function: (FinancialState) → FinancialState.

Node pipeline:
  evidence_extraction_node
  → event_normalization_node
  → financial_twin_node
  → strategy_generation_node
  → strategy_ranking_node
  → safety_proof_node
  → explanation_node
  → output_assembly_node

The pipeline is deterministic except for evidence_extraction_node and
explanation_node (both LLM-backed). Every financial calculation is
performed in pure Python nodes.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from buywise.ai.extractor import DecisionExplainer, EvidenceExtractor
from buywise.ai.provider import LLMProvider
from buywise.core.models import OutputRow
from buywise.finance.affordability import (
    determine_affordability_status,
    validate_strategy,
)
from buywise.finance.events import normalize_events
from buywise.finance.forecast import run_financial_twin
from buywise.finance.ranking import select_best_strategy
from buywise.finance.strategies import generate_and_validate_strategies, make_not_recommended_strategy
from buywise.graph.state import (
    CandidateStrategy,
    FinancialState,
    PaymentEntry,
    PaymentPlan,
    SafetyProof,
    SpendingChange,
)

logger = logging.getLogger(__name__)


# ── Node 1: Evidence extraction ───────────────────────────────────────────────

def evidence_extraction_node(state: FinancialState, extractor: EvidenceExtractor) -> FinancialState:
    """
    Extract structured evidence from messages and images.
    Populates state.evidence_graph.
    """
    ctx = state.context
    try:
        graph = extractor.extract(
            request_id=ctx.request_id,
            user_id=ctx.user_id,
            messages=ctx.messages,
            images=ctx.images,
            events_by_id=ctx.user_events and {e.event_id: e for e in ctx.user_events} or {},
            home_currency=ctx.home_currency,
        )
        state.evidence_graph = graph
        if graph.has_suspicious_instructions():
            state.add_warning(
                f"Instruction injection detected in source for {ctx.request_id}"
            )
    except Exception as exc:
        logger.error("Evidence extraction failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Evidence extraction failed: {exc}")
        # Continue with empty evidence graph — pipeline is still functional

    return state


# ── Node 2: Event normalization ───────────────────────────────────────────────

def event_normalization_node(state: FinancialState) -> FinancialState:
    """
    Normalize raw financial events using the evidence graph.
    Populates state.normalized_events.
    """
    ctx = state.context
    try:
        state.normalized_events = normalize_events(
            events=ctx.user_events,
            home_currency=ctx.home_currency,
            exchange_rates=ctx.exchange_rates,
            evidence_graph=state.evidence_graph,
            events_by_id={e.event_id: e for e in ctx.user_events},
        )
    except Exception as exc:
        logger.error("Event normalization failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Event normalization failed: {exc}")
        state.normalized_events = []

    return state


# ── Node 3: Financial Twin ────────────────────────────────────────────────────

def financial_twin_node(state: FinancialState) -> FinancialState:
    """
    Run the 90-day Financial Twin.
    Populates state.amount_safe_to_pay and state.earliest_full_payment_date.
    """
    ctx = state.context
    profile = ctx.profile
    try:
        twin = run_financial_twin(
            starting_balance=profile.current_available_balance,
            minimum_balance=profile.minimum_balance_to_keep,
            events=state.normalized_events,
            request_date=ctx.request_date,
            requested_amount=ctx.requested_amount,
            desired_completion_date=ctx.desired_completion_date,
        )
        state.amount_safe_to_pay = twin.amount_safe_to_pay
        state.earliest_full_payment_date = twin.earliest_full_payment_date
        # Store daily balance for potential use in explanation
        state.daily_balance_projection = {
            str(k): v for k, v in twin.base_simulation.daily_balance.items()
        }
    except Exception as exc:
        logger.error("Financial Twin failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Financial Twin failed: {exc}")
        state.amount_safe_to_pay = 0.0
        state.earliest_full_payment_date = None

    return state


# ── Node 4: Strategy generation ───────────────────────────────────────────────

def strategy_generation_node(state: FinancialState) -> FinancialState:
    """
    Generate and validate all candidate strategies.
    Populates state.candidate_strategies and state.valid_strategies.
    """
    ctx = state.context
    profile = ctx.profile
    try:
        valid = generate_and_validate_strategies(
            profile=profile,
            events=state.normalized_events,
            request_date=ctx.request_date,
            requested_amount=ctx.requested_amount,
            desired_completion_date=ctx.desired_completion_date,
            allows_partial_payment=ctx.request.allows_partial_payment,
            payment_options=ctx.payment_options,
        )
        state.valid_strategies = valid
        state.candidate_strategies = valid  # All generated are valid
    except Exception as exc:
        logger.error("Strategy generation failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Strategy generation failed: {exc}")
        state.valid_strategies = []

    return state


# ── Node 5: Strategy ranking ──────────────────────────────────────────────────

def strategy_ranking_node(state: FinancialState) -> FinancialState:
    """
    Select the best strategy from valid candidates.
    Populates state.selected_strategy.
    """
    ctx = state.context
    try:
        state.selected_strategy = select_best_strategy(
            state.valid_strategies,
            ctx.desired_completion_date,
        )
    except Exception as exc:
        logger.error("Strategy ranking failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Strategy ranking failed: {exc}")
        state.selected_strategy = make_not_recommended_strategy()

    return state


# ── Node 6: Safety proof ──────────────────────────────────────────────────────

def safety_proof_node(state: FinancialState) -> FinancialState:
    """
    Build the machine-generated safety proof for the selected strategy.
    Populates state.safety_proof.
    """
    ctx = state.context
    profile = ctx.profile
    strategy = state.selected_strategy

    if strategy is None:
        strategy = make_not_recommended_strategy()
        state.selected_strategy = strategy

    try:
        # Find the payment option for installments
        payment_option = None
        if strategy.payment_method == "installments" and strategy.payment_option_id:
            for opt in ctx.payment_options:
                if opt.payment_option_id == strategy.payment_option_id:
                    payment_option = opt
                    break

        # Run final validation to get proof data
        validation = validate_strategy(
            strategy=strategy,
            profile=profile,
            events=state.normalized_events,
            request_date=ctx.request_date,
            requested_amount=ctx.requested_amount,
            desired_completion_date=ctx.desired_completion_date,
            allows_partial_payment=ctx.request.allows_partial_payment,
            amount_safe_to_pay=state.amount_safe_to_pay,
            earliest_full_payment_date=state.earliest_full_payment_date,
            payment_option=payment_option,
        )

        min_proj = validation.minimum_projected_balance
        if min_proj is None:
            min_proj = profile.current_available_balance

        proof = SafetyProof(
            requested_amount=ctx.requested_amount,
            amount_safe_to_pay=state.amount_safe_to_pay,
            strategy=strategy.payment_method,
            payment_plan_string=strategy.payment_plan.to_string(),
            minimum_projected_balance=min_proj,
            required_minimum_balance=profile.minimum_balance_to_keep,
            plan_completion_date=strategy.payment_plan.last_payment_date,
            desired_completion_date=ctx.desired_completion_date,
            spending_changes=strategy.spending_changes,
            minimum_balance_ok=validation.checks.get("minimum_balance", True),
            completion_deadline_ok=validation.checks.get("completion_deadline", True),
            payment_method_ok=validation.checks.get("payment_method", True),
            essential_expense_ok=validation.checks.get("spending_changes", True),
            spending_change_count_ok=validation.checks.get("max_spending_changes", True),
            amount_bounds_ok=validation.checks.get("amount_bounds", True),
        )
        state.safety_proof = proof
    except Exception as exc:
        logger.error("Safety proof failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Safety proof failed: {exc}")

    return state


# ── Node 7: Explanation generation ────────────────────────────────────────────

def explanation_node(state: FinancialState, explainer: DecisionExplainer) -> FinancialState:
    """
    Generate the human-readable decision_explanation.
    Uses LLM to write a grounded explanation from the safety proof.
    """
    ctx = state.context
    profile = ctx.profile
    strategy = state.selected_strategy or make_not_recommended_strategy()
    proof = state.safety_proof

    # Determine affordability status from results
    has_valid_real_strategy = bool(state.valid_strategies) and any(
        s.payment_method != "not_recommended" for s in state.valid_strategies
    )
    affordability_status = determine_affordability_status(
        amount_safe_to_pay=state.amount_safe_to_pay,
        requested_amount=ctx.requested_amount,
        earliest_full_payment_date=state.earliest_full_payment_date,
        request_date=ctx.request_date,
        profile=profile,
        has_valid_strategy=has_valid_real_strategy,
        selected_method=strategy.payment_method,
    )

    try:
        explanation = explainer.explain(
            requested_amount=ctx.requested_amount,
            currency=ctx.home_currency,
            request_type=ctx.request.request_type,
            request_date=str(ctx.request_date),
            desired_completion_date=str(ctx.desired_completion_date),
            affordability_status=affordability_status,
            recommended_payment_method=strategy.payment_method,
            amount_safe_to_pay=state.amount_safe_to_pay,
            payment_plan=strategy.payment_plan.to_string(),
            earliest_date=str(state.earliest_full_payment_date) if state.earliest_full_payment_date else "",
            spending_changes=strategy.spending_changes_string(),
            min_projected_balance=proof.minimum_projected_balance if proof else profile.minimum_balance_to_keep,
            required_minimum_balance=profile.minimum_balance_to_keep,
            all_constraints_pass=proof.all_constraints_pass if proof else False,
        )
        state.decision_explanation = explanation
        state.explanation_fallback_used = bool(
            getattr(explainer, "last_used_fallback", False)
        )
    except Exception as exc:
        logger.error("Explanation failed for %s: %s", ctx.request_id, exc)
        state.explanation_fallback_used = True
        state.decision_explanation = (
            f"{affordability_status.replace('_', ' ').title()} — "
            f"see payment plan for details."
        )

    # Store affordability_status on state for output assembly
    state._affordability_status = affordability_status  # type: ignore[attr-defined]
    return state


# ── Node 8: Output assembly ───────────────────────────────────────────────────

def output_assembly_node(state: FinancialState) -> FinancialState:
    """
    Assemble the final OutputRow from all pipeline outputs.
    This is the last node — populates state.output_row.
    """
    ctx = state.context
    profile = ctx.profile
    strategy = state.selected_strategy or make_not_recommended_strategy()
    proof = state.safety_proof

    # Affordability status (may have been computed in explanation node)
    has_valid_real_strategy = bool(state.valid_strategies) and any(
        s.payment_method != "not_recommended" for s in state.valid_strategies
    )
    affordability_status = getattr(state, "_affordability_status", None) or determine_affordability_status(
        amount_safe_to_pay=state.amount_safe_to_pay,
        requested_amount=ctx.requested_amount,
        earliest_full_payment_date=state.earliest_full_payment_date,
        request_date=ctx.request_date,
        profile=profile,
        has_valid_strategy=has_valid_real_strategy,
        selected_method=strategy.payment_method,
    )

    # Clamp amount_safe_to_pay to [0, requested_amount]
    amount_safe = max(0.0, min(state.amount_safe_to_pay, ctx.requested_amount))
    amount_safe = round(amount_safe, 2)

    # Earliest date string
    if affordability_status == "affordable_now":
        earliest_str = str(ctx.request_date)
    elif affordability_status == "not_affordable":
        earliest_str = ""  # Spec: leave empty when full payment never safe within forecast
    elif state.earliest_full_payment_date:
        earliest_str = str(state.earliest_full_payment_date)
    else:
        earliest_str = ""

    try:
        state.output_row = OutputRow(
            request_id=ctx.request_id,
            amount_safe_to_pay=amount_safe,
            affordability_status=affordability_status,
            recommended_payment_method=strategy.payment_method,
            payment_plan=strategy.payment_plan.to_string(),
            earliest_date_for_full_payment=earliest_str,
            spending_changes_needed=strategy.spending_changes_string(),
            decision_explanation=state.decision_explanation or "See payment plan for details.",
        )
    except Exception as exc:
        logger.error("Output assembly failed for %s: %s", ctx.request_id, exc)
        state.add_error(f"Output assembly failed: {exc}")

    return state
