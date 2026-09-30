"""
Strategy generation + counterfactual engine.

Generates every candidate payment strategy that is plausible for a given
request, validates each one deterministically, and returns a list of
valid candidates ready for ranking.

Strategy types generated:
1. full_payment   — pay requested_amount on request_date
2. installments   — iterate over every eligible installment option
3. partial_payment — pay amount_safe_to_pay now, rest on earliest_full_payment_date
4. wait           — pay full amount on earliest_full_payment_date
5. full_payment + spending_changes — spend-down variants (stop/reduce 1-3 events)
6. not_recommended — fallback when nothing else works

Counterfactual engine:
  If no strategy passes without spending changes, enumerate permutations of
  permitted flexible expenses (up to MAX_SPENDING_CHANGES changes) and
  re-run the Financial Twin + validator for each combination.

This module is PURELY DETERMINISTIC — no LLM calls here.
Every strategy it proposes is either accepted or rejected by the
deterministic validator.
"""

from __future__ import annotations

import itertools
import logging
from datetime import date
from typing import Optional

from buywise.core.config import FORECAST_DAYS, MAX_SPENDING_CHANGES
from buywise.core.models import FinancialProfile, PaymentOption
from buywise.finance.affordability import (
    is_installment_option_eligible,
    is_partial_payment_eligible,
    validate_strategy,
)
from buywise.finance.events import NormalizedEvent
from buywise.finance.forecast import run_financial_twin
from buywise.graph.state import (
    CandidateStrategy,
    PaymentEntry,
    PaymentPlan,
    SpendingChange,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plan(entries: list[tuple[date, float]], option_id: Optional[str] = None) -> PaymentPlan:
    plan = PaymentPlan(entries=[PaymentEntry(d, a) for d, a in entries])
    plan.payment_option_id = option_id
    return plan


def _sc_lookup(changes: list[SpendingChange]) -> dict[str, Optional[float]]:
    """Convert SpendingChange list to the dict format used by the simulator."""
    result: dict[str, Optional[float]] = {}
    for sc in changes:
        result[sc.event_id] = sc.new_amount if sc.action == "reduce_to" else None
    return result


def _get_stoppable_events(
    events: list[NormalizedEvent],
    profile: FinancialProfile,
) -> list[NormalizedEvent]:
    """Return recurring expenses that the user is willing to stop."""
    return [
        e for e in events
        if e.is_recurring
        and e.is_expense
        and e.is_stoppable
        and e.category in profile.expense_categories_user_is_willing_to_stop
    ]


def _get_reducible_events(
    events: list[NormalizedEvent],
    profile: FinancialProfile,
) -> list[NormalizedEvent]:
    """Return recurring expenses that the user is willing to reduce, with a viable reduction."""
    result = []
    for e in events:
        if (
            e.is_recurring
            and e.is_expense
            and e.is_reducible
            and e.category in profile.expense_categories_user_is_willing_to_reduce
        ):
            # Only include if there is room to reduce
            min_amt = e.minimum_allowed_home or 0.0
            if e.amount_home > min_amt:
                result.append(e)
    return result


def _typical_reduction_amount(event: NormalizedEvent) -> float:
    """
    For a reduce_to change, use minimum_allowed_home if set,
    otherwise reduce by 50% (floor at 0).
    """
    if event.minimum_allowed_home is not None:
        return event.minimum_allowed_home
    return round(event.amount_home * 0.5, 2)


# ── Core strategy builder ─────────────────────────────────────────────────────

def _build_full_payment_strategy(
    request_date: date,
    requested_amount: float,
    spending_changes: list[SpendingChange],
) -> CandidateStrategy:
    return CandidateStrategy(
        payment_method="full_payment",
        payment_plan=_make_plan([(request_date, requested_amount)]),
        spending_changes=spending_changes,
    )


def _build_wait_strategy(
    earliest_date: date,
    requested_amount: float,
) -> CandidateStrategy:
    return CandidateStrategy(
        payment_method="wait",
        payment_plan=_make_plan([(earliest_date, requested_amount)]),
        spending_changes=[],
    )


def _build_installment_strategy(
    option: PaymentOption,
) -> CandidateStrategy:
    schedule = [(d, option.payment_amount) for d in option.payment_dates()]
    return CandidateStrategy(
        payment_method="installments",
        payment_plan=_make_plan(schedule, option_id=option.payment_option_id),
        payment_option_id=option.payment_option_id,
        spending_changes=[],
    )


def _build_partial_strategy(
    request_date: date,
    amount_safe_to_pay: float,
    requested_amount: float,
    earliest_full_date: date,
) -> CandidateStrategy:
    remaining = round(requested_amount - amount_safe_to_pay, 2)
    return CandidateStrategy(
        payment_method="partial_payment",
        payment_plan=_make_plan([
            (request_date, amount_safe_to_pay),
            (earliest_full_date, remaining),
        ]),
        spending_changes=[],
    )


# ── Main strategy generator ───────────────────────────────────────────────────

def generate_and_validate_strategies(
    profile: FinancialProfile,
    events: list[NormalizedEvent],
    request_date: date,
    requested_amount: float,
    desired_completion_date: date,
    allows_partial_payment: bool,
    payment_options: list[PaymentOption],
    forecast_days: int = FORECAST_DAYS,
) -> list[CandidateStrategy]:
    """
    Generate all plausible strategies for a request and validate each one.

    Returns a list of CandidateStrategy objects where is_valid=True.

    Generation order:
    1. Full payment (no changes)
    2. All eligible installment options (no changes)
    3. Partial payment (no changes)
    4. Wait / full payment on earliest_date
    5. Spending-change variants: stop/reduce combinations × each of the above
    6. not_recommended (always added as last-resort fallback)

    The financial twin is re-run for spending-change variants to get the
    correct amount_safe_to_pay and earliest_date under the new expense profile.
    """
    valid_strategies: list[CandidateStrategy] = []

    # ── Baseline Financial Twin run (no spending changes) ─────────────────────
    baseline = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=events,
        request_date=request_date,
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        forecast_days=forecast_days,
    )
    amount_safe = baseline.amount_safe_to_pay
    earliest_full = baseline.earliest_full_payment_date

    # ── 1. Full payment (no changes) ──────────────────────────────────────────
    if "full_payment" in profile.payment_methods_user_will_consider:
        strat = _build_full_payment_strategy(request_date, requested_amount, [])
        result = validate_strategy(
            strategy=strat,
            profile=profile,
            events=events,
            request_date=request_date,
            requested_amount=requested_amount,
            desired_completion_date=desired_completion_date,
            allows_partial_payment=allows_partial_payment,
            amount_safe_to_pay=amount_safe,
            earliest_full_payment_date=earliest_full,
            forecast_days=forecast_days,
        )
        strat.is_valid = result.is_valid
        strat.validation_violations = result.violations
        strat.minimum_projected_balance = result.minimum_projected_balance
        if result.is_valid:
            valid_strategies.append(strat)

    # ── 2. Installment options (no changes) ───────────────────────────────────
    if "installments" in profile.payment_methods_user_will_consider:
        for opt in payment_options:
            if opt.payment_method != "installments":
                continue
            eligible, _ = is_installment_option_eligible(
                opt, profile, desired_completion_date
            )
            if not eligible:
                continue

            strat = _build_installment_strategy(opt)
            result = validate_strategy(
                strategy=strat,
                profile=profile,
                events=events,
                request_date=request_date,
                requested_amount=requested_amount,
                desired_completion_date=desired_completion_date,
                allows_partial_payment=allows_partial_payment,
                amount_safe_to_pay=amount_safe,
                earliest_full_payment_date=earliest_full,
                payment_option=opt,
                forecast_days=forecast_days,
            )
            strat.is_valid = result.is_valid
            strat.validation_violations = result.violations
            strat.minimum_projected_balance = result.minimum_projected_balance
            if result.is_valid:
                valid_strategies.append(strat)

    # ── 3. Partial payment (no changes) ───────────────────────────────────────
    if "partial_payment" in profile.payment_methods_user_will_consider:
        partial_eligible, _ = is_partial_payment_eligible(
            allows_partial_payment,
            profile,
            amount_safe,
            requested_amount,
            earliest_full,
            desired_completion_date,
        )
        if partial_eligible:
            strat = _build_partial_strategy(
                request_date, amount_safe, requested_amount, earliest_full  # type: ignore[arg-type]
            )
            result = validate_strategy(
                strategy=strat,
                profile=profile,
                events=events,
                request_date=request_date,
                requested_amount=requested_amount,
                desired_completion_date=desired_completion_date,
                allows_partial_payment=allows_partial_payment,
                amount_safe_to_pay=amount_safe,
                earliest_full_payment_date=earliest_full,
                forecast_days=forecast_days,
            )
            strat.is_valid = result.is_valid
            strat.validation_violations = result.violations
            strat.minimum_projected_balance = result.minimum_projected_balance
            if result.is_valid:
                valid_strategies.append(strat)

    # ── 4. Wait (no changes) ──────────────────────────────────────────────────
    if (
        "full_payment" in profile.payment_methods_user_will_consider
        and earliest_full is not None
        and earliest_full > request_date
        and earliest_full <= desired_completion_date
    ):
        strat = _build_wait_strategy(earliest_full, requested_amount)
        result = validate_strategy(
            strategy=strat,
            profile=profile,
            events=events,
            request_date=request_date,
            requested_amount=requested_amount,
            desired_completion_date=desired_completion_date,
            allows_partial_payment=allows_partial_payment,
            amount_safe_to_pay=amount_safe,
            earliest_full_payment_date=earliest_full,
            forecast_days=forecast_days,
        )
        strat.is_valid = result.is_valid
        strat.validation_violations = result.violations
        strat.minimum_projected_balance = result.minimum_projected_balance
        if result.is_valid:
            valid_strategies.append(strat)

    # ── 5. Spending-change variants (counterfactuals) ─────────────────────────
    # Only explore if no valid strategy found yet without spending changes.
    # Spending changes are a last resort — exhaustive search only when needed.
    # Also limit the candidate pool: only use the highest-impact events
    # (largest amount for stoppable, most savings for reducible).
    stoppable = _get_stoppable_events(events, profile)
    reducible = _get_reducible_events(events, profile)

    if valid_strategies:
        # Already have valid strategies without changes — skip expensive search
        pass
    elif stoppable or reducible:
        # Limit search space: top 3 stoppable + top 3 reducible by savings
        top_stoppable = sorted(stoppable, key=lambda e: e.amount_home, reverse=True)[:3]
        top_reducible = sorted(
            reducible,
            key=lambda e: e.amount_home - (e.minimum_allowed_home or 0),
            reverse=True,
        )[:3]

        spending_change_variants = _enumerate_spending_changes(
            top_stoppable, top_reducible, max_changes=MAX_SPENDING_CHANGES
        )

        for changes in spending_change_variants:
            sc_dict = _sc_lookup(changes)

            # Re-run Financial Twin with spending changes applied
            twin_with_changes = run_financial_twin(
                starting_balance=profile.current_available_balance,
                minimum_balance=profile.minimum_balance_to_keep,
                events=events,
                request_date=request_date,
                requested_amount=requested_amount,
                desired_completion_date=desired_completion_date,
                spending_changes=sc_dict,
                forecast_days=forecast_days,
            )
            safe_with_changes = twin_with_changes.amount_safe_to_pay
            earliest_with_changes = twin_with_changes.earliest_full_payment_date

            # Full payment + spending changes
            if "full_payment" in profile.payment_methods_user_will_consider:
                strat = _build_full_payment_strategy(request_date, requested_amount, changes)
                result = validate_strategy(
                    strategy=strat,
                    profile=profile,
                    events=events,
                    request_date=request_date,
                    requested_amount=requested_amount,
                    desired_completion_date=desired_completion_date,
                    allows_partial_payment=allows_partial_payment,
                    amount_safe_to_pay=safe_with_changes,
                    earliest_full_payment_date=earliest_with_changes,
                    forecast_days=forecast_days,
                )
                strat.is_valid = result.is_valid
                strat.validation_violations = result.violations
                strat.minimum_projected_balance = result.minimum_projected_balance
                if result.is_valid:
                    valid_strategies.append(strat)

            # Installments + spending changes
            if "installments" in profile.payment_methods_user_will_consider:
                for opt in payment_options:
                    if opt.payment_method != "installments":
                        continue
                    eligible, _ = is_installment_option_eligible(
                        opt, profile, desired_completion_date
                    )
                    if not eligible:
                        continue
                    strat = _build_installment_strategy(opt)
                    strat.spending_changes = changes
                    result = validate_strategy(
                        strategy=strat,
                        profile=profile,
                        events=events,
                        request_date=request_date,
                        requested_amount=requested_amount,
                        desired_completion_date=desired_completion_date,
                        allows_partial_payment=allows_partial_payment,
                        amount_safe_to_pay=safe_with_changes,
                        earliest_full_payment_date=earliest_with_changes,
                        payment_option=opt,
                        forecast_days=forecast_days,
                    )
                    strat.is_valid = result.is_valid
                    strat.validation_violations = result.violations
                    strat.minimum_projected_balance = result.minimum_projected_balance
                    if result.is_valid:
                        valid_strategies.append(strat)

            # Wait + spending changes
            if (
                "full_payment" in profile.payment_methods_user_will_consider
                and earliest_with_changes is not None
                and earliest_with_changes > request_date
                and earliest_with_changes <= desired_completion_date
            ):
                strat = _build_wait_strategy(earliest_with_changes, requested_amount)
                strat.spending_changes = changes
                result = validate_strategy(
                    strategy=strat,
                    profile=profile,
                    events=events,
                    request_date=request_date,
                    requested_amount=requested_amount,
                    desired_completion_date=desired_completion_date,
                    allows_partial_payment=allows_partial_payment,
                    amount_safe_to_pay=safe_with_changes,
                    earliest_full_payment_date=earliest_with_changes,
                    forecast_days=forecast_days,
                )
                strat.is_valid = result.is_valid
                strat.validation_violations = result.violations
                strat.minimum_projected_balance = result.minimum_projected_balance
                if result.is_valid:
                    valid_strategies.append(strat)

    logger.debug(
        "Generated %d valid strategies for request (balance=%.2f, requested=%.2f)",
        len(valid_strategies),
        profile.current_available_balance,
        requested_amount,
    )

    return valid_strategies


def _enumerate_spending_changes(
    stoppable: list[NormalizedEvent],
    reducible: list[NormalizedEvent],
    max_changes: int = MAX_SPENDING_CHANGES,
) -> list[list[SpendingChange]]:
    """
    Enumerate all valid spending change combinations up to max_changes.

    Returns a list of change-sets, each being a list of SpendingChange objects.
    The same event is never both stopped and reduced in the same combination.
    """
    # Build candidate individual changes
    stop_changes = [
        SpendingChange(action="stop", event_id=e.event_id) for e in stoppable
    ]
    reduce_changes = [
        SpendingChange(
            action="reduce_to",
            event_id=e.event_id,
            new_amount=_typical_reduction_amount(e),
        )
        for e in reducible
    ]

    all_single_changes = stop_changes + reduce_changes
    if not all_single_changes:
        return []

    results: list[list[SpendingChange]] = []
    seen: set[frozenset] = set()  # deduplicate by event_id set

    for n in range(1, min(max_changes, len(all_single_changes)) + 1):
        for combo in itertools.combinations(all_single_changes, n):
            # Ensure no event appears twice (stop + reduce same event)
            event_ids = [c.event_id for c in combo]
            if len(event_ids) != len(set(event_ids)):
                continue
            key = frozenset(f"{c.action}:{c.event_id}" for c in combo)
            if key in seen:
                continue
            seen.add(key)
            results.append(list(combo))

    return results


# ── Not-recommended fallback ──────────────────────────────────────────────────

def make_not_recommended_strategy() -> CandidateStrategy:
    """Return the fallback not_recommended strategy."""
    return CandidateStrategy(
        payment_method="not_recommended",
        payment_plan=PaymentPlan(entries=[]),
        spending_changes=[],
        is_valid=True,  # Always "valid" as a fallback
    )
