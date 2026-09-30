"""
Deterministic strategy ranking.

Ranks valid candidate strategies according to the exact priority order
specified in the problem statement.  The LLM plays NO role here.

Official ranking order (lower sort key = better):
  1. Completes the full request by desired_completion_date.
  2. Requires no spending changes.
  3. Minimizes total amount paid.
  4. Starts payment earlier.
  5. Uses fewer payments.
  6. Lowest payment_option_id as final tie-breaker.

A strategy is eligible for wait only if user accepts full_payment.
not_recommended is always the last resort (rank = infinity).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from buywise.graph.state import CandidateStrategy

logger = logging.getLogger(__name__)

_NEVER = date(9999, 12, 31)   # Sentinel for strategies with no payments


def _sort_key(
    strategy: CandidateStrategy,
    desired_completion_date: date,
) -> tuple:
    """
    Build a tuple sort key for a candidate strategy.

    Lower is better.

    Dimensions (all ascending — smaller = preferred):
      0: not_recommended last (0 = real strategy, 1 = not_recommended)
      1: completes by desired_completion_date (0 = yes, 1 = no)
      2: requires spending changes (0 = no, 1 = yes)
      3: total amount paid (float)
      4: first payment date (date — earlier is better)
      5: number of payments (int — fewer is better)
      6: payment_option_id (str — lexicographic, lower is better)
    """
    if strategy.payment_method == "not_recommended":
        return (1, 1, 1, float("inf"), _NEVER, 9999, "zzz")

    # Completion by deadline
    last_date = strategy.payment_plan.last_payment_date or _NEVER
    completes_by_deadline = 0 if last_date <= desired_completion_date else 1

    # Spending changes
    has_changes = 1 if strategy.spending_changes else 0

    # Total paid
    total_paid = strategy.payment_plan.total_amount

    # First payment date
    first_date = strategy.payment_plan.first_payment_date or _NEVER

    # Number of payments
    n_payments = strategy.payment_plan.num_payments

    # Tie-breaker: payment_option_id (lower wins)
    option_id = strategy.payment_option_id or ""

    return (0, completes_by_deadline, has_changes, total_paid, first_date, n_payments, option_id)


def rank_strategies(
    strategies: list[CandidateStrategy],
    desired_completion_date: date,
) -> list[CandidateStrategy]:
    """
    Sort valid candidate strategies by the official priority order.

    Returns the strategies in ranked order (best first).
    Attaches rank and rank_reason to each strategy.
    """
    if not strategies:
        return []

    ranked = sorted(
        strategies,
        key=lambda s: _sort_key(s, desired_completion_date),
    )

    for i, strat in enumerate(ranked):
        strat.rank = i + 1
        strat.rank_reason = _describe_rank_reason(strat, desired_completion_date)

    logger.debug("Ranked %d strategies. Winner: %s", len(ranked), ranked[0].payment_method)
    return ranked


def _describe_rank_reason(
    strategy: CandidateStrategy,
    desired_completion_date: date,
) -> str:
    """Generate a short human-readable description of why this strategy ranked where it did."""
    if strategy.payment_method == "not_recommended":
        return "no safe payment option available"

    reasons = []
    last_date = strategy.payment_plan.last_payment_date
    if last_date and last_date <= desired_completion_date:
        reasons.append("completes by deadline")
    else:
        reasons.append("misses deadline")

    if not strategy.spending_changes:
        reasons.append("no spending changes needed")
    else:
        changes = ", ".join(sc.to_string() for sc in strategy.spending_changes)
        reasons.append(f"requires {len(strategy.spending_changes)} spending change(s): {changes}")

    reasons.append(f"total={strategy.payment_plan.total_amount:.2f}")
    reasons.append(f"{strategy.payment_plan.num_payments} payment(s)")
    return "; ".join(reasons)


def select_best_strategy(
    valid_strategies: list[CandidateStrategy],
    desired_completion_date: date,
) -> CandidateStrategy:
    """
    Select the best strategy from a list of valid candidates.

    If no valid strategies exist, returns a not_recommended fallback.
    """
    from buywise.finance.strategies import make_not_recommended_strategy

    if not valid_strategies:
        fallback = make_not_recommended_strategy()
        fallback.rank = 1
        fallback.rank_reason = "no safe payment option available"
        return fallback

    ranked = rank_strategies(valid_strategies, desired_completion_date)
    return ranked[0]
