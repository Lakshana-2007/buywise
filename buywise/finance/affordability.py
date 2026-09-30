"""
Affordability calculator and deterministic safety validator.

This module is the financial safety gate of BUYWISE.

The LLM proposes → the Financial Twin simulates → THIS MODULE VALIDATES.

Every candidate strategy must pass all applicable checks before it can
be recommended.  The validator returns structured results, not just a
boolean — so explanations can cite exactly which constraint was satisfied
or violated.

Design:
- Pure functions; no I/O; no side effects.
- All constraints are explicit and testable in isolation.
- The validator never modifies strategies; it only evaluates them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from buywise.core.config import FORECAST_DAYS, MAX_SPENDING_CHANGES
from buywise.core.models import FinancialProfile, PaymentOption
from buywise.finance.events import NormalizedEvent
from buywise.finance.forecast import SimulationResult, simulate_payment_plan
from buywise.graph.state import CandidateStrategy, PaymentEntry, PaymentPlan, SpendingChange

logger = logging.getLogger(__name__)


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Structured result from the deterministic safety validator.

    is_valid=True only when ALL applicable checks pass.
    violations holds the list of failed check names with human-readable reasons.
    """
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    minimum_projected_balance: Optional[float] = None
    required_minimum_balance: float = 0.0

    # Individual check results
    checks: dict[str, bool] = field(default_factory=dict)

    def add_violation(self, check: str, reason: str) -> None:
        self.checks[check] = False
        self.violations.append(f"{check}: {reason}")
        self.is_valid = False

    def add_pass(self, check: str) -> None:
        self.checks[check] = True

    def __repr__(self) -> str:
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"violations={self.violations}, "
            f"min_projected={self.minimum_projected_balance:.2f if self.minimum_projected_balance else 'N/A'})"
        )


# ── Installment eligibility ───────────────────────────────────────────────────

def installment_duration_months(option: PaymentOption) -> float:
    """
    Calculate the duration of an installment plan in months.
    Uses (number_of_payments - 1) * payment_frequency_days / 30.
    A single payment has duration 0.
    """
    if option.number_of_payments <= 1 or not option.payment_frequency_days:
        return 0.0
    total_days = (option.number_of_payments - 1) * option.payment_frequency_days
    return total_days / 30.0


def is_installment_option_eligible(
    option: PaymentOption,
    profile: FinancialProfile,
    desired_completion_date: date,
) -> tuple[bool, str]:
    """
    Check whether an installment option is eligible for this user and request.

    Rules:
    1. User must accept 'installments' in payment_methods_user_will_consider.
    2. The plan must not exceed max_installment_months (if set).
    3. The last payment date must be <= desired_completion_date.

    Returns (is_eligible, reason_if_not).
    """
    if "installments" not in profile.payment_methods_user_will_consider:
        return False, "user does not accept installments"

    # Check max_installment_months
    if profile.max_installment_months is not None:
        duration = installment_duration_months(option)
        if duration > profile.max_installment_months:
            return False, (
                f"plan duration {duration:.1f} months exceeds "
                f"user max {profile.max_installment_months} months"
            )

    # Check completion deadline
    payment_dates = option.payment_dates()
    if payment_dates:
        last_payment = max(payment_dates)
        if last_payment > desired_completion_date:
            return False, (
                f"last payment {last_payment} is after "
                f"desired_completion_date {desired_completion_date}"
            )

    return True, ""


# ── Partial payment eligibility ───────────────────────────────────────────────

def is_partial_payment_eligible(
    request_allows: bool,
    profile: FinancialProfile,
    amount_safe_to_pay: float,
    requested_amount: float,
    earliest_full_payment_date: Optional[date],
    desired_completion_date: date,
) -> tuple[bool, str]:
    """
    Check whether a partial payment recommendation is eligible.

    Spec rules (all must hold):
    1. request.allows_partial_payment == True
    2. user accepts 'partial_payment'
    3. 0 < amount_safe_to_pay < requested_amount
    4. earliest_full_payment_date is not None
    5. earliest_full_payment_date <= desired_completion_date
    """
    if not request_allows:
        return False, "request does not allow partial payment"

    if "partial_payment" not in profile.payment_methods_user_will_consider:
        return False, "user does not accept partial_payment"

    if amount_safe_to_pay <= 0:
        return False, f"amount_safe_to_pay={amount_safe_to_pay:.2f} must be > 0"

    if amount_safe_to_pay >= requested_amount:
        return False, (
            f"amount_safe_to_pay={amount_safe_to_pay:.2f} >= "
            f"requested_amount={requested_amount:.2f}; full payment is available"
        )

    if earliest_full_payment_date is None:
        return False, "no safe date for full payment within forecast window"

    if earliest_full_payment_date > desired_completion_date:
        return False, (
            f"earliest_full_payment_date {earliest_full_payment_date} is after "
            f"desired_completion_date {desired_completion_date}"
        )

    return True, ""


# ── Spending change eligibility ───────────────────────────────────────────────

def validate_spending_changes(
    changes: list[SpendingChange],
    events_by_id: dict[str, NormalizedEvent],
    profile: FinancialProfile,
) -> tuple[bool, list[str]]:
    """
    Validate that all spending changes are permitted.

    Rules:
    1. At most MAX_SPENDING_CHANGES (3) changes.
    2. Each event must exist and be a recurring expense.
    3. Stop changes: event must be stoppable; category must be in
       expense_categories_user_is_willing_to_stop.
    4. Reduce changes: event must be reducible; category must be in
       expense_categories_user_is_willing_to_reduce; new_amount >= minimum_allowed.
    5. Same event cannot be both stopped AND reduced.
    """
    violations = []

    if len(changes) > MAX_SPENDING_CHANGES:
        violations.append(f"too many spending changes: {len(changes)} > {MAX_SPENDING_CHANGES}")

    # Check stop/reduce mutual exclusivity for same event
    stop_events = {c.event_id for c in changes if c.action == "stop"}
    reduce_events = {c.event_id for c in changes if c.action == "reduce_to"}
    overlap = stop_events & reduce_events
    if overlap:
        violations.append(f"events {overlap} are both stopped and reduced (mutually exclusive)")

    for change in changes:
        event = events_by_id.get(change.event_id)
        if event is None:
            violations.append(f"event {change.event_id} not found")
            continue

        if not event.is_recurring:
            violations.append(f"event {change.event_id} is not a recurring event")

        if not event.is_expense:
            violations.append(f"event {change.event_id} is not an expense (cannot be modified)")

        if change.action == "stop":
            if not event.is_stoppable:
                violations.append(
                    f"event {change.event_id} (flexibility={event.flexibility}) is not stoppable"
                )
            if event.category not in profile.expense_categories_user_is_willing_to_stop:
                violations.append(
                    f"category {event.category} of {change.event_id} is not in user's willing-to-stop list"
                )

        elif change.action == "reduce_to":
            if not event.is_reducible:
                violations.append(
                    f"event {change.event_id} (flexibility={event.flexibility}) is not reducible"
                )
            if event.category not in profile.expense_categories_user_is_willing_to_reduce:
                violations.append(
                    f"category {event.category} of {change.event_id} is not in user's willing-to-reduce list"
                )
            if change.new_amount is None:
                violations.append(f"reduce_to:{change.event_id} has no new_amount")
            elif event.minimum_allowed_home is not None and change.new_amount < event.minimum_allowed_home:
                violations.append(
                    f"reduce_to:{change.event_id}:{change.new_amount:.2f} < "
                    f"minimum_allowed {event.minimum_allowed_home:.2f}"
                )
            elif change.new_amount is not None and change.new_amount >= event.amount_home:
                violations.append(
                    f"reduce_to:{change.event_id}:{change.new_amount:.2f} >= "
                    f"current amount {event.amount_home:.2f} (no reduction)"
                )

    return len(violations) == 0, violations


# ── Main strategy validator ───────────────────────────────────────────────────

def validate_strategy(
    strategy: CandidateStrategy,
    profile: FinancialProfile,
    events: list[NormalizedEvent],
    request_date: date,
    requested_amount: float,
    desired_completion_date: date,
    allows_partial_payment: bool,
    amount_safe_to_pay: float,
    earliest_full_payment_date: Optional[date],
    payment_option: Optional[PaymentOption] = None,
    forecast_days: int = FORECAST_DAYS,
) -> ValidationResult:
    """
    Deterministically validate a candidate strategy.

    Checks:
    1. amount_safe_to_pay bounds: 0 <= amount_safe_to_pay <= requested_amount
    2. payment_method is in user's accepted methods
    3. spending_changes are valid
    4. essential expenses are protected
    5. installment plan exactly matches supplied option
    6. partial payment rules
    7. plan completes by desired_completion_date
    8. minimum balance never breached (simulation)
    9. max spending changes <= 3

    Returns ValidationResult.
    """
    result = ValidationResult(
        is_valid=True,
        required_minimum_balance=profile.minimum_balance_to_keep,
    )

    events_by_id = {e.event_id: e for e in events}

    # ── Check 1: amount_safe_to_pay bounds ────────────────────────────────────
    if amount_safe_to_pay < 0 or amount_safe_to_pay > requested_amount:
        result.add_violation(
            "amount_bounds",
            f"amount_safe_to_pay={amount_safe_to_pay:.2f} must be in [0, {requested_amount:.2f}]",
        )
    else:
        result.add_pass("amount_bounds")

    # ── Check 2: payment method accepted by user ──────────────────────────────
    method = strategy.payment_method
    if method not in ("wait", "not_recommended"):
        if method not in profile.payment_methods_user_will_consider:
            result.add_violation(
                "payment_method",
                f"method {method!r} not in user's accepted methods "
                f"{profile.payment_methods_user_will_consider}",
            )
        else:
            result.add_pass("payment_method")
    else:
        result.add_pass("payment_method")

    # ── Check 3: spending changes valid ───────────────────────────────────────
    if strategy.spending_changes:
        sc_valid, sc_violations = validate_spending_changes(
            strategy.spending_changes, events_by_id, profile
        )
        if sc_valid:
            result.add_pass("spending_changes")
        else:
            for v in sc_violations:
                result.add_violation("spending_changes", v)
    else:
        result.add_pass("spending_changes")

    # ── Check 4: max spending changes ────────────────────────────────────────
    if len(strategy.spending_changes) > MAX_SPENDING_CHANGES:
        result.add_violation(
            "max_spending_changes",
            f"{len(strategy.spending_changes)} changes exceed max {MAX_SPENDING_CHANGES}",
        )
    else:
        result.add_pass("max_spending_changes")

    # ── Check 5: plan completes by desired_completion_date ───────────────────
    if method not in ("wait", "not_recommended"):
        last_payment = strategy.payment_plan.last_payment_date
        if last_payment is None:
            result.add_violation("completion_deadline", "payment plan is empty")
        elif last_payment > desired_completion_date:
            result.add_violation(
                "completion_deadline",
                f"last payment {last_payment} is after desired {desired_completion_date}",
            )
        else:
            result.add_pass("completion_deadline")
    else:
        result.add_pass("completion_deadline")

    # ── Check 6: installment plan matches supplied option ─────────────────────
    if method == "installments":
        if payment_option is None:
            result.add_violation("installment_match", "no payment option provided for installments")
        else:
            # Verify payment count
            expected_count = payment_option.number_of_payments
            actual_count = strategy.payment_plan.num_payments
            if actual_count != expected_count:
                result.add_violation(
                    "installment_match",
                    f"plan has {actual_count} payments but option specifies {expected_count}",
                )
            else:
                result.add_pass("installment_match")

            # Verify each payment amount (within 0.02 tolerance for rounding)
            for entry in strategy.payment_plan.entries:
                diff = abs(entry.amount - payment_option.payment_amount)
                if diff > 0.02:
                    result.add_violation(
                        "installment_amount",
                        f"payment {entry.amount:.2f} doesn't match option amount "
                        f"{payment_option.payment_amount:.2f}",
                    )
                    break
            else:
                result.add_pass("installment_amount")

            # Verify option eligibility (duration, deadline)
            eligible, reason = is_installment_option_eligible(
                payment_option, profile, desired_completion_date
            )
            if not eligible:
                result.add_violation("installment_eligibility", reason)
            else:
                result.add_pass("installment_eligibility")

    # ── Check 7: partial payment rules ────────────────────────────────────────
    if method == "partial_payment":
        eligible, reason = is_partial_payment_eligible(
            allows_partial_payment,
            profile,
            amount_safe_to_pay,
            requested_amount,
            earliest_full_payment_date,
            desired_completion_date,
        )
        if not eligible:
            result.add_violation("partial_payment_rules", reason)
        else:
            # Verify two payments that sum to requested_amount
            entries = strategy.payment_plan.entries
            if len(entries) != 2:
                result.add_violation(
                    "partial_payment_rules",
                    f"partial payment must have exactly 2 payments, got {len(entries)}",
                )
            else:
                total = sum(e.amount for e in entries)
                diff = abs(total - requested_amount)
                if diff > 0.02:
                    result.add_violation(
                        "partial_payment_rules",
                        f"payments sum {total:.2f} != requested_amount {requested_amount:.2f}",
                    )
                else:
                    result.add_pass("partial_payment_rules")

    # ── Check 8: minimum balance never breached (simulation) ──────────────────
    if method not in ("not_recommended",) and strategy.payment_plan.entries:
        # Build spending change lookup
        sc_lookup: dict[str, Optional[float]] = {}
        for sc in strategy.spending_changes:
            sc_lookup[sc.event_id] = sc.new_amount if sc.action == "reduce_to" else None

        payment_schedule = [
            (e.payment_date, e.amount) for e in strategy.payment_plan.entries
        ]

        sim = simulate_payment_plan(
            starting_balance=profile.current_available_balance,
            minimum_balance=profile.minimum_balance_to_keep,
            events=events,
            request_date=request_date,
            payment_schedule=payment_schedule,
            spending_changes=sc_lookup,
            forecast_days=forecast_days,
        )

        result.minimum_projected_balance = sim.minimum_projected_balance

        if not sim.is_safe:
            result.add_violation(
                "minimum_balance",
                f"projected balance {sim.minimum_projected_balance:.2f} < "
                f"required {profile.minimum_balance_to_keep:.2f}",
            )
        else:
            result.add_pass("minimum_balance")
    else:
        result.add_pass("minimum_balance")

    return result


# ── Affordability status determination ────────────────────────────────────────

def determine_affordability_status(
    amount_safe_to_pay: float,
    requested_amount: float,
    earliest_full_payment_date: Optional[date],
    request_date: date,
    profile: FinancialProfile,
    has_valid_strategy: bool,
    selected_method: str = "",
) -> str:
    """
    Determine the affordability_status enum value.

    Rules (in priority order):
    - affordable_now: full amount is safe on request_date AND user accepts full_payment
    - affordable_with_plan: full amount completable via partial/installments/spending_changes
      (NOT wait — wait maps to affordable_later)
    - affordable_later: full amount becomes safe after request_date within forecast window
    - not_affordable: none of the above
    """
    # affordable_now: can pay full amount today AND user accepts full_payment
    if (
        amount_safe_to_pay >= requested_amount - 0.01
        and earliest_full_payment_date == request_date
        and "full_payment" in profile.payment_methods_user_will_consider
    ):
        return "affordable_now"

    # affordable_with_plan: a valid non-wait plan exists that completes the full request
    if has_valid_strategy and selected_method not in ("", "wait", "not_recommended"):
        return "affordable_with_plan"

    # affordable_later: full payment becomes safe later (covers wait strategy too)
    if (
        earliest_full_payment_date is not None
        and earliest_full_payment_date > request_date
        and "full_payment" in profile.payment_methods_user_will_consider
    ):
        return "affordable_later"

    return "not_affordable"
