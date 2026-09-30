"""
90-Day Financial Twin — deterministic balance simulator.

This is the financial heart of BUYWISE. It answers two key questions:

  1. amount_safe_to_pay: How much can the user safely pay RIGHT NOW
     (on request_date) without the balance ever dropping below minimum
     during the next 90 days?

  2. earliest_full_payment_date: What is the first date in the forecast
     window when the user can safely pay the FULL requested amount as a
     single payment?

Algorithm
---------
For each candidate date D in [request_date, request_date + 90 days]:
  1. Build a projected cash-flow for the 90-day window starting from D,
     assuming a payment of `requested_amount` is made on D.
  2. For each day T in the window:
       projected_balance[T] = projected_balance[T-1]
                              + confirmed_income_on_T
                              - essential_expenses_on_T
                              - mandatory_payments_on_T
                              - purchase_payment_on_T    (if T == D)
                              - other_confirmed_debits_on_T
  3. If min(projected_balance) >= minimum_balance_to_keep → D is safe.

For the `amount_safe_to_pay` calculation we use D = request_date and
binary-search over the payment amount to find the maximum safe amount.

Key spec rules enforced here:
- Current balance is the starting point.
- Pending debits are reserved (deducted on their expected date).
- Pending credits are NOT counted.
- Scheduled salary IS counted on its settlement_date.
- Recurring events are projected forward from their last known date
  using day-of-month and ~30-day interval.
- No future income is invented beyond confirmed scheduled events.
- Spending changes (stop/reduce) are applied before simulation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from buywise.core.config import FORECAST_DAYS
from buywise.finance.events import NormalizedEvent

logger = logging.getLogger(__name__)


# ── Day entry ─────────────────────────────────────────────────────────────────

@dataclass
class DayEntry:
    """
    Events projected onto a single calendar day.
    Credits (income) increase the balance; debits decrease it.
    """
    income: float = 0.0
    expenses: float = 0.0

    @property
    def net(self) -> float:
        return self.income - self.expenses


# ── Simulation result ─────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Full result of one Financial Twin simulation run.
    """
    start_date: date
    end_date: date
    starting_balance: float
    minimum_balance: float
    payment_amount: float       # The payment tested on start_date
    payment_date: date          # When this payment was applied

    # Daily balance projection: date -> end-of-day balance
    daily_balance: dict[date, float] = field(default_factory=dict)

    @property
    def minimum_projected_balance(self) -> float:
        if not self.daily_balance:
            return self.starting_balance
        return min(self.daily_balance.values())

    @property
    def is_safe(self) -> bool:
        return self.minimum_projected_balance >= self.minimum_balance

    @property
    def balance_on_payment_date(self) -> float:
        return self.daily_balance.get(self.payment_date, self.starting_balance)

    def balance_on(self, d: date) -> float:
        return self.daily_balance.get(d, self.starting_balance)

    def __repr__(self) -> str:
        return (
            f"SimulationResult(payment={self.payment_amount:.2f} on {self.payment_date}, "
            f"min_balance={self.minimum_projected_balance:.2f}, "
            f"safe={self.is_safe})"
        )


# ── Projection builder ────────────────────────────────────────────────────────

def _build_cash_flow(
    events: list[NormalizedEvent],
    start_date: date,
    end_date: date,
    spending_changes: Optional[dict[str, float | None]] = None,
) -> dict[date, DayEntry]:
    """
    Project cash flows from `events` into a day-keyed dict for
    [start_date, end_date] (inclusive).

    spending_changes: dict mapping event_id -> new_amount (or None = stop).
    Recurring events are projected forward using their day-of-month.
    Non-recurring future events are placed on their effective_date.

    Returns dict[date -> DayEntry].
    """
    flow: dict[date, DayEntry] = defaultdict(DayEntry)
    sc = spending_changes or {}

    # Determine unique recurring templates (latest occurrence per category+direction+day)
    # Use the modal (most common) amount for projection, not the latest event's amount
    recurring_templates: dict[tuple, NormalizedEvent] = {}
    recurring_modal_amounts: dict[tuple, float] = {}

    # First pass: group by key, find modal amount
    from collections import defaultdict as _defaultdict
    recurring_groups: dict[tuple, list[NormalizedEvent]] = _defaultdict(list)
    for e in events:
        if e.is_recurring:
            key = (e.category, e.direction, e.recurrence_day)
            recurring_groups[key].append(e)

    for key, group in recurring_groups.items():
        # Pick the most recent template event (for date tracking)
        latest = max(group, key=lambda e: e.effective_date)
        recurring_templates[key] = latest
        # Find modal amount (most common amount in the series)
        from collections import Counter as _Counter
        # Round to nearest integer for grouping near-equal amounts
        amount_counts = _Counter(round(e.amount_home, 0) for e in group if e.amount_home > 0)
        modal_amount = amount_counts.most_common(1)[0][0] if amount_counts else latest.amount_home
        recurring_modal_amounts[key] = float(modal_amount)

    # ── Project recurring events ──────────────────────────────────────────────
    for key, template in recurring_templates.items():
        # Use modal amount for projection (most representative amount in series)
        modal_amount = recurring_modal_amounts[key]

        # Apply spending change if applicable
        if template.event_id in sc:
            new_amt = sc[template.event_id]
            if new_amt is None:
                continue  # stopped
            effective_amount = new_amt
        else:
            effective_amount = modal_amount

        if effective_amount <= 0:
            continue

        # Generate occurrences in the forecast window
        # Use day-of-month anchoring
        recur_day = template.recurrence_day or template.effective_date.day

        # Start projecting from the month AFTER the last known occurrence
        # to avoid double-counting already-settled events in the history
        last_known = template.effective_date
        projection_start = _next_occurrence(last_known, recur_day)

        d = projection_start
        while d <= end_date:
            if d >= start_date:
                entry = flow[d]
                if template.is_income:
                    entry.income += effective_amount
                else:
                    entry.expenses += effective_amount
            d = _next_monthly(d, recur_day)

    # ── Project non-recurring future events ───────────────────────────────────
    for e in events:
        if e.is_recurring:
            continue  # already handled above

        event_date = e.effective_date
        if event_date < start_date or event_date > end_date:
            continue

        # Apply spending change
        if e.event_id in sc:
            new_amt = sc[e.event_id]
            if new_amt is None:
                continue
            effective_amount = new_amt
        else:
            effective_amount = e.amount_home

        if effective_amount <= 0:
            continue

        entry = flow[event_date]
        if e.is_income:
            entry.income += effective_amount
        else:
            entry.expenses += effective_amount

    return flow


def _next_occurrence(last_date: date, day_of_month: int) -> date:
    """Return the first occurrence AFTER last_date on day_of_month."""
    # Move to next month
    year = last_date.year
    month = last_date.month + 1
    if month > 12:
        month = 1
        year += 1
    return _safe_date(year, month, day_of_month)


def _next_monthly(d: date, day_of_month: int) -> date:
    """Advance one month, keeping the same day-of-month."""
    month = d.month + 1
    year = d.year
    if month > 12:
        month = 1
        year += 1
    return _safe_date(year, month, day_of_month)


def _safe_date(year: int, month: int, day: int) -> date:
    """Clamp day to the last valid day of the month (handles Feb 29/30 etc.)."""
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, max_day))


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(
    starting_balance: float,
    minimum_balance: float,
    events: list[NormalizedEvent],
    request_date: date,
    payment_amount: float,
    payment_date: date,
    spending_changes: Optional[dict[str, float | None]] = None,
    forecast_days: int = FORECAST_DAYS,
) -> SimulationResult:
    """
    Run one simulation: apply `payment_amount` on `payment_date` and
    project the balance for `forecast_days` from `request_date`.

    Parameters
    ----------
    starting_balance    : Current available balance on request_date.
    minimum_balance     : Balance must never fall below this.
    events              : Normalized events (post-filtering).
    request_date        : The evaluation start date.
    payment_amount      : The payment to test.
    payment_date        : When the payment is made (>= request_date).
    spending_changes    : {event_id: new_amount_or_None} (stop=None, reduce=amount).
    forecast_days       : How many days to project.

    Returns SimulationResult with daily_balance populated.
    """
    end_date = request_date + timedelta(days=forecast_days)

    # Build projected cash flows (without the purchase payment)
    flow = _build_cash_flow(events, request_date, end_date, spending_changes)

    # Run day-by-day simulation
    balance = starting_balance
    daily_balance: dict[date, float] = {}

    current = request_date
    while current <= end_date:
        day_entry = flow.get(current, DayEntry())

        # Apply income
        balance += day_entry.income

        # Deduct expenses
        balance -= day_entry.expenses

        # Apply the purchase payment on its scheduled date
        if current == payment_date:
            balance -= payment_amount

        daily_balance[current] = balance
        current += timedelta(days=1)

    return SimulationResult(
        start_date=request_date,
        end_date=end_date,
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        payment_amount=payment_amount,
        payment_date=payment_date,
        daily_balance=daily_balance,
    )


# ── Financial Twin ────────────────────────────────────────────────────────────

@dataclass
class FinancialTwinResult:
    """Output of the Financial Twin for one request."""
    amount_safe_to_pay: float           # Max safe payment on request_date (no spending changes)
    earliest_full_payment_date: Optional[date]  # First date full amount is safe
    base_simulation: SimulationResult   # Simulation with zero payment (baseline)
    full_payment_simulation: Optional[SimulationResult]  # Simulation at amount_safe_to_pay


def run_financial_twin(
    starting_balance: float,
    minimum_balance: float,
    events: list[NormalizedEvent],
    request_date: date,
    requested_amount: float,
    desired_completion_date: date,
    spending_changes: Optional[dict[str, float | None]] = None,
    forecast_days: int = FORECAST_DAYS,
) -> FinancialTwinResult:
    """
    Run the Financial Twin for a request.

    Computes:
    1. amount_safe_to_pay: binary search for the maximum safe amount on
       request_date without any spending changes (unless provided).
    2. earliest_full_payment_date: first date in forecast window where
       the full requested_amount can be safely paid.

    Parameters
    ----------
    spending_changes : Optional spending changes to apply (for strategy evaluation).
                       When None, computes the baseline (no changes).
    """
    end_date = request_date + timedelta(days=forecast_days)

    # ── Step 1: Find amount_safe_to_pay ──────────────────────────────────────
    # Binary search: find max amount where simulation on request_date is safe
    amount_safe = _binary_search_safe_amount(
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        events=events,
        request_date=request_date,
        max_amount=requested_amount,
        spending_changes=spending_changes,
        forecast_days=forecast_days,
    )

    # ── Step 2: Base simulation (zero payment for baseline projection) ────────
    base_sim = simulate(
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        events=events,
        request_date=request_date,
        payment_amount=0.0,
        payment_date=request_date,
        spending_changes=spending_changes,
        forecast_days=forecast_days,
    )

    # ── Step 3: Find earliest_full_payment_date ───────────────────────────────
    # Spec: earliest date when full requested_amount passes safety check
    # WITHOUT spending changes (independent of selected strategy)
    earliest = _find_earliest_full_payment_date(
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        events=events,
        request_date=request_date,
        requested_amount=requested_amount,
        end_date=end_date,
        spending_changes=None,  # Always without spending changes per spec
        forecast_days=forecast_days,
    )

    # ── Step 4: Simulation at found safe amount ───────────────────────────────
    full_sim: Optional[SimulationResult] = None
    if amount_safe > 0:
        full_sim = simulate(
            starting_balance=starting_balance,
            minimum_balance=minimum_balance,
            events=events,
            request_date=request_date,
            payment_amount=amount_safe,
            payment_date=request_date,
            spending_changes=spending_changes,
            forecast_days=forecast_days,
        )

    return FinancialTwinResult(
        amount_safe_to_pay=amount_safe,
        earliest_full_payment_date=earliest,
        base_simulation=base_sim,
        full_payment_simulation=full_sim,
    )


def _binary_search_safe_amount(
    starting_balance: float,
    minimum_balance: float,
    events: list[NormalizedEvent],
    request_date: date,
    max_amount: float,
    spending_changes: Optional[dict[str, float | None]],
    forecast_days: int,
) -> float:
    """
    Binary search for the maximum amount that can be safely paid on request_date.

    Returns value in [0.0, max_amount] rounded to 2 decimal places.
    """
    # Quick check: can we afford the full amount immediately?
    result = simulate(
        starting_balance, minimum_balance, events, request_date,
        max_amount, request_date, spending_changes, forecast_days,
    )
    if result.is_safe:
        return round(max_amount, 2)

    # Quick check: can we afford zero? (baseline safety)
    result_zero = simulate(
        starting_balance, minimum_balance, events, request_date,
        0.0, request_date, spending_changes, forecast_days,
    )
    if not result_zero.is_safe:
        # Even with no payment, balance would breach minimum
        # (extreme edge case — essential expenses exceed balance)
        return 0.0

    # Binary search between 0 and max_amount
    lo, hi = 0.0, max_amount
    best = 0.0
    iterations = 0
    while hi - lo > 0.01 and iterations < 50:
        mid = (lo + hi) / 2.0
        result = simulate(
            starting_balance, minimum_balance, events, request_date,
            mid, request_date, spending_changes, forecast_days,
        )
        if result.is_safe:
            best = mid
            lo = mid
        else:
            hi = mid
        iterations += 1

    return round(best, 2)


def _find_earliest_full_payment_date(
    starting_balance: float,
    minimum_balance: float,
    events: list[NormalizedEvent],
    request_date: date,
    requested_amount: float,
    end_date: date,
    spending_changes: Optional[dict[str, float | None]],
    forecast_days: int,
) -> Optional[date]:
    """
    Find the first date >= request_date where paying requested_amount
    as a single payment would be safe for the next 90 days.

    Returns None if no such date exists within the forecast window.

    Spec: this is calculated WITHOUT spending changes (it measures
    raw financial capacity, not strategy-dependent capacity).
    """
    # The forecast window runs forecast_days from request_date.
    # A payment date is only meaningful if it falls within this window.
    forecast_end = request_date + timedelta(days=forecast_days)
    # Use the tighter of end_date and forecast_end as the search boundary
    search_end = min(end_date, forecast_end)

    # First check request_date itself
    result = simulate(
        starting_balance, minimum_balance, events, request_date,
        requested_amount, request_date, None, forecast_days,
    )
    if result.is_safe:
        return request_date

    # Walk forward day by day up to search_end
    current = request_date + timedelta(days=1)
    while current <= search_end:
        result = simulate(
            starting_balance, minimum_balance, events, request_date,
            requested_amount, current, None, forecast_days,
        )
        if result.is_safe:
            return current
        current += timedelta(days=1)

    return None  # Not affordable within forecast window


# ── Strategy-aware simulation ─────────────────────────────────────────────────

def simulate_payment_plan(
    starting_balance: float,
    minimum_balance: float,
    events: list[NormalizedEvent],
    request_date: date,
    payment_schedule: list[tuple[date, float]],   # [(date, amount), ...]
    spending_changes: Optional[dict[str, float | None]] = None,
    forecast_days: int = FORECAST_DAYS,
) -> SimulationResult:
    """
    Simulate a multi-payment plan (installments, partial payment).

    Each payment in payment_schedule is applied on its specified date.
    All payments must be within the forecast window.

    Returns a SimulationResult where is_safe tells you whether the entire
    plan keeps the balance above minimum throughout.
    """
    if not payment_schedule:
        return simulate(starting_balance, minimum_balance, events,
                        request_date, 0.0, request_date, spending_changes, forecast_days)

    end_date = request_date + timedelta(days=forecast_days)

    # Build cash flow
    flow = _build_cash_flow(events, request_date, end_date, spending_changes)

    # Run simulation
    balance = starting_balance
    daily_balance: dict[date, float] = {}

    # Payment schedule as a lookup
    payments_by_date: dict[date, float] = defaultdict(float)
    for pdate, pamount in payment_schedule:
        payments_by_date[pdate] += pamount

    current = request_date
    while current <= end_date:
        day_entry = flow.get(current, DayEntry())
        balance += day_entry.income
        balance -= day_entry.expenses
        if current in payments_by_date:
            balance -= payments_by_date[current]
        daily_balance[current] = balance
        current += timedelta(days=1)

    # Use first payment date as the "payment_date" for result
    first_payment_date = min(d for d, _ in payment_schedule)
    total_payment = sum(a for _, a in payment_schedule)

    return SimulationResult(
        start_date=request_date,
        end_date=end_date,
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        payment_amount=total_payment,
        payment_date=first_payment_date,
        daily_balance=daily_balance,
    )
