"""
Deterministic financial event normalization.

Transforms the raw list of FinancialEvents for a user into a clean,
actionable set of NormalizedEvents ready for the 90-day simulator.

Pipeline:
1. IGNORE:  cancelled, failed, unrealized, non_cash events.
2. IGNORE:  pending CREDITS (spec: do not count pending credits).
3. IGNORE:  duplicate records (expense->expense via linked_event_id where
            parent is cancelled and child is the real settled record).
4. APPLY:   evidence amendments — update amounts, mark cancellations.
5. RESOLVE: blank amounts from evidence graph.
6. CONVERT: all amounts to home_currency using dated exchange rates.
7. DETECT:  recurrence (same category + amount + day-of-month, 3+ history).
8. CLASSIFY: recurring income / recurring expense / one-time / future-confirmed.

A NormalizedEvent is the canonical representation used by the Financial Twin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from buywise.core.enums import (
    IGNORED_STATUSES,
    EventDirection,
    EventFlexibility,
    EventStatus,
    EventType,
)
from buywise.core.models import ExchangeRate, FinancialEvent
from buywise.data.evidence import EvidenceGraph, EvidenceType, resolve_conflicts
from buywise.finance.currency import ConversionError, convert

logger = logging.getLogger(__name__)


# ── Normalized event ──────────────────────────────────────────────────────────

@dataclass
class NormalizedEvent:
    """
    A single financial event that has been cleaned, converted to home_currency,
    and classified for the cash-flow simulator.
    """
    event_id: str
    user_id: str
    event_type: str
    category: str
    direction: str          # "credit" | "debit"
    amount_home: float      # Amount in home_currency (always positive)
    original_amount: float  # Original amount before conversion
    original_currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    flexibility: str
    minimum_allowed_amount: Optional[float]  # In original_currency
    minimum_allowed_home: Optional[float]    # In home_currency
    description: str

    # Classification flags
    is_recurring: bool = False
    recurrence_day: Optional[int] = None     # Day-of-month for monthly recurrence
    recurrence_interval_days: Optional[int] = None  # ~30 for monthly

    # Source tracking
    amount_from_image: bool = False    # True if amount was extracted from image
    amount_from_message: bool = False  # True if amount was updated by a message

    @property
    def is_income(self) -> bool:
        return self.direction == EventDirection.CREDIT

    @property
    def is_expense(self) -> bool:
        return self.direction == EventDirection.DEBIT

    @property
    def is_protected(self) -> bool:
        """Protected events cannot be modified by spending changes."""
        return self.flexibility == EventFlexibility.FIXED

    @property
    def is_stoppable(self) -> bool:
        return self.flexibility in (
            EventFlexibility.STOPPABLE,
            EventFlexibility.REDUCIBLE_OR_STOPPABLE,
        )

    @property
    def is_reducible(self) -> bool:
        return self.flexibility in (
            EventFlexibility.REDUCIBLE,
            EventFlexibility.REDUCIBLE_OR_STOPPABLE,
        )

    @property
    def effective_date(self) -> date:
        """The date that matters for cash-flow: settlement_date when available."""
        return self.settlement_date if self.settlement_date else self.event_date

    def __repr__(self) -> str:
        rec = f"(recurring d{self.recurrence_day})" if self.is_recurring else ""
        return (
            f"NormalizedEvent({self.event_id}, {self.category}, "
            f"{'+' if self.is_income else '-'}{self.amount_home:.2f} {self.original_currency}, "
            f"status={self.status}, date={self.effective_date}{rec})"
        )


# ── Ignore-filter logic ───────────────────────────────────────────────────────

def _should_ignore(event: FinancialEvent) -> tuple[bool, str]:
    """
    Returns (should_ignore, reason).

    Spec rules:
    - Ignore: cancelled, failed, unrealized events.
    - Ignore: non_cash direction (investment valuations).
    - Ignore: pending CREDITS (do not count pending income).
    - Keep:   pending DEBITS (reserve them — conservative).
    - Keep:   settled, scheduled events.
    """
    # Ignore by status
    if event.status in IGNORED_STATUSES:
        return True, f"status={event.status}"

    # Ignore non-cash direction (investment valuations etc.)
    if event.direction == EventDirection.NON_CASH:
        return True, "non_cash direction"

    # Ignore pending credits (spec: do not count pending credits)
    if event.status == EventStatus.PENDING and event.direction == EventDirection.CREDIT:
        return True, "pending credit"

    # Ignore investment valuations (non-cash by type too)
    if event.event_type == EventType.INVESTMENT_VALUATION:
        return True, "investment_valuation (unrealized)"

    return False, ""


def _is_duplicate_record(
    event: FinancialEvent,
    events_by_id: dict[str, FinancialEvent],
) -> tuple[bool, str]:
    """
    Detect duplicate records linked via linked_event_id.

    Pattern 1: parent=cancelled, child=settled  → parent is a duplicate of child.
    Pattern 2: parent=settled/pending, child=pending + "duplicate" in description.
    """
    if not event.linked_event_id:
        return False, ""

    parent = events_by_id.get(event.linked_event_id)
    if parent is None:
        return False, ""

    # The child (current event) is the real settled version;
    # the parent was cancelled — parent is the duplicate
    # We mark the PARENT as the duplicate; child is kept.
    # This function is called on each event; we ignore cancelled parents
    # because _should_ignore already handles cancelled status.

    # Pattern 2: current event is pending and description says "duplicate"
    if (
        event.status == EventStatus.PENDING
        and "duplicate" in event.description.lower()
        and parent.status in (EventStatus.SETTLED, EventStatus.PENDING)
    ):
        return True, f"duplicate of {parent.event_id}"

    return False, ""


# ── Recurrence detection ──────────────────────────────────────────────────────

def _detect_recurrence(
    events: list[FinancialEvent],
) -> dict[str, tuple[bool, Optional[int], Optional[int]]]:
    """
    Detect recurring events from the history.

    An event is considered recurring if there are 3+ settled/scheduled
    occurrences of the same (category, direction) on the same day-of-month
    with the same or very similar amount (within 1% tolerance).

    Returns dict[event_id -> (is_recurring, recurrence_day, interval_days)].
    """
    from collections import defaultdict

    result: dict[str, tuple[bool, Optional[int], Optional[int]]] = {}

    # Group by (category, direction, day_of_month) for settled events
    groups: dict[tuple, list[FinancialEvent]] = defaultdict(list)
    for e in events:
        if e.status in (EventStatus.SETTLED, EventStatus.SCHEDULED) and e.direction != EventDirection.NON_CASH:
            day = e.event_date.day
            groups[(e.category, e.direction, day)].append(e)

    # For each group, check if amounts are consistent (within 5%)
    recurring_event_ids: set[str] = set()
    recurrence_info: dict[str, tuple[int, int]] = {}  # event_id -> (day, interval)

    for (cat, direction, day), group in groups.items():
        if len(group) < 3:
            continue

        # Check amount consistency within 5% of median
        # Use the mode-like approach: at least 60% of amounts must be within 5% of each other
        amounts = sorted(e.amount for e in group if e.amount is not None)
        if not amounts:
            continue

        # Find the most common amount cluster (within 5% band)
        best_count = 0
        best_representative = amounts[len(amounts) // 2]
        for candidate in amounts:
            band_members = [a for a in amounts if abs(a - candidate) / (candidate + 1e-9) <= 0.05]
            if len(band_members) > best_count:
                best_count = len(band_members)
                best_representative = candidate

        # Require at least 60% of occurrences to be consistent
        consistent = best_count / len(amounts) >= 0.6
        if not consistent:
            continue

        # Use best_representative as the canonical recurring amount
        # (mark ALL events in the group as recurring, not just the consistent ones)
        median_amt = best_representative

        # Mark all events in this group as recurring
        for e in group:
            recurring_event_ids.add(e.event_id)
            recurrence_info[e.event_id] = (day, 30)  # approximately monthly

    for e in events:
        if e.event_id in recurring_event_ids:
            day, interval = recurrence_info[e.event_id]
            result[e.event_id] = (True, day, interval)
        else:
            result[e.event_id] = (False, None, None)

    return result


# ── Main normalization function ───────────────────────────────────────────────

def normalize_events(
    events: list[FinancialEvent],
    home_currency: str,
    exchange_rates: list[ExchangeRate],
    evidence_graph: Optional[EvidenceGraph] = None,
    events_by_id: Optional[dict[str, FinancialEvent]] = None,
) -> list[NormalizedEvent]:
    """
    Normalize a user's raw financial events into the canonical form
    used by the 90-day Financial Twin.

    Parameters
    ----------
    events          : All raw events for this user (unsorted).
    home_currency   : User's home currency for conversion.
    exchange_rates  : Available exchange rates.
    evidence_graph  : Extracted evidence facts (amendments, cancellations).
    events_by_id    : Global event lookup for linked_event_id resolution.

    Returns a list of NormalizedEvents, sorted by effective_date ascending.
    """
    if events_by_id is None:
        events_by_id = {e.event_id: e for e in events}

    # Step 1: Collect evidence-based cancellations and amendments
    cancelled_by_evidence: set[str] = set()
    amended_amounts: dict[str, float] = {}  # event_id -> new amount in original currency

    if evidence_graph:
        resolved_facts = resolve_conflicts(evidence_graph.facts)
        for fact in resolved_facts:
            if fact.evidence_type == EvidenceType.EVENT_CANCELLED and fact.is_trustworthy():
                if fact.related_event_id:
                    cancelled_by_evidence.add(fact.related_event_id)

        for fact in resolved_facts:
            if fact.evidence_type == EvidenceType.EVENT_CANCELLED and fact.is_trustworthy():
                continue
            if fact.related_event_id:
                if fact.amount is not None and fact.evidence_type in (
                    EvidenceType.EVENT_AMENDED,
                    EvidenceType.SALARY_UPDATE,
                ) and fact.is_trustworthy():
                    # Only override if not already cancelled
                    if fact.related_event_id not in cancelled_by_evidence:
                        amended_amounts[fact.related_event_id] = fact.amount

    # Step 2: Detect recurrence on the raw event list (before filtering)
    recurrence_map = _detect_recurrence(events)

    # Step 3: Process each event
    normalized: list[NormalizedEvent] = []

    for event in events:
        # --- Ignore filters ---
        should_ignore, reason = _should_ignore(event)
        if should_ignore:
            logger.debug("Ignoring %s: %s", event.event_id, reason)
            continue

        # --- Evidence-based cancellation ---
        if event.event_id in cancelled_by_evidence:
            logger.debug("Cancelling %s via evidence", event.event_id)
            continue

        # --- Duplicate record check ---
        is_dup, dup_reason = _is_duplicate_record(event, events_by_id)
        if is_dup:
            logger.debug("Skipping duplicate %s: %s", event.event_id, dup_reason)
            continue

        # --- Resolve amount ---
        amount_from_image = False
        amount_from_message = False
        raw_amount = event.amount
        event_currency = event.currency

        if raw_amount is None:
            # Try evidence graph for image-extracted amount
            if evidence_graph:
                amount_fact = evidence_graph.get_amount_fact_for_event(event.event_id)
                if amount_fact is not None:
                    raw_amount = amount_fact.amount
                    if amount_fact.currency:
                        event_currency = amount_fact.currency
                    amount_from_image = True
                    logger.debug(
                        "Resolved blank amount for %s from evidence: %.2f",
                        event.event_id, raw_amount,
                    )
                else:
                    # Cannot resolve — skip this event conservatively
                    # (blank amount != zero; we cannot assume)
                    logger.warning(
                        "Cannot resolve blank amount for %s — skipping", event.event_id
                    )
                    continue
            else:
                logger.warning(
                    "Blank amount on %s with no evidence graph — skipping", event.event_id
                )
                continue

        # --- Apply evidence amendment ---
        if event.event_id in amended_amounts:
            raw_amount = amended_amounts[event.event_id]
            amount_from_message = True

        # --- Currency conversion ---
        amount_home = raw_amount

        if event_currency != home_currency:
            effective_date = event.settlement_date or event.event_date
            try:
                amount_home = convert(
                    raw_amount, event_currency, home_currency, effective_date, exchange_rates
                )
            except ConversionError as exc:
                logger.warning(
                    "Currency conversion failed for %s (%s->%s): %s. Using raw amount.",
                    event.event_id, event_currency, home_currency, exc,
                )
                # Conservative fallback: use raw amount (same magnitude, wrong currency)
                # This is flagged but does not crash the pipeline
                amount_home = raw_amount

        # --- Minimum allowed amount conversion ---
        min_allowed_home: Optional[float] = None
        if event.minimum_allowed_amount is not None:
            if event_currency != home_currency:
                effective_date = event.settlement_date or event.event_date
                try:
                    min_allowed_home = convert(
                        event.minimum_allowed_amount,
                        event_currency,
                        home_currency,
                        effective_date,
                        exchange_rates,
                    )
                except ConversionError:
                    min_allowed_home = event.minimum_allowed_amount
            else:
                min_allowed_home = event.minimum_allowed_amount

        # --- Recurrence info ---
        is_recurring, recurrence_day, recurrence_interval = recurrence_map.get(
            event.event_id, (False, None, None)
        )

        # --- Build NormalizedEvent ---
        normalized.append(NormalizedEvent(
            event_id=event.event_id,
            user_id=event.user_id,
            event_type=event.event_type,
            category=event.category,
            direction=event.direction,
            amount_home=amount_home,
            original_amount=raw_amount,
            original_currency=event_currency,
            event_date=event.event_date,
            settlement_date=event.settlement_date,
            status=event.status,
            flexibility=event.flexibility,
            minimum_allowed_amount=event.minimum_allowed_amount,
            minimum_allowed_home=min_allowed_home,
            description=event.description,
            is_recurring=is_recurring,
            recurrence_day=recurrence_day,
            recurrence_interval_days=recurrence_interval,
            amount_from_image=amount_from_image,
            amount_from_message=amount_from_message,
        ))

    # Step 4: Sort by effective date
    normalized.sort(key=lambda e: e.effective_date)

    logger.debug(
        "Normalized %d events from %d raw (%d recurring)",
        len(normalized), len(events),
        sum(1 for e in normalized if e.is_recurring),
    )

    return normalized


# ── Helpers for the Financial Twin ───────────────────────────────────────────

def get_recurring_expenses(
    events: list[NormalizedEvent],
    protect_categories: list[str],
) -> list[NormalizedEvent]:
    """Return recurring debits, sorted by category protection status."""
    return [
        e for e in events
        if e.is_recurring and e.is_expense
        and e.status in (EventStatus.SETTLED, EventStatus.SCHEDULED, EventStatus.PENDING)
    ]


def get_recurring_income(events: list[NormalizedEvent]) -> list[NormalizedEvent]:
    """Return recurring credits."""
    return [
        e for e in events
        if e.is_recurring and e.is_income
        and e.status in (EventStatus.SETTLED, EventStatus.SCHEDULED)
    ]


def get_future_confirmed_payments(
    events: list[NormalizedEvent],
    from_date: date,
) -> list[NormalizedEvent]:
    """
    Return non-recurring future debits that are confirmed (scheduled).
    These are mandatory payments — loan instalments, confirmed bills, etc.
    """
    return [
        e for e in events
        if not e.is_recurring
        and e.is_expense
        and e.effective_date >= from_date
        and e.status == EventStatus.SCHEDULED
    ]


def get_pending_debits(
    events: list[NormalizedEvent],
    from_date: date,
) -> list[NormalizedEvent]:
    """Return pending debits on or after from_date (reserved — conservative)."""
    return [
        e for e in events
        if e.is_expense
        and e.effective_date >= from_date
        and e.status == EventStatus.PENDING
    ]
