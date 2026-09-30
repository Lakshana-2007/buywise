"""
Output validator and CSV writer.

Validates the final output.csv before writing it.
Fails loudly on structural violations.
Warns on content violations (wrong enums, malformed plans) but still writes.

The validator is the final quality gate of the pipeline.
"""

from __future__ import annotations

import csv
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from buywise.core.models import OutputRow, PaymentOption, Request, FinancialProfile
from buywise.finance.events import NormalizedEvent
from buywise.finance.forecast import simulate_payment_plan

logger = logging.getLogger(__name__)

# Official allowed values
ALLOWED_STATUSES = frozenset({
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
})

ALLOWED_METHODS = frozenset({
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
})

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PLAN_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:[0-9]+(\.[0-9]+)?$")
SPENDING_CHANGE_RE = re.compile(
    r"^(stop:[a-zA-Z0-9_]+|reduce_to:[a-zA-Z0-9_]+:[0-9]+(\.[0-9]+)?)$"
)


class ValidationError(Exception):
    """Raised when a structural violation prevents writing valid output."""


def validate_output_row(row: OutputRow, requested_amount: float) -> list[str]:
    """
    Validate one output row.  Returns a list of warning strings (empty = clean).
    Does NOT raise — warnings are collected and reported.
    """
    warnings = []
    rid = row.request_id

    # amount_safe_to_pay bounds
    if row.amount_safe_to_pay < 0:
        warnings.append(f"{rid}: amount_safe_to_pay={row.amount_safe_to_pay} < 0")
    if row.amount_safe_to_pay > requested_amount + 0.01:
        warnings.append(
            f"{rid}: amount_safe_to_pay={row.amount_safe_to_pay} > requested_amount={requested_amount}"
        )

    # Enum validation
    if row.affordability_status not in ALLOWED_STATUSES:
        warnings.append(f"{rid}: invalid affordability_status={row.affordability_status!r}")
    if row.recommended_payment_method not in ALLOWED_METHODS:
        warnings.append(f"{rid}: invalid recommended_payment_method={row.recommended_payment_method!r}")

    # payment_plan format
    if row.payment_plan != "none":
        parts = row.payment_plan.split("|")
        for part in parts:
            if not PLAN_ENTRY_RE.match(part.strip()):
                warnings.append(f"{rid}: malformed payment_plan entry: {part!r}")
                break
        # Check chronological order
        try:
            plan_dates = [date.fromisoformat(p.split(":")[0]) for p in parts]
            if plan_dates != sorted(plan_dates):
                warnings.append(f"{rid}: payment_plan entries not in chronological order")
        except Exception:
            pass

    # spending_changes_needed format
    if row.spending_changes_needed != "none":
        changes = row.spending_changes_needed.split("|")
        if len(changes) > 3:
            warnings.append(f"{rid}: too many spending_changes ({len(changes)} > 3)")
        for change in changes:
            if not SPENDING_CHANGE_RE.match(change.strip()):
                warnings.append(f"{rid}: malformed spending_change: {change!r}")

    # earliest_date_for_full_payment format
    if row.earliest_date_for_full_payment:
        if not DATE_RE.match(row.earliest_date_for_full_payment):
            warnings.append(
                f"{rid}: malformed earliest_date_for_full_payment: "
                f"{row.earliest_date_for_full_payment!r}"
            )

    # affordable_now must have earliest = request_date (enforced in assembly)
    if row.affordability_status == "affordable_now" and not row.earliest_date_for_full_payment:
        warnings.append(f"{rid}: affordable_now must have earliest_date_for_full_payment set")

    # not_affordable / not_recommended should have empty earliest
    if row.affordability_status == "not_affordable" and row.earliest_date_for_full_payment:
        warnings.append(
            f"{rid}: not_affordable should have empty earliest_date_for_full_payment, "
            f"got {row.earliest_date_for_full_payment!r}"
        )

    # decision_explanation must be non-empty
    if not row.decision_explanation or len(row.decision_explanation.strip()) < 5:
        warnings.append(f"{rid}: decision_explanation is empty or too short")

    return warnings


def _parse_plan(plan: str, rid: str) -> tuple[list[tuple[date, float]], list[str]]:
    """Parse a plan once so semantic checks use the same validated values."""
    if plan == "none":
        return [], []
    entries: list[tuple[date, float]] = []
    errors: list[str] = []
    for part in plan.split("|"):
        if not PLAN_ENTRY_RE.fullmatch(part.strip()):
            errors.append(f"{rid}: malformed payment_plan entry: {part!r}")
            continue
        raw_date, raw_amount = part.split(":", 1)
        try:
            entries.append((date.fromisoformat(raw_date), float(raw_amount)))
        except ValueError:
            errors.append(f"{rid}: invalid payment_plan date or amount: {part!r}")
    if entries and [entry[0] for entry in entries] != sorted(entry[0] for entry in entries):
        errors.append(f"{rid}: payment_plan entries not in chronological order")
    return entries, errors


def validate_output_semantics(
    row: OutputRow,
    request: Request,
    profile: FinancialProfile,
    payment_options: list[PaymentOption],
    normalized_events: list[NormalizedEvent],
) -> list[str]:
    """Validate strategy contracts against the deterministic financial context."""
    warnings: list[str] = []
    rid = row.request_id
    if rid != request.request_id:
        warnings.append(f"{rid}: request_id does not match request context {request.request_id}")

    entries, plan_errors = _parse_plan(row.payment_plan, rid)
    warnings.extend(plan_errors)
    if row.earliest_date_for_full_payment:
        try:
            date.fromisoformat(row.earliest_date_for_full_payment)
        except ValueError:
            warnings.append(f"{rid}: invalid earliest_date_for_full_payment")

    method = row.recommended_payment_method
    status = row.affordability_status
    amount = row.amount_safe_to_pay
    tolerance = 0.02

    if method == "full_payment":
        if len(entries) != 1 or entries[0][0] != request.request_date:
            warnings.append(f"{rid}: full_payment must have one payment on request_date")
        elif abs(entries[0][1] - request.requested_amount) > tolerance:
            warnings.append(f"{rid}: full_payment plan does not equal requested_amount")
    elif method == "installments":
        matching = any(
            len(entries) == option.number_of_payments
            and all(
                payment_date == expected_date
                and abs(payment_amount - option.payment_amount) <= tolerance
                for (payment_date, payment_amount), expected_date in zip(
                    entries, option.payment_dates()
                )
            )
            for option in payment_options
            if option.payment_method == "installments"
        )
        if not matching:
            warnings.append(f"{rid}: installment plan does not match a supplied option")
    elif method == "partial_payment":
        if len(entries) != 2:
            warnings.append(f"{rid}: partial_payment must contain exactly two payments")
        else:
            first_date, first_amount = entries[0]
            second_date, second_amount = entries[1]
            if first_date != request.request_date:
                warnings.append(f"{rid}: partial_payment must start on request_date")
            if not 0 < amount < request.requested_amount:
                warnings.append(f"{rid}: partial_payment safe amount must be between zero and requested_amount")
            if abs(first_amount - amount) > tolerance:
                warnings.append(f"{rid}: first partial payment must equal amount_safe_to_pay")
            if abs(first_amount + second_amount - request.requested_amount) > tolerance:
                warnings.append(f"{rid}: partial payments do not sum to requested_amount")
            if row.earliest_date_for_full_payment != second_date.isoformat():
                warnings.append(f"{rid}: second partial payment must equal earliest safe full-payment date")
            if second_date > request.desired_completion_date:
                warnings.append(f"{rid}: partial payment misses desired completion date")
    elif method == "wait":
        if len(entries) != 1 or entries[0][1] != request.requested_amount:
            warnings.append(f"{rid}: wait must contain one full requested payment")
        elif row.earliest_date_for_full_payment != entries[0][0].isoformat():
            warnings.append(f"{rid}: wait date must equal earliest safe full-payment date")
        if entries and entries[0][0] > request.desired_completion_date:
            warnings.append(f"{rid}: wait misses desired completion date")
    elif method == "not_recommended" and entries:
        warnings.append(f"{rid}: not_recommended must not contain a payment plan")

    expected_methods = {
        "affordable_now": {"full_payment"},
        "affordable_with_plan": {"full_payment", "partial_payment", "installments"},
        "affordable_later": {"wait"},
        "not_affordable": {"not_recommended"},
    }
    status_method_allowed = method in expected_methods.get(status, set())
    if (
        status == "affordable_later"
        and method == "not_recommended"
        and row.earliest_date_for_full_payment
    ):
        # Financial capacity exists, but the safe date may miss the user's
        # requested deadline, leaving no actionable payment recommendation.
        status_method_allowed = date.fromisoformat(
            row.earliest_date_for_full_payment
        ) > request.desired_completion_date
    if not status_method_allowed:
        warnings.append(f"{rid}: status {status} is inconsistent with method {method}")

    changes = [] if row.spending_changes_needed == "none" else row.spending_changes_needed.split("|")
    if len(changes) > 3:
        warnings.append(f"{rid}: too many spending changes")
    events_by_id = {event.event_id: event for event in normalized_events}
    seen_events: set[str] = set()
    actions_by_event: dict[str, set[str]] = {}
    spending_lookup: dict[str, Optional[float]] = {}
    for change in changes:
        try:
            parts = change.split(":")
            action, event_id = parts[0], parts[1]
            event = events_by_id.get(event_id)
            if event_id in seen_events:
                warnings.append(f"{rid}: event {event_id} has multiple spending changes")
            seen_events.add(event_id)
            actions_by_event.setdefault(event_id, set()).add(action)
            if event is None:
                warnings.append(f"{rid}: spending change targets unknown event {event_id}")
                continue
            if not event.is_recurring or not event.is_expense or event.is_protected:
                warnings.append(f"{rid}: spending change targets a non-flexible recurring expense {event_id}")
            if action == "stop":
                if event.category not in profile.expense_categories_user_is_willing_to_stop or not event.is_stoppable:
                    warnings.append(f"{rid}: stop change is not permitted for {event_id}")
                spending_lookup[event_id] = None
            elif action == "reduce_to":
                new_amount = float(parts[2])
                if event.category not in profile.expense_categories_user_is_willing_to_reduce or not event.is_reducible:
                    warnings.append(f"{rid}: reduce change is not permitted for {event_id}")
                if new_amount >= event.amount_home or (event.minimum_allowed_home is not None and new_amount < event.minimum_allowed_home):
                    warnings.append(f"{rid}: reduce amount is invalid for {event_id}")
                spending_lookup[event_id] = new_amount
            else:
                warnings.append(f"{rid}: unknown spending change action {action}")
        except (IndexError, ValueError):
            warnings.append(f"{rid}: malformed spending change {change!r}")
    if any(len(actions) > 1 for actions in actions_by_event.values()):
        warnings.append(f"{rid}: stop and reduce cannot target the same event")

    if method != "not_recommended" and entries:
        simulation = simulate_payment_plan(
            starting_balance=profile.current_available_balance,
            minimum_balance=profile.minimum_balance_to_keep,
            events=normalized_events,
            request_date=request.request_date,
            payment_schedule=entries,
            spending_changes=spending_lookup,
        )
        if not simulation.is_safe:
            warnings.append(
                f"{rid}: payment plan breaches minimum balance "
                f"({simulation.minimum_projected_balance:.2f} < {profile.minimum_balance_to_keep:.2f})"
            )
    return warnings


def validate_and_write_output(
    output_rows: list[OutputRow],
    requested_amounts: dict[str, float],  # request_id -> requested_amount
    output_path: Path,
    expected_request_ids: list[str],
    validation_contexts: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[bool, list[str]]:
    """
    Validate all output rows and write to output_path.

    Performs structural checks (raises ValidationError on failure) and
    content checks (collects warnings).

    Parameters
    ----------
    output_rows         : All generated output rows.
    requested_amounts   : {request_id: requested_amount} for bounds checking.
    output_path         : Where to write output.csv.
    expected_request_ids: The 250 request IDs that must all be present.

    Returns (success, warnings_list).
    """
    # ── Structural checks (raise on failure) ──────────────────────────────────

    # Check row count
    if len(output_rows) != len(expected_request_ids):
        raise ValidationError(
            f"Expected {len(expected_request_ids)} rows, got {len(output_rows)}"
        )

    # Check for duplicate request IDs
    row_ids = [r.request_id for r in output_rows]
    if len(row_ids) != len(set(row_ids)):
        from collections import Counter
        dupes = [rid for rid, cnt in Counter(row_ids).items() if cnt > 1]
        raise ValidationError(f"Duplicate request_ids: {dupes}")

    # Check for missing or extra request IDs
    expected_set = set(expected_request_ids)
    actual_set = set(row_ids)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing:
        raise ValidationError(f"Missing request_ids: {sorted(missing)[:10]}")
    if extra:
        raise ValidationError(f"Extra request_ids: {sorted(extra)[:10]}")

    # ── Content validation (collect warnings) ─────────────────────────────────
    all_warnings: list[str] = []
    for row in output_rows:
        req_amount = requested_amounts.get(row.request_id, float("inf"))
        row_warnings = validate_output_row(row, req_amount)
        if validation_contexts and row.request_id in validation_contexts:
            context = validation_contexts[row.request_id]
            row_warnings.extend(validate_output_semantics(row=row, **context))
        all_warnings.extend(row_warnings)

    if all_warnings:
        logger.warning("Output validation warnings (%d):", len(all_warnings))
        for w in all_warnings[:20]:
            logger.warning("  %s", w)
        if len(all_warnings) > 20:
            logger.warning("  ... and %d more", len(all_warnings) - 20)

    # ── Write output CSV ───────────────────────────────────────────────────────
    # Sort by original request order
    id_order = {rid: i for i, rid in enumerate(expected_request_ids)}
    sorted_rows = sorted(output_rows, key=lambda r: id_order.get(r.request_id, 9999))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row.to_csv_row())

    logger.info("Wrote %d rows to %s", len(sorted_rows), output_path)
    return True, all_warnings
