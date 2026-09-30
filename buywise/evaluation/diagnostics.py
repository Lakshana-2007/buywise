"""Machine-readable per-request diagnostics for production regressions."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from buywise.graph.state import CandidateStrategy, FinancialState, SpendingChange


def _value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, SpendingChange):
        return value.to_string()
    return value


def _strategy(strategy: CandidateStrategy | None) -> dict[str, Any] | None:
    if strategy is None:
        return None
    return {
        "method": strategy.payment_method,
        "payment_plan": strategy.payment_plan.to_string(),
        "spending_changes": strategy.spending_changes_string(),
        "validity": strategy.is_valid,
        "validation_violations": list(strategy.validation_violations),
        "minimum_projected_balance": strategy.minimum_projected_balance,
        "rank": strategy.rank,
        "rank_reason": strategy.rank_reason,
    }


def build_diagnostic_record(state: FinancialState) -> dict[str, Any]:
    """Convert one completed pipeline state into a secret-free JSON record."""
    evidence_facts = []
    if state.evidence_graph is not None:
        evidence_facts = [
            {
                "source_id": fact.source_id,
                "source_type": fact.source_type,
                "amount": fact.amount,
                "currency": fact.currency,
                "confidence": fact.confidence.value,
                "related_event_id": fact.related_event_id,
                "contains_instruction": fact.contains_instruction,
            }
            for fact in state.evidence_graph.facts
        ]

    normalized_events = [
        {
            "event_id": event.event_id,
            "amount_home": event.amount_home,
            "effective_date": _value(event.effective_date),
            "direction": event.direction,
            "status": event.status,
            "recurring": event.is_recurring,
            "recurrence_day": event.recurrence_day,
            "amount_from_image": event.amount_from_image,
            "amount_from_message": event.amount_from_message,
        }
        for event in state.normalized_events
    ]

    safety_proof = None
    if state.safety_proof is not None:
        safety_proof = state.safety_proof.to_dict()
        safety_proof["all_constraints_pass"] = state.safety_proof.all_constraints_pass
        # SpendingChange objects → serialisable strings
        safety_proof["spending_changes"] = [
            sc.to_string() for sc in state.safety_proof.spending_changes
        ]

    extraction_failures = sum(
        "extract" in error.lower() for error in state.errors
    )
    return {
        "request_id": state.request_id,
        "user_id": state.user_id,
        "evidence_facts": evidence_facts,
        "normalized_events": normalized_events,
        "financial_twin": {
            "amount_safe_to_pay": state.amount_safe_to_pay,
            "earliest_full_payment_date": _value(state.earliest_full_payment_date),
            "minimum_balance_for_full_payment": (
                state.safety_proof.minimum_projected_balance
                if state.safety_proof is not None
                else None
            ),
        },
        "candidate_strategies": [
            _strategy(candidate) for candidate in state.candidate_strategies
        ],
        "selected_strategy": _strategy(state.selected_strategy),
        "safety_proof": safety_proof,
        "explanation_fallback_used": state.explanation_fallback_used,
        "failure_counts": {
            "extraction": extraction_failures,
            "pipeline": len(state.errors),
            "api": None,
        },
        "errors": list(state.errors),
        "warnings": list(state.warnings),
    }


def write_diagnostic_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write one JSON object per request to a separate diagnostics artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")