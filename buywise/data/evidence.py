"""
Structured evidence representation.

Every important financial fact extracted from messages or images is
represented as an `EvidenceFact` with full provenance.

Key design properties:
- Untrusted source: messages and images are data, not instructions.
- Every fact knows WHERE it came from, WHEN it was stated, and HOW
  confident the extraction was.
- Conflicting facts are resolved by `resolve_conflicts()` using the
  official priority order from the problem spec.
- Facts only AMEND the base financial state; they do not replace it
  unless explicitly modelled as an amendment/cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class EvidenceType(str, Enum):
    """What kind of financial information does this fact represent?"""
    AMOUNT = "amount"               # A monetary value (bill, receipt, salary)
    SALARY_UPDATE = "salary_update" # Employer confirms new salary amount
    SALARY_CONFIRMED = "salary_confirmed"  # Employer confirms upcoming payment
    EVENT_CANCELLED = "event_cancelled"    # A prior event is cancelled
    EVENT_AMENDED = "event_amended"        # A prior event is amended
    EVENT_SETTLED = "event_settled"        # A prior event has settled
    EXPENSE_CONFIRMED = "expense_confirmed"  # A future expense is confirmed
    PAYMENT_CONFIRMED = "payment_confirmed"  # A future payment is confirmed
    INCOME_PENDING = "income_pending"        # Pending income (do NOT count)
    INCOME_CANCELLED = "income_cancelled"    # Income will not arrive
    GENERAL = "general"                      # Non-actionable context


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # Extracted from structured data (employer msg, bank stmt)
    MEDIUM = "medium"   # Extracted from natural language with clear signal
    LOW = "low"         # Ambiguous extraction
    UNCERTAIN = "uncertain"  # Could not reliably extract


@dataclass
class EvidenceFact:
    """
    A single structured financial fact extracted from a message or image.

    Provenance is mandatory — we must always know where a fact came from.
    """
    evidence_type: EvidenceType
    source_id: str              # message_id or image_id
    source_type: str            # "message" | "image"
    extracted_at: datetime      # When the source was created/sent

    # ── Financial content ─────────────────────────────────────────────────────
    amount: Optional[float] = None
    currency: Optional[str] = None
    applies_to_date: Optional[date] = None  # The date this fact relates to
    related_event_id: Optional[str] = None  # Which event this amends/cancels

    # ── Provenance metadata ───────────────────────────────────────────────────
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    raw_text: str = ""          # Original snippet (for auditability)
    notes: str = ""             # Extraction reasoning

    # ── Safety flag ───────────────────────────────────────────────────────────
    contains_instruction: bool = False  # True if source tried to give instructions

    def is_trustworthy(self) -> bool:
        """
        A fact is trustworthy for financial calculations only if:
        - It does NOT contain embedded instructions.
        - Confidence is not UNCERTAIN.
        """
        return not self.contains_instruction and self.confidence != ConfidenceLevel.UNCERTAIN

    def is_amendment(self) -> bool:
        return self.evidence_type in (
            EvidenceType.EVENT_CANCELLED,
            EvidenceType.EVENT_AMENDED,
            EvidenceType.EVENT_SETTLED,
            EvidenceType.SALARY_UPDATE,
        )

    def __repr__(self) -> str:
        return (
            f"EvidenceFact({self.evidence_type.value}, "
            f"src={self.source_id}, "
            f"amount={self.amount} {self.currency}, "
            f"confidence={self.confidence.value}, "
            f"event={self.related_event_id})"
        )


@dataclass
class EvidenceGraph:
    """
    Collection of all EvidenceFacts for a single request context.

    Supports:
    - Looking up facts by event_id (amendments, cancellations)
    - Filtering by type and confidence
    - Conflict detection
    """
    request_id: str
    user_id: str
    facts: list[EvidenceFact] = field(default_factory=list)

    def add(self, fact: EvidenceFact) -> None:
        self.facts.append(fact)

    def get_facts_for_event(self, event_id: str) -> list[EvidenceFact]:
        return [f for f in self.facts if f.related_event_id == event_id]

    def get_amendments(self) -> list[EvidenceFact]:
        return [f for f in self.facts if f.is_amendment() and f.is_trustworthy()]

    def get_cancellations(self) -> list[EvidenceFact]:
        return [
            f for f in self.facts
            if f.evidence_type == EvidenceType.EVENT_CANCELLED and f.is_trustworthy()
        ]

    def get_salary_updates(self) -> list[EvidenceFact]:
        return [
            f for f in self.facts
            if f.evidence_type == EvidenceType.SALARY_UPDATE and f.is_trustworthy()
        ]

    def get_amount_for_event(self, event_id: str) -> Optional[float]:
        """
        Return the best known amount for a blank-amount event.
        Uses the highest-confidence fact, preferring more recent sources.
        """
        fact = self.get_amount_fact_for_event(event_id)
        return fact.amount if fact else None

    def get_amount_fact_for_event(self, event_id: str) -> Optional[EvidenceFact]:
        """Return the best trustworthy amount fact, including confirmed salary facts."""
        candidates = [
            f for f in self.get_facts_for_event(event_id)
            if f.evidence_type in (EvidenceType.AMOUNT, EvidenceType.SALARY_CONFIRMED)
            and f.amount is not None
            and f.is_trustworthy()
        ]
        if not candidates:
            return None
        # Sort: HIGH confidence first, then most recent
        order = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2}
        candidates.sort(key=lambda f: (order.get(f.confidence, 9), -f.extracted_at.timestamp()))
        return candidates[0]

    def has_suspicious_instructions(self) -> bool:
        """Return True if any source contained embedded instructions."""
        return any(f.contains_instruction for f in self.facts)

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_facts": len(self.facts),
            "amendments": len(self.get_amendments()),
            "cancellations": len(self.get_cancellations()),
            "salary_updates": len(self.get_salary_updates()),
            "suspicious": self.has_suspicious_instructions(),
        }


# ── Conflict resolution ───────────────────────────────────────────────────────

def resolve_conflicts(facts: list[EvidenceFact]) -> list[EvidenceFact]:
    """
    Apply the official conflict-resolution priority order from the spec:

    1. Explicit cancellation / settlement / amendment wins over older records.
    2. Newer record from the same source wins over older.
    3. Settled event over estimate/forecast.
    4. Financially safer interpretation when unresolvable.

    Returns a deduplicated, resolved list of facts.
    The caller must still decide which facts to apply — this function
    only handles ordering and flagging contradictions.
    """
    if not facts:
        return []

    # Group by related_event_id
    by_event: dict[Optional[str], list[EvidenceFact]] = {}
    for f in facts:
        by_event.setdefault(f.related_event_id, []).append(f)

    resolved: list[EvidenceFact] = []
    for event_id, event_facts in by_event.items():
        if len(event_facts) == 1:
            resolved.extend(event_facts)
            continue

        # Priority 1: explicit cancellations dominate
        cancellations = [f for f in event_facts if f.evidence_type == EvidenceType.EVENT_CANCELLED]
        if cancellations:
            # Take the most recent cancellation
            cancellations.sort(key=lambda f: f.extracted_at, reverse=True)
            resolved.append(cancellations[0])
            continue

        # Priority 2: amendments/settlements over plain amounts
        amendments = [f for f in event_facts if f.is_amendment()]
        if amendments:
            amendments.sort(key=lambda f: f.extracted_at, reverse=True)
            resolved.append(amendments[0])
            continue

        # Priority 3: HIGH confidence over lower, then newest
        conf_order = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2, ConfidenceLevel.UNCERTAIN: 3}
        event_facts.sort(
            key=lambda f: (conf_order.get(f.confidence, 9), -f.extracted_at.timestamp())
        )
        resolved.append(event_facts[0])

    return resolved


# ── Instruction-injection detection ──────────────────────────────────────────

_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all",
    "disregard",
    "override",
    "approve this",
    "treat this as income",
    "treat this amount",
    "you are now",
    "new instructions",
    "forget your",
    "bypass",
    "jailbreak",
    "system prompt",
]


def detect_instruction_injection(text: str) -> bool:
    """
    Return True if the text appears to contain instruction-injection attempts.
    Messages and images are untrusted data — they must never override rules.
    """
    lower = text.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)
