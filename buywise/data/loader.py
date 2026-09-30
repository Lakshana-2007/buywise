"""
Dataset loader.

Reads every CSV from dataset/ into typed Pydantic models.
All I/O is centralised here — the rest of the codebase works
with typed objects, never raw strings from CSV rows.

Key design decisions:
- Encoding is explicitly UTF-8 (dataset has non-ASCII characters).
- Blank fields are parsed to None, not empty strings.
- No financial logic here — pure I/O and type coercion.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from buywise.core.config import (
    EXCHANGE_RATES_CSV,
    FINANCIAL_EVENTS_CSV,
    FINANCIAL_PROFILES_CSV,
    IMAGES_CSV,
    MESSAGES_CSV,
    PAYMENT_OPTIONS_CSV,
    REQUESTS_CSV,
    SAMPLE_REQUESTS_CSV,
)
from buywise.core.models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageReference,
    Message,
    OutputRow,
    PaymentOption,
    Request,
)

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a CSV file and return a list of raw row dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ── Public loaders ────────────────────────────────────────────────────────────

def load_requests(sample: bool = False) -> list[Request]:
    """Load all 250 requests (or sample_requests when sample=True)."""
    path = SAMPLE_REQUESTS_CSV if sample else REQUESTS_CSV
    rows = _read_csv(path)
    requests = []
    for row in rows:
        try:
            requests.append(Request(**{k: v for k, v in row.items() if k in Request.model_fields}))
        except Exception as exc:
            logger.warning("Skipping malformed request row %s: %s", row.get("request_id"), exc)
    logger.info("Loaded %d requests from %s", len(requests), path.name)
    return requests


def load_financial_profiles() -> dict[str, FinancialProfile]:
    """Load all financial profiles, keyed by user_id."""
    rows = _read_csv(FINANCIAL_PROFILES_CSV)
    profiles: dict[str, FinancialProfile] = {}
    for row in rows:
        try:
            profile = FinancialProfile(**row)
            profiles[profile.user_id] = profile
        except Exception as exc:
            logger.warning("Skipping malformed profile row %s: %s", row.get("user_id"), exc)
    logger.info("Loaded %d financial profiles", len(profiles))
    return profiles


def load_financial_events() -> dict[str, list[FinancialEvent]]:
    """
    Load all financial events, grouped by user_id.
    Returns dict[user_id -> list[FinancialEvent]].
    """
    rows = _read_csv(FINANCIAL_EVENTS_CSV)
    events: dict[str, list[FinancialEvent]] = {}
    skipped = 0
    for row in rows:
        try:
            event = FinancialEvent(**row)
            events.setdefault(event.user_id, []).append(event)
        except Exception as exc:
            logger.warning(
                "Skipping malformed event row %s: %s", row.get("event_id"), exc
            )
            skipped += 1
    total = sum(len(v) for v in events.values())
    logger.info("Loaded %d financial events across %d users (%d skipped)", total, len(events), skipped)
    return events


def load_financial_events_flat() -> list[FinancialEvent]:
    """Load all financial events as a flat list (for index building)."""
    rows = _read_csv(FINANCIAL_EVENTS_CSV)
    events = []
    for row in rows:
        try:
            events.append(FinancialEvent(**row))
        except Exception as exc:
            logger.warning("Skipping event %s: %s", row.get("event_id"), exc)
    return events


def load_exchange_rates() -> list[ExchangeRate]:
    """Load all exchange rates."""
    rows = _read_csv(EXCHANGE_RATES_CSV)
    rates = []
    for row in rows:
        try:
            rates.append(ExchangeRate(**row))
        except Exception as exc:
            logger.warning("Skipping exchange rate row: %s", exc)
    logger.info("Loaded %d exchange rates", len(rates))
    return rates


def load_payment_options() -> dict[str, list[PaymentOption]]:
    """
    Load payment options, grouped by request_id.
    Returns dict[request_id -> list[PaymentOption]].
    """
    rows = _read_csv(PAYMENT_OPTIONS_CSV)
    options: dict[str, list[PaymentOption]] = {}
    for row in rows:
        try:
            opt = PaymentOption(**row)
            options.setdefault(opt.request_id, []).append(opt)
        except Exception as exc:
            logger.warning(
                "Skipping payment option %s: %s", row.get("payment_option_id"), exc
            )
    total = sum(len(v) for v in options.values())
    logger.info("Loaded %d payment options across %d requests", total, len(options))
    return options


def load_messages() -> dict[str, list[Message]]:
    """
    Load all messages.
    Returns a dict with three indices merged into one multi-key dict:
    - by user_id
    - by request_id (when present)
    - by related_event_id (when present)

    For simplicity, returns dict[key -> list[Message]] where key can be
    a user_id, request_id, or event_id.  Callers should query by the most
    specific key they have.
    """
    rows = _read_csv(MESSAGES_CSV)
    index: dict[str, list[Message]] = {}
    for row in rows:
        try:
            msg = Message(**row)
            # Index by user_id
            index.setdefault(msg.user_id, []).append(msg)
            # Also index by request_id if present
            if msg.request_id:
                index.setdefault(msg.request_id, []).append(msg)
            # Also index by related_event_id if present
            if msg.related_event_id:
                index.setdefault(msg.related_event_id, []).append(msg)
        except Exception as exc:
            logger.warning("Skipping message %s: %s", row.get("message_id"), exc)
    logger.info("Loaded %d messages", len(rows))
    return index


def load_images() -> dict[str, list[ImageReference]]:
    """
    Load all image references.
    Returns dict keyed by related_event_id, request_id, and user_id.
    """
    rows = _read_csv(IMAGES_CSV)
    index: dict[str, list[ImageReference]] = {}
    for row in rows:
        try:
            img = ImageReference(**row)
            index.setdefault(img.user_id, []).append(img)
            if img.request_id:
                index.setdefault(img.request_id, []).append(img)
            if img.related_event_id:
                index.setdefault(img.related_event_id, []).append(img)
        except Exception as exc:
            logger.warning("Skipping image row %s: %s", row.get("image_id"), exc)
    logger.info("Loaded %d image references", len(rows))
    return index


def load_sample_outputs() -> dict[str, dict]:
    """
    Load sample_requests.csv output columns for regression testing.
    Returns dict[request_id -> output field dict].
    """
    rows = _read_csv(SAMPLE_REQUESTS_CSV)
    output_fields = [
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    return {
        row["request_id"]: {k: row.get(k, "") for k in output_fields}
        for row in rows
    }


# ── Dataset bundle ────────────────────────────────────────────────────────────

class DatasetBundle:
    """
    A fully-loaded, cross-referenced dataset bundle.

    Load once at startup; pass around by reference.
    All fields are typed and validated.
    """

    def __init__(self) -> None:
        logger.info("Loading dataset bundle...")
        self.requests: list[Request] = load_requests()
        self.profiles: dict[str, FinancialProfile] = load_financial_profiles()
        self.events_by_user: dict[str, list[FinancialEvent]] = load_financial_events()
        self.events_flat: list[FinancialEvent] = load_financial_events_flat()
        self.events_by_id: dict[str, FinancialEvent] = {
            e.event_id: e for e in self.events_flat
        }
        self.exchange_rates: list[ExchangeRate] = load_exchange_rates()
        self.payment_options: dict[str, list[PaymentOption]] = load_payment_options()
        self.messages: dict[str, list[Message]] = load_messages()
        self.images: dict[str, list[ImageReference]] = load_images()
        logger.info("Dataset bundle loaded successfully.")

    def get_profile(self, user_id: str) -> FinancialProfile:
        if user_id not in self.profiles:
            raise KeyError(f"No financial profile found for user_id={user_id!r}")
        return self.profiles[user_id]

    def get_user_events(self, user_id: str) -> list[FinancialEvent]:
        return self.events_by_user.get(user_id, [])

    def get_payment_options(self, request_id: str) -> list[PaymentOption]:
        return self.payment_options.get(request_id, [])

    def get_messages_for_user(self, user_id: str) -> list[Message]:
        return self.messages.get(user_id, [])

    def get_messages_for_request(self, request_id: str) -> list[Message]:
        return self.messages.get(request_id, [])

    def get_messages_for_event(self, event_id: str) -> list[Message]:
        return self.messages.get(event_id, [])

    def get_images_for_event(self, event_id: str) -> list[ImageReference]:
        return self.images.get(event_id, [])

    def get_images_for_request(self, request_id: str) -> list[ImageReference]:
        return self.images.get(request_id, [])

    def get_event_by_id(self, event_id: str) -> FinancialEvent | None:
        return self.events_by_id.get(event_id)

    def summary(self) -> str:
        return (
            f"DatasetBundle: {len(self.requests)} requests, "
            f"{len(self.profiles)} profiles, "
            f"{len(self.events_flat)} events, "
            f"{len(self.exchange_rates)} exchange rates, "
            f"{sum(len(v) for v in self.payment_options.values())} payment options, "
            f"{sum(1 for e in self.events_flat if e.amount is None)} blank-amount events"
        )
