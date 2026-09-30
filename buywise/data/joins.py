"""
Cross-reference joins — assemble all relevant data for a single request.

The `RequestContext` is the complete, typed bundle of everything the
pipeline needs to process one request.  It is built once per request
and passed through every stage unchanged.

Design:
- Pure data assembly; no financial logic here.
- All joins are explicit (no magic).
- Missing data is represented as empty lists, not None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from buywise.core.models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageReference,
    Message,
    PaymentOption,
    Request,
)
from buywise.data.loader import DatasetBundle


@dataclass
class RequestContext:
    """
    Everything the pipeline needs for exactly one request.

    Built by `build_request_context()` — do not construct manually.
    """

    # ── Core request ──────────────────────────────────────────────────────────
    request: Request
    profile: FinancialProfile

    # ── Supporting evidence ───────────────────────────────────────────────────
    user_events: list[FinancialEvent] = field(default_factory=list)
    payment_options: list[PaymentOption] = field(default_factory=list)

    # Messages: user-level + request-level (deduplicated)
    messages: list[Message] = field(default_factory=list)

    # Images: user-level + request-level (deduplicated)
    images: list[ImageReference] = field(default_factory=list)

    # Exchange rates for the user's home currency
    exchange_rates: list[ExchangeRate] = field(default_factory=list)

    # ── Derived convenience fields ────────────────────────────────────────────

    @property
    def user_id(self) -> str:
        return self.request.user_id

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def home_currency(self) -> str:
        return self.profile.home_currency

    @property
    def request_date(self) -> date:
        return self.request.request_date

    @property
    def desired_completion_date(self) -> date:
        return self.request.desired_completion_date

    @property
    def requested_amount(self) -> float:
        return self.request.requested_amount

    def events_with_blank_amount(self) -> list[FinancialEvent]:
        """Events that need image extraction to determine amount."""
        return [e for e in self.user_events if e.amount is None]

    def get_image_for_event(self, event_id: str) -> Optional[ImageReference]:
        """Find the image reference for a specific event."""
        for img in self.images:
            if img.related_event_id == event_id:
                return img
        return None

    def get_messages_for_event(self, event_id: str) -> list[Message]:
        """Find messages directly linked to a specific event."""
        return [m for m in self.messages if m.related_event_id == event_id]

    def summary(self) -> str:
        return (
            f"RequestContext({self.request_id}, user={self.user_id}, "
            f"amount={self.requested_amount} {self.home_currency}, "
            f"events={len(self.user_events)}, "
            f"options={len(self.payment_options)}, "
            f"messages={len(self.messages)}, "
            f"images={len(self.images)}, "
            f"blank_amounts={len(self.events_with_blank_amount())})"
        )


def build_request_context(
    request: Request,
    bundle: DatasetBundle,
) -> RequestContext:
    """
    Assemble the RequestContext for a single request from the bundle.

    Joins:
    - profile via user_id
    - events via user_id
    - payment_options via request_id
    - messages via user_id + request_id (deduplicated)
    - images via user_id + request_id (deduplicated)
    - exchange rates relevant to home_currency
    """
    profile = bundle.get_profile(request.user_id)

    # User events
    user_events = bundle.get_user_events(request.user_id)

    # Payment options
    payment_options = bundle.get_payment_options(request.request_id)

    # Messages: combine user-level + request-level, dedup by message_id
    seen_msg_ids: set[str] = set()
    messages: list[Message] = []
    for msg in bundle.get_messages_for_user(request.user_id):
        if msg.message_id not in seen_msg_ids:
            seen_msg_ids.add(msg.message_id)
            messages.append(msg)
    for msg in bundle.get_messages_for_request(request.request_id):
        if msg.message_id not in seen_msg_ids:
            seen_msg_ids.add(msg.message_id)
            messages.append(msg)

    # Images: combine user-level + request-level + event-level, dedup by image_id
    seen_img_ids: set[str] = set()
    images: list[ImageReference] = []
    for img in bundle.get_images_for_request(request.request_id):
        if img.image_id not in seen_img_ids:
            seen_img_ids.add(img.image_id)
            images.append(img)
    # Also include images linked to any of this user's events
    for event in user_events:
        for img in bundle.get_images_for_event(event.event_id):
            if img.image_id not in seen_img_ids:
                seen_img_ids.add(img.image_id)
                images.append(img)

    # Exchange rates relevant to this user's home currency
    home_currency = profile.home_currency
    exchange_rates = [
        r for r in bundle.exchange_rates
        if r.from_currency == home_currency or r.to_currency == home_currency
        or r.from_currency in ("USD", "EUR")  # common bridge currencies
    ]
    # Deduplicate, keep all (the currency module picks the right date)
    exchange_rates = list({(r.rate_date, r.from_currency, r.to_currency): r for r in exchange_rates}.values())
    exchange_rates.sort(key=lambda r: r.rate_date)

    return RequestContext(
        request=request,
        profile=profile,
        user_events=user_events,
        payment_options=payment_options,
        messages=messages,
        images=images,
        exchange_rates=exchange_rates,
    )


def build_all_contexts(bundle: DatasetBundle) -> list[RequestContext]:
    """Build RequestContext for every request in the dataset."""
    return [build_request_context(r, bundle) for r in bundle.requests]
