"""
Pydantic models that represent every dataset row type.

These are the canonical typed representations of the raw CSV data.
They validate types but do NOT apply financial logic — that belongs
in finance/.

All amounts are stored as floats in the event's original currency.
Currency conversion to home_currency happens in finance/currency.py.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Financial Profile ─────────────────────────────────────────────────────────

class FinancialProfile(BaseModel):
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str] = Field(default_factory=list)
    expense_categories_to_protect: list[str] = Field(default_factory=list)
    expense_categories_user_is_willing_to_reduce: list[str] = Field(default_factory=list)
    expense_categories_user_is_willing_to_stop: list[str] = Field(default_factory=list)
    payment_methods_user_will_consider: list[str] = Field(default_factory=list)
    max_installment_months: Optional[int] = None  # None = will not consider installments

    @field_validator(
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        mode="before",
    )
    @classmethod
    def split_pipe(cls, v: str | list) -> list[str]:
        if isinstance(v, list):
            return v
        if not v or not v.strip():
            return []
        return [x.strip() for x in v.split("|") if x.strip()]

    @field_validator("max_installment_months", mode="before")
    @classmethod
    def parse_optional_int(cls, v) -> Optional[int]:
        if v is None or str(v).strip() == "":
            return None
        return int(v)


# ── Financial Event ───────────────────────────────────────────────────────────

class FinancialEvent(BaseModel):
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]          # None when blank (must be extracted from image)
    currency: str
    event_date: date
    settlement_date: Optional[date]  # None for unrealized/non-cash events
    status: str
    linked_event_id: Optional[str]   # Points to earlier event in same lifecycle
    flexibility: str
    minimum_allowed_amount: Optional[float]  # For reducible events

    @field_validator("amount", "minimum_allowed_amount", mode="before")
    @classmethod
    def parse_optional_float(cls, v) -> Optional[float]:
        if v is None or str(v).strip() == "":
            return None
        return float(v)

    @field_validator("event_date", mode="before")
    @classmethod
    def parse_event_date(cls, v) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("settlement_date", mode="before")
    @classmethod
    def parse_settlement_date(cls, v) -> Optional[date]:
        """settlement_date may be blank for unrealized/non-cash investment events."""
        if v is None or str(v).strip() == "":
            return None
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("linked_event_id", mode="before")
    @classmethod
    def empty_to_none(cls, v) -> Optional[str]:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip()

    @property
    def amount_is_missing(self) -> bool:
        return self.amount is None


# ── Exchange Rate ─────────────────────────────────────────────────────────────

class ExchangeRate(BaseModel):
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float

    @field_validator("rate_date", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())


# ── Request ───────────────────────────────────────────────────────────────────

class Request(BaseModel):
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str

    @field_validator("request_date", "desired_completion_date", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("allows_partial_payment", mode="before")
    @classmethod
    def parse_bool(cls, v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "1", "yes")


# ── Payment Option ────────────────────────────────────────────────────────────

class PaymentOption(BaseModel):
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]  # None for full_payment (single payment)
    financing_fee: float
    total_payable_amount: float

    @field_validator("first_payment_date", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("payment_frequency_days", mode="before")
    @classmethod
    def parse_optional_int(cls, v) -> Optional[int]:
        if v is None or str(v).strip() == "":
            return None
        return int(v)

    @field_validator("financing_fee", "total_payable_amount", "payment_amount", mode="before")
    @classmethod
    def parse_float(cls, v) -> float:
        if v is None or str(v).strip() == "":
            return 0.0
        return float(v)

    def payment_dates(self) -> list[date]:
        """Generate the complete list of payment dates for this option."""
        from datetime import timedelta
        dates = [self.first_payment_date]
        if self.payment_frequency_days and self.number_of_payments > 1:
            for i in range(1, self.number_of_payments):
                dates.append(
                    self.first_payment_date
                    + timedelta(days=self.payment_frequency_days * i)
                )
        return dates


# ── Message ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str

    @field_validator("request_id", "related_event_id", mode="before")
    @classmethod
    def empty_to_none(cls, v) -> Optional[str]:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip()

    @field_validator("sent_at", mode="before")
    @classmethod
    def parse_datetime(cls, v) -> datetime:
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))


# ── Image Reference ───────────────────────────────────────────────────────────

class ImageReference(BaseModel):
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]

    @field_validator("request_id", "related_event_id", mode="before")
    @classmethod
    def empty_to_none(cls, v) -> Optional[str]:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip()

    def image_path(self, media_dir) -> "Path":
        from pathlib import Path
        return Path(media_dir) / f"{self.image_id}.png"


# ── Output Row ────────────────────────────────────────────────────────────────

class OutputRow(BaseModel):
    """Represents one row in the final output.csv."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str           # "none" or "YYYY-MM-DD:amount|..."
    earliest_date_for_full_payment: str  # "YYYY-MM-DD" or ""
    spending_changes_needed: str  # "none" or "stop:eid|reduce_to:eid:amt"
    decision_explanation: str

    @model_validator(mode="after")
    def validate_amount_bounds(self) -> "OutputRow":
        if self.amount_safe_to_pay < 0:
            raise ValueError(
                f"amount_safe_to_pay must be >= 0, got {self.amount_safe_to_pay}"
            )
        return self

    def to_csv_row(self) -> dict:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": self.amount_safe_to_pay,
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": self.earliest_date_for_full_payment,
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }
