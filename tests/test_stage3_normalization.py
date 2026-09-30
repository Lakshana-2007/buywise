"""
Stage 3 tests: financial event normalization and currency conversion.

Run with:  python -m pytest tests/test_stage3_normalization.py -v
"""

import os
import sys
from datetime import date, datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from buywise.core.models import ExchangeRate, FinancialEvent
from buywise.data.loader import DatasetBundle
from buywise.data.joins import build_request_context
from buywise.data.evidence import (
    ConfidenceLevel, EvidenceFact, EvidenceGraph, EvidenceType,
)
from buywise.finance.currency import ConversionError, convert
from buywise.finance.events import (
    NormalizedEvent,
    normalize_events,
    get_recurring_expenses,
    get_recurring_income,
    get_future_confirmed_payments,
    get_pending_debits,
    _should_ignore,
    _detect_recurrence,
)


@pytest.fixture(scope="module")
def bundle():
    return DatasetBundle()


# ── Currency conversion ───────────────────────────────────────────────────────

@pytest.fixture
def sample_rates():
    """A minimal set of exchange rates for testing."""
    return [
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="EUR", to_currency="ZAR", rate=20.0),
        ExchangeRate(rate_date=date(2024, 2, 15), from_currency="EUR", to_currency="ZAR", rate=21.0),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="EUR", rate=0.92),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="IDR", rate=15833.33),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="INR", rate=83.5),
    ]


def test_convert_same_currency():
    rates = []
    result = convert(100.0, "ZAR", "ZAR", date(2024, 1, 1), rates)
    assert result == 100.0


def test_convert_direct_rate(sample_rates):
    result = convert(100.0, "EUR", "ZAR", date(2024, 1, 20), sample_rates)
    assert result == 2000.0  # 100 * 20


def test_convert_uses_most_recent_prior_rate(sample_rates):
    # On 2024-02-20, the most recent EUR->ZAR rate is the Feb rate (21.0)
    result = convert(100.0, "EUR", "ZAR", date(2024, 2, 20), sample_rates)
    assert result == 2100.0  # 100 * 21


def test_convert_falls_back_to_earliest_when_no_prior(sample_rates):
    # Date before any rate — should fall back to earliest
    result = convert(100.0, "EUR", "ZAR", date(2023, 1, 1), sample_rates)
    assert result == 2000.0  # earliest rate


def test_convert_inverse_rate(sample_rates):
    # ZAR -> EUR using inverse of EUR->ZAR
    result = convert(2000.0, "ZAR", "EUR", date(2024, 1, 20), sample_rates)
    assert abs(result - 100.0) < 0.1


def test_convert_bridge_via_usd(sample_rates):
    # EUR -> IDR: no direct rate, bridge via USD
    # EUR->USD: 1/0.92 ~ 1.087; USD->IDR: 15833.33
    eur_to_usd = 1 / 0.92
    usd_to_idr = 15833.33
    expected = 100.0 * eur_to_usd * usd_to_idr
    result = convert(100.0, "EUR", "IDR", date(2024, 1, 20), sample_rates)
    assert abs(result - expected) < 10  # within 10 IDR


def test_convert_unknown_pair_raises(sample_rates):
    with pytest.raises(ConversionError):
        convert(100.0, "GBP", "ZAR", date(2024, 1, 1), sample_rates)


# ── Ignore filter ─────────────────────────────────────────────────────────────

def make_event(**kwargs) -> FinancialEvent:
    """Helper to create a minimal FinancialEvent for testing."""
    defaults = dict(
        event_id="test_event",
        user_id="user_test",
        event_type="expense",
        description="Test",
        category="groceries",
        direction="debit",
        amount=100.0,
        currency="ZAR",
        event_date=date(2024, 1, 15),
        settlement_date=date(2024, 1, 15),
        status="settled",
        linked_event_id=None,
        flexibility="fixed",
        minimum_allowed_amount=None,
    )
    defaults.update(kwargs)
    return FinancialEvent(**defaults)


def test_ignore_cancelled():
    e = make_event(status="cancelled")
    should, reason = _should_ignore(e)
    assert should
    assert "cancelled" in reason


def test_ignore_failed():
    e = make_event(status="failed")
    should, reason = _should_ignore(e)
    assert should


def test_ignore_unrealized():
    e = make_event(status="unrealized", direction="non_cash")
    should, reason = _should_ignore(e)
    assert should


def test_ignore_non_cash_direction():
    e = make_event(direction="non_cash", status="settled")
    should, reason = _should_ignore(e)
    assert should


def test_ignore_pending_credit():
    e = make_event(status="pending", direction="credit")
    should, reason = _should_ignore(e)
    assert should
    assert "pending credit" in reason


def test_keep_pending_debit():
    """Pending debits must be KEPT (reserved conservatively)."""
    e = make_event(status="pending", direction="debit")
    should, reason = _should_ignore(e)
    assert not should


def test_keep_settled_expense():
    e = make_event(status="settled", direction="debit")
    should, reason = _should_ignore(e)
    assert not should


def test_keep_scheduled_income():
    e = make_event(status="scheduled", direction="credit")
    should, reason = _should_ignore(e)
    assert not should


# ── Recurrence detection ──────────────────────────────────────────────────────

def make_recurring_series(n=4) -> list[FinancialEvent]:
    """Create n monthly expenses of same amount on same day."""
    events = []
    for i in range(n):
        m = 10 + i
        if m > 12:
            m = m - 12
        y = 2023 if m >= 10 else 2024
        events.append(make_event(
            event_id=f"test_ev_{i}",
            category="rent",
            direction="debit",
            amount=5000.0,
            event_date=date(y, m, 2),
            settlement_date=date(y, m, 2),
            status="settled",
        ))
    return events


def test_detect_recurrence_monthly_pattern():
    events = make_recurring_series(4)
    result = _detect_recurrence(events)
    for e in events:
        is_rec, day, interval = result[e.event_id]
        assert is_rec, f"Expected {e.event_id} to be recurring"
        assert day == 2
        assert interval == 30


def test_detect_recurrence_not_recurring_if_variable_amounts():
    events = []
    for i, amt in enumerate([1000.0, 2000.0, 3000.0, 4000.0]):
        events.append(make_event(
            event_id=f"var_{i}",
            category="shopping",
            direction="debit",
            amount=amt,
            event_date=date(2024, i + 1, 5),
            settlement_date=date(2024, i + 1, 5),
            status="settled",
        ))
    result = _detect_recurrence(events)
    for e in events:
        is_rec, _, _ = result[e.event_id]
        assert not is_rec


def test_detect_recurrence_min_3_occurrences():
    events = make_recurring_series(2)  # Only 2 — not enough
    result = _detect_recurrence(events)
    for e in events:
        is_rec, _, _ = result[e.event_id]
        assert not is_rec


# ── Full normalization pipeline ───────────────────────────────────────────────

def test_normalize_removes_cancelled(sample_rates):
    events = [
        make_event(event_id="e1", status="settled", amount=500.0),
        make_event(event_id="e2", status="cancelled", amount=500.0),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    ids = [e.event_id for e in result]
    assert "e1" in ids
    assert "e2" not in ids


def test_normalize_removes_pending_credit(sample_rates):
    events = [
        make_event(event_id="income_pending", status="pending", direction="credit", amount=5000.0),
        make_event(event_id="expense_settled", status="settled", direction="debit", amount=100.0),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    ids = [e.event_id for e in result]
    assert "income_pending" not in ids
    assert "expense_settled" in ids


def test_normalize_keeps_pending_debit(sample_rates):
    events = [
        make_event(event_id="debit_pending", status="pending", direction="debit", amount=200.0),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    assert len(result) == 1
    assert result[0].event_id == "debit_pending"


def test_normalize_converts_currency(sample_rates):
    events = [
        make_event(
            event_id="eur_event",
            amount=100.0,
            currency="EUR",
            status="settled",
        ),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    assert len(result) == 1
    # 100 EUR * 20 ZAR/EUR = 2000 ZAR
    assert result[0].amount_home == 2000.0
    assert result[0].original_currency == "EUR"


def test_normalize_skips_blank_amount_without_evidence(sample_rates):
    events = [
        make_event(event_id="blank_amt", amount=None, status="settled"),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    # Should be skipped (blank amount != zero, can't assume)
    assert len(result) == 0


def test_normalize_resolves_blank_amount_from_evidence(sample_rates):
    events = [
        make_event(event_id="blank_evt", amount=None, status="settled"),
    ]
    graph = EvidenceGraph(request_id="r1", user_id="u1")
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.AMOUNT,
        source_id="image_01",
        source_type="image",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        amount=999.0,
        currency="ZAR",
        confidence=ConfidenceLevel.HIGH,
        related_event_id="blank_evt",
    ))
    result = normalize_events(events, "ZAR", sample_rates, evidence_graph=graph)
    assert len(result) == 1
    assert result[0].amount_home == 999.0
    assert result[0].amount_from_image is True


def test_normalize_blank_amount_uses_evidence_currency(sample_rates):
    events = [make_event(event_id="blank_foreign", amount=None, currency="ZAR", status="settled")]
    graph = EvidenceGraph(request_id="r1", user_id="u1")
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.AMOUNT,
        source_id="image_foreign",
        source_type="image",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        amount=100.0,
        currency="EUR",
        confidence=ConfidenceLevel.HIGH,
        related_event_id="blank_foreign",
    ))
    result = normalize_events(events, "ZAR", sample_rates, evidence_graph=graph)
    assert result[0].amount_home == 2000.0


def test_normalize_applies_evidence_amendment(sample_rates):
    events = [
        make_event(event_id="salary_evt", direction="credit", amount=40000.0, status="scheduled"),
    ]
    graph = EvidenceGraph(request_id="r1", user_id="u1")
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.SALARY_UPDATE,
        source_id="msg_01",
        source_type="message",
        extracted_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        amount=45000.0,
        currency="ZAR",
        confidence=ConfidenceLevel.HIGH,
        related_event_id="salary_evt",
    ))
    result = normalize_events(events, "ZAR", sample_rates, evidence_graph=graph)
    assert len(result) == 1
    assert result[0].amount_home == 45000.0
    assert result[0].amount_from_message is True


def test_normalize_uses_newer_amendment(sample_rates):
    events = [make_event(event_id="salary_evt", direction="credit", amount=40000.0, status="scheduled")]
    graph = EvidenceGraph(request_id="r1", user_id="u1")
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.SALARY_UPDATE,
        source_id="old",
        source_type="message",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        amount=41000.0,
        currency="ZAR",
        confidence=ConfidenceLevel.HIGH,
        related_event_id="salary_evt",
    ))
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.SALARY_UPDATE,
        source_id="new",
        source_type="message",
        extracted_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        amount=45000.0,
        currency="ZAR",
        confidence=ConfidenceLevel.HIGH,
        related_event_id="salary_evt",
    ))
    result = normalize_events(events, "ZAR", sample_rates, evidence_graph=graph)
    assert result[0].amount_home == 45000.0


def test_normalize_applies_evidence_cancellation(sample_rates):
    events = [
        make_event(event_id="evt_to_cancel", status="settled", amount=500.0),
    ]
    graph = EvidenceGraph(request_id="r1", user_id="u1")
    graph.add(EvidenceFact(
        evidence_type=EvidenceType.EVENT_CANCELLED,
        source_id="msg_cancel",
        source_type="message",
        extracted_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        confidence=ConfidenceLevel.HIGH,
        related_event_id="evt_to_cancel",
    ))
    result = normalize_events(events, "ZAR", sample_rates, evidence_graph=graph)
    assert len(result) == 0  # cancelled via evidence


def test_normalize_detects_recurring(sample_rates):
    recurring = make_recurring_series(4)
    result = normalize_events(recurring, "ZAR", sample_rates)
    assert all(e.is_recurring for e in result)
    assert all(e.recurrence_day == 2 for e in result)


def test_normalize_sorted_by_date(sample_rates):
    events = [
        make_event(event_id="e3", event_date=date(2024, 3, 1), settlement_date=date(2024, 3, 1), amount=100.0),
        make_event(event_id="e1", event_date=date(2024, 1, 1), settlement_date=date(2024, 1, 1), amount=100.0),
        make_event(event_id="e2", event_date=date(2024, 2, 1), settlement_date=date(2024, 2, 1), amount=100.0),
    ]
    result = normalize_events(events, "ZAR", sample_rates)
    dates = [e.effective_date for e in result]
    assert dates == sorted(dates)


# ── Integration with real dataset ────────────────────────────────────────────

def test_normalize_real_user_01(bundle):
    """Normalize user_01 events and verify basic properties."""
    profile = bundle.get_profile("user_01")
    events = bundle.get_user_events("user_01")
    result = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    # Should have events
    assert len(result) > 0

    # All amounts should be positive
    for e in result:
        assert e.amount_home > 0, f"Negative amount on {e.event_id}: {e.amount_home}"

    # Should detect rent as recurring (6 occurrences)
    rent_events = [e for e in result if e.category == "rent"]
    assert len(rent_events) >= 5
    assert any(e.is_recurring for e in rent_events)


def test_normalize_real_no_cancelled_events(bundle):
    """No cancelled events should appear in normalized output."""
    for user_id in ["user_01", "user_02", "user_03"]:
        events = bundle.get_user_events(user_id)
        profile = bundle.get_profile(user_id)
        result = normalize_events(events, profile.home_currency, bundle.exchange_rates)
        for e in result:
            assert e.status != "cancelled", f"Cancelled event {e.event_id} survived normalization"
            assert e.status != "failed"


def test_normalize_real_no_non_cash(bundle):
    """No non_cash events should appear in normalized output."""
    for user_id in ["user_01", "user_02", "user_03"]:
        events = bundle.get_user_events(user_id)
        profile = bundle.get_profile(user_id)
        result = normalize_events(events, profile.home_currency, bundle.exchange_rates)
        for e in result:
            assert e.direction != "non_cash"


def test_normalize_real_no_pending_credits(bundle):
    """No pending credits should appear in normalized output."""
    for user_id in ["user_01", "user_02", "user_03"]:
        events = bundle.get_user_events(user_id)
        profile = bundle.get_profile(user_id)
        result = normalize_events(events, profile.home_currency, bundle.exchange_rates)
        for e in result:
            if e.status == "pending":
                assert e.direction == "debit", (
                    f"Pending credit {e.event_id} survived normalization"
                )


def test_normalize_real_user_02_recurring_salary(bundle):
    """user_02 has 5+ monthly salary entries — should be detected as recurring."""
    profile = bundle.get_profile("user_02")
    events = bundle.get_user_events("user_02")
    result = normalize_events(events, profile.home_currency, bundle.exchange_rates)
    salary = [e for e in result if e.category == "salary" and e.is_income]
    assert len(salary) > 0
    recurring_salary = [e for e in salary if e.is_recurring]
    assert len(recurring_salary) > 0, "user_02 salary should be detected as recurring"


def test_normalize_all_amounts_positive(bundle):
    """All normalized event amounts must be positive."""
    for user_id in ["user_01", "user_02", "user_05"]:
        events = bundle.get_user_events(user_id)
        profile = bundle.get_profile(user_id)
        result = normalize_events(events, profile.home_currency, bundle.exchange_rates)
        for e in result:
            assert e.amount_home > 0


# ── Helper functions ─────────────────────────────────────────────────────────

def test_get_recurring_expenses_filters_income(bundle):
    events = bundle.get_user_events("user_01")
    profile = bundle.get_profile("user_01")
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)
    recurring_exp = get_recurring_expenses(normalized, profile.expense_categories_to_protect)
    for e in recurring_exp:
        assert e.is_expense
        assert e.is_recurring


def test_get_recurring_income(bundle):
    events = bundle.get_user_events("user_02")
    profile = bundle.get_profile("user_02")
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)
    rec_income = get_recurring_income(normalized)
    for e in rec_income:
        assert e.is_income
        assert e.is_recurring


def test_get_future_confirmed_payments(bundle):
    profile = bundle.get_profile("user_01")
    events = bundle.get_user_events("user_01")
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)
    future = get_future_confirmed_payments(normalized, date(2024, 3, 1))
    for e in future:
        assert e.effective_date >= date(2024, 3, 1)
        assert e.status == "scheduled"
