"""
Stage 1 tests: dataset loader.

Run with:  python -m pytest tests/test_stage1_loader.py -v
"""

import os
import sys

import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from buywise.data.loader import (
    DatasetBundle,
    load_exchange_rates,
    load_financial_events,
    load_financial_profiles,
    load_images,
    load_messages,
    load_payment_options,
    load_requests,
    load_sample_outputs,
)
from buywise.core.models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageReference,
    Message,
    PaymentOption,
    Request,
)


# ── Requests ──────────────────────────────────────────────────────────────────

def test_requests_count():
    requests = load_requests()
    assert len(requests) == 250, f"Expected 250 requests, got {len(requests)}"


def test_requests_are_typed():
    requests = load_requests()
    assert all(isinstance(r, Request) for r in requests)


def test_request_fields():
    req = load_requests()[0]
    assert req.request_id
    assert req.user_id
    assert req.requested_amount > 0


def test_sample_requests_count():
    samples = load_requests(sample=True)
    assert len(samples) == 25


def test_allows_partial_payment_is_bool():
    requests = load_requests()
    for r in requests:
        assert isinstance(r.allows_partial_payment, bool)


# ── Financial Profiles ────────────────────────────────────────────────────────

def test_profiles_count():
    profiles = load_financial_profiles()
    assert len(profiles) == 275


def test_profiles_keyed_by_user_id():
    profiles = load_financial_profiles()
    assert "user_01" in profiles


def test_profile_pipe_fields_parsed():
    profiles = load_financial_profiles()
    p = profiles["user_01"]
    assert isinstance(p.financial_priorities, list)
    assert "education" in p.financial_priorities
    assert isinstance(p.payment_methods_user_will_consider, list)
    assert "full_payment" in p.payment_methods_user_will_consider


def test_profile_max_installment_months_optional():
    profiles = load_financial_profiles()
    # user_01 has no max_installment_months
    assert profiles["user_01"].max_installment_months is None
    # user_02 has 7
    assert profiles["user_02"].max_installment_months == 7


def test_profile_balance_positive():
    profiles = load_financial_profiles()
    for p in profiles.values():
        assert p.current_available_balance >= 0
        assert p.minimum_balance_to_keep >= 0


# ── Financial Events ──────────────────────────────────────────────────────────

def test_events_total_count():
    events_by_user = load_financial_events()
    total = sum(len(v) for v in events_by_user.values())
    assert total == 25342, f"Expected 25342 events, got {total}"


def test_blank_amount_events():
    events_by_user = load_financial_events()
    all_events = [e for evts in events_by_user.values() for e in evts]
    blank = [e for e in all_events if e.amount is None]
    assert len(blank) == 16, f"Expected 16 blank-amount events, got {len(blank)}"


def test_blank_amount_not_zero():
    """Blank amounts must be None, NOT 0.0 — critical spec requirement."""
    events_by_user = load_financial_events()
    all_events = [e for evts in events_by_user.values() for e in evts]
    blank = [e for e in all_events if e.amount is None]
    for e in blank:
        assert e.amount is None, f"Blank amount must be None, not 0 for {e.event_id}"


def test_event_statuses_are_known():
    known = {"settled", "pending", "scheduled", "cancelled", "failed", "unrealized"}
    events_by_user = load_financial_events()
    for evts in events_by_user.values():
        for e in evts:
            assert e.status in known, f"Unknown status: {e.status} on {e.event_id}"


def test_event_directions_are_known():
    known = {"credit", "debit", "non_cash"}
    events_by_user = load_financial_events()
    for evts in events_by_user.values():
        for e in evts:
            assert e.direction in known, f"Unknown direction: {e.direction}"


def test_event_flexibility_values():
    known = {"fixed", "reducible", "reducible_or_stoppable", "stoppable"}
    events_by_user = load_financial_events()
    for evts in events_by_user.values():
        for e in evts:
            assert e.flexibility in known, f"Unknown flexibility: {e.flexibility}"


# ── Exchange Rates ────────────────────────────────────────────────────────────

def test_exchange_rates_count():
    rates = load_exchange_rates()
    assert len(rates) == 134


def test_exchange_rate_fields():
    rates = load_exchange_rates()
    r = rates[0]
    assert r.rate > 0
    assert r.from_currency
    assert r.to_currency
    assert r.rate_date


def test_exchange_rate_pairs():
    rates = load_exchange_rates()
    pairs = set((r.from_currency, r.to_currency) for r in rates)
    expected = {("EUR", "USD"), ("EUR", "ZAR"), ("USD", "EUR"), ("USD", "IDR"), ("USD", "INR")}
    assert pairs == expected


# ── Payment Options ───────────────────────────────────────────────────────────

def test_payment_options_total():
    opts = load_payment_options()
    total = sum(len(v) for v in opts.values())
    assert total == 790


def test_payment_options_per_request():
    opts = load_payment_options()
    counts = [len(v) for v in opts.values()]
    assert min(counts) >= 2
    assert max(counts) <= 4


def test_payment_option_dates():
    """Test that payment_dates() generates the right number of dates."""
    opts = load_payment_options()
    for req_opts in opts.values():
        for o in req_opts:
            if o.payment_method == "installments" and o.number_of_payments > 1:
                dates = o.payment_dates()
                assert len(dates) == o.number_of_payments
                assert dates[0] == o.first_payment_date
                return
    pytest.fail("No installment option found to test")


def test_payment_methods_in_options():
    opts = load_payment_options()
    methods = set()
    for req_opts in opts.values():
        for o in req_opts:
            methods.add(o.payment_method)
    assert methods == {"full_payment", "installments"}


# ── Messages ──────────────────────────────────────────────────────────────────

def test_messages_total():
    msgs = load_messages()
    seen = set()
    for msg_list in msgs.values():
        for m in msg_list:
            seen.add(m.message_id)
    assert len(seen) == 215


def test_messages_indexed_by_event_id():
    msgs = load_messages()
    event_keys = [k for k in msgs.keys() if k.startswith("event_")]
    assert len(event_keys) > 0


def test_messages_indexed_by_request_id():
    msgs = load_messages()
    req_keys = [k for k in msgs.keys() if k.startswith("request_")]
    assert len(req_keys) > 0


# ── Images ────────────────────────────────────────────────────────────────────

def test_images_count():
    imgs = load_images()
    seen = set()
    for img_list in imgs.values():
        for img in img_list:
            seen.add(img.image_id)
    assert len(seen) == 16


def test_blank_amount_events_have_images():
    """Every blank-amount event must have a matching image reference."""
    events_by_user = load_financial_events()
    imgs = load_images()
    all_events = [e for evts in events_by_user.values() for e in evts]
    blank = [e for e in all_events if e.amount is None]

    for e in blank:
        matching = imgs.get(e.event_id, [])
        assert len(matching) > 0, (
            f"Blank-amount event {e.event_id} has no image reference"
        )


def test_images_files_exist():
    """All referenced image files must exist on disk."""
    from buywise.core.config import MEDIA_DIR
    imgs = load_images()
    seen = set()
    for img_list in imgs.values():
        for img in img_list:
            if img.image_id not in seen:
                seen.add(img.image_id)
                path = MEDIA_DIR / f"{img.image_id}.png"
                assert path.exists(), f"Image file missing: {path}"


# ── Dataset Bundle ────────────────────────────────────────────────────────────

def test_dataset_bundle_loads():
    bundle = DatasetBundle()
    assert len(bundle.requests) == 250
    assert len(bundle.profiles) == 275
    assert len(bundle.events_flat) == 25342
    assert len(bundle.exchange_rates) == 134


def test_bundle_get_profile():
    bundle = DatasetBundle()
    profile = bundle.get_profile("user_01")
    assert profile.home_currency == "ZAR"


def test_bundle_get_user_events():
    bundle = DatasetBundle()
    events = bundle.get_user_events("user_01")
    assert len(events) > 0


def test_bundle_get_payment_options():
    bundle = DatasetBundle()
    opts = bundle.get_payment_options("request_01")
    assert len(opts) >= 1


def test_bundle_event_by_id():
    bundle = DatasetBundle()
    e = bundle.get_event_by_id("event_01")
    assert e is not None
    assert e.user_id == "user_01"


def test_bundle_summary_string():
    bundle = DatasetBundle()
    s = bundle.summary()
    assert "250 requests" in s
    assert "blank-amount events" in s


def test_bundle_missing_user_raises():
    bundle = DatasetBundle()
    with pytest.raises(KeyError):
        bundle.get_profile("user_NONEXISTENT")


# ── Sample outputs for regression testing ────────────────────────────────────

def test_sample_outputs_load():
    samples = load_sample_outputs()
    assert len(samples) == 25
    assert "request_01" in samples
    assert samples["request_01"]["affordability_status"] == "affordable_now"
    assert samples["request_01"]["recommended_payment_method"] == "full_payment"


def test_sample_outputs_all_statuses_valid():
    valid = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    samples = load_sample_outputs()
    for rid, row in samples.items():
        assert row["affordability_status"] in valid, (
            f"{rid} has unexpected status: {row['affordability_status']}"
        )
