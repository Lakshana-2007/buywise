"""
Stage 4 tests: 90-day Financial Twin simulation engine.

Tests cover:
- Basic balance simulation
- Recurring expense projection
- Income projection
- amount_safe_to_pay calculation
- earliest_full_payment_date calculation
- Spending changes (stop/reduce)
- Multi-payment plan simulation
- Regression against sample answers

Run with:  python -m pytest tests/test_stage4_financial_twin.py -v
"""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from buywise.data.loader import DatasetBundle
from buywise.finance.events import NormalizedEvent, normalize_events
from buywise.finance.forecast import (
    DayEntry,
    SimulationResult,
    FinancialTwinResult,
    _binary_search_safe_amount,
    _build_cash_flow,
    _find_earliest_full_payment_date,
    run_financial_twin,
    simulate,
    simulate_payment_plan,
)


@pytest.fixture(scope="module")
def bundle():
    return DatasetBundle()


# ── DayEntry ──────────────────────────────────────────────────────────────────

def test_day_entry_net():
    entry = DayEntry(income=5000.0, expenses=3000.0)
    assert entry.net == 2000.0


# ── Basic simulation ──────────────────────────────────────────────────────────

def test_simulate_no_events_no_payment():
    """With no events and no payment, balance stays constant."""
    result = simulate(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        payment_amount=0.0,
        payment_date=date(2024, 1, 1),
        forecast_days=30,
    )
    assert result.is_safe
    assert result.minimum_projected_balance == pytest.approx(10000.0, abs=1.0)


def test_simulate_payment_reduces_balance():
    result = simulate(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        payment_amount=3000.0,
        payment_date=date(2024, 1, 1),
        forecast_days=30,
    )
    assert result.is_safe
    # Balance on day 1 should be 7000
    assert result.balance_on(date(2024, 1, 1)) == pytest.approx(7000.0, abs=1.0)


def test_simulate_payment_breaches_minimum():
    result = simulate(
        starting_balance=10000.0,
        minimum_balance=8000.0,
        events=[],
        request_date=date(2024, 1, 1),
        payment_amount=5000.0,  # leaves only 5000 < min 8000
        payment_date=date(2024, 1, 1),
        forecast_days=30,
    )
    assert not result.is_safe
    assert result.minimum_projected_balance == pytest.approx(5000.0, abs=1.0)


def test_simulate_future_expense_causes_breach():
    """A future expense should be reflected in the simulation."""
    # A recurring expense fires at beginning of each month
    event = NormalizedEvent(
        event_id="future_exp",
        user_id="test",
        event_type="expense",
        category="rent",
        direction="debit",
        amount_home=8000.0,
        original_amount=8000.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 1),
        settlement_date=date(2023, 12, 1),
        status="settled",
        flexibility="fixed",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="rent",
        is_recurring=True,
        recurrence_day=1,
        recurrence_interval_days=30,
    )
    result = simulate(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[event],
        request_date=date(2024, 1, 2),   # After last known occurrence
        payment_amount=0.0,
        payment_date=date(2024, 1, 2),
        forecast_days=30,
    )
    # Rent of 8000 fires on 2024-02-01, balance would be 10000 - 8000 = 2000 < 5000
    assert not result.is_safe


def test_simulate_income_prevents_breach():
    """Income should raise the balance before an expense fires."""
    salary = NormalizedEvent(
        event_id="salary_ev",
        user_id="test",
        event_type="income",
        category="salary",
        direction="credit",
        amount_home=10000.0,
        original_amount=10000.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 15),
        settlement_date=date(2023, 12, 15),
        status="settled",
        flexibility="fixed",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="salary",
        is_recurring=True,
        recurrence_day=15,
        recurrence_interval_days=30,
    )
    rent = NormalizedEvent(
        event_id="rent_ev",
        user_id="test",
        event_type="expense",
        category="rent",
        direction="debit",
        amount_home=8000.0,
        original_amount=8000.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 1),
        settlement_date=date(2023, 12, 1),
        status="settled",
        flexibility="fixed",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="rent",
        is_recurring=True,
        recurrence_day=1,
        recurrence_interval_days=30,
    )
    result = simulate(
        starting_balance=9000.0,
        minimum_balance=5000.0,
        events=[salary, rent],
        request_date=date(2024, 1, 2),
        payment_amount=0.0,
        payment_date=date(2024, 1, 2),
        forecast_days=30,
    )
    # 2024-01-15: +10000 salary → balance = 19000
    # 2024-02-01: -8000 rent → balance = 11000
    assert result.is_safe


# ── Binary search for safe amount ─────────────────────────────────────────────

def test_binary_search_full_amount_affordable():
    result = _binary_search_safe_amount(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        max_amount=4000.0,
        spending_changes=None,
        forecast_days=30,
    )
    assert result == pytest.approx(4000.0, abs=0.1)


def test_binary_search_partial_amount():
    """When full amount not safe, returns partial amount."""
    result = _binary_search_safe_amount(
        starting_balance=10000.0,
        minimum_balance=8000.0,
        events=[],
        request_date=date(2024, 1, 1),
        max_amount=5000.0,
        spending_changes=None,
        forecast_days=30,
    )
    # Can only afford 10000 - 8000 = 2000
    assert result == pytest.approx(2000.0, abs=10.0)


def test_binary_search_zero_if_no_headroom():
    result = _binary_search_safe_amount(
        starting_balance=5000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        max_amount=1000.0,
        spending_changes=None,
        forecast_days=30,
    )
    assert result == pytest.approx(0.0, abs=0.1)


# ── Earliest full payment date ────────────────────────────────────────────────

def test_earliest_full_payment_request_date_when_affordable():
    result = _find_earliest_full_payment_date(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=4000.0,
        end_date=date(2024, 4, 1),
        spending_changes=None,
        forecast_days=90,
    )
    assert result == date(2024, 1, 1)


def test_earliest_full_payment_after_income():
    """With a future salary, earliest date should be after salary arrives."""
    salary = NormalizedEvent(
        event_id="sal1",
        user_id="test",
        event_type="income",
        category="salary",
        direction="credit",
        amount_home=5000.0,
        original_amount=5000.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 15),
        settlement_date=date(2023, 12, 15),
        status="settled",
        flexibility="fixed",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="salary",
        is_recurring=True,
        recurrence_day=15,
        recurrence_interval_days=30,
    )
    result = _find_earliest_full_payment_date(
        starting_balance=3000.0,   # Not enough to pay 2000 and keep 2000 min
        minimum_balance=2000.0,
        events=[salary],
        request_date=date(2024, 1, 1),
        requested_amount=2000.0,   # 3000 - 2000 = 1000 < min(2000), not safe yet
        end_date=date(2024, 4, 1),
        spending_changes=None,
        forecast_days=90,
    )
    # After salary on Jan 15: balance = 3000 + 5000 = 8000, can pay 2000 and keep 6000 > 2000
    assert result is not None
    assert result >= date(2024, 1, 15)


def test_earliest_full_payment_returns_none_if_never_safe():
    """When there's no income and balance insufficient, should return None."""
    result = _find_earliest_full_payment_date(
        starting_balance=1000.0,
        minimum_balance=900.0,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=500.0,   # Would leave only 500, below 900 min — no income to recover
        end_date=date(2024, 3, 31),   # Within 90-day window
        spending_changes=None,
        forecast_days=90,
    )
    assert result is None


# ── Multi-payment simulation ──────────────────────────────────────────────────

def test_simulate_payment_plan_two_payments():
    result = simulate_payment_plan(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        payment_schedule=[
            (date(2024, 1, 1), 2000.0),
            (date(2024, 2, 1), 2000.0),
        ],
        forecast_days=90,
    )
    # After first payment: 8000; after second: 6000 > 5000 min
    assert result.is_safe
    assert result.balance_on(date(2024, 1, 1)) == pytest.approx(8000.0, abs=1.0)


def test_simulate_payment_plan_breaches_on_second_payment():
    result = simulate_payment_plan(
        starting_balance=10000.0,
        minimum_balance=8000.0,
        events=[],
        request_date=date(2024, 1, 1),
        payment_schedule=[
            (date(2024, 1, 1), 1000.0),   # OK: 9000 > 8000
            (date(2024, 2, 1), 1500.0),   # Breach: 7500 < 8000
        ],
        forecast_days=90,
    )
    assert not result.is_safe


# ── Financial Twin integration ─────────────────────────────────────────────────

def test_run_financial_twin_immediately_affordable():
    result = run_financial_twin(
        starting_balance=10000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=4000.0,
        desired_completion_date=date(2024, 3, 1),
    )
    assert result.amount_safe_to_pay == pytest.approx(4000.0, abs=1.0)
    assert result.earliest_full_payment_date == date(2024, 1, 1)


def test_run_financial_twin_not_immediately_affordable():
    result = run_financial_twin(
        starting_balance=6000.0,
        minimum_balance=5000.0,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=3000.0,  # 6000 - 3000 = 3000 < 5000 min
        desired_completion_date=date(2024, 3, 1),
    )
    assert result.amount_safe_to_pay == pytest.approx(1000.0, abs=10.0)  # 6000 - 5000
    assert result.earliest_full_payment_date is None  # no future income


def test_run_financial_twin_with_spending_change():
    """Stopping an expense should increase amount_safe_to_pay."""
    # A subscription fires on day 5, costing 800/month
    subscription = NormalizedEvent(
        event_id="sub_ev",
        user_id="test",
        event_type="subscription",
        category="streaming",
        direction="debit",
        amount_home=800.0,
        original_amount=800.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 5),
        settlement_date=date(2023, 12, 5),
        status="settled",
        flexibility="stoppable",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="streaming",
        is_recurring=True,
        recurrence_day=5,
        recurrence_interval_days=30,
    )
    baseline = run_financial_twin(
        starting_balance=8500.0,
        minimum_balance=8000.0,
        events=[subscription],
        request_date=date(2024, 1, 1),
        requested_amount=1000.0,
        desired_completion_date=date(2024, 3, 1),
    )
    with_change = run_financial_twin(
        starting_balance=8500.0,
        minimum_balance=8000.0,
        events=[subscription],
        request_date=date(2024, 1, 1),
        requested_amount=1000.0,
        desired_completion_date=date(2024, 3, 1),
        spending_changes={"sub_ev": None},  # stop the subscription
    )
    # Stopping the subscription should allow a larger safe payment
    assert with_change.amount_safe_to_pay >= baseline.amount_safe_to_pay


# ── Regression tests against sample answers ──────────────────────────────────

@pytest.mark.parametrize("request_id,expected_status,expected_earliest", [
    # request_01: affordable_now, earliest=request_date — works without LLM
    ("request_01", "affordable_now", "2024-03-03"),
])
def test_regression_earliest_date(bundle, request_id, expected_status, expected_earliest):
    """Verify earliest_full_payment_date against sample answers (no LLM)."""
    from buywise.data.loader import load_sample_outputs, load_requests
    samples = load_sample_outputs()
    sample_requests = load_requests(sample=True)

    req = next(r for r in sample_requests if r.request_id == request_id)
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    twin = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
    )

    expected_date = date.fromisoformat(expected_earliest) if expected_earliest else None

    if expected_earliest:
        assert twin.earliest_full_payment_date is not None, (
            f"{request_id}: Expected earliest={expected_earliest}, got None"
        )
        # Allow ±1 day tolerance for boundary effects
        delta = abs((twin.earliest_full_payment_date - expected_date).days)
        assert delta <= 1, (
            f"{request_id}: Expected {expected_earliest}, got {twin.earliest_full_payment_date}"
        )


def test_regression_request_03_needs_salary_projection(bundle):
    """
    request_03: user_03 has a blank salary event (needs image extraction).
    Without LLM extraction, salary is unknown → verify recurring salary IS projected
    once recurrence is detected from the 4 known salary amounts.
    """
    from buywise.data.loader import load_requests
    sample_requests = load_requests(sample=True)
    req = next(r for r in sample_requests if r.request_id == "request_03")
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    # Salary should be detected as recurring from the non-blank occurrences
    recurring_income = [e for e in normalized if e.is_recurring and e.is_income]
    # After fixing recurrence detection, salary (4365000 IDR, day 15) should be detected
    # Even if one entry (the blank one) is skipped
    assert len(recurring_income) > 0, "Salary should be detected as recurring"

    twin = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
    )
    # With salary projected, earliest date should be within 90 days
    assert twin.earliest_full_payment_date is not None


def test_regression_request_08_recurring_salary_detection(bundle):
    """
    request_08: user_08 has salary entries including one amended amount.
    With fixed recurrence detection (60% threshold), salary should be detected.
    """
    from buywise.data.loader import load_requests
    sample_requests = load_requests(sample=True)
    req = next(r for r in sample_requests if r.request_id == "request_08")
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    salary = [e for e in normalized if e.category == "salary"]
    recurring_salary = [e for e in salary if e.is_recurring]
    assert len(recurring_salary) > 0, (
        "user_08 salary (4/5 entries at 1422.85) should be detected as recurring"
    )

    twin = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
    )
    # With recurring salary projected, earliest date should be found
    assert twin.earliest_full_payment_date is not None


def test_request_17_daily_safety_overrides_payment_date_only(bundle):
    """Option 47 is unsafe between payments; option 49 misses the deadline."""
    from buywise.data.loader import load_requests

    req = next(r for r in load_requests(sample=True) if r.request_id == "request_17")
    profile = bundle.get_profile(req.user_id)
    normalized = normalize_events(
        bundle.get_user_events(req.user_id),
        profile.home_currency,
        bundle.exchange_rates,
    )
    options = {o.payment_option_id: o for o in bundle.get_payment_options(req.request_id)}

    option_47 = options["payment_option_47"]
    schedule_47 = [(d, option_47.payment_amount) for d in option_47.payment_dates()]
    sim_47 = simulate_payment_plan(
        profile.current_available_balance,
        profile.minimum_balance_to_keep,
        normalized,
        req.request_date,
        schedule_47,
    )
    payment_date_minimum = min(sim_47.daily_balance[d] for d, _ in schedule_47)
    assert payment_date_minimum == pytest.approx(253319.24, abs=0.01)
    assert sim_47.minimum_projected_balance == pytest.approx(146648.24, abs=0.01)
    assert profile.minimum_balance_to_keep == pytest.approx(166100.00, abs=0.01)
    assert payment_date_minimum >= profile.minimum_balance_to_keep
    assert sim_47.minimum_projected_balance < profile.minimum_balance_to_keep
    assert not sim_47.is_safe

    option_49 = options["payment_option_49"]
    assert max(option_49.payment_dates()) > req.desired_completion_date


def test_request_21_expected_spending_changes_do_not_pass_90_day_safety(bundle):
    """The sample plan for request_21 conflicts with the 90-day invariant."""
    from buywise.data.loader import load_requests

    req = next(r for r in load_requests(sample=True) if r.request_id == "request_21")
    profile = bundle.get_profile(req.user_id)
    normalized = normalize_events(
        bundle.get_user_events(req.user_id),
        profile.home_currency,
        bundle.exchange_rates,
    )
    cloud_storage = max(
        (event for event in normalized if event.category == "cloud_storage"),
        key=lambda event: event.effective_date,
    )
    streaming = max(
        (event for event in normalized if event.category == "streaming"),
        key=lambda event: event.effective_date,
    )
    changes = {
        cloud_storage.event_id: None,
        streaming.event_id: round(streaming.amount_home * 0.5, 2),
    }
    simulation = simulate_payment_plan(
        profile.current_available_balance,
        profile.minimum_balance_to_keep,
        normalized,
        req.request_date,
        [(req.request_date, req.requested_amount)],
        changes,
    )
    assert simulation.minimum_projected_balance < profile.minimum_balance_to_keep
    assert not simulation.is_safe


def test_regression_request_01_amount_safe(bundle):
    """request_01: user_01 can afford full ZAR 25256 immediately."""
    from buywise.data.loader import load_requests
    sample_requests = load_requests(sample=True)
    req = next(r for r in sample_requests if r.request_id == "request_01")
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    twin = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
    )

    assert twin.amount_safe_to_pay == pytest.approx(req.requested_amount, abs=1.0)
    assert twin.earliest_full_payment_date == req.request_date


def test_regression_request_05_not_affordable(bundle):
    """
    request_05: Sample says not_affordable with amount_safe=737.
    Note: the sample answer may depend on message/image evidence not yet
    processed (Stage 8). This test verifies the core simulator behavior
    with available data only — full accuracy requires LLM evidence extraction.
    """
    from buywise.data.loader import load_requests
    sample_requests = load_requests(sample=True)
    req = next(r for r in sample_requests if r.request_id == "request_05")
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    twin = run_financial_twin(
        starting_balance=profile.current_available_balance,
        minimum_balance=profile.minimum_balance_to_keep,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
    )
    # Core invariants that must always hold regardless of evidence
    assert twin.amount_safe_to_pay >= 0
    assert twin.amount_safe_to_pay <= req.requested_amount


def test_spending_change_stop_event():
    """Stopping an event should completely remove its future cash flows."""
    monthly_fee = NormalizedEvent(
        event_id="fee_ev",
        user_id="test",
        event_type="subscription",
        category="cloud_storage",
        direction="debit",
        amount_home=500.0,
        original_amount=500.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 10),
        settlement_date=date(2023, 12, 10),
        status="settled",
        flexibility="stoppable",
        minimum_allowed_amount=None,
        minimum_allowed_home=None,
        description="cloud",
        is_recurring=True,
        recurrence_day=10,
        recurrence_interval_days=30,
    )
    # Without stop: subscription fires and may breach minimum
    # With stop: subscription removed
    flow_without = _build_cash_flow([monthly_fee], date(2024, 1, 1), date(2024, 4, 1))
    flow_with = _build_cash_flow([monthly_fee], date(2024, 1, 1), date(2024, 4, 1),
                                  spending_changes={"fee_ev": None})

    # With stop, there should be no expense entries from this event
    total_without = sum(e.expenses for e in flow_without.values())
    total_with = sum(e.expenses for e in flow_with.values())
    assert total_with < total_without


def test_spending_change_reduce_event():
    """Reducing an event should use the new (lower) amount."""
    monthly_fee = NormalizedEvent(
        event_id="dining_ev",
        user_id="test",
        event_type="expense",
        category="dining",
        direction="debit",
        amount_home=2000.0,
        original_amount=2000.0,
        original_currency="ZAR",
        event_date=date(2023, 12, 15),
        settlement_date=date(2023, 12, 15),
        status="settled",
        flexibility="reducible",
        minimum_allowed_amount=500.0,
        minimum_allowed_home=500.0,
        description="dining",
        is_recurring=True,
        recurrence_day=15,
        recurrence_interval_days=30,
    )
    flow_full = _build_cash_flow([monthly_fee], date(2024, 1, 1), date(2024, 4, 1))
    flow_reduced = _build_cash_flow([monthly_fee], date(2024, 1, 1), date(2024, 4, 1),
                                     spending_changes={"dining_ev": 500.0})

    total_full = sum(e.expenses for e in flow_full.values())
    total_reduced = sum(e.expenses for e in flow_reduced.values())
    assert total_reduced < total_full
    # Reduced amount should be 500, not 2000
    assert total_reduced == pytest.approx(500.0 * 3, abs=50.0)  # ~3 months in window
