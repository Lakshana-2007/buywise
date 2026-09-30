"""
Stage 5 tests: affordability calculator and safety validator.

Run with:  python -m pytest tests/test_stage5_affordability.py -v
"""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from buywise.core.models import FinancialProfile, PaymentOption
from buywise.data.loader import DatasetBundle, load_requests
from buywise.finance.affordability import (
    ValidationResult,
    determine_affordability_status,
    installment_duration_months,
    is_installment_option_eligible,
    is_partial_payment_eligible,
    validate_spending_changes,
    validate_strategy,
)
from buywise.finance.events import NormalizedEvent, normalize_events
from buywise.finance.forecast import run_financial_twin
from buywise.graph.state import (
    CandidateStrategy,
    PaymentEntry,
    PaymentPlan,
    SpendingChange,
)


@pytest.fixture(scope="module")
def bundle():
    return DatasetBundle()


def make_profile(**kwargs) -> FinancialProfile:
    defaults = dict(
        user_id="test_user",
        home_currency="ZAR",
        current_available_balance=50000.0,
        minimum_balance_to_keep=10000.0,
        financial_priorities=["education"],
        expense_categories_to_protect=["rent", "groceries"],
        expense_categories_user_is_willing_to_reduce=["dining"],
        expense_categories_user_is_willing_to_stop=["streaming"],
        payment_methods_user_will_consider=["full_payment", "partial_payment", "installments"],
        max_installment_months=6,
    )
    defaults.update(kwargs)
    return FinancialProfile(**defaults)


def make_payment_option(**kwargs) -> PaymentOption:
    defaults = dict(
        payment_option_id="opt_01",
        request_id="req_01",
        payment_method="installments",
        payment_amount=1000.0,
        number_of_payments=3,
        first_payment_date=date(2024, 1, 1),
        payment_frequency_days=30,
        financing_fee=100.0,
        total_payable_amount=3100.0,
    )
    defaults.update(kwargs)
    return PaymentOption(**defaults)


def make_strategy(
    method: str,
    entries: list[tuple[date, float]],
    changes: list[SpendingChange] | None = None,
    option_id: str | None = None,
) -> CandidateStrategy:
    plan = PaymentPlan(entries=[PaymentEntry(d, a) for d, a in entries])
    if option_id:
        plan.payment_option_id = option_id
    return CandidateStrategy(
        payment_method=method,
        payment_plan=plan,
        spending_changes=changes or [],
        payment_option_id=option_id,
    )


def make_recurring_event(
    event_id="ev_recurring",
    category="dining",
    direction="debit",
    amount=1000.0,
    flexibility="reducible_or_stoppable",
    min_amount=None,
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=event_id,
        user_id="test_user",
        event_type="expense",
        category=category,
        direction=direction,
        amount_home=amount,
        original_amount=amount,
        original_currency="ZAR",
        event_date=date(2023, 12, 5),
        settlement_date=date(2023, 12, 5),
        status="settled",
        flexibility=flexibility,
        minimum_allowed_amount=min_amount,
        minimum_allowed_home=min_amount,
        description=category,
        is_recurring=True,
        recurrence_day=5,
        recurrence_interval_days=30,
    )


# ── installment_duration_months ───────────────────────────────────────────────

def test_installment_duration_single_payment():
    opt = make_payment_option(number_of_payments=1, payment_frequency_days=None)
    assert installment_duration_months(opt) == 0.0


def test_installment_duration_3_months():
    opt = make_payment_option(number_of_payments=3, payment_frequency_days=30)
    # (3-1)*30/30 = 2.0 months
    assert installment_duration_months(opt) == pytest.approx(2.0)


def test_installment_duration_18_months():
    opt = make_payment_option(number_of_payments=18, payment_frequency_days=31)
    # (18-1)*31/30 = 17.57 months
    assert installment_duration_months(opt) == pytest.approx(17.57, abs=0.01)


# ── is_installment_option_eligible ───────────────────────────────────────────

def test_installment_eligible_within_limits():
    profile = make_profile(max_installment_months=6)
    opt = make_payment_option(
        number_of_payments=3,
        payment_frequency_days=30,
        first_payment_date=date(2024, 1, 1),
    )
    eligible, reason = is_installment_option_eligible(opt, profile, date(2024, 12, 31))
    assert eligible, reason


def test_installment_ineligible_exceeds_max_months():
    profile = make_profile(max_installment_months=4)
    opt = make_payment_option(
        number_of_payments=18,
        payment_frequency_days=30,
        first_payment_date=date(2024, 1, 1),
    )
    eligible, reason = is_installment_option_eligible(opt, profile, date(2025, 12, 31))
    assert not eligible
    assert "max" in reason.lower()


def test_installment_ineligible_user_rejects():
    profile = make_profile(payment_methods_user_will_consider=["full_payment"])
    opt = make_payment_option(number_of_payments=3, payment_frequency_days=30)
    eligible, reason = is_installment_option_eligible(opt, profile, date(2024, 12, 31))
    assert not eligible


def test_installment_ineligible_after_deadline():
    profile = make_profile(max_installment_months=12)
    opt = make_payment_option(
        number_of_payments=3,
        payment_frequency_days=30,
        first_payment_date=date(2024, 11, 1),
    )
    # Last payment = Jan 1, 2025 (Nov 1 + 60 days = Dec 31, Nov 1 + 61 = Jan 1)
    # With 3 payments, freq=30: dates = Nov 1, Dec 1, Jan 1, 2025
    # desired = Dec 15, 2024 -> last payment Jan 1 2025 > deadline
    eligible, reason = is_installment_option_eligible(opt, profile, date(2024, 12, 15))
    assert not eligible
    assert "desired" in reason.lower() or "after" in reason.lower()


def test_installment_no_max_months_any_duration():
    profile = make_profile(max_installment_months=None)
    opt = make_payment_option(number_of_payments=24, payment_frequency_days=30)
    eligible, _ = is_installment_option_eligible(opt, profile, date(2030, 12, 31))
    assert eligible


# ── is_partial_payment_eligible ───────────────────────────────────────────────

def test_partial_eligible():
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 1),
        desired_completion_date=date(2024, 3, 1),
    )
    assert eligible, reason


def test_partial_ineligible_request_disallows():
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=False,
        profile=profile,
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 1),
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


def test_partial_ineligible_user_rejects():
    profile = make_profile(payment_methods_user_will_consider=["full_payment", "installments"])
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 1),
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


def test_partial_ineligible_safe_amount_zero():
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=0.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 1),
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


def test_partial_ineligible_safe_equals_requested():
    """If amount_safe >= requested, full payment is available — not partial."""
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=10000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 1, 1),
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


def test_partial_ineligible_no_earliest_date():
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=None,
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


def test_partial_ineligible_earliest_after_desired():
    profile = make_profile()
    eligible, reason = is_partial_payment_eligible(
        request_allows=True,
        profile=profile,
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 4, 1),  # After desired
        desired_completion_date=date(2024, 3, 1),
    )
    assert not eligible


# ── validate_spending_changes ─────────────────────────────────────────────────

def test_spending_changes_none_is_valid():
    profile = make_profile()
    valid, violations = validate_spending_changes([], {}, profile)
    assert valid


def test_spending_changes_stop_valid():
    profile = make_profile()
    event = make_recurring_event(event_id="ev1", category="streaming", flexibility="stoppable")
    change = SpendingChange(action="stop", event_id="ev1")
    valid, violations = validate_spending_changes([change], {"ev1": event}, profile)
    assert valid, violations


def test_spending_changes_stop_wrong_category():
    profile = make_profile()
    # "rent" is protected, not in willing_to_stop
    event = make_recurring_event(event_id="ev1", category="rent", flexibility="stoppable")
    change = SpendingChange(action="stop", event_id="ev1")
    valid, violations = validate_spending_changes([change], {"ev1": event}, profile)
    assert not valid
    assert any("willing-to-stop" in v for v in violations)


def test_spending_changes_reduce_valid():
    profile = make_profile()
    event = make_recurring_event(event_id="ev1", category="dining", flexibility="reducible",
                                  amount=2000.0, min_amount=500.0)
    change = SpendingChange(action="reduce_to", event_id="ev1", new_amount=800.0)
    valid, violations = validate_spending_changes([change], {"ev1": event}, profile)
    assert valid, violations


def test_spending_changes_reduce_below_minimum():
    profile = make_profile()
    event = make_recurring_event(event_id="ev1", category="dining", flexibility="reducible",
                                  amount=2000.0, min_amount=500.0)
    change = SpendingChange(action="reduce_to", event_id="ev1", new_amount=100.0)
    valid, violations = validate_spending_changes([change], {"ev1": event}, profile)
    assert not valid
    assert any("minimum" in v.lower() for v in violations)


def test_spending_changes_stop_and_reduce_same_event():
    profile = make_profile()
    event = make_recurring_event(event_id="ev1", category="streaming",
                                  flexibility="reducible_or_stoppable")
    changes = [
        SpendingChange(action="stop", event_id="ev1"),
        SpendingChange(action="reduce_to", event_id="ev1", new_amount=500.0),
    ]
    valid, violations = validate_spending_changes(changes, {"ev1": event}, profile)
    assert not valid
    assert any("mutually exclusive" in v for v in violations)


def test_spending_changes_too_many():
    profile = make_profile()
    events = {f"ev{i}": make_recurring_event(f"ev{i}", "streaming") for i in range(4)}
    changes = [SpendingChange(action="stop", event_id=f"ev{i}") for i in range(4)]
    valid, violations = validate_spending_changes(changes, events, profile)
    assert not valid
    assert any("too many" in v for v in violations)


# ── validate_strategy ─────────────────────────────────────────────────────────

def test_validate_full_payment_affordable():
    profile = make_profile()
    strategy = make_strategy("full_payment", [(date(2024, 1, 1), 20000.0)])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=20000.0,
        desired_completion_date=date(2024, 3, 1),
        allows_partial_payment=True,
        amount_safe_to_pay=20000.0,
        earliest_full_payment_date=date(2024, 1, 1),
    )
    assert result.is_valid, result.violations
    assert result.minimum_projected_balance is not None
    assert result.minimum_projected_balance >= profile.minimum_balance_to_keep


def test_validate_full_payment_breaches_minimum():
    profile = make_profile(
        current_available_balance=15000.0,
        minimum_balance_to_keep=10000.0,
    )
    # Paying 8000 leaves 7000 < 10000 min
    strategy = make_strategy("full_payment", [(date(2024, 1, 1), 8000.0)])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=8000.0,
        desired_completion_date=date(2024, 3, 1),
        allows_partial_payment=True,
        amount_safe_to_pay=4000.0,
        earliest_full_payment_date=None,
    )
    assert not result.is_valid
    assert any("minimum_balance" in v for v in result.violations)


def test_validate_method_not_accepted():
    profile = make_profile(payment_methods_user_will_consider=["full_payment"])
    strategy = make_strategy("installments", [
        (date(2024, 1, 1), 5000.0),
        (date(2024, 2, 1), 5000.0),
    ])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=10000.0,
        desired_completion_date=date(2024, 3, 1),
        allows_partial_payment=False,
        amount_safe_to_pay=10000.0,
        earliest_full_payment_date=date(2024, 1, 1),
    )
    assert not result.is_valid
    assert any("payment_method" in v for v in result.violations)


def test_validate_after_deadline():
    profile = make_profile()
    # Last payment after desired date
    strategy = make_strategy("installments", [
        (date(2024, 1, 1), 5000.0),
        (date(2024, 4, 1), 5000.0),  # After desired March 31
    ])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=10000.0,
        desired_completion_date=date(2024, 3, 31),
        allows_partial_payment=False,
        amount_safe_to_pay=10000.0,
        earliest_full_payment_date=date(2024, 1, 1),
    )
    assert not result.is_valid
    assert any("completion_deadline" in v for v in result.violations)


def test_validate_installment_wrong_payment_count():
    profile = make_profile()
    opt = make_payment_option(number_of_payments=3, payment_frequency_days=30)
    # Strategy has 2 payments but option requires 3
    strategy = make_strategy("installments", [
        (date(2024, 1, 1), 1000.0),
        (date(2024, 2, 1), 1000.0),
    ])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=3100.0,
        desired_completion_date=date(2024, 12, 31),
        allows_partial_payment=False,
        amount_safe_to_pay=3100.0,
        earliest_full_payment_date=date(2024, 1, 1),
        payment_option=opt,
    )
    assert not result.is_valid
    assert any("installment_match" in v for v in result.violations)


def test_validate_partial_payment_correct():
    profile = make_profile()
    strategy = make_strategy("partial_payment", [
        (date(2024, 1, 1), 5000.0),
        (date(2024, 2, 1), 5000.0),
    ])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=10000.0,
        desired_completion_date=date(2024, 3, 1),
        allows_partial_payment=True,
        amount_safe_to_pay=5000.0,
        earliest_full_payment_date=date(2024, 2, 1),
    )
    assert result.is_valid, result.violations


def test_validate_partial_payment_wrong_total():
    profile = make_profile()
    # Payments don't add up to requested_amount
    strategy = make_strategy("partial_payment", [
        (date(2024, 1, 1), 4000.0),
        (date(2024, 2, 1), 5000.0),  # Total = 9000 != 10000
    ])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=date(2024, 1, 1),
        requested_amount=10000.0,
        desired_completion_date=date(2024, 3, 1),
        allows_partial_payment=True,
        amount_safe_to_pay=4000.0,
        earliest_full_payment_date=date(2024, 2, 1),
    )
    assert not result.is_valid
    assert any("partial_payment_rules" in v for v in result.violations)


# ── determine_affordability_status ───────────────────────────────────────────

def test_status_affordable_now():
    profile = make_profile()
    status = determine_affordability_status(
        amount_safe_to_pay=10000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 1, 1),
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=False,
    )
    assert status == "affordable_now"


def test_status_affordable_now_requires_full_payment_method():
    """User who doesn't accept full_payment cannot get affordable_now."""
    profile = make_profile(payment_methods_user_will_consider=["installments"])
    status = determine_affordability_status(
        amount_safe_to_pay=10000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 1, 1),
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=False,
    )
    assert status != "affordable_now"


def test_status_affordable_with_plan():
    profile = make_profile()
    status = determine_affordability_status(
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 1),
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=True,
        selected_method="installments",  # non-wait method
    )
    assert status == "affordable_with_plan"


def test_status_affordable_later():
    profile = make_profile()
    status = determine_affordability_status(
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 15),  # In the future
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=False,
    )
    assert status == "affordable_later"


def test_status_not_affordable():
    profile = make_profile()
    status = determine_affordability_status(
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=None,
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=False,
    )
    assert status == "not_affordable"


def test_status_affordable_later_requires_full_payment_method():
    """Wait strategy only works if user accepts full_payment."""
    profile = make_profile(payment_methods_user_will_consider=["partial_payment"])
    status = determine_affordability_status(
        amount_safe_to_pay=5000.0,
        requested_amount=10000.0,
        earliest_full_payment_date=date(2024, 2, 15),
        request_date=date(2024, 1, 1),
        profile=profile,
        has_valid_strategy=False,
    )
    assert status == "not_affordable"


# ── Integration: validate real sample cases ──────────────────────────────────

def test_validate_request_01_full_payment(bundle):
    """request_01: full_payment should be valid for user_01."""
    sample_reqs = load_requests(sample=True)
    req = next(r for r in sample_reqs if r.request_id == "request_01")
    profile = bundle.get_profile(req.user_id)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    strategy = make_strategy("full_payment", [(req.request_date, req.requested_amount)])
    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
        allows_partial_payment=req.allows_partial_payment,
        amount_safe_to_pay=req.requested_amount,
        earliest_full_payment_date=req.request_date,
    )
    assert result.is_valid, result.violations
    assert result.minimum_projected_balance >= profile.minimum_balance_to_keep


def test_validate_request_02_installments(bundle):
    """request_02: 3-payment installment plan should be valid."""
    sample_reqs = load_requests(sample=True)
    req = next(r for r in sample_reqs if r.request_id == "request_02")
    profile = bundle.get_profile(req.user_id)
    opts = bundle.get_payment_options(req.request_id)
    # Payment option 05: 3 payments x 15952906.67
    opt = next(o for o in opts if o.number_of_payments == 3)
    events = bundle.get_user_events(req.user_id)
    normalized = normalize_events(events, profile.home_currency, bundle.exchange_rates)

    plan_entries = [(d, opt.payment_amount) for d in opt.payment_dates()]
    strategy = make_strategy("installments", plan_entries, option_id=opt.payment_option_id)

    twin = run_financial_twin(
        profile.current_available_balance, profile.minimum_balance_to_keep,
        normalized, req.request_date, req.requested_amount, req.desired_completion_date,
    )

    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=normalized,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
        allows_partial_payment=req.allows_partial_payment,
        amount_safe_to_pay=twin.amount_safe_to_pay,
        earliest_full_payment_date=twin.earliest_full_payment_date,
        payment_option=opt,
    )
    assert result.is_valid, result.violations


def test_validate_installment_exceeds_max_months(bundle):
    """request_10: 15-payment installment plan exceeds max_installment_months=6."""
    sample_reqs = load_requests(sample=True)
    req = next(r for r in sample_reqs if r.request_id == "request_10")
    profile = bundle.get_profile(req.user_id)
    opts = bundle.get_payment_options(req.request_id)
    # Only installment option has 15 payments (14 months > max 6)
    inst_opts = [o for o in opts if o.payment_method == "installments"]
    assert len(inst_opts) > 0

    opt = inst_opts[0]
    plan_entries = [(d, opt.payment_amount) for d in opt.payment_dates()]
    strategy = make_strategy("installments", plan_entries, option_id=opt.payment_option_id)

    result = validate_strategy(
        strategy=strategy,
        profile=profile,
        events=[],
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        desired_completion_date=req.desired_completion_date,
        allows_partial_payment=req.allows_partial_payment,
        amount_safe_to_pay=0.0,
        earliest_full_payment_date=None,
        payment_option=opt,
    )
    assert not result.is_valid
    assert any("installment_eligibility" in v for v in result.violations)
