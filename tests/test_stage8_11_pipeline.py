"""
Stage 8–11 tests: end-to-end pipeline, output validation, and regression.

Tests the complete BUYWISE pipeline in MOCK mode (no LLM calls).

Run with:  python -m pytest tests/test_stage8_11_pipeline.py -v
"""

import csv
import os
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["MOCK_LLM"] = "true"

from buywise.ai.extractor import DecisionExplainer, EvidenceExtractor, _deterministic_explanation
from buywise.ai.provider import GeminiProvider, MockProvider, get_llm_provider
from buywise.core.models import OutputRow
from buywise.data.loader import DatasetBundle, load_requests, load_sample_outputs
from buywise.data.joins import build_request_context
from buywise.graph.state import FinancialState
from buywise.graph.workflow import BuywisePipeline, build_pipeline
from buywise.validation.output_validator import (
    ValidationError,
    validate_and_write_output,
    validate_output_semantics,
    validate_output_row,
)


@pytest.fixture(scope="module")
def bundle():
    return DatasetBundle()


@pytest.fixture(scope="module")
def pipeline():
    provider = MockProvider()
    return build_pipeline(provider)


# ── Provider tests ─────────────────────────────────────────────────────────────

def test_mock_provider_instantiates():
    p = MockProvider()
    assert p.provider_name == "mock"
    assert p.model_name == "mock-deterministic"


def test_get_llm_provider_returns_mock():
    p = get_llm_provider()
    assert p.provider_name == "mock"


def test_get_llm_provider_requires_explicit_real_configuration(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_PROVIDER is required"):
        get_llm_provider()


def test_get_llm_provider_returns_gemini(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    provider = get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-flash"


def test_gemini_provider_constructs_chat_model_without_network(monkeypatch):
    import types

    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        types.SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    provider = GeminiProvider(model="gemini-2.5-flash")
    provider.get_chat_model()
    assert captured == {
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "google_api_key": "test-google-key",
    }


# ── Deterministic explanation ─────────────────────────────────────────────────

def test_explanation_affordable_now():
    text = _deterministic_explanation(
        affordability_status="affordable_now",
        recommended_payment_method="full_payment",
        amount_safe_to_pay=25256.0,
        currency="ZAR",
        payment_plan="2024-03-03:25256.0",
        earliest_date="2024-03-03",
        spending_changes="none",
        required_minimum_balance=18000.0,
        min_projected_balance=25000.0,
    )
    assert "25,256.00" in text or "25256" in text
    assert "ZAR" in text


def test_explanation_not_affordable():
    text = _deterministic_explanation(
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        amount_safe_to_pay=100.0,
        currency="EUR",
        payment_plan="none",
        earliest_date="",
        spending_changes="none",
        required_minimum_balance=800.0,
        min_projected_balance=850.0,
    )
    assert "800" in text


def test_explanation_affordable_later():
    text = _deterministic_explanation(
        affordability_status="affordable_later",
        recommended_payment_method="wait",
        amount_safe_to_pay=5000.0,
        currency="IDR",
        payment_plan="2019-11-15:5491000.0",
        earliest_date="2019-11-15",
        spending_changes="none",
        required_minimum_balance=2668700.0,
        min_projected_balance=2800000.0,
    )
    assert "2019-11-15" in text


def test_explanation_installments():
    text = _deterministic_explanation(
        affordability_status="affordable_with_plan",
        recommended_payment_method="installments",
        amount_safe_to_pay=46018000.0,
        currency="IDR",
        payment_plan="2025-08-08:15952906.67|2025-09-07:15952906.67|2025-10-07:15952906.67",
        earliest_date="2025-09-15",
        spending_changes="none",
        required_minimum_balance=29158400.0,
        min_projected_balance=30000000.0,
    )
    assert "installment" in text.lower() or "3" in text


def test_explanation_with_spending_change():
    text = _deterministic_explanation(
        affordability_status="affordable_with_plan",
        recommended_payment_method="full_payment",
        amount_safe_to_pay=620.4,
        currency="EUR",
        payment_plan="2026-01-03:620.4",
        earliest_date="2026-01-03",
        spending_changes="stop:event_476",
        required_minimum_balance=800.0,
        min_projected_balance=850.0,
    )
    assert "event_476" in text


# ── Pipeline end-to-end ───────────────────────────────────────────────────────

def test_pipeline_processes_single_request(bundle, pipeline):
    """Single request processes without error and produces valid output."""
    req = bundle.requests[0]
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    assert state.output_row is not None
    row = state.output_row
    assert row.request_id == req.request_id
    assert 0 <= row.amount_safe_to_pay <= req.requested_amount
    assert row.affordability_status in (
        "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
    )
    assert row.recommended_payment_method in (
        "full_payment", "partial_payment", "installments", "wait", "not_recommended"
    )
    assert row.payment_plan  # not empty
    assert row.decision_explanation  # not empty


def test_pipeline_processes_all_sample_requests(bundle, pipeline):
    """All 25 sample requests process without errors."""
    sample_reqs = load_requests(sample=True)
    errors = 0
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        assert state.output_row is not None, f"No output for {req.request_id}"
        if state.errors:
            errors += 1
    assert errors == 0, f"{errors} requests had pipeline errors"


def test_pipeline_amount_safe_bounded(bundle, pipeline):
    """amount_safe_to_pay must always be in [0, requested_amount]."""
    sample_reqs = load_requests(sample=True)
    for req in sample_reqs[:10]:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        row = state.output_row
        assert row.amount_safe_to_pay >= 0, f"{req.request_id}: negative safe amount"
        assert row.amount_safe_to_pay <= req.requested_amount + 0.01, (
            f"{req.request_id}: safe amount {row.amount_safe_to_pay} > requested {req.requested_amount}"
        )


def test_pipeline_affordable_now_has_earliest_date(bundle, pipeline):
    """affordable_now must always have earliest_date_for_full_payment = request_date."""
    sample_reqs = load_requests(sample=True)
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        row = state.output_row
        if row.affordability_status == "affordable_now":
            assert row.earliest_date_for_full_payment == str(req.request_date), (
                f"{req.request_id}: affordable_now must have earliest={req.request_date}"
            )


def test_pipeline_not_affordable_empty_earliest_date(bundle, pipeline):
    """not_affordable must have empty earliest_date_for_full_payment."""
    sample_reqs = load_requests(sample=True)
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        row = state.output_row
        if row.affordability_status == "not_affordable":
            assert row.earliest_date_for_full_payment == "", (
                f"{req.request_id}: not_affordable should have empty earliest date, "
                f"got {row.earliest_date_for_full_payment!r}"
            )


def test_pipeline_payment_plan_format(bundle, pipeline):
    """Payment plans must be 'none' or valid date:amount entries."""
    import re
    plan_re = re.compile(r"^\d{4}-\d{2}-\d{2}:[0-9]+(\.[0-9]+)?$")
    sample_reqs = load_requests(sample=True)
    for req in sample_reqs[:15]:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        row = state.output_row
        if row.payment_plan != "none":
            parts = row.payment_plan.split("|")
            for p in parts:
                assert plan_re.match(p.strip()), (
                    f"{req.request_id}: malformed plan entry: {p!r}"
                )


# ── Output validator tests ─────────────────────────────────────────────────────

def make_valid_row(**kwargs) -> OutputRow:
    defaults = dict(
        request_id="req_test",
        amount_safe_to_pay=1000.0,
        affordability_status="affordable_now",
        recommended_payment_method="full_payment",
        payment_plan="2024-01-01:1000.0",
        earliest_date_for_full_payment="2024-01-01",
        spending_changes_needed="none",
        decision_explanation="Pay 1000.0 today. This leaves enough buffer.",
    )
    defaults.update(kwargs)
    return OutputRow(**defaults)


def test_validator_clean_row():
    row = make_valid_row()
    warnings = validate_output_row(row, requested_amount=1000.0)
    assert warnings == []


def test_validator_amount_below_zero():
    """OutputRow Pydantic model enforces amount_safe_to_pay >= 0."""
    from pydantic import ValidationError as PydanticValidationError
    with pytest.raises(PydanticValidationError):
        make_valid_row(amount_safe_to_pay=-1.0)


def test_validator_amount_above_requested():
    row = make_valid_row(amount_safe_to_pay=1500.0)
    warnings = validate_output_row(row, requested_amount=1000.0)
    assert any("amount_safe_to_pay" in w for w in warnings)


def test_validator_invalid_status():
    row = make_valid_row(affordability_status="unknown_status")
    warnings = validate_output_row(row, requested_amount=1000.0)
    assert any("affordability_status" in w for w in warnings)


def test_validator_malformed_plan():
    row = make_valid_row(payment_plan="not-a-date:1000")
    warnings = validate_output_row(row, requested_amount=1000.0)
    assert any("payment_plan" in w for w in warnings)


def test_semantic_validator_rejects_wrong_full_payment_plan(bundle):
    req = load_requests(sample=True)[0]
    ctx = build_request_context(req, bundle)
    row = make_valid_row(
        request_id=req.request_id,
        amount_safe_to_pay=req.requested_amount,
        payment_plan=f"{req.request_date}:1.0",
        earliest_date_for_full_payment=str(req.request_date),
    )
    warnings = validate_output_semantics(
        row=row,
        request=ctx.request,
        profile=ctx.profile,
        payment_options=ctx.payment_options,
        normalized_events=[],
    )
    assert any("full_payment plan does not equal" in warning for warning in warnings)


def test_validator_too_many_changes():
    row = make_valid_row(spending_changes_needed="stop:e1|stop:e2|stop:e3|stop:e4")
    warnings = validate_output_row(row, requested_amount=1000.0)
    assert any("spending_change" in w for w in warnings)


def test_validate_and_write_output(tmp_path, bundle):
    """Full CSV write roundtrip."""
    provider = MockProvider()
    pl = build_pipeline(provider)
    sample_reqs = load_requests(sample=True)[:5]
    rows = []
    amounts = {}
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pl.run(state)
        rows.append(state.output_row)
        amounts[req.request_id] = req.requested_amount

    out_path = tmp_path / "test_output.csv"
    success, warnings = validate_and_write_output(
        output_rows=rows,
        requested_amounts=amounts,
        output_path=out_path,
        expected_request_ids=[r.request_id for r in sample_reqs],
    )
    assert success
    assert out_path.exists()

    # Read back and verify
    with open(out_path, encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 5
    expected_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation",
    ]
    assert list(written[0].keys()) == expected_cols


def test_validate_wrong_count_raises(tmp_path):
    rows = [make_valid_row(request_id="req_1"), make_valid_row(request_id="req_2")]
    with pytest.raises(ValidationError, match="Expected 3"):
        validate_and_write_output(
            output_rows=rows,
            requested_amounts={"req_1": 1000.0, "req_2": 1000.0, "req_3": 1000.0},
            output_path=tmp_path / "out.csv",
            expected_request_ids=["req_1", "req_2", "req_3"],
        )


def test_validate_duplicate_ids_raises(tmp_path):
    rows = [make_valid_row(request_id="req_1"), make_valid_row(request_id="req_1")]
    with pytest.raises(ValidationError, match="Duplicate"):
        validate_and_write_output(
            output_rows=rows,
            requested_amounts={"req_1": 1000.0},
            output_path=tmp_path / "out.csv",
            expected_request_ids=["req_1", "req_1"],
        )


# ── Regression against sample answers ────────────────────────────────────────

@pytest.mark.parametrize("request_id,expected_status,expected_method", [
    ("request_01", "affordable_now", "full_payment"),
    ("request_02", "affordable_with_plan", "installments"),
    ("request_03", "affordable_later", "wait"),
    ("request_07", "affordable_with_plan", "installments"),
    ("request_08", "affordable_later", "wait"),
    ("request_09", "affordable_now", "full_payment"),
    ("request_10", "not_affordable", "not_recommended"),
    ("request_12", "affordable_with_plan", "installments"),
    ("request_13", "affordable_later", "wait"),
    ("request_22", "affordable_with_plan", "installments"),
])
def test_regression_status_and_method(bundle, pipeline, request_id, expected_status, expected_method):
    """Core regression: status and method match sample answers."""
    sample_reqs = load_requests(sample=True)
    req = next(r for r in sample_reqs if r.request_id == request_id)
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)
    row = state.output_row
    assert row.affordability_status == expected_status, (
        f"{request_id}: expected status={expected_status}, got={row.affordability_status}"
    )
    assert row.recommended_payment_method == expected_method, (
        f"{request_id}: expected method={expected_method}, got={row.recommended_payment_method}"
    )


# ── Diagnostic artifact tests ─────────────────────────────────────────────────

import json
from buywise.evaluation.diagnostics import build_diagnostic_record, write_diagnostic_jsonl
from buywise.graph.state import SafetyProof, SpendingChange, CandidateStrategy, PaymentPlan


# ── build_diagnostic_record unit tests ───────────────────────────────────────

def test_build_diagnostic_record_returns_dict(bundle, pipeline):
    """build_diagnostic_record must return a dict without raising."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_01")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    assert isinstance(rec, dict)


def test_build_diagnostic_record_required_keys(bundle, pipeline):
    """Every diagnostic record must contain the fields needed for downstream analysis."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_01")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)

    required_keys = {
        "request_id",
        "user_id",
        "evidence_facts",
        "normalized_events",
        "financial_twin",
        "candidate_strategies",
        "selected_strategy",
        "safety_proof",
        "explanation_fallback_used",
        "failure_counts",
        "errors",
        "warnings",
    }
    missing = required_keys - rec.keys()
    assert not missing, f"Diagnostic record missing keys: {missing}"


def test_build_diagnostic_record_financial_twin_subkeys(bundle, pipeline):
    """The financial_twin sub-dict must contain the three key observability fields."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_01")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    twin = rec["financial_twin"]
    assert "amount_safe_to_pay" in twin
    assert "earliest_full_payment_date" in twin
    assert "minimum_balance_for_full_payment" in twin


def test_build_diagnostic_record_safety_proof_serialisable(bundle, pipeline):
    """safety_proof must be JSON-serialisable with spending_changes as strings."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_01")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    # Must not raise
    json_str = json.dumps(rec)
    parsed = json.loads(json_str)
    # spending_changes in safety_proof must be a list of strings (not SpendingChange objects)
    if parsed.get("safety_proof"):
        changes = parsed["safety_proof"].get("spending_changes", [])
        for change in changes:
            assert isinstance(change, str), f"spending_change not serialised as string: {change!r}"


def test_build_diagnostic_record_normalized_events_have_required_fields(bundle, pipeline):
    """Each normalized event entry must carry the fields needed to diagnose simulation."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_02")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    assert len(rec["normalized_events"]) > 0, "request_02 should have normalized events"
    for evt in rec["normalized_events"]:
        for field in ("event_id", "amount_home", "effective_date", "direction",
                      "status", "recurring", "recurrence_day",
                      "amount_from_image", "amount_from_message"):
            assert field in evt, f"normalized event missing field: {field}"


def test_build_diagnostic_record_candidate_strategies_have_required_fields(bundle, pipeline):
    """Each candidate strategy entry must have method, plan, validity, and min balance."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_07")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    assert len(rec["candidate_strategies"]) > 0, "request_07 should have candidate strategies"
    for cand in rec["candidate_strategies"]:
        for field in ("method", "payment_plan", "spending_changes",
                      "validity", "validation_violations",
                      "minimum_projected_balance", "rank", "rank_reason"):
            assert field in cand, f"candidate strategy missing field: {field}"


def test_build_diagnostic_record_selected_strategy_present(bundle, pipeline):
    """selected_strategy must be non-None for requests that have a valid strategy."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_01")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    assert rec["selected_strategy"] is not None
    assert "method" in rec["selected_strategy"]
    assert "payment_plan" in rec["selected_strategy"]
    assert "spending_changes" in rec["selected_strategy"]


def test_build_diagnostic_record_evidence_facts_have_required_fields(bundle, pipeline):
    """Evidence facts (when present) must carry provenance fields."""
    # Run all 25 sample requests; at least some will have evidence from messages/images
    # In MOCK mode evidence_facts will be empty, but the structure must be valid
    sample_reqs = load_requests(sample=True)
    for req in sample_reqs[:5]:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        rec = build_diagnostic_record(state)
        for fact in rec.get("evidence_facts", []):
            for field in ("source_id", "source_type", "amount", "currency",
                          "confidence", "related_event_id", "contains_instruction"):
                assert field in fact, f"evidence fact missing field: {field}"


def test_build_diagnostic_record_no_safe_strategy(bundle, pipeline):
    """A request with not_recommended should still produce a valid diagnostic record."""
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_05")
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    rec = build_diagnostic_record(state)
    assert rec["request_id"] == "request_05"
    # selected_strategy may be not_recommended but must not be absent
    assert rec["selected_strategy"] is not None
    assert rec["selected_strategy"]["method"] in (
        "not_recommended", "full_payment", "partial_payment",
        "installments", "wait",
    )


# ── write_diagnostic_jsonl tests ──────────────────────────────────────────────

def test_write_diagnostic_jsonl_creates_file(tmp_path, bundle, pipeline):
    """write_diagnostic_jsonl must create the output file."""
    sample_reqs = load_requests(sample=True)[:3]
    records = []
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        records.append(build_diagnostic_record(state))

    out = tmp_path / "diag.jsonl"
    write_diagnostic_jsonl(records, out)
    assert out.exists()


def test_write_diagnostic_jsonl_one_line_per_record(tmp_path, bundle, pipeline):
    """Each diagnostic record occupies exactly one line in the JSONL file."""
    n = 5
    sample_reqs = load_requests(sample=True)[:n]
    records = []
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        records.append(build_diagnostic_record(state))

    out = tmp_path / "diag.jsonl"
    write_diagnostic_jsonl(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n, f"Expected {n} lines, got {len(lines)}"


def test_write_diagnostic_jsonl_each_line_valid_json(tmp_path, bundle, pipeline):
    """Every line in the JSONL output must be valid JSON."""
    sample_reqs = load_requests(sample=True)[:4]
    records = []
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        records.append(build_diagnostic_record(state))

    out = tmp_path / "diag.jsonl"
    write_diagnostic_jsonl(records, out)

    for i, line in enumerate(out.read_text(encoding="utf-8").splitlines()):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Line {i} is not valid JSON: {exc}")


def test_write_diagnostic_jsonl_request_ids_match(tmp_path, bundle, pipeline):
    """request_id in each JSONL record must match the processed request."""
    sample_reqs = load_requests(sample=True)[:5]
    records = []
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pipeline.run(state)
        records.append(build_diagnostic_record(state))

    out = tmp_path / "diag.jsonl"
    write_diagnostic_jsonl(records, out)

    written_ids = [
        json.loads(line)["request_id"]
        for line in out.read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = [r.request_id for r in sample_reqs]
    assert written_ids == expected_ids, (
        f"Written IDs {written_ids} do not match expected {expected_ids}"
    )


def test_write_diagnostic_jsonl_creates_parent_dirs(tmp_path, bundle, pipeline):
    """write_diagnostic_jsonl must create parent directories if missing."""
    req = load_requests(sample=True)[0]
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)
    rec = build_diagnostic_record(state)

    deep_path = tmp_path / "a" / "b" / "c" / "diag.jsonl"
    assert not deep_path.parent.exists()
    write_diagnostic_jsonl([rec], deep_path)
    assert deep_path.exists()


# ── Non-interference: diagnostics must not alter output.csv decisions ─────────

def test_diagnostic_does_not_alter_output_row(bundle, pipeline):
    """
    The output_row produced by the pipeline must be identical whether or not
    build_diagnostic_record is called on the resulting state.

    We run the same request twice and compare output rows.
    build_diagnostic_record must be a pure reader — it must not modify state.
    """
    req = next(r for r in load_requests(sample=True) if r.request_id == "request_02")

    # Run 1: call build_diagnostic_record
    ctx1 = build_request_context(req, bundle)
    state1 = FinancialState(context=ctx1)
    state1 = pipeline.run(state1)
    rec = build_diagnostic_record(state1)  # called
    row1 = state1.output_row

    # Run 2: do NOT call build_diagnostic_record
    ctx2 = build_request_context(req, bundle)
    state2 = FinancialState(context=ctx2)
    state2 = pipeline.run(state2)
    row2 = state2.output_row

    assert row1 is not None
    assert row2 is not None
    assert row1.affordability_status == row2.affordability_status
    assert row1.recommended_payment_method == row2.recommended_payment_method
    assert row1.payment_plan == row2.payment_plan
    assert row1.amount_safe_to_pay == row2.amount_safe_to_pay
    assert row1.earliest_date_for_full_payment == row2.earliest_date_for_full_payment
    assert row1.spending_changes_needed == row2.spending_changes_needed


def test_diagnostic_does_not_mutate_state(bundle, pipeline):
    """
    build_diagnostic_record must not mutate FinancialState fields after the
    pipeline has completed.  Check that output_row and safety_proof are unchanged.
    """
    req = load_requests(sample=True)[0]
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)

    # Snapshot key state fields before calling build_diagnostic_record
    row_before = state.output_row
    proof_before = state.safety_proof
    safe_before = state.amount_safe_to_pay
    earliest_before = state.earliest_full_payment_date

    _ = build_diagnostic_record(state)  # call it

    # Nothing should have changed
    assert state.output_row is row_before
    assert state.safety_proof is proof_before
    assert state.amount_safe_to_pay == safe_before
    assert state.earliest_full_payment_date == earliest_before


def test_full_sample_run_produces_one_diagnostic_per_request(tmp_path, bundle):
    """
    Running the pipeline over all 25 sample requests and collecting diagnostics
    must produce exactly 25 records — one per request, in order.
    """
    provider = MockProvider()
    pl = build_pipeline(provider)
    sample_reqs = load_requests(sample=True)

    records = []
    for req in sample_reqs:
        ctx = build_request_context(req, bundle)
        state = FinancialState(context=ctx)
        state = pl.run(state)
        records.append(build_diagnostic_record(state))

    out = tmp_path / "sample_diag.jsonl"
    write_diagnostic_jsonl(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(sample_reqs), (
        f"Expected {len(sample_reqs)} diagnostic records, got {len(lines)}"
    )

    written_ids = [json.loads(line)["request_id"] for line in lines]
    expected_ids = [r.request_id for r in sample_reqs]
    assert written_ids == expected_ids


def test_diagnostic_output_csv_identical_with_and_without_diagnostics(tmp_path, bundle):
    """
    Running the full 25-request sample with and without diagnostic collection
    must produce byte-identical output.csv content.
    This proves diagnostics are a side-car and never influence decisions.
    """
    from buywise.core.models import OutputRow
    provider = MockProvider()

    def run_and_collect(with_diag: bool) -> list[dict]:
        pl = build_pipeline(provider)
        sample_reqs = load_requests(sample=True)
        rows = []
        for req in sample_reqs:
            ctx = build_request_context(req, bundle)
            state = FinancialState(context=ctx)
            state = pl.run(state)
            if with_diag:
                _ = build_diagnostic_record(state)  # called but result ignored
            if state.output_row:
                rows.append(state.output_row.to_csv_row())
        return rows

    rows_with = run_and_collect(with_diag=True)
    rows_without = run_and_collect(with_diag=False)

    assert len(rows_with) == len(rows_without)
    for a, b in zip(rows_with, rows_without):
        assert a == b, (
            f"Output row differs: {a['request_id']} — "
            f"with_diag={a} vs without_diag={b}"
        )
