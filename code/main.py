"""
BUYWISE — Evidence-to-Proof Financial Agent
HackerRank Orchestrate September 2026

Entry point for the full 250-request production run.

Usage:
    python code/main.py                    # Full run with configured LLM
    MOCK_LLM=true python code/main.py      # Offline/test run (no LLM calls)
    python code/main.py --sample           # Run on 25 sample requests only

Output:
    output.csv                in repo root
    evaluation/usage_report.md in code/evaluation/

Environment variables:
    LLM_PROVIDER    openai | groq | gemini  (default: openai)
    LLM_MODEL       model name     (default: gpt-4o-mini)
    OPENAI_API_KEY  required for openai provider
    GROQ_API_KEY    required for groq provider
    GOOGLE_API_KEY  required for gemini provider
    MOCK_LLM        true           (skip all LLM calls, deterministic mode)
    LOG_LEVEL       DEBUG|INFO|WARNING (default: INFO)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buywise.core.logger import setup_logging
from buywise.core.agent_log import write_turn
setup_logging()

import logging
logger = logging.getLogger(__name__)


def main(sample_only: bool = False) -> None:
    import os
    from buywise.core.config import DIAGNOSTIC_OUTPUT_PATH, OUTPUT_PATH
    from buywise.ai.provider import get_llm_provider, usage_tracker
    from buywise.data.loader import DatasetBundle
    from buywise.data.joins import build_request_context
    from buywise.evaluation.diagnostics import build_diagnostic_record, write_diagnostic_jsonl
    from buywise.evaluation.report import write_usage_report
    from buywise.graph.state import FinancialState
    from buywise.graph.workflow import build_pipeline
    from buywise.validation.output_validator import validate_and_write_output

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("BUYWISE Financial Agent - Production Run")
    logger.info("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    bundle = DatasetBundle()
    logger.info(bundle.summary())

    requests = bundle.requests
    if sample_only:
        from buywise.data.loader import load_requests
        requests = load_requests(sample=True)
        logger.info("Sample mode: processing %d sample requests", len(requests))
    else:
        logger.info("Processing %d requests", len(requests))

    # ── Build pipeline ────────────────────────────────────────────────────────
    provider = get_llm_provider()
    logger.info("Provider: %s, Model: %s", provider.provider_name, provider.model_name)
    pipeline = build_pipeline(provider)

    # ── Process all requests ──────────────────────────────────────────────────
    output_rows = []
    n_errors = 0
    requested_amounts: dict[str, float] = {}
    validation_contexts = {}
    diagnostic_records = []

    for i, req in enumerate(requests, 1):
        req_start = time.time()
        logger.info("[%d/%d] Processing %s (user=%s, type=%s, amount=%s %s)",
                    i, len(requests), req.request_id, req.user_id,
                    req.request_type, req.requested_amount,
                    bundle.get_profile(req.user_id).home_currency)

        requested_amounts[req.request_id] = req.requested_amount

        try:
            ctx = build_request_context(req, bundle)
            state = FinancialState(context=ctx)
            state = pipeline.run(state)
            diagnostic_records.append(build_diagnostic_record(state))

            if state.output_row:
                output_rows.append(state.output_row)
                logger.info(
                    "  -> %s | %s | safe=%.2f | plan=%s",
                    state.output_row.affordability_status,
                    state.output_row.recommended_payment_method,
                    state.output_row.amount_safe_to_pay,
                    state.output_row.payment_plan[:60],
                )
            else:
                logger.error("  -> No output row produced for %s", req.request_id)
                n_errors += 1
                # Emergency fallback
                from buywise.core.models import OutputRow
                output_rows.append(OutputRow(
                    request_id=req.request_id,
                    amount_safe_to_pay=0.0,
                    affordability_status="not_affordable",
                    recommended_payment_method="not_recommended",
                    payment_plan="none",
                    earliest_date_for_full_payment="",
                    spending_changes_needed="none",
                    decision_explanation="Processing error - could not determine affordability.",
                ))

            if state.errors:
                n_errors += 1
                for err in state.errors:
                    logger.warning("  Error: %s", err)

            validation_contexts[req.request_id] = {
                "request": req,
                "profile": bundle.get_profile(req.user_id),
                "payment_options": ctx.payment_options,
                "normalized_events": state.normalized_events,
            }

        except Exception as exc:
            logger.error("Unhandled error for %s: %s", req.request_id, exc, exc_info=True)
            n_errors += 1
            from buywise.core.models import OutputRow
            output_rows.append(OutputRow(
                request_id=req.request_id,
                amount_safe_to_pay=0.0,
                affordability_status="not_affordable",
                recommended_payment_method="not_recommended",
                payment_plan="none",
                earliest_date_for_full_payment="",
                spending_changes_needed="none",
                decision_explanation="Processing error - could not determine affordability.",
            ))
            diagnostic_records.append({
                "request_id": req.request_id,
                "user_id": req.user_id,
                "errors": [f"Unhandled error: {exc}"],
                "failure_counts": {"extraction": 0, "pipeline": 1, "api": None},
            })

        elapsed = time.time() - req_start
        logger.debug("  Processed in %.2fs", elapsed)

    # ── Validate and write output ─────────────────────────────────────────────
    logger.info("Validating and writing output...")
    expected_ids = [r.request_id for r in requests]

    try:
        success, warnings = validate_and_write_output(
            output_rows=output_rows,
            requested_amounts=requested_amounts,
            output_path=OUTPUT_PATH,
            expected_request_ids=expected_ids,
            validation_contexts=validation_contexts,
        )
        if warnings:
            logger.warning("%d validation warnings (see above)", len(warnings))
        logger.info("Output written to: %s", OUTPUT_PATH)
    except Exception as exc:
        logger.error("Output validation/write failed: %s", exc)
        sys.exit(1)

    write_diagnostic_jsonl(diagnostic_records, DIAGNOSTIC_OUTPUT_PATH)
    logger.info("Diagnostics written to: %s", DIAGNOSTIC_OUTPUT_PATH)

    # ── Write evaluation report ───────────────────────────────────────────────
    duration = time.time() - start_time
    write_usage_report(
        n_requests=len(requests),
        n_errors=n_errors,
        run_duration_seconds=duration,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Run complete in %.1fs", duration)
    logger.info("Requests processed: %d", len(requests))
    logger.info("Errors/fallbacks: %d", n_errors)
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("=" * 60)

    write_turn(
        title=f"Production run: {len(requests)} requests processed",
        user_prompt="python code/main.py",
        summary=f"Processed {len(requests)} requests in {duration:.1f}s. "
                f"{n_errors} errors. Output written to output.csv.",
        actions=[f"Ran pipeline on {len(requests)} requests", f"Wrote {OUTPUT_PATH}"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BUYWISE Financial Agent")
    parser.add_argument("--sample", action="store_true",
                        help="Run on sample_requests.csv only (25 requests)")
    args = parser.parse_args()
    main(sample_only=args.sample)
