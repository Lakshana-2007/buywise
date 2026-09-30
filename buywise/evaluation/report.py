"""
Evaluation report generator.

Writes evaluation/usage_report.md after a full dataset run.
Tracks token usage, cost, and run statistics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from buywise.ai.provider import usage_tracker
from buywise.core.config import EVALUATION_DIR

logger = logging.getLogger(__name__)


def write_usage_report(
    n_requests: int,
    n_errors: int,
    run_duration_seconds: float,
    output_path: Path | None = None,
) -> None:
    """
    Write the token usage report to evaluation/usage_report.md.

    Parameters
    ----------
    n_requests         : Total requests processed.
    n_errors           : Requests that failed or used fallback.
    run_duration_seconds : Wall-clock time for the full run.
    output_path        : Override output path (default: evaluation/usage_report.md).
    """
    report_path = output_path or (EVALUATION_DIR / "usage_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = usage_tracker
    data = tracker.to_report_dict(n_requests)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    provider_name = str(data.get('provider') or 'mock/deterministic')
    model_name = str(data.get('model') or 'N/A')
    total_calls = int(data.get('total_calls', 0))
    input_tokens = int(data.get('input_tokens', 0))
    output_tokens = int(data.get('output_tokens', 0))
    total_tokens = int(data.get('total_tokens', 0))
    estimated_cost = float(data.get('estimated_cost_usd', 0.0))
    avg_tokens = float(data.get('avg_tokens_per_request', 0.0))
    avg_cost = float(data.get('avg_cost_per_request_usd', 0.0))

    lines = [
        "# BUYWISE — Token Usage Report",
        "",
        f"**Generated:** {now}",
        f"**Run type:** Full dataset ({n_requests} requests)",
        f"**Errors / fallbacks:** {n_errors}",
        f"**Wall-clock duration:** {run_duration_seconds:.1f}s ({run_duration_seconds / max(n_requests, 1):.2f}s avg/request)",
        "",
        "---",
        "",
        "## Model Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Provider | {provider_name} |",
        f"| Model | {model_name} |",
        "| Temperature | 0.0 (deterministic) |",
        "",
        "---",
        "",
        "## Token Usage",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total LLM calls | {total_calls} |",
        f"| Input tokens | {input_tokens:,} |",
        f"| Output tokens | {output_tokens:,} |",
        f"| **Total tokens** | **{total_tokens:,}** |",
        f"| Avg tokens / request | {avg_tokens} |",
        "",
        "---",
        "",
        "## Cost Estimate",
        "",
        "> Pricing basis: GPT-4o-mini ($0.15/1M input, $0.60/1M output).",
        "> Actual cost depends on the model used.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total estimated cost (USD) | ${estimated_cost:.4f} |",
        f"| Avg cost per request (USD) | ${avg_cost:.6f} |",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Token counts are collected from LangChain response metadata.",
        "- If `usage_metadata` is unavailable (some providers), counts default to 0.",
        "  In that case, the actual usage is non-zero but cannot be reported accurately.",
        "- The financial simulation engine is fully deterministic and makes zero LLM calls.",
        "- LLM calls are made only for evidence extraction and explanation generation.",
        "- The pipeline runs correctly with MOCK_LLM=true (zero LLM calls).",
        "",
    ]
    report = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    logger.info("Usage report written to %s", report_path)
