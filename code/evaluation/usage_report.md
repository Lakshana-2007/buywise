# BUYWISE — Token Usage Report

**Generated:** 2026-09-14 09:02:06 UTC
**Run type:** Full dataset (25 requests)
**Errors / fallbacks:** 0
**Wall-clock duration:** 2.7s (0.11s avg/request)

---

## Model Configuration

| Field | Value |
|---|---|
| Provider | mock/deterministic |
| Model | N/A |
| Temperature | 0.0 (deterministic) |

---

## Token Usage

| Metric | Value |
|---|---|
| Total LLM calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| **Total tokens** | **0** |
| Avg tokens / request | 0.0 |

---

## Cost Estimate

> Pricing basis: GPT-4o-mini ($0.15/1M input, $0.60/1M output).
> Actual cost depends on the model used.

| Metric | Value |
|---|---|
| Total estimated cost (USD) | $0.0000 |
| Avg cost per request (USD) | $0.000000 |

---

## Notes

- Token counts are collected from LangChain response metadata.
- If `usage_metadata` is unavailable (some providers), counts default to 0.
  In that case, the actual usage is non-zero but cannot be reported accurately.
- The financial simulation engine is fully deterministic and makes zero LLM calls.
- LLM calls are made only for evidence extraction and explanation generation.
- The pipeline runs correctly with MOCK_LLM=true (zero LLM calls).
