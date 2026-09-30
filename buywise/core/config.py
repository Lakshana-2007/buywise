"""
Central configuration.  All paths and tuneable constants live here.
No hardcoded paths anywhere else in the codebase.
"""

import os
from pathlib import Path

# ── Repository layout ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]          # d:/hacker/repo  (buywise/core/config.py -> repo)
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"
CODE_DIR = REPO_ROOT / "code"
EVALUATION_DIR = CODE_DIR / "evaluation"
DIAGNOSTIC_OUTPUT_PATH = EVALUATION_DIR / "regression_diagnostics.jsonl"
OUTPUT_PATH = REPO_ROOT / "output.csv"

# ── Dataset files ─────────────────────────────────────────────────────────────
REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
TEMPLATE_OUTPUT_CSV = DATASET_DIR / "output.csv"

# ── Financial simulation constants ───────────────────────────────────────────
FORECAST_DAYS: int = 90          # Safety check horizon
MAX_SPENDING_CHANGES: int = 3    # Maximum flexible expense modifications

# ── LLM configuration ────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Enable mock/deterministic mode (no LLM calls) — useful for tests
MOCK_LLM: bool = os.getenv("MOCK_LLM", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
AGENT_LOG_PATH = REPO_ROOT / "log.txt"
