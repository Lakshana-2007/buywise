"""
Append-only agent log writer — satisfies the AGENTS.md §5 contract.

Creates log.txt next to AGENTS.md (repo root) and appends
SESSION START and per-turn entries in the required format.

Never logs secrets.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from buywise.core.config import AGENT_LOG_PATH

TOOL_NAME = "Kiro"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(text: str) -> None:
    with open(AGENT_LOG_PATH, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


def write_session_start(branch: str = "unknown") -> None:
    """Write §5.1 SESSION START entry."""
    repo_root = str(AGENT_LOG_PATH.parent)
    entry = f"""
## [{_now_iso()}] SESSION START

tool={TOOL_NAME}
Repo Root: {repo_root}
Branch: {branch}
Worktree: main
Parent Agent: none
Language: py
Time Remaining: see challenge deadline
"""
    _append(entry)


def write_turn(
    title: str,
    user_prompt: str,
    summary: str,
    actions: list[str],
    branch: str = "unknown",
) -> None:
    """Write §5.2 per-turn log entry."""
    repo_root = str(AGENT_LOG_PATH.parent)
    actions_str = "\n".join(f"* {a}" for a in actions)
    # Redact common secret patterns
    safe_prompt = _redact_secrets(user_prompt)
    entry = f"""
## [{_now_iso()}] {title[:80]}

User Prompt (verbatim, secrets redacted):
{safe_prompt}

Agent Response Summary:
{summary}

Actions:
{actions_str}

Context:
tool={TOOL_NAME}
branch={branch}
repo_root={repo_root}
worktree=main
parent_agent=none
"""
    _append(entry)


def _redact_secrets(text: str) -> str:
    """Redact obvious secret patterns — API keys, tokens, etc."""
    import re
    patterns = [
        r"sk-[A-Za-z0-9]{20,}",    # OpenAI keys
        r"gsk_[A-Za-z0-9]{20,}",   # Groq keys
        r"Bearer [A-Za-z0-9\-_.~+/=]{20,}",
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "[REDACTED]", result)
    return result
