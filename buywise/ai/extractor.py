"""
LLM evidence extractor.

Extracts structured financial facts from messages and images.
Returns EvidenceGraph objects that the normalization engine uses
to resolve blank amounts, apply amendments, and apply cancellations.

Security contract:
- All message/image content is treated as untrusted data.
- Instruction injection is detected and flagged.
- Extracted facts only amend the deterministic financial model —
  they never directly produce output values.

Architecture:
- RequestAnalyzer: extracts evidence from ALL messages+images for a request
- MessageExtractor: processes one message at a time
- ImageExtractor: processes one image at a time (multimodal or fallback)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from buywise.ai.prompts import (
    EVIDENCE_EXTRACTION_SYSTEM,
    EXPLANATION_PROMPT,
    IMAGE_EXTRACTION_PROMPT,
    MESSAGE_EXTRACTION_PROMPT,
)
from buywise.ai.provider import LLMProvider, usage_tracker
from buywise.core.config import MEDIA_DIR
from buywise.data.evidence import (
    ConfidenceLevel,
    EvidenceFact,
    EvidenceGraph,
    EvidenceType,
    detect_instruction_injection,
)
from buywise.core.models import ImageReference, Message

logger = logging.getLogger(__name__)


def _parse_confidence(raw: str) -> ConfidenceLevel:
    mapping = {
        "high": ConfidenceLevel.HIGH,
        "medium": ConfidenceLevel.MEDIUM,
        "low": ConfidenceLevel.LOW,
        "uncertain": ConfidenceLevel.UNCERTAIN,
    }
    return mapping.get(str(raw).lower(), ConfidenceLevel.MEDIUM)


def _parse_evidence_type(raw: str) -> EvidenceType:
    mapping = {
        "amount": EvidenceType.AMOUNT,
        "salary_update": EvidenceType.SALARY_UPDATE,
        "salary_confirmed": EvidenceType.SALARY_CONFIRMED,
        "event_cancelled": EvidenceType.EVENT_CANCELLED,
        "event_amended": EvidenceType.EVENT_AMENDED,
        "event_settled": EvidenceType.EVENT_SETTLED,
        "expense_confirmed": EvidenceType.EXPENSE_CONFIRMED,
        "payment_confirmed": EvidenceType.PAYMENT_CONFIRMED,
        "income_pending": EvidenceType.INCOME_PENDING,
        "income_cancelled": EvidenceType.INCOME_CANCELLED,
        "general": EvidenceType.GENERAL,
    }
    return mapping.get(str(raw).lower(), EvidenceType.GENERAL)


_UNCONFIRMED_INCOME = re.compile(
    r"\b(pending|not approved|awaiting approval|subject to|still processing|"
    r"not yet credited|not been credited|open deals|unearned|uncertain|"
    r"cancelled|canceled|will not arrive|ended)\b",
    re.IGNORECASE,
)
_INCOME_TERMS = re.compile(
    r"\b(salary|payroll|bonus|commission|invoice payment|refund|prize|income|payout)\b",
    re.IGNORECASE,
)


def _income_uncertainty_type(text: str) -> Optional[EvidenceType]:
    """Classify uncertain income conservatively without treating it as an instruction."""
    if not (_UNCONFIRMED_INCOME.search(text) and _INCOME_TERMS.search(text)):
        return None
    if re.search(r"\b(cancelled|canceled|will not arrive|ended)\b", text, re.IGNORECASE):
        return EvidenceType.INCOME_CANCELLED
    return EvidenceType.INCOME_PENDING


def _parse_optional_date(raw: Any) -> Optional[date]:
    if raw is None or not str(raw).strip():
        return None
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError:
        return None


def _normalize_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _safe_json_parse(text: str) -> Optional[dict]:
    """Parse JSON from LLM output, tolerating minor formatting issues."""
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _track_usage(response: Any) -> None:
    """Extract token counts from LangChain response metadata."""
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            if isinstance(meta, dict):
                inp = meta.get("input_tokens", 0) or 0
                out = meta.get("output_tokens", 0) or 0
            else:
                inp = getattr(meta, "input_tokens", 0) or 0
                out = getattr(meta, "output_tokens", 0) or 0
            # Only record if we got real integers
            if isinstance(inp, int) and isinstance(out, int):
                usage_tracker.record(inp, out)
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if isinstance(meta, dict):
                token_usage = meta.get("token_usage", {})
                if isinstance(token_usage, dict):
                    inp = int(token_usage.get("prompt_tokens", 0))
                    out = int(token_usage.get("completion_tokens", 0))
                    usage_tracker.record(inp, out)
    except Exception:
        pass  # Token tracking is best-effort; never block the pipeline


class MessageExtractor:
    """Extracts EvidenceFacts from a single financial message."""

    def __init__(self, provider: LLMProvider):
        self._llm = provider.get_chat_model()
        self.last_used_fallback = False

    def extract(
        self,
        message: Message,
        home_currency: str,
    ) -> list[EvidenceFact]:
        """
        Extract financial facts from one message.
        Returns a list of EvidenceFact objects.
        """
        # Quick injection check before sending to LLM
        raw_injection = detect_instruction_injection(message.message_text)

        prompt = MESSAGE_EXTRACTION_PROMPT.format(
            message_id=message.message_id,
            source_type=message.source_type,
            sent_at=message.sent_at.isoformat(),
            related_event_id=message.related_event_id or "none",
            home_currency=home_currency,
            message_text=message.message_text[:2000],  # Cap at 2000 chars
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = self._llm.invoke([
                SystemMessage(content=EVIDENCE_EXTRACTION_SYSTEM),
                HumanMessage(content=prompt),
            ])
            _track_usage(response)
            parsed = _safe_json_parse(_normalize_response_content(response.content))
        except Exception as exc:
            logger.warning("Message extraction failed for %s: %s", message.message_id, exc)
            return []

        if not parsed or "facts" not in parsed:
            return []

        facts = []
        for item in parsed.get("facts", []):
            try:
                contains_instruction = bool(item.get("contains_instruction", False)) or raw_injection
                evidence_type = _parse_evidence_type(item.get("evidence_type", "general"))
                income_uncertainty = _income_uncertainty_type(message.message_text)
                if income_uncertainty and evidence_type in (
                    EvidenceType.AMOUNT,
                    EvidenceType.SALARY_CONFIRMED,
                ):
                    evidence_type = income_uncertainty
                fact = EvidenceFact(
                    evidence_type=evidence_type,
                    source_id=message.message_id,
                    source_type="message",
                    extracted_at=message.sent_at,
                    amount=float(item["amount"]) if item.get("amount") is not None else None,
                    currency=item.get("currency") or home_currency,
                    applies_to_date=_parse_optional_date(item.get("applies_to_date")),
                    related_event_id=item.get("related_event_id") or message.related_event_id,
                    confidence=_parse_confidence(item.get("confidence", "medium")),
                    raw_text=str(item.get("raw_text", ""))[:200],
                    notes=str(item.get("notes", ""))[:500],
                    contains_instruction=contains_instruction,
                )
                facts.append(fact)
            except Exception as exc:
                logger.debug("Skipping malformed fact from %s: %s", message.message_id, exc)

        return facts


class ImageExtractor:
    """
    Extracts EvidenceFacts from financial images (receipts, payslips, bills).

    Uses multimodal LLM when available (GPT-4o / LLaMA-3.2-Vision).
    Falls back to conservative handling if extraction fails.
    """

    def __init__(self, provider: LLMProvider):
        self._provider = provider
        self._llm = provider.get_chat_model()

    def extract(
        self,
        image_ref: ImageReference,
        event_type: str,
        event_category: str,
        home_currency: str,
    ) -> Optional[EvidenceFact]:
        """
        Extract the financial amount from an image.
        Returns a single EvidenceFact, or None if extraction fails/is uncertain.
        """
        image_path = MEDIA_DIR / f"{image_ref.image_id}.png"
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            return None

        # Encode image as base64
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.warning("Cannot read image %s: %s", image_ref.image_id, exc)
            return None

        prompt_text = IMAGE_EXTRACTION_PROMPT.format(
            image_id=image_ref.image_id,
            related_event_id=image_ref.related_event_id or "none",
            event_type=event_type,
            event_category=event_category,
            home_currency=home_currency,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            # Build multimodal message with image
            message_content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
            response = self._llm.invoke([
                SystemMessage(content=EVIDENCE_EXTRACTION_SYSTEM),
                HumanMessage(content=message_content),
            ])
            _track_usage(response)
            parsed = _safe_json_parse(_normalize_response_content(response.content))
        except Exception as exc:
            logger.warning("Image extraction failed for %s: %s", image_ref.image_id, exc)
            return None

        if not parsed:
            return None

        amount = parsed.get("amount")
        if amount is None:
            logger.info("Image %s: amount is null (uncertain extraction)", image_ref.image_id)
            return None

        contains_instruction = bool(parsed.get("contains_instruction", False))
        confidence = _parse_confidence(parsed.get("confidence", "medium"))

        if confidence == ConfidenceLevel.UNCERTAIN:
            logger.info("Image %s: uncertain confidence, not using amount", image_ref.image_id)
            return None

        return EvidenceFact(
            evidence_type=EvidenceType.AMOUNT,
            source_id=image_ref.image_id,
            source_type="image",
            extracted_at=datetime.now(timezone.utc),
            amount=float(amount),
            currency=parsed.get("currency") or home_currency,
            related_event_id=image_ref.related_event_id,
            confidence=confidence,
            raw_text=f"Extracted from {image_ref.image_id}",
            notes=str(parsed.get("notes", ""))[:500],
            contains_instruction=contains_instruction,
        )


class EvidenceExtractor:
    """
    Top-level extractor: processes all messages and images for a request
    and returns a populated EvidenceGraph.
    """

    def __init__(self, provider: LLMProvider):
        self._msg_extractor = MessageExtractor(provider)
        self._img_extractor = ImageExtractor(provider)

    def extract(
        self,
        request_id: str,
        user_id: str,
        messages: list[Message],
        images: list[ImageReference],
        events_by_id: dict,
        home_currency: str,
    ) -> EvidenceGraph:
        """
        Extract all evidence facts from messages and images for a request.
        Returns a populated EvidenceGraph.
        """
        graph = EvidenceGraph(request_id=request_id, user_id=user_id)

        # Extract from messages
        for msg in messages:
            facts = self._msg_extractor.extract(msg, home_currency)
            for fact in facts:
                if fact.is_trustworthy() or fact.contains_instruction:
                    graph.add(fact)
            if facts:
                logger.debug(
                    "Extracted %d facts from message %s", len(facts), msg.message_id
                )

        # Extract from images (only for blank-amount events)
        for img in images:
            if not img.related_event_id:
                continue
            event = events_by_id.get(img.related_event_id)
            if event is None or event.amount is not None:
                continue  # Skip images for events that already have amounts

            fact = self._img_extractor.extract(
                image_ref=img,
                event_type=event.event_type,
                event_category=event.category,
                home_currency=home_currency,
            )
            if fact is not None:
                graph.add(fact)
                logger.info(
                    "Extracted amount %.2f from image %s for event %s",
                    fact.amount, img.image_id, img.related_event_id,
                )

        return graph


class DecisionExplainer:
    """
    Generates the human-readable decision_explanation field.

    Uses an LLM to write a concise, grounded explanation based on
    the deterministic safety proof. The LLM explains — it does NOT decide.
    """

    def __init__(self, provider: LLMProvider):
        self._llm = provider.get_chat_model()

    def explain(
        self,
        requested_amount: float,
        currency: str,
        request_type: str,
        request_date: str,
        desired_completion_date: str,
        affordability_status: str,
        recommended_payment_method: str,
        amount_safe_to_pay: float,
        payment_plan: str,
        earliest_date: str,
        spending_changes: str,
        min_projected_balance: float,
        required_minimum_balance: float,
        all_constraints_pass: bool,
    ) -> str:
        """
        Generate a 1-3 sentence explanation for the recommendation.
        Falls back to a deterministic template if LLM call fails.
        """
        self.last_used_fallback = False
        prompt = EXPLANATION_PROMPT.format(
            requested_amount=f"{requested_amount:,.2f}",
            currency=currency,
            request_type=request_type,
            request_date=request_date,
            desired_completion_date=desired_completion_date,
            affordability_status=affordability_status,
            recommended_payment_method=recommended_payment_method,
            amount_safe_to_pay=f"{amount_safe_to_pay:,.2f}",
            payment_plan=payment_plan,
            earliest_date=earliest_date or "N/A",
            spending_changes=spending_changes,
            min_projected_balance=f"{min_projected_balance:,.2f}",
            required_minimum_balance=f"{required_minimum_balance:,.2f}",
            all_constraints_pass=all_constraints_pass,
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            _track_usage(response)
            text = _normalize_response_content(response.content).strip()
            # Basic sanity: must be a non-empty string without JSON artifacts
            if text and len(text) > 10 and not text.startswith("{"):
                return text[:500]
        except Exception as exc:
            logger.warning("Explanation generation failed: %s", exc)

        # Deterministic fallback explanation
        self.last_used_fallback = True
        return _deterministic_explanation(
            affordability_status=affordability_status,
            recommended_payment_method=recommended_payment_method,
            amount_safe_to_pay=amount_safe_to_pay,
            currency=currency,
            payment_plan=payment_plan,
            earliest_date=earliest_date,
            spending_changes=spending_changes,
            required_minimum_balance=required_minimum_balance,
            min_projected_balance=min_projected_balance,
        )


def _deterministic_explanation(
    affordability_status: str,
    recommended_payment_method: str,
    amount_safe_to_pay: float,
    currency: str,
    payment_plan: str,
    earliest_date: str,
    spending_changes: str,
    required_minimum_balance: float,
    min_projected_balance: float,
) -> str:
    """
    Fallback deterministic explanation when LLM is unavailable.
    Produces clear, factual text using only the safety proof data.
    """
    changes_text = ""
    if spending_changes and spending_changes != "none":
        parts = spending_changes.split("|")
        change_desc = []
        for p in parts:
            if p.startswith("stop:"):
                change_desc.append(f"stop {p.split(':')[1]}")
            elif p.startswith("reduce_to:"):
                bits = p.split(":")
                change_desc.append(f"reduce {bits[1]} to {bits[2]}")
        changes_text = "After " + " and ".join(change_desc) + ", "

    if affordability_status == "affordable_now":
        return (
            f"{changes_text}Pay {currency} {amount_safe_to_pay:,.2f} today. "
            f"This leaves at least {currency} {min_projected_balance:,.2f} available over the next 90 days."
        )
    elif affordability_status == "affordable_with_plan":
        if recommended_payment_method == "installments" and payment_plan != "none":
            parts = payment_plan.split("|")
            n = len(parts)
            first_amt = parts[0].split(":")[1] if parts else ""
            first_date = parts[0].split(":")[0] if parts else ""
            return (
                f"{changes_text}Use {n} installment{'s' if n > 1 else ''} of "
                f"{currency} {first_amt}, starting {first_date}. "
                f"This leaves at least {currency} {min_projected_balance:,.2f} available."
            )
        elif recommended_payment_method == "partial_payment":
            return (
                f"Pay {currency} {amount_safe_to_pay:,.2f} today, "
                f"then the remaining amount on {earliest_date}. "
                f"This leaves at least {currency} {min_projected_balance:,.2f} available."
            )
        else:
            return (
                f"{changes_text}Pay {currency} {amount_safe_to_pay:,.2f} today. "
                f"This keeps the {currency} {required_minimum_balance:,.2f} minimum protected."
            )
    elif affordability_status == "affordable_later":
        pay_date = earliest_date or "a future date"
        return (
            f"Wait until {pay_date}, then pay {currency} {amount_safe_to_pay:,.2f} in full. "
            f"Paying earlier would put the {currency} {required_minimum_balance:,.2f} minimum at risk."
        )
    else:  # not_affordable / not_recommended
        return (
            f"Do not proceed with this payment. "
            f"None of the available options keeps the {currency} {required_minimum_balance:,.2f} "
            f"minimum protected over the forecast period."
        )
