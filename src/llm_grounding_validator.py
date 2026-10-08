from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:₹|\$|€|£)?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:%|(?:M|MM|B|K|million|billion|thousand))?",
    re.IGNORECASE,
)

_SPECULATION_PATTERNS = [
    r"\bcould be due to\b",
    r"\bmay be due to\b",
    r"\bmight be due to\b",
    r"\bpossibly due to\b",
    r"\bperhaps due to\b",
    r"\blikely due to\b",
    r"\bprobably due to\b",
    r"\bdue to\b",
    r"\bcaused by\b",
    r"\bdriven by\b",
    r"\bexplained by\b",
    r"\bresulting from\b",
    r"\battributable to\b",
]

_TREND_PATTERNS = [
    r"\btrend\b",
    r"\btrending\b",
    r"\bcontinues?\b",
    r"\bcontinuing\b",
    r"\btrajectory\b",
    r"\bhistoric(?:al|ally)?\b",
    r"\bhistorical performance\b",
    r"\byear-over-year\b",
    r"\bmonth-over-month\b",
    r"\byoy\b",
    r"\bmom\b",
    r"\bpattern\b",
]

_DIRECTION_PATTERNS = {
    "unfavorable": [
        r"\boverspend(?:ing)?\b",
        r"\bover budget\b",
        r"\babove budget\b",
        r"\bexceeded budget\b",
        r"\bunfavorable\b",
        r"\bunfavourable\b",
    ],
    "favorable": [
        r"\bunderspend(?:ing)?\b",
        r"\bunder budget\b",
        r"\bbelow budget\b",
        r"\bfavorable\b",
        r"\bfavourable\b",
        r"\bsaving(?:s)?\b",
    ],
}

_MULTIPLIERS = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "b": 1_000_000_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "thousand": 1_000.0,
}


def _compact_snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - 90)
    right = min(len(text), end + 140)
    return text[left:right].replace("\n", " ").strip()


@dataclass
class GroundingValidationResult:
    passed: bool
    unsupported_numbers: list[str]
    unsupported_percentages: list[str]
    speculative_claims: list[str]
    trend_claims_without_evidence: list[str]
    evidence_numbers: list[str]
    response_numbers: list[str]
    checks: dict[str, bool]
    schema_errors: list[str] = field(default_factory=list)
    financial_semantic_errors: list[str] = field(default_factory=list)
    evidence_coverage_errors: list[str] = field(default_factory=list)
    referenced_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMGroundingValidator:
    """
    Deterministic post-generation validator for VARIA LLM outputs.

    Validation is deliberately stricter than generic factual grounding:
      1. numeric values must be supported by evidence;
      2. causal language is rejected unless evidence explicitly supports a cause;
      3. trend language is rejected unless evidence explicitly supports a trend;
      4. cited evidence IDs must exist;
      5. financial direction (overspend vs underspend) must agree with cited
         evidence semantics;
      6. planning exceptions are never treated as overspend by accident.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Numeric parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _numbers_from_text(text: str) -> list[str]:
        return [m.group(0).strip() for m in _NUMBER_RE.finditer(text or "")]

    @staticmethod
    def _is_list_enumerator(
        text: str,
        match_start: int,
        match_end: int,
        token: str,
    ) -> bool:
        normalized = token.strip()
        if not normalized.isdigit():
            return False

        # Ignore common list markers such as "1." / "2)" / "3:".
        suffix = text[match_end:match_end + 1]
        return suffix in {".", ")", ":"}

    @staticmethod
    def _is_identifier_number(text: str, match_start: int, match_end: int, token: str) -> bool:
        """Ignore numeric suffixes that are labels/IDs, not financial claims."""
        normalized = token.strip()
        if not normalized.isdigit():
            return False

        prefix = text[max(0, match_start - 24):match_start]
        # Examples: Scenario 001, SCN-001, PLAN-002, ISS-001.
        if re.search(
            r"(?:\bscenario\s+|\bscn-?|\bplan-?|\biss(?:ue)?-?|\bexec-?)$",
            prefix,
            flags=re.IGNORECASE,
        ):
            return True

        # Numeric suffixes immediately following a hyphen/underscore identifier.
        if match_start > 0 and text[match_start - 1] in {"-", "_"}:
            left = text[max(0, match_start - 12):match_start - 1]
            if re.search(r"[A-Za-z]$", left):
                return True

        return False

    @classmethod
    def _response_number_tokens(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for match in _NUMBER_RE.finditer(text or ""):
            token = match.group(0).strip()
            if cls._is_list_enumerator(
                text,
                match.start(),
                match.end(),
                token,
            ):
                continue
            if cls._is_identifier_number(
                text,
                match.start(),
                match.end(),
                token,
            ):
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _normalize_number(token: str) -> tuple[float | None, str, int]:
        cleaned = (
            token.strip()
            .replace(",", "")
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
        )

        match = re.fullmatch(
            r"(-?\d+(?:\.\d+)?)\s*(%|k|m|mm|b|million|billion|thousand)?",
            cleaned,
            re.IGNORECASE,
        )
        if not match:
            return None, "", 0

        number_text = match.group(1)
        value = float(number_text)
        suffix = (match.group(2) or "").lower()
        decimals = (
            len(number_text.split(".", 1)[1])
            if "." in number_text
            else 0
        )

        if suffix == "%":
            return value, "%", decimals

        value *= _MULTIPLIERS.get(suffix, 1.0)
        return value, "", decimals

    @classmethod
    def _structured_evidence_numbers(
        cls,
        context_payload: dict[str, Any],
    ) -> list[tuple[float, str, str]]:
        entries: list[tuple[float, str, str]] = []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                # Evidence-ledger records carry semantic metadata alongside a
                # scalar `value`. Use the record's `field` / `semantic_type` to
                # preserve percentage semantics (e.g. `Current Variance %`).
                if "value" in value and (
                    "field" in value or "semantic_type" in value
                ):
                    record_value = value.get("value")
                    field_name = str(value.get("field", "")).lower()
                    semantic_type = str(value.get("semantic_type", "")).lower()
                    unit = (
                        "%"
                        if (
                            "pct" in field_name
                            or "percent" in field_name
                            or "percentage" in field_name
                            or "pct" in semantic_type
                            or "percent" in semantic_type
                        )
                        else ""
                    )
                    if isinstance(record_value, bool):
                        return
                    if isinstance(record_value, (int, float)):
                        entries.append((float(record_value), unit, path + ".value"))
                    elif isinstance(record_value, str):
                        for token in cls._numbers_from_text(record_value):
                            number, token_unit, _ = cls._normalize_number(token)
                            if number is not None:
                                entries.append((number, token_unit or unit, path + ".value"))
                    elif isinstance(record_value, (dict, list, tuple, set)):
                        walk(record_value, path + ".value")
                    return

                for key, child in value.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    walk(child, child_path)
                return

            if isinstance(value, (list, tuple, set)):
                for index, child in enumerate(value):
                    walk(child, f"{path}[{index}]")
                return

            if isinstance(value, bool):
                return

            key = path.split(".")[-1].lower()
            unit = (
                "%"
                if any(
                    marker in key
                    for marker in ("pct", "percent", "percentage")
                )
                else ""
            )

            if isinstance(value, (int, float)):
                entries.append((float(value), unit, path))
                return

            if isinstance(value, str):
                for token in cls._numbers_from_text(value):
                    number, token_unit, _ = cls._normalize_number(token)
                    if number is not None:
                        entries.append((number, token_unit or unit, path))

        walk(context_payload, "")
        return entries

    @classmethod
    def _evidence_display_numbers(
        cls,
        context_payload: dict[str, Any],
    ) -> list[str]:
        entries = cls._structured_evidence_numbers(context_payload)
        return [
            f"{value:g}%" if unit == "%" else f"{value:g}"
            for value, unit, _ in entries
        ]

    @classmethod
    def _matches_evidence(
        cls,
        response_token: str,
        evidence_entries: list[tuple[float, str, str]],
    ) -> bool:
        response_value, response_unit, response_decimals = (
            cls._normalize_number(response_token)
        )
        if response_value is None:
            return True

        for evidence_value, evidence_unit, _ in evidence_entries:
            if response_unit != evidence_unit:
                continue
            if round(evidence_value, response_decimals) == round(
                response_value,
                response_decimals,
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Evidence ledger
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_ledger(context_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        ledger = context_payload.get("evidence_ledger")
        if isinstance(ledger, list):
            out: dict[str, dict[str, Any]] = {}
            for item in ledger:
                if not isinstance(item, dict):
                    continue
                evidence_id = item.get("id")
                if isinstance(evidence_id, str) and evidence_id.strip():
                    out[evidence_id] = item
            return out
        return {}

    @staticmethod
    def _direction_supported(
        evidence_items: list[dict[str, Any]],
        expected: str,
    ) -> bool:
        for item in evidence_items:
            direction = str(item.get("direction", "")).upper()
            if expected == "unfavorable" and direction in {
                "UNFAVORABLE",
                "OVERSPEND",
            }:
                return True
            if expected == "favorable" and direction in {
                "FAVORABLE",
                "UNDERSPEND",
                "SAVINGS",
            }:
                return True
        return False

    @staticmethod
    def _claim_has_pattern(text: str, patterns: list[str]) -> bool:
        return any(
            re.search(pattern, text or "", flags=re.IGNORECASE)
            for pattern in patterns
        )

    def _validate_block_semantics(
        self,
        block_text: str,
        evidence_ids: list[str],
        ledger: dict[str, dict[str, Any]],
        label: str,
    ) -> tuple[list[str], list[str], list[str]]:
        semantic_errors: list[str] = []
        coverage_errors: list[str] = []

        cited = [ledger[eid] for eid in evidence_ids if eid in ledger]

        if not cited:
            coverage_errors.append(
                f"{label}: no resolvable evidence records were cited."
            )
            return [], [], coverage_errors

        cited_payload = {"evidence": cited}
        cited_entries = self._structured_evidence_numbers(cited_payload)
        response_numbers = self._response_number_tokens(block_text)

        for token in response_numbers:
            if not self._matches_evidence(token, cited_entries):
                semantic_errors.append(
                    f"{label}: numeric claim '{token}' is not supported by its cited evidence."
                )

        # Directional wording must agree with the cited evidence semantics.
        if self._claim_has_pattern(
            block_text,
            _DIRECTION_PATTERNS["unfavorable"],
        ) and not self._direction_supported(cited, "unfavorable"):
            semantic_errors.append(
                f"{label}: unfavorable/overspend language conflicts with cited evidence."
            )

        if self._claim_has_pattern(
            block_text,
            _DIRECTION_PATTERNS["favorable"],
        ) and not self._direction_supported(cited, "favorable"):
            # "savings" is acceptable only when evidence explicitly says savings.
            semantic_errors.append(
                f"{label}: favorable/underspend language conflicts with cited evidence."
            )

        has_cause_language = self._claim_has_pattern(
            block_text,
            _SPECULATION_PATTERNS,
        )
        if has_cause_language and not any(
            bool(item.get("supports_cause")) for item in cited
        ):
            semantic_errors.append(
                f"{label}: causal language is not supported by cited evidence."
            )

        has_trend_language = self._claim_has_pattern(
            block_text,
            _TREND_PATTERNS,
        )
        if has_trend_language and not any(
            bool(item.get("supports_trend")) for item in cited
        ):
            semantic_errors.append(
                f"{label}: trend/continuation language is not supported by cited evidence."
            )

        return semantic_errors, [], coverage_errors

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_for_schema_validation(payload: Any) -> str:
        if not isinstance(payload, dict):
            return str(payload)

        parts: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"evidence_ids"}:
                        continue
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
            elif value is not None:
                parts.append(str(value))

        collect(payload)
        return " ".join(parts)

    def validate(
        self,
        response_text: str,
        context_payload: dict[str, Any],
        parsed_payload: dict[str, Any] | None = None,
        schema_errors: list[str] | None = None,
    ) -> GroundingValidationResult:
        ledger_payload = (
            {"evidence_ledger": context_payload.get("evidence_ledger")}
            if isinstance(context_payload.get("evidence_ledger"), list)
            else context_payload
        )

        evidence_entries = self._structured_evidence_numbers(ledger_payload)
        evidence_numbers = self._evidence_display_numbers(ledger_payload)

        text_for_validation = (
            self._extract_text_for_schema_validation(parsed_payload)
            if isinstance(parsed_payload, dict)
            else response_text
        )
        response_numbers = self._response_number_tokens(text_for_validation)

        unsupported_numbers = [
            token
            for token in response_numbers
            if not self._matches_evidence(token, evidence_entries)
        ]

        speculative_claims = self._find_phrases(
            text_for_validation,
            _SPECULATION_PATTERNS,
        )
        trend_claims = self._find_phrases(
            text_for_validation,
            _TREND_PATTERNS,
        )

        ledger = self._evidence_ledger(context_payload)
        financial_semantic_errors: list[str] = []
        evidence_coverage_errors: list[str] = []
        referenced_ids: list[str] = []

        if isinstance(parsed_payload, dict):
            factual_blocks: list[tuple[str, Any]] = []

            if "executive_observation" in parsed_payload:
                factual_blocks.append(
                    ("executive_observation", parsed_payload["executive_observation"])
                )

            for index, block in enumerate(parsed_payload.get("key_issues", [])):
                factual_blocks.append((f"key_issues[{index}]", block))

            for index, block in enumerate(
                parsed_payload.get("evidence_based_interpretation", [])
            ):
                factual_blocks.append(
                    (f"evidence_based_interpretation[{index}]", block)
                )

            if "scenario_implication" in parsed_payload:
                factual_blocks.append(
                    ("scenario_implication", parsed_payload["scenario_implication"])
                )

            for label, block in factual_blocks:
                if not isinstance(block, dict):
                    continue
                block_text = str(block.get("text", ""))
                ids = block.get("evidence_ids", [])
                if isinstance(ids, list):
                    referenced_ids.extend(
                        item for item in ids if isinstance(item, str)
                    )
                    sem_errors, _, cov_errors = self._validate_block_semantics(
                        block_text,
                        [item for item in ids if isinstance(item, str)],
                        ledger,
                        label,
                    )
                    financial_semantic_errors.extend(sem_errors)
                    evidence_coverage_errors.extend(cov_errors)

            # Recommendations have no evidence_ids by design, but any figures
            # they introduce still have to exist somewhere in VARIA evidence.
            for index, recommendation in enumerate(
                parsed_payload.get("recommendations", [])
            ):
                rec_text = str(recommendation)
                rec_numbers = self._response_number_tokens(rec_text)
                for token in rec_numbers:
                    if not self._matches_evidence(token, evidence_entries):
                        financial_semantic_errors.append(
                            f"recommendations[{index}]: numeric claim '{token}' is not supported by VARIA evidence."
                        )

        evidence_has_time_series = any(
            marker in repr(context_payload).lower()
            for marker in (
                "month",
                "monthly",
                "year",
                "fiscal_year",
                "time_series",
                "historical",
                "history",
                "trend",
            )
        )

        # Legacy global trend control remains, but semantic validation is now
        # stricter when a structured response cites evidence.
        trend_claims_without_evidence = (
            []
            if evidence_has_time_series
            else trend_claims
        )

        unsupported_percentages = [
            token
            for token in unsupported_numbers
            if "%" in token
        ]

        unsupported_causal_blocks = [
            error
            for error in financial_semantic_errors
            if "causal language is not supported" in error
        ]
        unsupported_trend_blocks = [
            error
            for error in financial_semantic_errors
            if "trend/continuation language is not supported" in error
        ]

        checks = {
            "numeric_grounding": len(unsupported_numbers) == 0,
            "no_unsupported_causal_speculation": (
                len(unsupported_causal_blocks) == 0
                if isinstance(parsed_payload, dict)
                else len(speculative_claims) == 0
            ),
            "no_unsupported_trend_claims": (
                len(unsupported_trend_blocks) == 0
                and len(trend_claims_without_evidence) == 0
            ),
            "financial_semantics": len(financial_semantic_errors) == 0,
            "evidence_coverage": len(evidence_coverage_errors) == 0,
            "schema": not schema_errors,
        }

        return GroundingValidationResult(
            passed=all(checks.values()),
            unsupported_numbers=unsupported_numbers,
            unsupported_percentages=unsupported_percentages,
            speculative_claims=speculative_claims,
            trend_claims_without_evidence=trend_claims_without_evidence,
            evidence_numbers=evidence_numbers,
            response_numbers=response_numbers,
            checks=checks,
            schema_errors=list(schema_errors or []),
            financial_semantic_errors=financial_semantic_errors,
            evidence_coverage_errors=evidence_coverage_errors,
            referenced_evidence_ids=sorted(set(referenced_ids)),
        )

    @staticmethod
    def _find_phrases(
        text: str,
        patterns: list[str],
    ) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            for match in re.finditer(
                pattern,
                text or "",
                flags=re.IGNORECASE,
            ):
                snippet = _compact_snippet(text or "", match.start(), match.end())
                if snippet not in found:
                    found.append(snippet)
        return found
