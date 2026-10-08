from __future__ import annotations

import json
from typing import Any, Protocol

from src.llm_grounding_validator import LLMGroundingValidator
from src.ollama_llm_engine import OllamaLLMEngine


VARIA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "executive_observation",
        "key_issues",
        "evidence_based_interpretation",
        "recommendations",
        "scenario_implication",
        "evidence_gaps",
    ],
    "properties": {
        "executive_observation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "evidence_ids"],
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 500},
                "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "key_issues": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "evidence_based_interpretation": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "recommendations": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "string",
                "minLength": 15,
                "maxLength": 400,
                "pattern": r"^Recommendation:",
            },
        },
        "scenario_implication": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "evidence_ids"],
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 500},
                "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "evidence_gaps": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 300,
            },
        },
    },
}


class _LLMClient(Protocol):
    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool | dict[str, Any] = False,
    ) -> str:
        ...


class VARIAGroundedAIEngine:
    """
    VARIA strict grounded AI layer.

    Controls:
      - Ollama receives the complete VARIA JSON Schema, not generic JSON mode.
      - Post-generation schema validation is fail-closed.
      - Every factual block must cite real evidence IDs.
      - Financial numbers must match cited evidence.
      - Overspend / underspend semantics are checked against evidence metadata.
      - Causal and trend claims require explicit supporting evidence.

    VARIA remains the source of truth for all financial decisions.
    """

    def __init__(
        self,
        llm_client: _LLMClient | None = None,
        validator: LLMGroundingValidator | None = None,
        max_retries: int = 2,
    ) -> None:
        self.llm = llm_client or OllamaLLMEngine()
        self.validator = validator or LLMGroundingValidator()
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _flatten_evidence(
        value: Any,
        prefix: str = "E",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for key, child in obj.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    walk(child, child_path)
            elif isinstance(obj, (list, tuple)):
                for i, child in enumerate(obj):
                    walk(child, f"{path}[{i}]")
            else:
                rows.append(
                    {
                        "id": f"{prefix}{len(rows) + 1}",
                        "section": "Legacy Flattened Context",
                        "field": path,
                        "value": str(obj),
                        "semantic_type": "LEGACY",
                        "direction": "NOT_APPLICABLE",
                        "supports_cause": False,
                        "supports_trend": False,
                    }
                )

        walk(value, "")
        return rows

    @classmethod
    def _prompt_context(
        cls,
        context_payload: dict[str, Any],
    ) -> dict[str, Any]:
        explicit_ledger = context_payload.get("evidence_ledger")
        evidence = (
            explicit_ledger
            if isinstance(explicit_ledger, list)
            else cls._flatten_evidence(context_payload)
        )

        rules = [
            "VARIA is the financial source of truth.",
            "Return only the exact JSON shape defined by the supplied schema.",
            "Every factual block must cite one or more existing evidence IDs.",
            "Never introduce a financial number that is absent from VARIA evidence.",
            "Never reverse overspend/unfavorable into underspend/favorable or vice versa.",
            "A planning exception amount is not an overspend unless evidence explicitly says so.",
            "Do not infer a cause. When the cause is not explicitly supported, use exactly: Cause cannot be established from the available evidence.",
            "Do not state that a trend, continuation, trajectory, sustainability, or historical pattern exists unless cited evidence explicitly supports it.",
            "Recommendations are actions, not facts, and must begin with 'Recommendation:'.",
            "Do not place recommendations in factual blocks.",
        ]

        return {
            "schema": VARIA_OUTPUT_SCHEMA,
            "system_instruction": "\n".join(rules),
            "model_role": "Conservative FP&A reporting assistant. VARIA decides WHAT matters; the LLM only communicates verified evidence.",
            "evidence_ledger": evidence,
            "verified_financial_rules": context_payload.get(
                "verified_financial_rules",
                {
                    "expense_direction_rule": "positive variance = unfavorable / overspend; negative variance = favorable / underspend",
                    "planning_exception_rule": "exception amounts are reconciliation exposure, not overspend by themselves",
                },
            ),
        }

    @staticmethod
    def _validate_json_schema(
        response: str,
        valid_evidence_ids: set[str],
    ) -> tuple[bool, dict[str, Any] | None, list[str]]:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            return False, None, [f"invalid_json: {exc}"]

        errors: list[str] = []

        if not isinstance(payload, dict):
            return False, None, ["root_must_be_object"]

        required_root = set(VARIA_OUTPUT_SCHEMA["required"])
        actual_root = set(payload.keys())
        missing = sorted(required_root - actual_root)
        extra = sorted(actual_root - required_root)
        if missing:
            errors.append(f"missing_required_fields: {missing}")
        if extra:
            errors.append(f"unexpected_fields: {extra}")

        def validate_fact_block(block: Any, label: str) -> None:
            if not isinstance(block, dict):
                errors.append(f"{label}: must_be_object")
                return

            allowed = {"text", "evidence_ids"}
            actual = set(block.keys())
            if "text" not in block:
                errors.append(f"{label}: missing_text")
            if "evidence_ids" not in block:
                errors.append(f"{label}: missing_evidence_ids")
            extra_keys = sorted(actual - allowed)
            if extra_keys:
                errors.append(f"{label}: unexpected_fields={extra_keys}")

            text_value = block.get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                errors.append(f"{label}: text_must_be_nonempty_string")
            elif len(text_value) > 500:
                errors.append(f"{label}: text_too_long")

            ids = block.get("evidence_ids")
            if not isinstance(ids, list) or not ids:
                errors.append(f"{label}: evidence_ids_must_be_nonempty_list")
                return
            if len(ids) > 8:
                errors.append(f"{label}: too_many_evidence_ids")
            for evidence_id in ids:
                if not isinstance(evidence_id, str) or not evidence_id.strip():
                    errors.append(f"{label}: evidence_id_must_be_nonempty_string")
                elif evidence_id not in valid_evidence_ids:
                    errors.append(f"{label}: unknown_evidence_id={evidence_id}")

        validate_fact_block(
            payload.get("executive_observation"),
            "executive_observation",
        )

        for field_name in (
            "key_issues",
            "evidence_based_interpretation",
        ):
            blocks = payload.get(field_name)
            if not isinstance(blocks, list):
                errors.append(f"{field_name}: must_be_array")
                continue
            if len(blocks) > 5:
                errors.append(f"{field_name}: too_many_items")
            for idx, block in enumerate(blocks):
                validate_fact_block(block, f"{field_name}[{idx}]")

        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, list):
            errors.append("recommendations: must_be_array")
        else:
            if len(recommendations) > 5:
                errors.append("recommendations: too_many_items")
            for idx, item in enumerate(recommendations):
                if not isinstance(item, str):
                    errors.append(f"recommendations[{idx}]: must_be_string")
                    continue
                if not item.strip().startswith("Recommendation:"):
                    errors.append(
                        f"recommendations[{idx}]: must_start_with_Recommendation"
                    )
                if len(item) < 15:
                    errors.append(f"recommendations[{idx}]: too_short")
                if len(item) > 400:
                    errors.append(f"recommendations[{idx}]: too_long")

        validate_fact_block(
            payload.get("scenario_implication"),
            "scenario_implication",
        )

        gaps = payload.get("evidence_gaps")
        if not isinstance(gaps, list):
            errors.append("evidence_gaps: must_be_array")
        else:
            if len(gaps) > 6:
                errors.append("evidence_gaps: too_many_items")
            for idx, item in enumerate(gaps):
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"evidence_gaps[{idx}]: must_be_nonempty_string")
                elif len(item) > 300:
                    errors.append(f"evidence_gaps[{idx}]: too_long")

        return len(errors) == 0, payload, errors

    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        prompt_context = self._prompt_context(context_payload)

        valid_ids = {
            item["id"]
            for item in prompt_context["evidence_ledger"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }

        attempts: list[dict[str, Any]] = []
        response = ""

        for attempt_number in range(self.max_retries + 1):
            response = self.llm.generate(
                context_payload=prompt_context,
                temperature=temperature,
                format_json=VARIA_OUTPUT_SCHEMA,
            )

            structure_ok, payload, schema_errors = self._validate_json_schema(
                response,
                valid_ids,
            )

            validation = self.validator.validate(
                response_text=response,
                context_payload=prompt_context,
                parsed_payload=payload,
                schema_errors=schema_errors,
            )

            passed = structure_ok and validation.passed

            attempts.append(
                {
                    "attempt": attempt_number + 1,
                    "schema_ok": structure_ok,
                    "schema_errors": schema_errors,
                    "financial_semantics_ok": validation.checks.get(
                        "financial_semantics", False
                    ),
                    "evidence_coverage_ok": validation.checks.get(
                        "evidence_coverage", False
                    ),
                    "validation": validation.to_dict(),
                }
            )

            if passed:
                return {
                    "status": "PASS",
                    "grounded": True,
                    "structured": payload,
                    "response": response,
                    "validation": validation.to_dict(),
                    "attempts": attempts,
                }

            rejection_reasons = schema_errors[:4]
            rejection_reasons.extend(validation.financial_semantic_errors[:4])
            rejection_reasons.extend(validation.evidence_coverage_errors[:3])
            if validation.unsupported_numbers:
                rejection_reasons.append(
                    f"unsupported_numbers={validation.unsupported_numbers[:5]}"
                )
            if not rejection_reasons:
                rejection_reasons.append("grounding_controls_failed")

            prompt_context["system_instruction"] += (
                "\n\nPREVIOUS RESPONSE REJECTED. Fix these exact control failures and "
                "return the same strict JSON schema.\n- "
                + "\n- ".join(rejection_reasons)
            )

        return {
            "status": "REVIEW",
            "grounded": False,
            "structured": None,
            "response": response,
            "validation": attempts[-1]["validation"],
            "attempts": attempts,
        }
