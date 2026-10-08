from __future__ import annotations

import json
from typing import Any

from src.grounded_ai_engine import VARIA_OUTPUT_SCHEMA, VARIAGroundedAIEngine
from src.llm_grounding_validator import LLMGroundingValidator


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0
        self.formats: list[Any] = []

    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool | dict[str, Any] = False,
    ) -> str:
        self.formats.append(format_json)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def context() -> dict[str, Any]:
    return {
        "pack_version": "2.0",
        "verified_financial_rules": {
            "expense_direction_rule": "positive variance = unfavorable / overspend; negative variance = favorable / underspend",
            "planning_exception_rule": "a planning exception amount is reconciliation exposure and must not be treated as operating overspend without explicit evidence",
        },
        "evidence_ledger": [
            {
                "id": "EXEC-003",
                "section": "Executive Position",
                "field": "Current Variance",
                "value": 100000.0,
                "semantic_type": "CURRENT_VARIANCE",
                "direction": "UNFAVORABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "EXEC-004",
                "section": "Executive Position",
                "field": "Current Variance %",
                "value": 10.0,
                "semantic_type": "CURRENT_VARIANCE_PCT",
                "direction": "UNFAVORABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "ISS-001-B",
                "section": "Top CFO Issues",
                "field": "Projected Variance",
                "value": 120000.0,
                "semantic_type": "PROJECTED_VARIANCE",
                "direction": "UNFAVORABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "ISS-001-C",
                "section": "Top CFO Issues",
                "field": "Projected Variance %",
                "value": 12.0,
                "semantic_type": "PROJECTED_VARIANCE_PCT",
                "direction": "UNFAVORABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "ISS-001-D",
                "section": "Top CFO Issues",
                "field": "Early Warning",
                "value": "RED",
                "semantic_type": "EARLY_WARNING",
                "direction": "NOT_APPLICABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "PLAN-001",
                "section": "Planning Exceptions",
                "field": "Exception Transactions",
                "value": 2,
                "semantic_type": "PLANNING_EXCEPTION_COUNT",
                "direction": "NOT_APPLICABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "PLAN-002",
                "section": "Planning Exceptions",
                "field": "Exception Amount",
                "value": 60569.56,
                "semantic_type": "PLANNING_EXCEPTION_AMOUNT",
                "direction": "NOT_APPLICABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
            {
                "id": "SCN-001-Scenario_Projected_Variance",
                "section": "Scenario Results",
                "field": "Scenario Projected Variance",
                "value": 120000.0,
                "semantic_type": "SCENARIO_PROJECTED_VARIANCE",
                "direction": "UNFAVORABLE",
                "supports_cause": False,
                "supports_trend": False,
            },
        ],
    }


def valid_response() -> str:
    return json.dumps(
        {
            "executive_observation": {
                "text": "The current variance is ₹100,000, or 10%, and is unfavorable.",
                "evidence_ids": ["EXEC-003", "EXEC-004"],
            },
            "key_issues": [
                {
                    "text": "The projected variance is ₹120,000, or 12%, with a RED warning.",
                    "evidence_ids": ["ISS-001-B", "ISS-001-C", "ISS-001-D"],
                },
                {
                    "text": "The reconciliation contains 2 planning exception transactions totaling ₹60,569.56.",
                    "evidence_ids": ["PLAN-001", "PLAN-002"],
                },
            ],
            "evidence_based_interpretation": [
                {
                    "text": "Cause cannot be established from the available evidence.",
                    "evidence_ids": ["ISS-001-B"],
                }
            ],
            "recommendations": [
                "Recommendation: Review the RED issue using the underlying finance records."
            ],
            "scenario_implication": {
                "text": "The scenario projected variance is ₹120,000 and is unfavorable.",
                "evidence_ids": ["SCN-001-Scenario_Projected_Variance"],
            },
            "evidence_gaps": [
                "Detailed causal evidence is not supplied."
            ],
        }
    )


def test_ollama_receives_real_schema_and_valid_response_passes():
    llm = ScriptedLLM([valid_response()])
    result = VARIAGroundedAIEngine(llm_client=llm, max_retries=0).generate(context())

    assert result["status"] == "PASS", result
    assert result["grounded"] is True
    assert llm.formats[0] == VARIA_OUTPUT_SCHEMA
    assert result["validation"]["checks"]["schema"] is True
    assert result["validation"]["checks"]["financial_semantics"] is True


def test_exception_amount_cannot_be_relabelled_as_overspend():
    bad = json.loads(valid_response())
    bad["key_issues"][1]["text"] = (
        "The reconciliation exception is an overspend of ₹60,569.56."
    )

    llm = ScriptedLLM([json.dumps(bad)])
    result = VARIAGroundedAIEngine(llm_client=llm, max_retries=0).generate(context())

    assert result["status"] == "REVIEW"
    assert result["grounded"] is False
    errors = result["validation"]["financial_semantic_errors"]
    assert any("conflicts with cited evidence" in e for e in errors)


def test_causal_claim_without_causal_evidence_fails():
    bad = json.loads(valid_response())
    bad["evidence_based_interpretation"][0]["text"] = (
        "The projected overspend is due to higher vendor costs."
    )

    llm = ScriptedLLM([json.dumps(bad)])
    result = VARIAGroundedAIEngine(llm_client=llm, max_retries=0).generate(context())

    assert result["status"] == "REVIEW"
    assert result["grounded"] is False
    assert any(
        "causal language is not supported" in e
        for e in result["validation"]["financial_semantic_errors"]
    )


def test_supported_cause_is_allowed_only_when_cited_evidence_supports_it():
    ctx = context()
    ctx["evidence_ledger"].append(
        {
            "id": "CAUSE-001",
            "section": "Validated Cause Evidence",
            "field": "Cause",
            "value": "supplier price increase",
            "semantic_type": "VALIDATED_CAUSE",
            "direction": "NOT_APPLICABLE",
            "supports_cause": True,
            "supports_trend": False,
        }
    )
    good = json.loads(valid_response())
    good["evidence_based_interpretation"][0] = {
        "text": "The overspend is caused by a supplier price increase.",
        "evidence_ids": ["ISS-001-B", "CAUSE-001"],
    }

    llm = ScriptedLLM([json.dumps(good)])
    result = VARIAGroundedAIEngine(
        llm_client=llm,
        max_retries=0,
    ).generate(ctx)

    assert result["status"] == "PASS", result
    assert result["grounded"] is True
    assert result["validation"]["checks"]["no_unsupported_causal_speculation"] is True


def test_schema_rejects_extra_fields():
    bad = json.loads(valid_response())
    bad["executive_observation"]["confidence"] = 0.91

    llm = ScriptedLLM([json.dumps(bad)])
    result = VARIAGroundedAIEngine(llm_client=llm, max_retries=0).generate(context())

    assert result["status"] == "REVIEW"
    assert result["grounded"] is False
    assert any(
        "unexpected_fields" in e
        for e in result["validation"]["schema_errors"]
    )


def test_validator_rejects_unsupported_numeric_recommendation():
    bad = json.loads(valid_response())
    bad["recommendations"] = [
        "Recommendation: Reduce the budget by 37%."
    ]

    validator = LLMGroundingValidator()
    parsed = bad
    result = validator.validate(
        response_text=json.dumps(bad),
        context_payload={"evidence_ledger": context()["evidence_ledger"]},
        parsed_payload=parsed,
        schema_errors=[],
    )

    assert result.passed is False
    assert any(
        "numeric claim '37%' is not supported" in e
        for e in result.financial_semantic_errors
    )


def test_trend_claim_without_trend_evidence_fails_at_block_level():
    bad = json.loads(valid_response())
    bad["executive_observation"]["text"] = (
        "The unfavorable variance continues."
    )

    llm = ScriptedLLM([json.dumps(bad)])
    result = VARIAGroundedAIEngine(llm_client=llm, max_retries=0).generate(context())

    assert result["status"] == "REVIEW"
    assert any(
        "trend/continuation language is not supported" in e
        for e in result["validation"]["financial_semantic_errors"]
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))


def test_scenario_identifier_numbers_are_not_financial_claims():
    validator = LLMGroundingValidator()
    assert validator._response_number_tokens(
        "Scenario 001 is ranked first; Scenario 004 is the base scenario."
    ) == []
