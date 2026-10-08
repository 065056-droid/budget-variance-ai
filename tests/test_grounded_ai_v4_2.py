import json
from typing import Any

from src.grounded_ai_engine import VARIAGroundedAIEngine


class GoodJSONLLM:
    def __init__(self):
        self.calls = 0
        self.format_json_values = []

    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        self.calls += 1
        self.format_json_values.append(format_json)

        if self.calls == 1:
            return json.dumps({
                "executive_observation": {
                    "text": "The issue requires attention.",
                    "evidence_ids": [],
                },
                "key_issues": [],
                "evidence_based_interpretation": [],
                "recommendations": [],
                "scenario_implication": {
                    "text": "",
                    "evidence_ids": [],
                },
                "evidence_gaps": [],
            })

        return json.dumps({
            "executive_observation": {
                "text": (
                    "The current variance is 100000 and the projected "
                    "variance is 120000, or 12%."
                ),
                "evidence_ids": ["E1", "E2", "E3"],
            },
            "key_issues": [
                {
                    "text": "The issue is RED.",
                    "evidence_ids": ["E4"],
                }
            ],
            "evidence_based_interpretation": [
                {
                    "text": (
                        "Cause cannot be established from the "
                        "available evidence."
                    ),
                    "evidence_ids": ["E1"],
                }
            ],
            "recommendations": [
                "Recommendation: Review the underlying spend."
            ],
            "scenario_implication": {
                "text": "No additional scenario conclusion is supplied.",
                "evidence_ids": ["E1"],
            },
            "evidence_gaps": [
                "Detailed causal evidence is not supplied."
            ],
        })


class BadDirectionLLM:
    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        return json.dumps({
            "executive_observation": {
                "text": "The underspend is due to competition.",
                "evidence_ids": ["E1"],
            },
            "key_issues": [],
            "evidence_based_interpretation": [],
            "recommendations": [],
            "scenario_implication": {
                "text": "No conclusion.",
                "evidence_ids": ["E1"],
            },
            "evidence_gaps": [],
        })


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V4.2 TEST")
    print("=" * 80)

    context = {
        "financial_position": {
            "current_variance": 100000.0,
            "projected_variance": 120000.0,
            "projected_variance_pct": 12.0,
        },
        "early_warning": "RED",
        "decision_priority": "HIGH PRIORITY",
    }

    good = GoodJSONLLM()

    result = VARIAGroundedAIEngine(
        llm_client=good,
        max_retries=2,
    ).generate(context)

    assert result["status"] == "PASS", result
    assert result["grounded"] is True
    assert len(result["attempts"]) == 2
    assert all(good.format_json_values)

    print("\nJSON MODE + EVIDENCE ANCHORING: PASS")
    print(f"Attempts required: {len(result['attempts'])}")

    bad = VARIAGroundedAIEngine(
        llm_client=BadDirectionLLM(),
        max_retries=0,
    ).generate(context)

    assert bad["status"] == "REVIEW"
    assert bad["grounded"] is False

    print("\nFAIL-CLOSED SEMANTIC CONTROL: PASS")
    print("Unsupported causal content remains REVIEW.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V4.2 TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
