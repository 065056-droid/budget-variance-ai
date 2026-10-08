import json
from typing import Any

from src.grounded_ai_engine import VARIAGroundedAIEngine


class GoodFakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        self.calls += 1

        if self.calls == 1:
            return (
                '{"executive_observation":{"text":"The variance is '
                'significant.","evidence_ids":[]},'
                '"key_issues":[],"evidence_based_interpretation":[],'
                '"recommendations":[],"scenario_implication":'
                '{"text":"","evidence_ids":[]},"evidence_gaps":[]}'
            )

        return json.dumps({
            "executive_observation": {
                "text": (
                    "The supplied evidence shows a current variance "
                    "of 100000 and a projected variance of 120000, "
                    "or 12%."
                ),
                "evidence_ids": ["E1", "E2", "E3"],
            },
            "key_issues": [
                {
                    "text": "The issue is RED and HIGH PRIORITY.",
                    "evidence_ids": ["E4", "E5"],
                }
            ],
            "evidence_based_interpretation": [
                {
                    "text": (
                        "Cause cannot be established from the "
                        "available evidence."
                    ),
                    "evidence_ids": ["E1", "E2", "E3"],
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


class AlwaysBadLLM:
    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        return json.dumps({
            "executive_observation": {
                "text": "It is likely due to competition.",
                "evidence_ids": ["E1"],
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


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V4 TEST")
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

    retry_engine = VARIAGroundedAIEngine(
        llm_client=GoodFakeLLM(),
        max_retries=2,
    )

    result = retry_engine.generate(context)

    assert result["status"] == "PASS"
    assert result["grounded"] is True
    assert len(result["attempts"]) == 2

    print("\nSTRUCTURED EVIDENCE ANCHORING: PASS")
    print(f"Attempts required: {len(result['attempts'])}")

    fail_engine = VARIAGroundedAIEngine(
        llm_client=AlwaysBadLLM(),
        max_retries=1,
    )

    failed = fail_engine.generate(context)

    assert failed["status"] == "REVIEW"
    assert failed["grounded"] is False

    print("\nFAIL-CLOSED: PASS")
    print("Unsupported causal output remains REVIEW.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V4 TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
