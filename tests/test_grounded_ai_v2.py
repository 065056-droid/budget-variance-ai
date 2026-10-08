from typing import Any

from src.grounded_ai_engine import VARIAGroundedAIEngine
from src.llm_grounding_validator import LLMGroundingValidator


class FakeLLM:
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
                "The variance could be due to market conditions. "
                "The trend may continue."
            )

        return (
            "Executive Observation\n"
            "The supplied evidence shows a current variance of 100000 "
            "and a projected variance of 120000, or 12%. "
            "Key Financial Issues\n"
            "The issue is RED and HIGH PRIORITY.\n"
            "Evidence-Based Interpretation\n"
            "Cause cannot be established from the available evidence.\n"
            "Management Actions\n"
            "Management should review the underlying spend and consider "
            "targeted cost controls.\n"
            "Scenario Implication\n"
            "No scenario conclusion is supplied.\n"
            "Key Caveats / Evidence Gaps\n"
            "Detailed causal evidence is not supplied."
        )


class AlwaysBadLLM:
    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        return "1. The variance could be due to competition."


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V2 TEST")
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

    validator = LLMGroundingValidator()

    rounded_response = (
        "The projected variance is 14.466% and the current variance "
        "is $994,433.79."
    )

    rounded_context = {
        "projected_variance_pct": 14.466254415264329,
        "current_variance": 994433.7878141887,
    }

    rounded_result = validator.validate(
        response_text=rounded_response,
        context_payload=rounded_context,
    )

    assert rounded_result.passed
    print("\nROUNDING TOLERANCE: PASS")
    print("Rounded financial values are matched to supplied evidence.")

    good_engine = VARIAGroundedAIEngine(
        llm_client=FakeLLM(),
        max_retries=2,
    )
    result = good_engine.generate(context)

    assert result["status"] == "PASS"
    assert result["grounded"] is True
    assert len(result["attempts"]) == 2

    print("\nRETRY-AND-VALIDATE: PASS")
    print(f"Attempts required: {len(result['attempts'])}")

    fail_engine = VARIAGroundedAIEngine(
        llm_client=AlwaysBadLLM(),
        max_retries=2,
    )
    failed = fail_engine.generate(context)

    assert failed["status"] == "REVIEW"
    assert failed["grounded"] is False
    assert failed["validation"]["passed"] is False

    print("\nFAIL-CLOSED BEHAVIOUR: PASS")
    print("Unsupported causal commentary remains REVIEW.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V2 TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
