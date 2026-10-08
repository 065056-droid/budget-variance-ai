from typing import Any

from src.grounded_ai_engine import VARIAGroundedAIEngine


class ConservativeFakeLLM:
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
                "Executive Observation\n"
                "The variance is significant.\n"
                "Evidence-Based Interpretation\n"
                "The variance could be due to market conditions."
            )

        return (
            "Executive Observation\n"
            "The supplied evidence shows a current variance of 100000 "
            "and a projected variance of 120000, or 12%.\n"
            "Key Financial Issues\n"
            "The issue is RED and HIGH PRIORITY.\n"
            "Evidence-Based Interpretation\n"
            "Cause cannot be established from the available evidence.\n"
            "Management Actions\n"
            "Recommendation: Review the underlying spend.\n"
            "Scenario Implication\n"
            "No additional scenario conclusion is supplied.\n"
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
        return (
            "Executive Observation\n"
            "The variance is high.\n"
            "Evidence-Based Interpretation\n"
            "It is likely due to competition."
        )


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V3 TEST")
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

    engine = VARIAGroundedAIEngine(
        llm_client=ConservativeFakeLLM(),
        max_retries=2,
    )

    result = engine.generate(context)

    assert result["status"] == "PASS"
    assert result["grounded"] is True
    assert len(result["attempts"]) == 2

    print("\nSTRICT RETRY: PASS")
    print(f"Attempts required: {len(result['attempts'])}")

    fail_engine = VARIAGroundedAIEngine(
        llm_client=AlwaysBadLLM(),
        max_retries=2,
    )

    failed = fail_engine.generate(context)

    assert failed["status"] == "REVIEW"
    assert failed["grounded"] is False

    print("\nFAIL-CLOSED: PASS")
    print("Persistent unsupported causal language remains REVIEW.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V3 TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
