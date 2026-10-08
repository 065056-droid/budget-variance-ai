from typing import Any

from src.grounded_ai_engine import VARIAGroundedAIEngine


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
                "The current trend may continue."
            )

        return (
            "The supplied evidence shows a current variance of 100000, "
            "a projected variance of 120000, and a projected variance "
            "of 12%. The issue is RED and HIGH PRIORITY. "
            "The available evidence does not establish the root cause. "
            "Management should review the underlying spend and consider "
            "targeted cost controls."
        )


class AlwaysBadLLM:
    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool = False,
    ) -> str:
        return "The variance could be due to competition and may continue."


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI ENGINE TEST")
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
        llm_client=FakeLLM(),
        max_retries=1,
    )

    result = retry_engine.generate(context)

    assert result["status"] == "PASS"
    assert result["grounded"] is True
    assert result["attempts"][-1]["validation"]["passed"] is True
    assert len(result["attempts"]) == 2

    print("\nRETRY-AND-VALIDATE: PASS")
    print(f"Attempts required: {len(result['attempts'])}")

    fail_engine = VARIAGroundedAIEngine(
        llm_client=AlwaysBadLLM(),
        max_retries=1,
    )

    failed = fail_engine.generate(context)

    assert failed["status"] == "REVIEW"
    assert failed["grounded"] is False
    assert failed["validation"]["passed"] is False

    print("\nFAIL-CLOSED BEHAVIOUR: PASS")
    print("Unresolved grounding problems correctly produce REVIEW.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI ENGINE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
