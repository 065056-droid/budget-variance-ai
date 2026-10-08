from src.llm_grounding_validator import LLMGroundingValidator


def main():
    print("\n" + "=" * 80)
    print("VARIA LLM GROUNDING VALIDATION TEST")
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

    bad_response = """
    The variance could be due to increased competition and higher media costs.
    If the current trend continues, the overrun may increase.
    """

    good_response = """
    The supplied evidence shows a current variance of 100000 and a
    projected variance of 120000, or 12%. The issue is marked RED and
    HIGH PRIORITY. The available evidence does not establish the root cause.
    Management should review the underlying advertising spend and consider
    targeted cost controls.
    """

    validator = LLMGroundingValidator()

    bad_result = validator.validate(
        response_text=bad_response,
        context_payload=context,
    )

    assert not bad_result.passed
    assert bad_result.speculative_claims
    assert bad_result.trend_claims_without_evidence

    print("\nUNSUPPORTED SPECULATION DETECTED: PASS")
    print(
        f"Speculative claims: {len(bad_result.speculative_claims)}"
    )
    print(
        "Trend claims without evidence: "
        f"{len(bad_result.trend_claims_without_evidence)}"
    )

    good_result = validator.validate(
        response_text=good_response,
        context_payload=context,
    )

    assert good_result.passed
    assert good_result.unsupported_numbers == []

    print("\nSUPPORTED RESPONSE VALIDATION: PASS")
    print("No unsupported numeric facts detected.")

    print("\n" + "=" * 80)
    print("VARIA LLM GROUNDING VALIDATION TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
