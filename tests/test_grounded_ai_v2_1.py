from src.llm_grounding_validator import LLMGroundingValidator


def main():
    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V2.1 TEST")
    print("=" * 80)

    validator = LLMGroundingValidator()

    rounded_context = {
        "projected_variance_pct": 14.466254415264329,
        "current_variance": 994433.7878141887,
    }

    rounded_response = (
        "The projected variance is 14.466% and the current variance "
        "is $994,433.79."
    )

    rounded_result = validator.validate(
        response_text=rounded_response,
        context_payload=rounded_context,
    )

    assert rounded_result.passed, rounded_result.to_dict()

    print("\nROUNDING TOLERANCE: PASS")
    print("Rounded financial values match supplied evidence.")

    context = {
        "financial_position": {
            "current_variance": 100000.0,
            "projected_variance": 120000.0,
            "projected_variance_pct": 12.0,
        },
        "early_warning": "RED",
        "decision_priority": "HIGH PRIORITY",
    }

    bad_response = (
        "1. The variance could be due to competition. "
        "2. The trend may continue."
    )

    bad_result = validator.validate(
        response_text=bad_response,
        context_payload=context,
    )

    assert not bad_result.passed
    assert bad_result.speculative_claims
    assert bad_result.trend_claims_without_evidence

    print("\nUNSUPPORTED CLAIMS DETECTED: PASS")
    print("Causal speculation and unsupported trend language are blocked.")

    good_response = (
        "Executive Observation\n"
        "The supplied evidence shows a current variance of 100000 "
        "and a projected variance of 120000, or 12%.\n"
        "Evidence-Based Interpretation\n"
        "Cause cannot be established from the available evidence.\n"
        "Management Actions\n"
        "Management should review the underlying spend."
    )

    good_result = validator.validate(
        response_text=good_response,
        context_payload=context,
    )

    assert good_result.passed, good_result.to_dict()

    print("\nSUPPORTED RESPONSE: PASS")
    print("Evidence-only commentary passes validation.")

    print("\n" + "=" * 80)
    print("VARIA GROUNDED AI V2.1 TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
