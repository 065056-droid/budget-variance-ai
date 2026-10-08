import json
import time

from src.data_loader import load_workbook
from src.decision_intelligence_pipeline import DecisionIntelligencePipeline
from src.grounded_ai_engine import VARIAGroundedAIEngine
from src.ollama_llm_engine import OllamaLLMEngine


MODELS = [
    "qwen2.5:7b",
    "granite4:tiny-h",
]


def main():
    print("\n" + "=" * 80)
    print("VARIA QWEN VS GRANITE A/B TEST")
    print("=" * 80)

    data = load_workbook()

    print("\nBUILDING VERIFIED VARIA CONTEXT...")
    pipeline = DecisionIntelligencePipeline(
        budget=data["Budget"],
        actuals=data["Actuals_Raw"],
    )

    # Build the full financial evidence once. No local LLM call here.
    outputs = pipeline.run(enable_local_llm=False)
    context = outputs["llm_context"]

    results = []

    for model in MODELS:
        print("\n" + "-" * 80)
        print(f"TESTING MODEL: {model}")
        print("-" * 80)

        client = OllamaLLMEngine(
            model=model,
            timeout=180,
        )

        health = client.health_check()
        print(f"Available: {health.get('ok')}")

        if not health.get("ok"):
            print("SKIPPED: model not available in Ollama.")
            continue

        engine = VARIAGroundedAIEngine(
            llm_client=client,
            max_retries=2,
        )

        start = time.perf_counter()

        result = engine.generate(
            context_payload=context,
            temperature=0.0,
        )

        elapsed = time.perf_counter() - start

        validation = result.get("validation", {})

        row = {
            "model": model,
            "status": result.get("status"),
            "grounded": result.get("grounded"),
            "elapsed_seconds": round(elapsed, 2),
            "attempts": len(result.get("attempts", [])),
            "unsupported_numbers": len(
                validation.get("unsupported_numbers", [])
            ),
            "speculative_claims": len(
                validation.get("speculative_claims", [])
            ),
            "unsupported_trends": len(
                validation.get("trend_claims_without_evidence", [])
            ),
            "response": result.get("response", ""),
        }

        results.append(row)

        print(f"Status:              {row['status']}")
        print(f"Grounded:            {row['grounded']}")
        print(f"Time (seconds):      {row['elapsed_seconds']}")
        print(f"Attempts:            {row['attempts']}")
        print(f"Unsupported numbers: {row['unsupported_numbers']}")
        print(f"Speculative claims:  {row['speculative_claims']}")
        print(f"Unsupported trends:  {row['unsupported_trends']}")

        print("\nMODEL RESPONSE")
        print(row["response"])

    print("\n" + "=" * 80)
    print("A/B COMPARISON")
    print("=" * 80)

    if not results:
        raise SystemExit("No models were available for testing.")

    for result in results:
        print(
            f"\n{result['model']}: "
            f"status={result['status']}, "
            f"grounded={result['grounded']}, "
            f"time={result['elapsed_seconds']}s, "
            f"attempts={result['attempts']}, "
            f"speculation={result['speculative_claims']}, "
            f"unsupported_numbers={result['unsupported_numbers']}"
        )

    print("\nRAW COMPARISON JSON")
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in result.items()
                    if k != "response"
                }
                for result in results
            ],
            indent=2,
        )
    )

    print("\n" + "=" * 80)
    print("VARIA QWEN VS GRANITE A/B TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
