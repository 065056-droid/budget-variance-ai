import json
import time

from src.cfo_evidence_pack_engine import CFOEvidencePackEngine
from src.data_loader import load_workbook
from src.decision_intelligence_pipeline import DecisionIntelligencePipeline
from src.grounded_ai_engine import VARIAGroundedAIEngine
from src.ollama_llm_engine import OllamaLLMEngine


MODELS = [
    "qwen2.5:7b",
    "granite4:tiny-h",
]


def run_model(model: str, context: dict) -> dict:
    client = OllamaLLMEngine(
        model=model,
        timeout=180,
    )

    health = client.health_check()
    if not health.get("ok"):
        return {
            "model": model,
            "status": "SKIPPED",
            "reason": "Model unavailable in Ollama.",
        }

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

    response = result.get("response", "")
    structured = result.get("structured")

    return {
        "model": model,
        "status": result.get("status"),
        "grounded": result.get("grounded"),
        "elapsed_seconds": round(elapsed, 2),
        "attempts": len(result.get("attempts", [])),
        "unsupported_numbers": len(
            validation.get("unsupported_numbers", [])
        ),
        "unsupported_percentages": len(
            validation.get("unsupported_percentages", [])
        ),
        "speculative_claims": len(
            validation.get("speculative_claims", [])
        ),
        "unsupported_trends": len(
            validation.get("trend_claims_without_evidence", [])
        ),
        "structured": structured,
        "response": response,
    }


def main():
    print("\n" + "=" * 80)
    print("VARIA FINAL QWEN VS GRANITE — JSON / EVIDENCE PACK TEST")
    print("=" * 80)

    data = load_workbook()

    pipeline = DecisionIntelligencePipeline(
        budget=data["Budget"],
        actuals=data["Actuals_Raw"],
    )

    # Run all deterministic VARIA calculations once. No LLM calls.
    outputs = pipeline.run(enable_local_llm=False)

    evidence_pack = outputs["cfo_evidence_pack"]

    print("\nVERIFIED VARIA CONTEXT")
    print(f"Evidence records: {len(evidence_pack['evidence_ledger'])}")
    print(
        "Current variance: "
        f"{evidence_pack['executive_position']['current_variance']:,.2f}"
    )
    print(
        "Planning exceptions: "
        f"{evidence_pack['planning_exceptions'].get('Exception Transactions')}"
    )

    results = []

    for model in MODELS:
        print("\n" + "-" * 80)
        print(f"MODEL: {model}")
        print("-" * 80)

        result = run_model(model, evidence_pack)
        results.append(result)

        if result["status"] == "SKIPPED":
            print("SKIPPED:", result["reason"])
            continue

        print(f"Status:                {result['status']}")
        print(f"Grounded:              {result['grounded']}")
        print(f"Time:                  {result['elapsed_seconds']} sec")
        print(f"Attempts:              {result['attempts']}")
        print(f"Unsupported numbers:   {result['unsupported_numbers']}")
        print(f"Unsupported percentages: {result['unsupported_percentages']}")
        print(f"Speculative claims:    {result['speculative_claims']}")
        print(f"Unsupported trends:    {result['unsupported_trends']}")

        print("\nSTRUCTURED OUTPUT")
        if result["structured"] is None:
            print("None")
        else:
            print(json.dumps(result["structured"], indent=2))

        print("\nRAW RESPONSE")
        print(result["response"])

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    for result in results:
        if result["status"] == "SKIPPED":
            print(f"{result['model']}: SKIPPED")
            continue

        print(
            f"{result['model']}: "
            f"grounded={result['grounded']}, "
            f"time={result['elapsed_seconds']}s, "
            f"speculation={result['speculative_claims']}, "
            f"unsupported_numbers={result['unsupported_numbers']}, "
            f"attempts={result['attempts']}"
        )

    passing = [
        r for r in results
        if r.get("status") == "PASS" and r.get("grounded") is True
    ]

    print("\nMODEL SELECTION")
    if len(passing) == 0:
        print("No model passed all grounding controls.")
        print("Keep deterministic VARIA output as authoritative.")
    elif len(passing) == 1:
        print(f"Grounding winner: {passing[0]['model']}")
    else:
        winner = min(
            passing,
            key=lambda r: r["elapsed_seconds"],
        )
        print(
            f"Both models passed. Faster grounded model: "
            f"{winner['model']}"
        )

    print("\n" + "=" * 80)
    print("FINAL QWEN VS GRANITE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
