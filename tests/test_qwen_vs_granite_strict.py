from __future__ import annotations

import json
import time

from src.data_loader import load_workbook
from src.decision_intelligence_pipeline import DecisionIntelligencePipeline
from src.ollama_llm_engine import OllamaLLMEngine
from src.grounded_ai_engine import VARIAGroundedAIEngine

MODELS = ["qwen2.5:7b", "granite4:tiny-h"]


def run_model(model: str, context: dict) -> dict:
    client = OllamaLLMEngine(model=model, timeout=180)
    health = client.health_check()
    if not health.get("ok"):
        return {
            "model": model,
            "status": "SKIPPED",
            "reason": health.get("error") or "Model unavailable in Ollama.",
        }

    started = time.perf_counter()
    result = VARIAGroundedAIEngine(
        llm_client=client,
        max_retries=2,
    ).generate(context_payload=context, temperature=0.0)
    elapsed = time.perf_counter() - started

    validation = result.get("validation", {})
    checks = validation.get("checks", {})

    return {
        "model": model,
        "status": result.get("status"),
        "grounded": result.get("grounded"),
        "elapsed_seconds": round(elapsed, 2),
        "attempts": len(result.get("attempts", [])),
        "schema_ok": checks.get("schema", False),
        "numeric_grounding": checks.get("numeric_grounding", False),
        "financial_semantics": checks.get("financial_semantics", False),
        "evidence_coverage": checks.get("evidence_coverage", False),
        "causal_control": checks.get("no_unsupported_causal_speculation", False),
        "trend_control": checks.get("no_unsupported_trend_claims", False),
        "schema_errors": validation.get("schema_errors", []),
        "semantic_errors": validation.get("financial_semantic_errors", []),
        "unsupported_numbers": validation.get("unsupported_numbers", []),
        "speculative_claims": validation.get("speculative_claims", []),
        "response": result.get("response", ""),
    }


def main() -> None:
    print("\n" + "=" * 90)
    print("VARIA STRICT QWEN VS GRANITE — STRUCTURED OUTPUT + FINANCIAL SEMANTICS")
    print("=" * 90)

    data = load_workbook()
    pipeline = DecisionIntelligencePipeline(
        budget=data["Budget"],
        actuals=data["Actuals_Raw"],
    )
    outputs = pipeline.run(enable_local_llm=False)
    evidence_pack = outputs["cfo_evidence_pack"]

    print("\nVERIFIED VARIA CONTEXT")
    print(f"Evidence records: {len(evidence_pack.get('evidence_ledger', []))}")
    print(
        "Current variance: "
        f"{evidence_pack['executive_position']['current_variance']:,.2f}"
    )
    print(
        "Planning exceptions: "
        f"{evidence_pack['planning_exceptions'].get('Exception Transactions')} "
        f"/ ₹{evidence_pack['planning_exceptions'].get('Exception Amount', 0):,.2f}"
    )

    results = [run_model(model, evidence_pack) for model in MODELS]

    for result in results:
        print("\n" + "-" * 90)
        print(f"MODEL: {result['model']}")
        print("-" * 90)

        if result["status"] == "SKIPPED":
            print("SKIPPED:", result["reason"])
            continue

        for label in (
            "status",
            "grounded",
            "elapsed_seconds",
            "attempts",
            "schema_ok",
            "numeric_grounding",
            "financial_semantics",
            "evidence_coverage",
            "causal_control",
            "trend_control",
        ):
            print(f"{label:24}: {result[label]}")

        print(f"unsupported_numbers      : {len(result['unsupported_numbers'])}")
        print(f"speculative_claims       : {len(result['speculative_claims'])}")
        print(f"schema_errors            : {len(result['schema_errors'])}")
        print(f"semantic_errors          : {len(result['semantic_errors'])}")

        if result["schema_errors"]:
            print("\nSCHEMA ERRORS")
            print("\n".join(f"- {x}" for x in result["schema_errors"][:10]))

        if result["semantic_errors"]:
            print("\nFINANCIAL SEMANTIC ERRORS")
            print("\n".join(f"- {x}" for x in result["semantic_errors"][:10]))

        print("\nRAW RESPONSE")
        print(result["response"])

    print("\n" + "=" * 90)
    print("GROUNDING WINNER")
    print("=" * 90)

    passing = [
        r
        for r in results
        if r.get("status") == "PASS" and r.get("grounded") is True
    ]

    if not passing:
        print("No local model passed every strict VARIA grounding control.")
        print("Deterministic VARIA analytics remain authoritative.")
    elif len(passing) == 1:
        print(f"{passing[0]['model']} passed all strict controls.")
    else:
        winner = min(passing, key=lambda row: row["elapsed_seconds"])
        print(
            f"Both models passed. Faster grounded model: {winner['model']}"
        )

    print("\n" + "=" * 90)
    print("STRICT QWEN VS GRANITE TEST COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
