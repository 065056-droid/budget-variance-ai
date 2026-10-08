from src.ollama_llm_engine import OllamaLLMEngine


def main():
    print("\n" + "=" * 80)
    print("VARIA OLLAMA / QWEN LOCAL AI TEST")
    print("=" * 80)

    engine = OllamaLLMEngine(
        model="qwen2.5:7b",
        timeout=120,
    )

    health = engine.health_check()

    print("\nOLLAMA HEALTH")
    print(f"Reachable: {health['ok']}")
    print(f"Model: {health['model']}")
    print(f"Available models: {health['available_models']}")

    if not health["ok"]:
        raise AssertionError(
            "Ollama is reachable but qwen2.5:7b is not available."
        )

    context = {
        "system_instruction": (
            "Use only supplied VARIA evidence. "
            "Do not invent figures or causes."
        ),
        "model_role": (
            "You are a grounded FP&A management commentary assistant."
        ),
        "issues": [
            {
                "scope": {
                    "business_unit": "Test BU",
                    "department": "Marketing",
                    "category": "Advertising",
                },
                "financial_position": {
                    "current_variance": 100000.0,
                    "projected_variance": 120000.0,
                    "projected_variance_pct": 12.0,
                },
                "early_warning": "RED",
                "decision_priority": "HIGH PRIORITY",
            }
        ],
    }

    response = engine.generate(
        context_payload=context,
        temperature=0.0,
    )

    assert isinstance(response, str)
    assert response.strip()

    print("\nLOCAL INFERENCE: PASS")
    print("\nQWEN RESPONSE")
    print(response)

    print("\n" + "=" * 80)
    print("VARIA OLLAMA / QWEN LOCAL AI TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
