import json
import urllib.error
import urllib.request
from typing import Any


class OllamaLLMEngine:
    """
    Local Ollama client for grounded VARIA management commentary.

    `format_json` can be:
      - False: no structured-output request;
      - True: Ollama JSON mode;
      - dict: an Ollama JSON Schema for strict structured outputs.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = int(timeout)

    def _build_prompt(
        self,
        context_payload: dict[str, Any],
    ) -> str:
        system_instruction = context_payload.get(
            "system_instruction",
            "Use only the supplied VARIA evidence.",
        )
        model_role = context_payload.get(
            "model_role",
            "You are a grounded FP&A management commentary assistant.",
        )
        schema = context_payload.get("schema")
        evidence = context_payload.get(
            "evidence_ledger",
            context_payload,
        )
        financial_rules = context_payload.get(
            "verified_financial_rules",
            {},
        )

        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        evidence_text = json.dumps(
            evidence,
            indent=2,
            default=str,
            ensure_ascii=False,
        )
        financial_rules_text = json.dumps(
            financial_rules,
            indent=2,
            default=str,
            ensure_ascii=False,
        )

        return f"""
SYSTEM INSTRUCTION:
{system_instruction}

ROLE:
{model_role}

STRICT OUTPUT SCHEMA:
{schema_text}

VERIFIED FINANCIAL RULES:
{financial_rules_text}

VARIA EVIDENCE LEDGER:
{evidence_text}

OUTPUT CONTROL:
- Return one JSON object only.
- Use only evidence IDs appearing in the ledger.
- Do not create explanations that are not explicitly supported.
- Do not change financial direction.
- Do not treat reconciliation exceptions as operating overspend.
- When a cause is not proven by evidence, state exactly: Cause cannot be established from the available evidence.
- Recommendations must begin with "Recommendation:".
""".strip()

    def generate(
        self,
        context_payload: dict[str, Any],
        temperature: float = 0.0,
        format_json: bool | dict[str, Any] = False,
    ) -> str:
        prompt = self._build_prompt(context_payload)

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
            },
        }

        if isinstance(format_json, dict):
            body["format"] = format_json
        elif format_json:
            body["format"] = "json"

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at "
                f"{self.host}. Make sure Ollama is running."
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        response_text = payload.get("response")

        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("Ollama returned no usable response.")

        return response_text.strip()

    def health_check(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}/api/tags",
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
        except Exception as exc:
            return {
                "ok": False,
                "model": self.model,
                "error": str(exc),
                "available_models": [],
            }

        models = [
            item.get("name")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        ]

        return {
            "ok": self.model in models,
            "model": self.model,
            "available_models": models,
        }
