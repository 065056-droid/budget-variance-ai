from pathlib import Path
import shutil

path = Path('src/decision_intelligence_pipeline.py')
if not path.exists():
    raise SystemExit(f'Not found: {path}')

text = path.read_text()

backup = path.with_name('decision_intelligence_pipeline.py.pre_grounded_ai.bak')
if not backup.exists():
    shutil.copy2(path, backup)

import_anchor = 'from src.llm_context_builder import LLMContextBuilder\n'
import_add = 'from src.grounded_ai_engine import VARIAGroundedAIEngine\n'
if import_add not in text:
    if import_anchor not in text:
        raise SystemExit('Import anchor not found. No changes made.')
    text = text.replace(import_anchor, import_anchor + import_add, 1)

run_anchor = '''    def run(\n        self,\n        category_mapping: dict | None = None,\n    ) -> dict:\n'''
run_replacement = '''    def run(\n        self,\n        category_mapping: dict | None = None,\n        enable_local_llm: bool = False,\n        llm_temperature: float = 0.0,\n    ) -> dict:\n'''
if 'enable_local_llm: bool = False' not in text:
    if run_anchor not in text:
        raise SystemExit('Run signature anchor not found. No changes made.')
    text = text.replace(run_anchor, run_replacement, 1)

context_anchor = '''        llm_context = (\n            llm_context_builder\n            .build_prompt_payload(\n                top_n=10\n            )\n        )\n\n        # Model governance / reproducibility manifest.\n'''
context_replacement = '''        llm_context = (\n            llm_context_builder\n            .build_prompt_payload(\n                top_n=10\n            )\n        )\n\n        # Optional local LLM layer. Disabled by default so the analytical\n        # regression suite remains deterministic and does not require Ollama.\n        # When enabled, Qwen receives only the verified VARIA context and its\n        # response is validated before being exposed as grounded commentary.\n        grounded_ai = {\n            "status": "DISABLED",\n            "grounded": False,\n            "response": None,\n            "validation": None,\n            "attempts": [],\n        }\n\n        if enable_local_llm:\n            grounded_ai = VARIAGroundedAIEngine().generate(\n                context_payload=llm_context,\n                temperature=llm_temperature,\n            )\n\n            self.audit.record(\n                "Grounded AI",\n                input_records=len(llm_context.get("issues", [])),\n                output_records=1,\n                status=(\n                    "PASS"\n                    if grounded_ai.get("grounded")\n                    else "REVIEW"\n                ),\n                metadata={\n                    "model": "qwen2.5:7b",\n                    "attempts": len(\n                        grounded_ai.get("attempts", [])\n                    ),\n                    "grounding_status": grounded_ai.get(\n                        "status"\n                    ),\n                },\n            )\n\n        # Model governance / reproducibility manifest.\n'''
if 'grounded_ai = {' not in text:
    if context_anchor not in text:
        raise SystemExit('LLM context anchor not found. No changes made.')
    text = text.replace(context_anchor, context_replacement, 1)

return_anchor = '''            "llm_context": llm_context,\n            "governance_manifest": governance_manifest,\n'''
return_replacement = '''            "llm_context": llm_context,\n            "grounded_ai": grounded_ai,\n            "governance_manifest": governance_manifest,\n'''
if '"grounded_ai": grounded_ai' not in text:
    if return_anchor not in text:
        raise SystemExit('Return anchor not found. No changes made.')
    text = text.replace(return_anchor, return_replacement, 1)

path.write_text(text)
print('GROUNDED AI INTEGRATED')
print(f'Pipeline backup: {backup}')
print('New run options: enable_local_llm=False, llm_temperature=0.0')
print('New output: grounded_ai')
