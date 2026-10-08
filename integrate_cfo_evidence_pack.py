from pathlib import Path
import shutil

path = Path("src/decision_intelligence_pipeline.py")

if not path.exists():
    raise SystemExit("src/decision_intelligence_pipeline.py not found.")

text = path.read_text()

backup = path.with_name(
    "decision_intelligence_pipeline.py.pre_cfo_evidence_pack.bak"
)
if not backup.exists():
    shutil.copy2(path, backup)

# 1. Import the evidence-pack engine.
import_anchor = "from src.llm_context_builder import LLMContextBuilder\n"
import_add = "from src.cfo_evidence_pack_engine import CFOEvidencePackEngine\n"

if import_add not in text:
    if import_anchor not in text:
        raise SystemExit(
            "LLMContextBuilder import anchor not found. No changes made."
        )
    text = text.replace(import_anchor, import_anchor + import_add, 1)

# 2. Build the compact evidence pack immediately after the existing
#    deterministic llm_context is created. This does not remove llm_context.
context_anchor = """        llm_context = (
            llm_context_builder
            .build_prompt_payload(
                top_n=10
            )
        )
"""

pack_block = """        llm_context = (
            llm_context_builder
            .build_prompt_payload(
                top_n=10
            )
        )

        # Compact, deterministic evidence selected by VARIA itself.
        # The LLM receives this pack instead of the broader raw context.
        cfo_evidence_pack = CFOEvidencePackEngine(
            outputs={
                "variance": variance,
                "materiality": materiality,
                "forecast": forecast,
                "risk": risk,
                "recommendations": recommendations,
                "action_sizing": action_sizing,
                "scenario_ranking": scenario_decision_pack[
                    "scenario_ranking"
                ],
                "recommended_scenario": scenario_decision_pack[
                    "recommended_scenario"
                ],
                "planning_summary": planning_summary,
            }
        ).build()
"""

if '"cfo_evidence_pack": cfo_evidence_pack' not in text:
    if context_anchor not in text:
        raise SystemExit(
            "Expected llm_context block not found. No changes made."
        )
    text = text.replace(context_anchor, pack_block, 1)

# 3. Make the local AI consume the compact evidence pack.
old_context_call = "context_payload=llm_context,"
new_context_call = "context_payload=cfo_evidence_pack,"

if new_context_call not in text:
    occurrences = text.count(old_context_call)
    if occurrences == 0:
        raise SystemExit(
            "No grounded-AI context_payload call found. No changes made."
        )
    if occurrences > 1:
        raise SystemExit(
            f"Found {occurrences} context_payload=llm_context occurrences. "
            "Refusing ambiguous patch. No changes made."
        )
    text = text.replace(old_context_call, new_context_call, 1)

# 4. Expose the pack for Streamlit, auditing, and future debugging.
return_anchor = '            "llm_context": llm_context,\n'
return_add = (
    '            "llm_context": llm_context,\n'
    '            "cfo_evidence_pack": cfo_evidence_pack,\n'
)

if '"cfo_evidence_pack": cfo_evidence_pack' not in text.split("return {", 1)[-1]:
    if return_anchor not in text:
        raise SystemExit(
            "Return llm_context anchor not found. No changes made."
        )
    text = text.replace(return_anchor, return_add, 1)

path.write_text(text)

print("CFO EVIDENCE PACK INTEGRATED")
print(f"Pipeline backup: {backup}")
print("LLM input changed: llm_context -> cfo_evidence_pack")
print("New output: cfo_evidence_pack")
