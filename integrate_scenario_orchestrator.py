from pathlib import Path
import shutil

path = Path("src/decision_intelligence_pipeline.py")
if not path.exists():
    raise SystemExit("src/decision_intelligence_pipeline.py was not found.")

text = path.read_text()

if "from src.scenario_orchestrator import ScenarioOrchestrator" in text:
    raise SystemExit("Scenario Orchestrator is already integrated. No changes made.")

# Always create a local rollback copy before modifying the pipeline.
backup = path.with_suffix(path.suffix + ".pre_orchestrator.bak")
shutil.copy2(path, backup)

import_anchor = "from src.scenario_engine import ScenarioEngine\n"
if import_anchor not in text:
    raise SystemExit(
        "Could not find the ScenarioEngine import anchor. "
        "Backup was created, but no changes were made."
    )

text = text.replace(
    import_anchor,
    import_anchor + "from src.scenario_orchestrator import ScenarioOrchestrator\n",
    1,
)

scenario_anchor = '''        scenario_summary = (\n            scenario_engine.summary(\n                base_scenario\n            )\n        )\n'''

scenario_block = scenario_anchor + '''\n        # ----------------------------------------------------\n        # Scenario Orchestration layer\n        #\n        # Coordinates the existing scenario, sensitivity, and action-sizing\n        # engines into one decision pack for the UI/reporting layer.\n        # Historical actuals are never modified by these scenarios.\n        # ----------------------------------------------------\n\n        scenario_orchestrator = ScenarioOrchestrator(\n            forecast\n        )\n\n        scenario_library = [\n            {\n                "name": "Base Scenario",\n                "future_spend_change_pct": 0.0,\n                "budget_change_pct": 0.0,\n                "scope": None,\n            },\n            {\n                "name": "5% Future Spend Control",\n                "future_spend_change_pct": -5.0,\n                "budget_change_pct": 0.0,\n                "scope": None,\n            },\n            {\n                "name": "10% Future Spend Control",\n                "future_spend_change_pct": -10.0,\n                "budget_change_pct": 0.0,\n                "scope": None,\n            },\n            {\n                "name": "15% Future Spend Control",\n                "future_spend_change_pct": -15.0,\n                "budget_change_pct": 0.0,\n                "scope": None,\n            },\n        ]\n\n        scenario_decision_pack = (\n            scenario_orchestrator.decision_pack(\n                scenarios=scenario_library,\n                sensitivity_dimension="Category",\n                sensitivity_reduction_pct=10.0,\n                sensitivity_top_n=10,\n            )\n        )\n'''

if scenario_anchor not in text:
    raise SystemExit(
        "Could not find the scenario summary anchor. "
        "Rollback backup was created, but no changes were made."
    )

text = text.replace(scenario_anchor, scenario_block, 1)

return_anchor = '            "scenario_summary": scenario_summary,\n'
return_block = return_anchor + '''            "scenario_orchestrator": scenario_orchestrator,\n            "scenario_decision_pack": scenario_decision_pack,\n            "scenario_comparison": scenario_decision_pack["scenario_comparison"],\n            "scenario_ranking": scenario_decision_pack["scenario_ranking"],\n            "recommended_scenario": scenario_decision_pack["recommended_scenario"],\n            "sensitivity": scenario_decision_pack["sensitivity"],\n'''

if return_anchor not in text:
    raise SystemExit(
        "Could not find the scenario_summary return anchor. "
        "Rollback backup was created, but no changes were made."
    )

text = text.replace(return_anchor, return_block, 1)

path.write_text(text)
print("SCENARIO ORCHESTRATOR INTEGRATED")
print(f"Pipeline backup: {backup}")
print("Added outputs: scenario_decision_pack, scenario_comparison, scenario_ranking, recommended_scenario, sensitivity")
