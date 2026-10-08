import pandas as pd

from src.data_loader import load_workbook
from src.decision_intelligence_pipeline import DecisionIntelligencePipeline


def main():
    print("\n" + "=" * 80)
    print("VARIA CFO EVIDENCE PACK PIPELINE INTEGRATION TEST")
    print("=" * 80)

    data = load_workbook()

    pipeline = DecisionIntelligencePipeline(
        budget=data["Budget"],
        actuals=data["Actuals_Raw"],
    )

    outputs = pipeline.run(
        enable_local_llm=False,
    )

    assert "cfo_evidence_pack" in outputs

    pack = outputs["cfo_evidence_pack"]

    required_sections = {
        "executive_position",
        "top_unfavorable_drivers",
        "top_cfo_issues",
        "scenario_results",
        "planning_exceptions",
        "evidence_ledger",
        "llm_rules",
    }

    missing = required_sections.difference(pack.keys())
    assert not missing, f"Missing sections: {sorted(missing)}"

    ledger = pack["evidence_ledger"]
    assert ledger
    ids = [item["id"] for item in ledger]
    assert len(ids) == len(set(ids))

    # Confirm the pack uses actual real-data planning exceptions.
    planning = pack["planning_exceptions"]
    assert planning["Exception Transactions"] == 2
    assert abs(float(planning["Exception Amount"]) - 60569.56) < 0.01

    # Confirm the executive totals remain tied to the real variance output.
    position = pack["executive_position"]
    assert abs(
        position["budget_total"] - 761323386.20
    ) < 0.01
    assert abs(
        position["actual_total"] - 787365053.22
    ) < 0.01
    assert abs(
        position["current_variance"] - 26041667.02
    ) < 0.01

    print("\nREAL-DATA EVIDENCE PACK: PASS")
    print("Executive totals preserved.")
    print("Planning exceptions preserved.")
    print(f"Evidence records: {len(ledger)}")

    print("\n" + "=" * 80)
    print("VARIA CFO EVIDENCE PACK PIPELINE INTEGRATION TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
