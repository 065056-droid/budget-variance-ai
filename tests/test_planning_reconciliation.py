import pandas as pd

from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.planning_reconciliation import PlanningReconciliation


def main() -> None:

    print("\n" + "=" * 80)
    print("VARIA PLANNING RECONCILIATION TEST")
    print("=" * 80)

    data = load_workbook()

    budget = data["Budget"]
    actuals_raw = data["Actuals_Raw"]

    cleaner = DataCleaner(
        actuals=actuals_raw,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        },
    )

    actuals = cleaner.cleaned

    engine = PlanningReconciliation(
        budget=budget,
        actuals=actuals,
    )

    reconciled = engine.run()
    matched = engine.matched_actuals()
    exceptions = engine.exception_queue()
    summary = engine.summary()

    assert len(reconciled) == 6685
    assert len(matched) == 6683
    assert len(exceptions) == 2

    assert summary[
        "Total Transactions"
    ] == 6685
    assert summary[
        "Matched Transactions"
    ] == 6683
    assert summary[
        "Exception Transactions"
    ] == 2

    assert abs(
        summary["Total Actual Amount"]
        - 787425622.78
    ) < 0.01

    assert abs(
        summary["Matched Amount"]
        - 787365053.22
    ) < 0.01

    assert abs(
        summary["Exception Amount"]
        - 60569.56
    ) < 0.01

    assert abs(
        summary["Gross Exception Exposure"]
        - 60569.56
    ) < 0.01

    exceptions_by_id = exceptions.set_index(
        "Transaction_ID"
    )

    assert (
        exceptions_by_id.loc[
            "TXN-0000051",
            "Planning_Exception_Type",
        ]
        == "UNMAPPED_BUDGET_LINE"
    )

    assert (
        exceptions_by_id.loc[
            "TXN-0000011",
            "Planning_Exception_Type",
        ]
        == "MISSING_DIMENSION"
    )

    assert abs(
        float(
            exceptions_by_id.loc[
                "TXN-0000051",
                "Actual_Amount",
            ]
        )
        - 21912.16
    ) < 0.01

    assert abs(
        float(
            exceptions_by_id.loc[
                "TXN-0000011",
                "Actual_Amount",
            ]
        )
        - 38657.40
    ) < 0.01

    assert not bool(
        matched["Planning_Exception_Type"].notna().any()
    )

    assert set(
        exceptions["Planning_Exception_Type"]
    ) == {
        "UNMAPPED_BUDGET_LINE",
        "MISSING_DIMENSION",
    }

    print("\nRECONCILIATION SUMMARY")
    for key, value in summary.items():
        print(f"{key:28s} {value}")

    print("\nEXCEPTION QUEUE")
    print(
        exceptions[
            [
                "Transaction_ID",
                "Business_Unit",
                "Department",
                "Category",
                "Actual_Amount",
                "Planning_Exception_Type",
                "Planning_Exception_Detail",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("PLANNING RECONCILIATION TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
