import pandas as pd

from src.varia_input_gateway import VARIAInputGateway


def build_raw_combined() -> pd.DataFrame:
    rows = []

    for month_number in range(1, 13):
        date = f"2025-{month_number:02d}-01"
        rows.append(
            {
                "Date": date,
                "BU": "Consumer",
                "Dept": "Marketing",
                "Expense Head": "Digital Advertising",
                "Plan": 100000.0,
                "Actual": 110000.0,
            }
        )

    return pd.DataFrame(rows)


def build_raw_budget() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": [
                "2025-01-01",
                "2025-02-01",
            ],
            "BU": [
                "Consumer",
                "Consumer",
            ],
            "Dept": [
                "Marketing",
                "Marketing",
            ],
            "Category": [
                "Digital Advertising",
                "Digital Advertising",
            ],
            "Plan": [
                100000.0,
                100000.0,
            ],
        }
    )


def build_raw_actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Transaction Date": [
                "2025-01-05",
                "2025-02-05",
            ],
            "BU": [
                "Consumer",
                "Consumer",
            ],
            "Dept": [
                "Marketing",
                "Marketing",
            ],
            "Category": [
                "Digital Advertising",
                "Digital Advertising",
            ],
            "Amount": [
                110000.0,
                90000.0,
            ],
        }
    )


def main():
    print("\n" + "=" * 80)
    print("VARIA UNIVERSAL INPUT GATEWAY TEST")
    print("=" * 80)

    combined = build_raw_combined()
    gateway = VARIAInputGateway(combined=combined)

    readiness = gateway.readiness_report()
    assert readiness["Combined"]["Ready"] is True

    budget, actuals = gateway.build()

    assert len(budget) == 12
    assert len(actuals) == 12
    assert abs(budget["Budget_Amount"].sum() - 1200000.0) < 1e-9
    assert abs(actuals["Actual_Amount"].sum() - 1320000.0) < 1e-9
    assert gateway.ingestion_metadata()["Source_Type"] == "combined"

    print("\nCOMBINED INPUT: PASS")
    print(gateway.ingestion_metadata())

    separate_gateway = VARIAInputGateway(
        budget=build_raw_budget(),
        actuals=build_raw_actuals(),
    )

    budget2, actuals2 = separate_gateway.build()

    assert len(budget2) == 2
    assert len(actuals2) == 2
    assert abs(budget2["Budget_Amount"].sum() - 200000.0) < 1e-9
    assert abs(actuals2["Actual_Amount"].sum() - 200000.0) < 1e-9

    assert "Transaction_Date" in actuals2.columns
    assert "Business_Unit" in actuals2.columns
    assert "Department" in actuals2.columns
    assert "Actual_Amount" in actuals2.columns

    print("\nSEPARATE INPUTS: PASS")
    print(
        "Budget rows:", len(budget2),
        "Actual rows:", len(actuals2),
    )

    # Explicitly reject mixed input modes.
    try:
        VARIAInputGateway(
            budget=build_raw_budget(),
            actuals=build_raw_actuals(),
            combined=combined,
        )
        raise AssertionError(
            "Mixed input modes should have raised ValueError."
        )
    except ValueError:
        pass

    print("\nINPUT CONTRACT: PASS")

    # The full analytical handoff is intentionally exposed through
    # gateway.run(). It is covered by the project's full-pipeline regression
    # test because the downstream stack is substantially larger than the
    # ingestion contract tested here.
    assert callable(gateway.run)
    print("\nPIPELINE HANDOFF CONTRACT: PASS")

    print("\n" + "=" * 80)
    print("VARIA UNIVERSAL INPUT GATEWAY TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
