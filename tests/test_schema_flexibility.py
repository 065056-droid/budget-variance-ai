import pandas as pd

from src.data_loader import load_workbook
from src.varia_input_gateway import VARIAInputGateway


def main():
    print("\n" + "=" * 80)
    print("VARIA REAL-DATA SCHEMA FLEXIBILITY TEST")
    print("=" * 80)

    data = load_workbook()
    budget = data["Budget"].copy()
    actuals = data["Actuals_Raw"].copy()

    original_budget_total = pd.to_numeric(
        budget["Budget_Amount"], errors="coerce"
    ).sum()
    original_actuals_total = pd.to_numeric(
        actuals["Actual_Amount"], errors="coerce"
    ).sum()

    # Deliberately rename important fields to realistic alternative names.
    renamed_budget = budget.rename(
        columns={
            "Business_Unit": "BU",
            "Department": "Function",
            "Category": "Expense_Type",
            "Budget_Amount": "Plan",
        }
    )

    renamed_actuals = actuals.rename(
        columns={
            "Transaction_Date": "Date",
            "Business_Unit": "BU",
            "Department": "Function",
            "Category": "Expense_Type",
            "Actual_Amount": "Spend",
        }
    )

    gateway = VARIAInputGateway(
        budget=renamed_budget,
        actuals=renamed_actuals,
    )

    budget_std, actuals_std = gateway.build()

    required_budget = {
        "Month",
        "Year",
        "Month_Number",
        "Fiscal_Year",
        "Business_Unit",
        "Department",
        "Category",
        "Budget_Amount",
    }

    required_actuals = {
        "Transaction_ID",
        "Transaction_Date",
        "Month",
        "Business_Unit",
        "Department",
        "Category",
        "Region",
        "Cost_Centre",
        "Vendor",
        "Actual_Amount",
        "Status",
    }

    assert required_budget.issubset(budget_std.columns)
    assert required_actuals.issubset(actuals_std.columns)

    standardized_budget_total = budget_std["Budget_Amount"].sum()
    standardized_actuals_total = actuals_std["Actual_Amount"].sum()

    assert abs(standardized_budget_total - original_budget_total) < 1e-6
    assert abs(standardized_actuals_total - original_actuals_total) < 1e-6

    assert len(actuals_std) == len(actuals)
    assert len(budget_std) <= len(budget)

    print("\nRENAMED COLUMNS: PASS")
    print("Budget mappings tested: BU, Function, Expense_Type, Plan")
    print("Actual mappings tested: Date, BU, Function, Expense_Type, Spend")

    print("\nVALUE PRESERVATION: PASS")
    print(f"Original budget total:       {original_budget_total:,.2f}")
    print(f"Standardized budget total:   {standardized_budget_total:,.2f}")
    print(f"Original actuals total:       {original_actuals_total:,.2f}")
    print(f"Standardized actuals total:   {standardized_actuals_total:,.2f}")

    print("\nSCHEMA CONTRACT: PASS")
    print(f"Standardized budget rows: {len(budget_std)}")
    print(f"Standardized actual rows: {len(actuals_std)}")

    print("\n" + "=" * 80)
    print("VARIA REAL-DATA SCHEMA FLEXIBILITY TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
