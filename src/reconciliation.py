from __future__ import annotations

import pandas as pd


class ReconciliationEngine:
    """
    Compares raw and cleaned financial data to ensure
    that the cleaning process has not silently altered
    financial totals.
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        cleaned_data: pd.DataFrame,
        amount_column: str = "Actual_Amount",
    ):

        self.raw_data = raw_data
        self.cleaned_data = cleaned_data
        self.amount_column = amount_column

    def calculate_raw_total(self) -> float:

        return float(
            pd.to_numeric(
                self.raw_data[self.amount_column],
                errors="coerce",
            ).sum()
        )

    def calculate_cleaned_total(self) -> float:

        return float(
            pd.to_numeric(
                self.cleaned_data[self.amount_column],
                errors="coerce",
            ).sum()
        )

    def calculate_raw_count(self) -> int:

        return len(self.raw_data)

    def calculate_cleaned_count(self) -> int:

        return len(self.cleaned_data)

    def calculate_total_difference(self) -> float:

        return (
            self.calculate_cleaned_total()
            - self.calculate_raw_total()
        )

    def calculate_review_amount(self) -> float:

        if "Requires_Review" not in self.cleaned_data.columns:
            return 0.0

        review_mask = self.cleaned_data["Requires_Review"]

        return float(
            pd.to_numeric(
                self.cleaned_data.loc[
                    review_mask,
                    self.amount_column,
                ],
                errors="coerce",
            ).sum()
        )

    def reconciliation_status(
        self,
        tolerance: float = 0.01,
    ) -> str:

        difference = abs(
            self.calculate_total_difference()
        )

        if difference <= tolerance:
            return "PASS"

        return "FAIL"

    def generate_report(self) -> dict:

        raw_total = self.calculate_raw_total()
        cleaned_total = self.calculate_cleaned_total()
        difference = self.calculate_total_difference()

        return {
            "Raw_Transaction_Count":
                self.calculate_raw_count(),

            "Cleaned_Transaction_Count":
                self.calculate_cleaned_count(),

            "Raw_Actual_Total":
                raw_total,

            "Cleaned_Actual_Total":
                cleaned_total,

            "Total_Difference":
                difference,

            "Amount_Requiring_Review":
                self.calculate_review_amount(),

            "Reconciliation_Status":
                self.reconciliation_status(),
        }