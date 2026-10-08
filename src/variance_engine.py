from __future__ import annotations

import numpy as np
import pandas as pd


class VarianceEngine:
    """
    FP&A Budget vs Actual Variance Engine.

    Responsibilities:
    1. Validate budget grain.
    2. Normalize actual transaction periods.
    3. Aggregate transaction-level actuals to budget grain.
    4. Validate actual aggregation.
    5. Match actuals to budget.
    6. Calculate absolute and percentage variance.
    7. Classify favorable/unfavorable variance.
    8. Report unmatched records.
    """

    PLANNING_KEY = [
        "Year",
        "Month_Number",
        "Fiscal_Year",
        "Business_Unit",
        "Department",
        "Category",
    ]

    def __init__(
        self,
        budget: pd.DataFrame,
        actuals: pd.DataFrame,
    ):

        self.budget = budget.copy()
        self.actuals = actuals.copy()

        self.validation = {}

    # =========================================================
    # 1. VALIDATE BUDGET GRAIN
    # =========================================================

    def validate_budget(self) -> None:

        missing_columns = [
            column
            for column in self.PLANNING_KEY + ["Budget_Amount"]
            if column not in self.budget.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Budget is missing required columns: "
                f"{missing_columns}"
            )

        duplicate_count = self.budget.duplicated(
            self.PLANNING_KEY
        ).sum()

        self.validation[
            "Budget_Duplicate_Keys"
        ] = int(duplicate_count)

        if duplicate_count > 0:
            raise ValueError(
                f"Budget grain validation failed. "
                f"{duplicate_count} duplicate planning keys found."
            )

    # =========================================================
    # 2. NORMALIZE ACTUAL PERIODS
    # =========================================================

    def normalize_actual_periods(self) -> pd.DataFrame:

        actuals = self.actuals.copy()

        # Prefer the existing Month field because it represents
        # the accounting month rather than the individual date.
        if "Month" in actuals.columns:

            actuals["Month"] = pd.to_datetime(
                actuals["Month"],
                errors="coerce",
            )

        elif "Transaction_Date" in actuals.columns:

            actuals["Month"] = pd.to_datetime(
                actuals["Transaction_Date"],
                errors="coerce",
            ).dt.to_period("M").dt.to_timestamp()

        else:

            raise ValueError(
                "Actuals must contain Month or Transaction_Date."
            )

        # Calendar year
        actuals["Year"] = (
            actuals["Month"].dt.year
        )

        # Calendar month number
        actuals["Month_Number"] = (
            actuals["Month"].dt.month
        )

        # Indian financial year:
        #
        # Apr 2023 - Mar 2024 = FY23-24
        # Apr 2024 - Mar 2025 = FY24-25

        fiscal_start_year = np.where(
            actuals["Month_Number"] >= 4,
            actuals["Year"],
            actuals["Year"] - 1,
        )

        actuals["Fiscal_Year"] = (
            "FY"
            + pd.Series(
                fiscal_start_year,
                index=actuals.index,
            )
            .astype(int)
            .mod(100)
            .astype(str)
            .str.zfill(2)
            + "-"
            + pd.Series(
                fiscal_start_year + 1,
                index=actuals.index,
            )
            .astype(int)
            .mod(100)
            .astype(str)
            .str.zfill(2)
        )

        return actuals

    # =========================================================
    # 3. AGGREGATE ACTUALS
    # =========================================================

    def aggregate_actuals(self) -> pd.DataFrame:

        actuals = self.normalize_actual_periods()

        required_columns = (
            self.PLANNING_KEY
            + ["Actual_Amount"]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in actuals.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Actuals are missing required columns: "
                f"{missing_columns}"
            )

        actuals["Actual_Amount"] = pd.to_numeric(
            actuals["Actual_Amount"],
            errors="coerce",
        )

        aggregated = (
            actuals
            .groupby(
                self.PLANNING_KEY,
                dropna=False,
                as_index=False,
            )
            ["Actual_Amount"]
            .sum()
        )

        return aggregated

    # =========================================================
    # 4. VALIDATE ACTUAL GRAIN
    # =========================================================

    def validate_actual_grain(
        self,
        aggregated_actuals: pd.DataFrame,
    ) -> None:

        duplicate_count = aggregated_actuals.duplicated(
            self.PLANNING_KEY
        ).sum()

        self.validation[
            "Actual_Duplicate_Keys_After_Aggregation"
        ] = int(duplicate_count)

        if duplicate_count > 0:

            raise ValueError(
                "Actual aggregation failed. "
                f"{duplicate_count} duplicate planning keys remain."
            )

    # =========================================================
    # 5. PREPARE BUDGET
    # =========================================================

    def prepare_budget(self) -> pd.DataFrame:

        budget = self.budget.copy()

        budget["Budget_Amount"] = pd.to_numeric(
            budget["Budget_Amount"],
            errors="coerce",
        )

        return budget

    # =========================================================
    # 6. MATCH BUDGET AND ACTUALS
    # =========================================================

    def merge_budget_actuals(
        self,
        budget: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> pd.DataFrame:

        variance = budget.merge(
            actuals,
            on=self.PLANNING_KEY,
            how="left",
            indicator=True,
        )

        # Number of budget combinations with actuals
        matched = (
            variance["_merge"] == "both"
        ).sum()

        unmatched_budget = (
            variance["_merge"] == "left_only"
        ).sum()

        self.validation[
            "Matched_Budget_Keys"
        ] = int(matched)

        self.validation[
            "Unmatched_Budget_Keys"
        ] = int(unmatched_budget)

        # Missing actual means zero spend.
        variance["Actual_Amount"] = (
            variance["Actual_Amount"]
            .fillna(0)
        )

        # Keep the match status for auditability.
        variance["Actual_Match_Status"] = np.where(
            variance["_merge"] == "both",
            "Matched",
            "No Actuals",
        )

        variance = variance.drop(
            columns=["_merge"]
        )

        return variance

    # =========================================================
    # 7. CALCULATE VARIANCE
    # =========================================================

    def calculate_variance(
        self,
        variance: pd.DataFrame,
    ) -> pd.DataFrame:

        variance = variance.copy()

        # Positive = overspend
        # Negative = underspend

        variance["Variance"] = (
            variance["Actual_Amount"]
            - variance["Budget_Amount"]
        )

        # Percentage variance
        variance["Variance_Pct"] = np.where(
            variance["Budget_Amount"] != 0,

            (
                variance["Variance"]
                / variance["Budget_Amount"]
            ) * 100,

            np.nan,
        )

        return variance

    # =========================================================
    # 8. CLASSIFY VARIANCE
    # =========================================================

    def classify_variance(
        self,
        variance: pd.DataFrame,
    ) -> pd.DataFrame:

        variance = variance.copy()

        variance["Variance_Direction"] = np.select(
            [
                variance["Variance"] > 0,
                variance["Variance"] < 0,
            ],
            [
                "Unfavorable",
                "Favorable",
            ],
            default="On Budget",
        )

        return variance

    # =========================================================
    # 9. RUN ENGINE
    # =========================================================

    def run(self) -> pd.DataFrame:

        # Validate budget first
        self.validate_budget()

        # Prepare budget
        budget = self.prepare_budget()

        # Aggregate actual transactions
        actuals = self.aggregate_actuals()

        # Validate aggregated actuals
        self.validate_actual_grain(
            actuals
        )

        # Match budget with actuals
        variance = self.merge_budget_actuals(
            budget,
            actuals,
        )

        # Calculate variance
        variance = self.calculate_variance(
            variance
        )

        # Classify
        variance = self.classify_variance(
            variance
        )

        return variance

    # =========================================================
    # 10. VALIDATION REPORT
    # =========================================================

    def validation_report(self) -> dict:

        return self.validation.copy()