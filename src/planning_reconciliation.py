from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PlanningReconciliation:
    """
    Reconcile transaction-level actuals to valid budget planning lines.

    The reconciliation happens BEFORE the variance engine so that actuals
    without a valid budget line cannot silently disappear during the
    budget-to-actual merge.

    Planning grain:
        Year + Month_Number + Fiscal_Year + Business_Unit
        + Department + Category

    Exception types:
        MISSING_DIMENSION
            A required planning dimension/date is missing.

        UNMAPPED_BUDGET_LINE
            All dimensions are present, but the combination does not exist
            in the budget planning grain.

    The engine never changes financial amounts and never creates budget
    records. Exceptions remain visible in a dedicated queue.
    """

    PLANNING_KEY = [
        "Year",
        "Month_Number",
        "Fiscal_Year",
        "Business_Unit",
        "Department",
        "Category",
    ]

    REQUIRED_BUDGET_COLUMNS = PLANNING_KEY + [
        "Budget_Amount"
    ]

    REQUIRED_ACTUAL_COLUMNS = [
        "Actual_Amount",
    ]

    TEXT_DIMENSIONS = [
        "Business_Unit",
        "Department",
        "Category",
    ]

    _MISSING_SENTINEL = "__VARIA_MISSING__"

    def __init__(
        self,
        budget: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> None:

        if not isinstance(budget, pd.DataFrame):
            raise TypeError("budget must be a pandas DataFrame.")

        if not isinstance(actuals, pd.DataFrame):
            raise TypeError("actuals must be a pandas DataFrame.")

        self.budget = budget.copy()
        self.actuals = actuals.copy()

        self.validation: dict[str, Any] = {}
        self._prepared: pd.DataFrame | None = None

    # =========================================================
    # Normalization helpers
    # =========================================================

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        """Trim text and convert blank strings to missing values."""
        normalized = series.astype("string").str.strip()
        return normalized.replace({"": pd.NA})

    @staticmethod
    def _fiscal_year(month: pd.Series) -> pd.Series:
        """Return Indian April-March fiscal year labels such as FY25-26."""
        month = pd.to_datetime(month, errors="coerce")
        year = month.dt.year.astype("Int64")
        month_number = month.dt.month.astype("Int64")

        fiscal_start_year = year.where(
            month_number >= 4,
            year - 1,
        )

        start = fiscal_start_year.astype("string").str[-2:].str.zfill(2)
        end = (fiscal_start_year + 1).astype("string").str[-2:].str.zfill(2)

        result = "FY" + start + "-" + end
        return result.where(fiscal_start_year.notna(), pd.NA)

    def _normalize_budget(self) -> pd.DataFrame:
        missing = [
            column
            for column in self.REQUIRED_BUDGET_COLUMNS
            if column not in self.budget.columns
        ]

        if missing:
            raise ValueError(
                "Budget is missing required columns: "
                + ", ".join(missing)
            )

        budget = self.budget.copy()

        for column in self.TEXT_DIMENSIONS:
            budget[column] = self._normalize_text(budget[column])

        budget["Year"] = pd.to_numeric(
            budget["Year"],
            errors="coerce",
        ).astype("Int64")

        budget["Month_Number"] = pd.to_numeric(
            budget["Month_Number"],
            errors="coerce",
        ).astype("Int64")

        budget["Fiscal_Year"] = self._normalize_text(
            budget["Fiscal_Year"]
        )

        duplicate_count = int(
            budget.duplicated(
                self.PLANNING_KEY,
                keep=False,
            ).sum()
        )

        self.validation["Budget_Duplicate_Rows"] = duplicate_count

        if duplicate_count > 0:
            raise ValueError(
                "Budget planning grain validation failed. "
                f"{duplicate_count} duplicate planning-grain rows found."
            )

        missing_budget_key_rows = int(
            budget[self.PLANNING_KEY]
            .isna()
            .any(axis=1)
            .sum()
        )

        self.validation[
            "Budget_Missing_Planning_Key_Rows"
        ] = missing_budget_key_rows

        if missing_budget_key_rows > 0:
            raise ValueError(
                "Budget contains rows with missing planning-key values: "
                f"{missing_budget_key_rows} row(s)."
            )

        budget["Budget_Amount"] = pd.to_numeric(
            budget["Budget_Amount"],
            errors="coerce",
        )

        return budget

    def _budget_key_set(self, budget: pd.DataFrame) -> set[tuple[str, ...]]:
        key_frame = (
            budget[self.PLANNING_KEY]
            .astype("string")
            .fillna(self._MISSING_SENTINEL)
        )

        return set(
            map(
                tuple,
                key_frame.to_numpy(),
            )
        )

    # =========================================================
    # Prepare actuals
    # =========================================================

    def prepare_actuals(self) -> pd.DataFrame:
        """
        Create a normalized planning key and classify each transaction.

        The result keeps every source transaction. Only matched rows are
        later passed into the variance engine.
        """

        missing = [
            column
            for column in self.REQUIRED_ACTUAL_COLUMNS
            if column not in self.actuals.columns
        ]

        if missing:
            raise ValueError(
                "Actuals are missing required columns: "
                + ", ".join(missing)
            )

        budget = self._normalize_budget()
        actuals = self.actuals.copy()

        # Normalize the accounting month. Prefer Month, but backfill it
        # from Transaction_Date where Month is absent/invalid.
        month = pd.Series(
            pd.NaT,
            index=actuals.index,
            dtype="datetime64[ns]",
        )

        if "Month" in actuals.columns:
            month = pd.to_datetime(
                actuals["Month"],
                errors="coerce",
            )

        if "Transaction_Date" in actuals.columns:
            transaction_date = pd.to_datetime(
                actuals["Transaction_Date"],
                errors="coerce",
            )
            fallback_month = (
                transaction_date
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            month = month.fillna(fallback_month)

        actuals["Month"] = month.dt.to_period("M").dt.to_timestamp()
        actuals["Year"] = actuals["Month"].dt.year.astype("Int64")
        actuals["Month_Number"] = (
            actuals["Month"].dt.month.astype("Int64")
        )
        actuals["Fiscal_Year"] = self._fiscal_year(
            actuals["Month"]
        )

        for column in self.TEXT_DIMENSIONS:
            if column in actuals.columns:
                actuals[column] = self._normalize_text(
                    actuals[column]
                )
            else:
                actuals[column] = pd.Series(
                    pd.NA,
                    index=actuals.index,
                    dtype="string",
                )

        actuals["Actual_Amount"] = pd.to_numeric(
            actuals["Actual_Amount"],
            errors="coerce",
        )

        missing_dimension_mask = (
            actuals[self.PLANNING_KEY]
            .isna()
            .any(axis=1)
        )

        budget_keys = self._budget_key_set(budget)

        key_frame = (
            actuals[self.PLANNING_KEY]
            .astype("string")
            .fillna(self._MISSING_SENTINEL)
        )

        key_values = list(
            map(
                tuple,
                key_frame.to_numpy(),
            )
        )

        budget_match_mask = pd.Series(
            [key in budget_keys for key in key_values],
            index=actuals.index,
            dtype="boolean",
        )

        # Missing required dimensions always take precedence over the
        # budget-match test. A missing Department, for example, must not be
        # mislabeled as merely an unmapped budget line.
        mapped_without_missing = (
            budget_match_mask
            & ~missing_dimension_mask
        )

        exception_type = pd.Series(
            pd.NA,
            index=actuals.index,
            dtype="string",
        )

        exception_detail = pd.Series(
            pd.NA,
            index=actuals.index,
            dtype="string",
        )

        if missing_dimension_mask.any():
            missing_rows = actuals.loc[
                missing_dimension_mask,
                self.PLANNING_KEY,
            ]

            missing_labels = missing_rows.apply(
                lambda row: ", ".join(
                    column
                    for column in self.PLANNING_KEY
                    if pd.isna(row[column])
                ),
                axis=1,
            )

            exception_type.loc[missing_dimension_mask] = (
                "MISSING_DIMENSION"
            )
            exception_detail.loc[missing_dimension_mask] = (
                "Missing planning field(s): "
                + missing_labels.astype("string")
            )

        unmapped_mask = (
            ~missing_dimension_mask
            & ~budget_match_mask.astype(bool)
        )

        if unmapped_mask.any():
            exception_type.loc[unmapped_mask] = (
                "UNMAPPED_BUDGET_LINE"
            )
            exception_detail.loc[unmapped_mask] = (
                "No matching budget planning line exists for the full "
                "planning key."
            )

        actuals["Budget_Match_Found"] = mapped_without_missing.astype(bool)
        actuals["Planning_Reconciliation_Status"] = np.where(
            mapped_without_missing,
            "MATCHED",
            "EXCEPTION",
        )
        actuals["Planning_Exception_Type"] = exception_type
        actuals["Planning_Exception_Detail"] = exception_detail

        self.validation["Total_Transactions"] = int(len(actuals))
        self.validation["Matched_Transactions"] = int(
            mapped_without_missing.sum()
        )
        self.validation["Exception_Transactions"] = int(
            (~mapped_without_missing).sum()
        )

        self._prepared = actuals
        return actuals.copy()

    # =========================================================
    # Public workflow
    # =========================================================

    def run(self) -> pd.DataFrame:
        """Run the reconciliation and return every prepared transaction."""
        return self.prepare_actuals()

    def _require_run(self) -> pd.DataFrame:
        if self._prepared is None:
            return self.run()
        return self._prepared

    def matched_actuals(self) -> pd.DataFrame:
        """Return only transactions with a valid budget planning line."""
        prepared = self._require_run()
        return prepared.loc[
            prepared["Planning_Reconciliation_Status"] == "MATCHED"
        ].copy()

    def exception_queue(self) -> pd.DataFrame:
        """
        Return all transactions that cannot safely enter variance analysis.
        """
        prepared = self._require_run()
        exceptions = prepared.loc[
            prepared["Planning_Reconciliation_Status"] == "EXCEPTION"
        ].copy()

        preferred = [
            "Transaction_ID",
            "Transaction_Date",
            "Month",
            "Year",
            "Month_Number",
            "Fiscal_Year",
            "Business_Unit",
            "Department",
            "Category",
            "Actual_Amount",
            "Status",
            "Planning_Exception_Type",
            "Planning_Exception_Detail",
        ]

        ordered = [
            column
            for column in preferred
            if column in exceptions.columns
        ]
        remaining = [
            column
            for column in exceptions.columns
            if column not in ordered
        ]

        return exceptions[ordered + remaining].copy()

    def summary(self) -> dict[str, Any]:
        """Return a compact reconciliation control summary."""
        prepared = self._require_run()

        amount = pd.to_numeric(
            prepared["Actual_Amount"],
            errors="coerce",
        ).fillna(0.0)

        matched_mask = (
            prepared["Planning_Reconciliation_Status"] == "MATCHED"
        )
        exception_mask = ~matched_mask

        total_transactions = int(len(prepared))
        matched_transactions = int(matched_mask.sum())
        exception_transactions = int(exception_mask.sum())

        total_amount = float(amount.sum())
        matched_amount = float(amount.loc[matched_mask].sum())
        exception_amount = float(amount.loc[exception_mask].sum())
        gross_exception_exposure = float(
            amount.loc[exception_mask].abs().sum()
        )

        type_counts = (
            prepared.loc[exception_mask, "Planning_Exception_Type"]
            .fillna("UNKNOWN")
            .value_counts()
            .to_dict()
        )

        matched_rate = (
            matched_transactions / total_transactions * 100.0
            if total_transactions
            else 100.0
        )

        return {
            "Total Transactions": total_transactions,
            "Matched Transactions": matched_transactions,
            "Exception Transactions": exception_transactions,
            "Matched Amount": matched_amount,
            "Exception Amount": exception_amount,
            "Gross Exception Exposure": gross_exception_exposure,
            "Total Actual Amount": total_amount,
            "Matched Rate %": matched_rate,
            "Exception Types": type_counts,
        }
