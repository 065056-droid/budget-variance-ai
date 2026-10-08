from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    """Stores the result of one data-quality check."""

    check_name: str
    status: str
    issue_count: int
    message: str
    affected_rows: list[int]


class DataValidator:
    """
    Validates raw financial transaction data before analysis.

    The validator does NOT modify the data.
    It only identifies problems.

    Cleaning is handled separately by data_cleaner.py.
    """

    REQUIRED_COLUMNS = [
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
    ]

    def __init__(
        self,
        actuals: pd.DataFrame,
        valid_departments: set[str] | None = None,
        valid_categories: set[str] | None = None,
        valid_business_units: set[str] | None = None,
        valid_regions: set[str] | None = None,
    ):
        self.actuals = actuals.copy()

        self.valid_departments = valid_departments
        self.valid_categories = valid_categories
        self.valid_business_units = valid_business_units
        self.valid_regions = valid_regions

        self.results: list[ValidationResult] = []

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def _add_result(
        self,
        check_name: str,
        status: str,
        issue_count: int,
        message: str,
        affected_rows: list[int] | None = None,
    ) -> None:

        self.results.append(
            ValidationResult(
                check_name=check_name,
                status=status,
                issue_count=issue_count,
                message=message,
                affected_rows=affected_rows or [],
            )
        )

    # ---------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------

    def check_required_columns(self) -> None:

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.actuals.columns
        ]

        if missing_columns:

            self._add_result(
                check_name="Required Columns",
                status="FAIL",
                issue_count=len(missing_columns),
                message=f"Missing columns: {missing_columns}",
            )

        else:

            self._add_result(
                check_name="Required Columns",
                status="PASS",
                issue_count=0,
                message="All required columns are present.",
            )

    # ---------------------------------------------------------
    # 2. Duplicate transaction IDs
    # ---------------------------------------------------------

    def check_duplicate_transactions(self) -> None:

        if "Transaction_ID" not in self.actuals.columns:
            return

        duplicate_mask = self.actuals["Transaction_ID"].duplicated(
            keep=False
        )

        duplicate_rows = self.actuals.index[
            duplicate_mask
        ].tolist()

        duplicate_ids = (
            self.actuals.loc[
                duplicate_mask, "Transaction_ID"
            ]
            .nunique()
        )

        if duplicate_ids > 0:

            self._add_result(
                check_name="Duplicate Transactions",
                status="WARNING",
                issue_count=duplicate_ids,
                message=(
                    f"{duplicate_ids} duplicate transaction ID(s) "
                    "were detected."
                ),
                affected_rows=duplicate_rows,
            )

        else:

            self._add_result(
                check_name="Duplicate Transactions",
                status="PASS",
                issue_count=0,
                message="No duplicate transaction IDs detected.",
            )

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    def check_missing_values(self) -> None:

        important_columns = [
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
        ]

        available_columns = [
            column
            for column in important_columns
            if column in self.actuals.columns
        ]

        missing_mask = self.actuals[available_columns].isna().any(axis=1)

        affected_rows = self.actuals.index[
            missing_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            missing_by_column = (
                self.actuals[available_columns]
                .isna()
                .sum()
            )

            missing_by_column = missing_by_column[
                missing_by_column > 0
            ].to_dict()

            self._add_result(
                check_name="Missing Values",
                status="WARNING",
                issue_count=issue_count,
                message=(
                    f"{issue_count} row(s) contain missing values. "
                    f"Columns affected: {missing_by_column}"
                ),
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Missing Values",
                status="PASS",
                issue_count=0,
                message="No missing values found in critical fields.",
            )

    # ---------------------------------------------------------
    # 4. Negative expenses
    # ---------------------------------------------------------

    def check_negative_amounts(self) -> None:

        if "Actual_Amount" not in self.actuals.columns:
            return

        negative_mask = self.actuals["Actual_Amount"] < 0

        affected_rows = self.actuals.index[
            negative_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            self._add_result(
                check_name="Negative Expenses",
                status="WARNING",
                issue_count=issue_count,
                message=(
                    f"{issue_count} transaction(s) contain "
                    "negative expense amounts."
                ),
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Negative Expenses",
                status="PASS",
                issue_count=0,
                message="No negative expense amounts detected.",
            )

    # ---------------------------------------------------------
    # 5. Invalid departments
    # ---------------------------------------------------------

    def check_departments(self) -> None:

        if (
            self.valid_departments is None
            or "Department" not in self.actuals.columns
        ):
            return

        invalid_mask = (
            self.actuals["Department"].notna()
            & ~self.actuals["Department"].isin(
                self.valid_departments
            )
        )

        affected_rows = self.actuals.index[
            invalid_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            invalid_values = (
                self.actuals.loc[
                    invalid_mask, "Department"
                ]
                .unique()
                .tolist()
            )

            self._add_result(
                check_name="Invalid Departments",
                status="WARNING",
                issue_count=issue_count,
                message=(
                    f"Invalid department value(s): "
                    f"{invalid_values}"
                ),
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Invalid Departments",
                status="PASS",
                issue_count=0,
                message="All departments match the master data.",
            )

    # ---------------------------------------------------------
    # 6. Invalid categories
    # ---------------------------------------------------------

    def check_categories(self) -> None:

        if (
            self.valid_categories is None
            or "Category" not in self.actuals.columns
        ):
            return

        invalid_mask = (
            self.actuals["Category"].notna()
            & ~self.actuals["Category"].isin(
                self.valid_categories
            )
        )

        affected_rows = self.actuals.index[
            invalid_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            invalid_values = (
                self.actuals.loc[
                    invalid_mask, "Category"
                ]
                .unique()
                .tolist()
            )

            self._add_result(
                check_name="Invalid Categories",
                status="WARNING",
                issue_count=issue_count,
                message=(
                    f"Invalid category value(s): "
                    f"{invalid_values}"
                ),
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Invalid Categories",
                status="PASS",
                issue_count=0,
                message="All categories match the master data.",
            )

    # ---------------------------------------------------------
    # 7. Invalid business units
    # ---------------------------------------------------------

    def check_business_units(self) -> None:

        if (
            self.valid_business_units is None
            or "Business_Unit" not in self.actuals.columns
        ):
            return

        invalid_mask = (
            self.actuals["Business_Unit"].notna()
            & ~self.actuals["Business_Unit"].isin(
                self.valid_business_units
            )
        )

        affected_rows = self.actuals.index[
            invalid_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            self._add_result(
                check_name="Invalid Business Units",
                status="WARNING",
                issue_count=issue_count,
                message="Some business units do not match the master data.",
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Invalid Business Units",
                status="PASS",
                issue_count=0,
                message="All business units match the master data.",
            )

    # ---------------------------------------------------------
    # 8. Invalid dates
    # ---------------------------------------------------------

    def check_dates(self) -> None:

        if "Transaction_Date" not in self.actuals.columns:
            return

        converted_dates = pd.to_datetime(
            self.actuals["Transaction_Date"],
            errors="coerce",
        )

        invalid_mask = converted_dates.isna()

        affected_rows = self.actuals.index[
            invalid_mask
        ].tolist()

        issue_count = len(affected_rows)

        if issue_count > 0:

            self._add_result(
                check_name="Invalid Dates",
                status="WARNING",
                issue_count=issue_count,
                message=(
                    f"{issue_count} transaction(s) contain "
                    "invalid dates."
                ),
                affected_rows=affected_rows,
            )

        else:

            self._add_result(
                check_name="Invalid Dates",
                status="PASS",
                issue_count=0,
                message="All transaction dates are valid.",
            )

    # ---------------------------------------------------------
    # Run every validation
    # ---------------------------------------------------------

    def run_all_checks(self) -> list[ValidationResult]:

        self.results = []

        self.check_required_columns()
        self.check_duplicate_transactions()
        self.check_missing_values()
        self.check_negative_amounts()
        self.check_departments()
        self.check_categories()
        self.check_business_units()
        self.check_dates()

        return self.results

    # ---------------------------------------------------------
    # Convert results to DataFrame
    # ---------------------------------------------------------

    def results_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    "Check": result.check_name,
                    "Status": result.status,
                    "Issue_Count": result.issue_count,
                    "Message": result.message,
                }
                for result in self.results
            ]
        )

    # ---------------------------------------------------------
    # Overall data quality status
    # ---------------------------------------------------------

    def overall_status(self) -> str:

        if not self.results:
            return "NOT_RUN"

        if any(
            result.status == "FAIL"
            for result in self.results
        ):
            return "FAIL"

        if any(
            result.status == "WARNING"
            for result in self.results
        ):
            return "REVIEW"

        return "PASS"