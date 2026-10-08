from __future__ import annotations

import pandas as pd


class DataCleaner:
    """
    Cleans validated transaction data using
    deterministic business rules.

    Important:
    The cleaner does not silently change financial
    amounts or delete questionable transactions.
    """

    def __init__(
        self,
        actuals: pd.DataFrame,
        category_mapping: dict[str, str] | None = None,
    ):
        self.original = actuals.copy()

        self.cleaned = actuals.copy()

        self.category_mapping = category_mapping or {}

        self.cleaning_log: list[dict] = []

    # ---------------------------------------------------------
    # 1. Standardize column names
    # ---------------------------------------------------------

    def standardize_columns(self) -> None:

        self.cleaned.columns = (
            self.cleaned.columns
            .str.strip()
            .str.replace(" ", "_")
        )

    # ---------------------------------------------------------
    # 2. Standardize text fields
    # ---------------------------------------------------------

    def standardize_text(self) -> None:

        text_columns = [
            "Business_Unit",
            "Department",
            "Category",
            "Region",
            "Cost_Centre",
            "Vendor",
            "Status",
        ]

        for column in text_columns:

            if column not in self.cleaned.columns:
                continue

            self.cleaned[column] = (
                self.cleaned[column]
                .apply(
                    lambda x: x.strip()
                    if isinstance(x, str)
                    else x
                )
            )

    # ---------------------------------------------------------
    # 3. Standardize category names
    # ---------------------------------------------------------

    def standardize_categories(self) -> None:

        if "Category" not in self.cleaned.columns:
            return

        for old_value, new_value in self.category_mapping.items():

            mask = self.cleaned["Category"] == old_value

            affected = int(mask.sum())

            if affected > 0:

                self.cleaned.loc[
                    mask,
                    "Category"
                ] = new_value

                self.cleaning_log.append(
                    {
                        "Rule": "Category Standardization",
                        "Original_Value": old_value,
                        "New_Value": new_value,
                        "Records_Affected": affected,
                        "Action": "Automatically corrected",
                    }
                )

    # ---------------------------------------------------------
    # 4. Convert dates
    # ---------------------------------------------------------

    def standardize_dates(self) -> None:

        date_columns = [
            "Transaction_Date",
            "Month",
        ]

        for column in date_columns:

            if column not in self.cleaned.columns:
                continue

            self.cleaned[column] = pd.to_datetime(
                self.cleaned[column],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # 5. Convert amount to numeric
    # ---------------------------------------------------------

    def standardize_amounts(self) -> None:

        if "Actual_Amount" not in self.cleaned.columns:
            return

        self.cleaned["Actual_Amount"] = pd.to_numeric(
            self.cleaned["Actual_Amount"],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # 6. Flag negative expenses
    # ---------------------------------------------------------

    def flag_negative_amounts(self) -> None:

        if "Actual_Amount" not in self.cleaned.columns:
            return

        mask = self.cleaned["Actual_Amount"] < 0

        self.cleaned["Negative_Amount_Flag"] = mask

        self.cleaning_log.append(
            {
                "Rule": "Negative Amount Detection",
                "Original_Value": None,
                "New_Value": None,
                "Records_Affected": int(mask.sum()),
                "Action": "Flagged for review",
            }
        )

    # ---------------------------------------------------------
    # 7. Flag missing critical fields
    # ---------------------------------------------------------

    def flag_missing_values(self) -> None:

        critical_columns = [
            "Transaction_ID",
            "Transaction_Date",
            "Month",
            "Business_Unit",
            "Department",
            "Category",
            "Actual_Amount",
        ]

        available_columns = [
            column
            for column in critical_columns
            if column in self.cleaned.columns
        ]

        mask = self.cleaned[
            available_columns
        ].isna().any(axis=1)

        self.cleaned["Missing_Critical_Field_Flag"] = mask

        self.cleaning_log.append(
            {
                "Rule": "Missing Critical Fields",
                "Original_Value": None,
                "New_Value": None,
                "Records_Affected": int(mask.sum()),
                "Action": "Flagged for review",
            }
        )

    # ---------------------------------------------------------
    # 8. Duplicate transaction flag
    # ---------------------------------------------------------

    def flag_duplicates(self) -> None:

        if "Transaction_ID" not in self.cleaned.columns:
            return

        mask = self.cleaned[
            "Transaction_ID"
        ].duplicated(
            keep=False
        )

        self.cleaned["Duplicate_Transaction_Flag"] = mask

        self.cleaning_log.append(
            {
                "Rule": "Duplicate Transaction Detection",
                "Original_Value": None,
                "New_Value": None,
                "Records_Affected": int(mask.sum()),
                "Action": "Flagged for review",
            }
        )

    # ---------------------------------------------------------
    # 9. Create overall review flag
    # ---------------------------------------------------------

    def create_review_flag(self) -> None:

        flag_columns = [
            "Negative_Amount_Flag",
            "Missing_Critical_Field_Flag",
            "Duplicate_Transaction_Flag",
        ]

        available_flags = [
            column
            for column in flag_columns
            if column in self.cleaned.columns
        ]

        if available_flags:

            self.cleaned["Requires_Review"] = (
                self.cleaned[available_flags]
                .any(axis=1)
            )

        else:

            self.cleaned["Requires_Review"] = False

    # ---------------------------------------------------------
    # 10. Run cleaning pipeline
    # ---------------------------------------------------------

    def run(self) -> pd.DataFrame:

        self.standardize_columns()

        self.standardize_text()

        self.standardize_categories()

        self.standardize_dates()

        self.standardize_amounts()

        self.flag_negative_amounts()

        self.flag_missing_values()

        self.flag_duplicates()

        self.create_review_flag()

        return self.cleaned

    # ---------------------------------------------------------
    # Cleaning log
    # ---------------------------------------------------------

    def cleaning_log_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(self.cleaning_log)
    