
import pandas as pd
import numpy as np

from src.column_mapping_engine import ColumnMappingEngine


class DataProfileEngine:
    """
    Profile a new financial dataset before VARIA runs analytics.

    Returns transparent information about:
      - row/column counts
      - date coverage
      - numeric fields
      - candidate budget/actual fields
      - missingness
      - duplicates
      - cardinality of business dimensions
      - likely dataset type
      - schema readiness

    It never alters the source data.
    """

    DIMENSIONS = [
        "Business_Unit",
        "Department",
        "Category",
        "Region",
        "Cost_Centre",
        "Vendor",
    ]

    def __init__(self, dataframe: pd.DataFrame):

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        self.data = dataframe.copy()

    def column_profile(self) -> pd.DataFrame:

        rows = []

        for col in self.data.columns:

            series = self.data[col]

            missing = int(
                series.isna().sum()
            )

            unique = int(
                series.nunique(
                    dropna=True
                )
            )

            dtype = str(
                series.dtype
            )

            rows.append(
                {
                    "Column": col,
                    "Data_Type": dtype,
                    "Rows": len(series),
                    "Missing": missing,
                    "Missing_Pct": (
                        missing
                        / len(series)
                        * 100
                        if len(series)
                        else 0.0
                    ),
                    "Unique_Values": unique,
                }
            )

        return pd.DataFrame(rows)

    def suggested_mapping(self) -> pd.DataFrame:

        mapper = ColumnMappingEngine(
            self.data
        )

        return mapper.suggestions()

    def summary(self) -> dict:

        rows = len(self.data)
        cols = len(self.data.columns)

        profile = self.column_profile()

        numeric_columns = (
            self.data.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        mapping = self.suggested_mapping()

        budget_found = bool(
            (
                mapping[
                    "VARIA_Field"
                ]
                == "Budget_Amount"
            )
            &
            (
                mapping[
                    "Suggested_Source_Column"
                ].notna()
            )
        ).any()
        actual_found = bool(
            (
                mapping[
                    "VARIA_Field"
                ]
                == "Actual_Amount"
            )
            &
            (
                mapping[
                    "Suggested_Source_Column"
                ].notna()
            )
        ).any()

        date_found = bool(
            (
                mapping[
                    "VARIA_Field"
                ]
                == "Transaction_Date"
            )
            &
            (
                mapping[
                    "Suggested_Source_Column"
                ].notna()
            )
        ).any()

        if budget_found and actual_found:
            dataset_type = "Combined Budget + Actual"
        elif budget_found:
            dataset_type = "Budget / Plan"
        elif actual_found:
            dataset_type = "Actuals / Transactions"
        else:
            dataset_type = "Unclassified"

        readiness = "READY"

        if rows == 0:
            readiness = "EMPTY"
        elif not date_found:
            readiness = "REVIEW — DATE REQUIRED"
        elif (
            not budget_found
            and not actual_found
        ):
            readiness = (
                "REVIEW — BUDGET/ACTUAL AMOUNT REQUIRED"
            )

        return {
            "Rows": rows,
            "Columns": cols,
            "Numeric_Columns": len(
                numeric_columns
            ),
            "Date_Detected": date_found,
            "Budget_Amount_Detected": budget_found,
            "Actual_Amount_Detected": actual_found,
            "Likely_Dataset_Type": dataset_type,
            "Schema_Readiness": readiness,
            "Total_Missing_Cells": int(
                profile["Missing"].sum()
            ),
            "Columns_With_Missing": int(
                (
                    profile["Missing"] > 0
                ).sum()
            ),
            "Duplicate_Rows": int(
                self.data.duplicated().sum()
            ),
        }

    def dimension_profile(self) -> pd.DataFrame:

        rows = []

        normalized = (
            self.data.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        rename = dict(
            zip(
                normalized,
                self.data.columns,
            )
        )

        aliases = {
            "Business_Unit": [
                "business_unit",
                "businessunit",
                "bu",
                "business",
            ],
            "Department": [
                "department",
                "dept",
                "function",
            ],
            "Category": [
                "category",
                "expense_category",
                "expense_type",
                "expense_head",
            ],
            "Region": [
                "region",
                "geography",
                "zone",
            ],
            "Cost_Centre": [
                "cost_centre",
                "cost_center",
                "costcentre",
                "costcenter",
            ],
            "Vendor": [
                "vendor",
                "supplier",
            ],
        }

        for target, candidates in aliases.items():

            source = None

            for candidate in candidates:

                if candidate in rename:
                    source = rename[candidate]
                    break

            if source is None:
                rows.append(
                    {
                        "Dimension": target,
                        "Detected": False,
                        "Unique_Values": None,
                    }
                )
            else:
                rows.append(
                    {
                        "Dimension": target,
                        "Detected": True,
                        "Unique_Values": int(
                            self.data[
                                source
                            ].nunique(
                                dropna=True
                            )
                        ),
                    }
                )

        return pd.DataFrame(rows)
