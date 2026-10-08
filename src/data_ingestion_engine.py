
import numpy as np
import pandas as pd

from src.column_mapping_engine import ColumnMappingEngine
from src.data_profile_engine import DataProfileEngine


class DataIngestionEngine:
    """
    Reusable dataset ingestion and schema-standardization layer for VARIA.

    Supported input patterns:
      1. Separate Budget + Actuals dataframes
      2. One combined dataframe containing Date + Budget + Actual

    Workflow:
        raw dataframe
            -> profile
            -> suggested mapping
            -> explicit/automatic mapping
            -> schema validation
            -> VARIA-standardized dataframe

    The engine never invents financial amounts. Missing optional descriptive
    fields may receive transparent defaults; missing core financial fields
    raise a validation error.
    """

    BUDGET_REQUIRED = [
        "Month",
        "Year",
        "Month_Number",
        "Fiscal_Year",
        "Business_Unit",
        "Department",
        "Category",
        "Budget_Amount",
    ]

    ACTUAL_REQUIRED = [
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

    DIMENSIONS = [
        "Business_Unit",
        "Department",
        "Category",
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_type: str = "auto",
    ):
        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        allowed = {
            "auto",
            "budget",
            "actuals",
            "combined",
        }

        dataset_type = dataset_type.lower()

        if dataset_type not in allowed:
            raise ValueError(
                "dataset_type must be one of: "
                + ", ".join(sorted(allowed))
            )

        self.raw = dataframe.copy()
        self.dataset_type = dataset_type

        self.profile_engine = DataProfileEngine(
            self.raw
        )

        self.mapping_engine = ColumnMappingEngine(
            self.raw
        )

    def profile(self) -> dict:
        return self.profile_engine.summary()

    def mapping_suggestions(self) -> pd.DataFrame:
        return self.mapping_engine.suggestions()

    def suggested_mapping(self) -> dict:
        return self.mapping_engine.mapping_dict()

    def detect_dataset_type(self) -> str:
        normalized = (
            self.raw.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        names = set(normalized)

        has_budget = bool(
            names.intersection(
                {
                    "budget",
                    "budget_amount",
                    "plan",
                    "planned",
                    "planned_amount",
                    "budgeted_amount",
                }
            )
        )

        has_actual = bool(
            names.intersection(
                {
                    "actual",
                    "actual_amount",
                    "actuals",
                    "expense",
                    "spend",
                    "actual_spend",
                    "amount",
                }
            )
        )

        if has_budget and has_actual:
            return "combined"

        if has_budget:
            return "budget"

        if has_actual:
            return "actuals"

        return "unknown"

    def resolve_dataset_type(self) -> str:
        if self.dataset_type == "auto":
            detected = self.detect_dataset_type()
            if detected == "unknown":
                raise ValueError(
                    "VARIA could not classify the dataset. "
                    "Provide dataset_type='budget', 'actuals' or 'combined'."
                )
            return detected

        return self.dataset_type

    def apply_mapping(
        self,
        mapping: dict | None = None,
    ) -> pd.DataFrame:

        selected_mapping = (
            self.suggested_mapping()
            if mapping is None
            else mapping
        )

        # Apply the mapping against the exact original column labels.
        # ColumnMappingEngine already preserves original labels, so this
        # keeps display labels such as "Transaction Date" valid.
        mapped = self.mapping_engine.apply(
            selected_mapping
        ).copy()

        # Defensive normalization for downstream matching only.
        # The actual data values are untouched.
        canonical_lookup = {
            str(col).strip().lower()
            .replace(" ", "_")
            .replace("-", "_"): col
            for col in mapped.columns
        }

        aliases = {
            "Transaction_Date": [
                "transaction_date",
                "transactiondate",
                "date",
                "period",
                "month",
            ],
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
            "Actual_Amount": [
                "actual_amount",
                "actual",
                "actuals",
                "expense",
                "spend",
                "amount",
            ],
            "Budget_Amount": [
                "budget_amount",
                "budget",
                "plan",
                "planned",
                "planned_amount",
            ],
        }

        final_rename = {}

        for target, candidate_names in aliases.items():

            if target in mapped.columns:
                continue

            for candidate in candidate_names:

                if candidate in canonical_lookup:

                    original = canonical_lookup[candidate]
                    final_rename[original] = target
                    break

        if final_rename:
            mapped = mapped.rename(
                columns=final_rename
            )

        return mapped

    @staticmethod
    def _fiscal_year(months: pd.Series) -> pd.Series:

        months = pd.to_datetime(
            months,
            errors="coerce",
        )

        year = months.dt.year
        month = months.dt.month

        start_year = np.where(
            month >= 4,
            year,
            year - 1,
        )

        return pd.Series(
            [
                (
                    f"FY{int(y) % 100:02d}-"
                    f"{(int(y) + 1) % 100:02d}"
                    if not pd.isna(y)
                    else np.nan
                )
                for y in start_year
            ],
            index=months.index,
        )

    @staticmethod
    def _require(
        dataframe: pd.DataFrame,
        required: list[str],
        dataset_type: str,
    ) -> None:

        missing = [
            c
            for c in required
            if c not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_type.title()} dataset is missing required "
                f"VARIA fields: "
                + ", ".join(missing)
            )

    def build_budget(
        self,
        mapping: dict | None = None,
    ) -> pd.DataFrame:

        mapped = self.apply_mapping(
            mapping
        ).copy()

        # If a direct Year/Month_Number representation is provided,
        # use it to create Month when no date field exists.
        if "Month" not in mapped.columns:

            if {
                "Year",
                "Month_Number",
            }.issubset(
                mapped.columns
            ):

                mapped["Month"] = pd.to_datetime(
                    dict(
                        year=pd.to_numeric(
                            mapped["Year"],
                            errors="coerce",
                        ),
                        month=pd.to_numeric(
                            mapped["Month_Number"],
                            errors="coerce",
                        ),
                        day=1,
                    ),
                    errors="coerce",
                )

            else:

                self._require(
                    mapped,
                    ["Transaction_Date"],
                    "budget",
                )

                mapped["Month"] = (
                    pd.to_datetime(
                        mapped[
                            "Transaction_Date"
                        ],
                        errors="coerce",
                    )
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )

        mapped["Month"] = (
            pd.to_datetime(
                mapped["Month"],
                errors="coerce",
            )
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        mapped["Year"] = (
            mapped["Month"].dt.year
        )

        mapped["Month_Number"] = (
            mapped["Month"].dt.month
        )

        mapped["Fiscal_Year"] = (
            self._fiscal_year(
                mapped["Month"]
            )
        )

        # Core fields must exist. Unlike descriptive fields, BU / Dept /
        # Category are not filled silently because doing so would alter
        # planning grain semantics.
        self._require(
            mapped,
            [
                "Month",
                "Business_Unit",
                "Department",
                "Category",
                "Budget_Amount",
            ],
            "budget",
        )

        mapped["Budget_Amount"] = pd.to_numeric(
            mapped["Budget_Amount"],
            errors="coerce",
        )

        if mapped["Budget_Amount"].isna().any():
            raise ValueError(
                "Budget_Amount contains non-numeric or missing values."
            )

        planning = [
            "Month",
            "Year",
            "Month_Number",
            "Fiscal_Year",
            "Business_Unit",
            "Department",
            "Category",
        ]

        result = (
            mapped.groupby(
                planning,
                as_index=False,
            )["Budget_Amount"]
            .sum()
        )

        return result[
            [
                "Month",
                "Year",
                "Month_Number",
                "Fiscal_Year",
                "Business_Unit",
                "Department",
                "Category",
                "Budget_Amount",
            ]
        ]

    def build_actuals(
        self,
        mapping: dict | None = None,
    ) -> pd.DataFrame:

        mapped = self.apply_mapping(
            mapping
        ).copy()

        self._require(
            mapped,
            [
                "Transaction_Date",
                "Business_Unit",
                "Department",
                "Category",
                "Actual_Amount",
            ],
            "actuals",
        )

        mapped["Transaction_Date"] = pd.to_datetime(
            mapped["Transaction_Date"],
            errors="coerce",
        )

        if mapped["Transaction_Date"].isna().any():
            raise ValueError(
                "Transaction_Date contains invalid or missing dates."
            )

        mapped["Month"] = (
            mapped["Transaction_Date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        if "Transaction_ID" not in mapped.columns:
            mapped["Transaction_ID"] = [
                f"UPLOAD-{i + 1:07d}"
                for i in range(len(mapped))
            ]

        mapped["Actual_Amount"] = pd.to_numeric(
            mapped["Actual_Amount"],
            errors="coerce",
        )

        if mapped["Actual_Amount"].isna().any():
            raise ValueError(
                "Actual_Amount contains non-numeric or missing values."
            )

        # Optional descriptive fields receive explicit defaults only.
        defaults = {
            "Region": "Unknown",
            "Cost_Centre": "Unknown",
            "Vendor": "Unknown",
            "Status": "Approved",
        }

        for col, default in defaults.items():
            if col not in mapped.columns:
                mapped[col] = default
            else:
                mapped[col] = mapped[col].fillna(
                    default
                )

        return mapped[
            [
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
        ].copy()

    def build_combined(
        self,
        mapping: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        mapped = self.apply_mapping(
            mapping
        ).copy()

        self._require(
            mapped,
            [
                "Transaction_Date",
                "Business_Unit",
                "Department",
                "Category",
                "Budget_Amount",
                "Actual_Amount",
            ],
            "combined",
        )

        actuals = self.build_actuals(
            mapping=mapping
        )

        budget = self.build_budget(
            mapping=mapping
        )

        return budget, actuals

    def build(
        self,
        mapping: dict | None = None,
    ):

        resolved = self.resolve_dataset_type()

        if resolved == "budget":
            return self.build_budget(
                mapping
            )

        if resolved == "actuals":
            return self.build_actuals(
                mapping
            )

        if resolved == "combined":
            return self.build_combined(
                mapping
            )

        raise ValueError(
            f"Unsupported resolved dataset type: {resolved}"
        )

    def readiness_report(
        self,
        mapping: dict | None = None,
    ) -> dict:

        selected_mapping = (
            self.suggested_mapping()
            if mapping is None
            else mapping
        )

        mapped = self.mapping_engine.apply(
            selected_mapping
        )

        resolved = self.resolve_dataset_type()

        required = (
            self.BUDGET_REQUIRED
            if resolved == "budget"
            else (
                self.ACTUAL_REQUIRED
                if resolved == "actuals"
                else [
                    "Transaction_Date",
                    "Business_Unit",
                    "Department",
                    "Category",
                    "Budget_Amount",
                    "Actual_Amount",
                ]
            )
        )

        missing = [
            c
            for c in required
            if c not in mapped.columns
        ]

        return {
            "Dataset_Type": resolved,
            "Rows": len(mapped),
            "Columns": len(mapped.columns),
            "Required_Fields": len(required),
            "Missing_Required_Fields": missing,
            "Ready": len(missing) == 0,
        }
