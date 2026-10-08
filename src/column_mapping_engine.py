
import pandas as pd


class ColumnMappingEngine:
    """
    Reusable column-mapping layer for uploaded financial datasets.

    It proposes mappings using transparent aliases and confidence levels.
    It does not silently create financial values.

    Typical mappings:
        Dept -> Department
        BU -> Business_Unit
        Plan -> Budget_Amount
        Actual -> Actual_Amount
        Date -> Transaction_Date
        Expense Head -> Category
    """

    TARGETS = {
        "Transaction_Date": [
            "transaction_date",
            "transactiondate",
            "date",
            "period",
            "month",
            "posting_date",
        ],
        "Transaction_ID": [
            "transaction_id",
            "transactionid",
            "txn_id",
            "transaction",
            "id",
        ],
        "Business_Unit": [
            "business_unit",
            "businessunit",
            "bu",
            "business",
            "business_segment",
            "segment",
        ],
        "Department": [
            "department",
            "dept",
            "function",
            "cost_function",
        ],
        "Category": [
            "category",
            "expense_category",
            "expense_type",
            "cost_category",
            "expense_head",
            "expense_head_name",
        ],
        "Region": [
            "region",
            "geography",
            "zone",
            "territory",
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
        "Actual_Amount": [
            "actual_amount",
            "actual",
            "actuals",
            "expense",
            "spend",
            "actual_spend",
            "amount",
        ],
        "Budget_Amount": [
            "budget_amount",
            "budget",
            "plan",
            "planned",
            "planned_amount",
            "budgeted_amount",
        ],
        "Status": [
            "status",
            "transaction_status",
            "approval_status",
        ],
    }

    REQUIRED_BUDGET = [
        "Transaction_Date",
        "Business_Unit",
        "Department",
        "Category",
        "Budget_Amount",
    ]

    REQUIRED_ACTUALS = [
        "Transaction_Date",
        "Business_Unit",
        "Department",
        "Category",
        "Actual_Amount",
    ]

    def __init__(self, dataframe: pd.DataFrame):

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        self.dataframe = dataframe.copy()
        self.normalized_columns = {
            str(c).strip().lower()
            .replace(" ", "_")
            .replace("-", "_"): c
            for c in dataframe.columns
        }

    def suggestions(self) -> pd.DataFrame:

        rows = []

        # A source column should be assigned to at most one VARIA field.
        # This prevents substring-based collisions such as:
        #   "Transaction Date" -> Transaction_Date
        #   "Transaction"      -> Transaction_ID
        # where both would otherwise select the same column.
        used_sources = set()

        normalized_items = list(
            self.normalized_columns.items()
        )

        # First pass: exact normalized matches. Exact matches are stronger
        # evidence than fuzzy/contains matches.
        for target, aliases in self.TARGETS.items():

            match = None
            confidence = "NOT FOUND"

            for alias in aliases:

                if alias in self.normalized_columns:

                    candidate = self.normalized_columns[
                        alias
                    ]

                    if candidate not in used_sources:

                        match = candidate
                        confidence = "HIGH"
                        break

            if match is not None:
                used_sources.add(match)

            rows.append(
                {
                    "VARIA_Field": target,
                    "Suggested_Source_Column": match,
                    "Confidence": confidence,
                }
            )

        # Second pass: conservative fuzzy/contains fallback for fields that
        # were not matched exactly. We only consider unassigned source columns.
        for row in rows:

            if row["Suggested_Source_Column"] is not None:
                continue

            target = row["VARIA_Field"]
            aliases = self.TARGETS[target]

            best_candidate = None
            best_score = 0.0

            for normalized_name, original_name in normalized_items:

                if original_name in used_sources:
                    continue

                # Prefer complete token overlap. This avoids using a short
                # token such as "bu" as a match for unrelated fields.
                name_tokens = set(
                    normalized_name.split("_")
                )

                for alias in aliases:

                    alias_tokens = set(
                        alias.split("_")
                    )

                    if not alias_tokens:
                        continue

                    overlap = len(
                        name_tokens
                        .intersection(
                            alias_tokens
                        )
                    )

                    score = (
                        overlap
                        / max(
                            len(alias_tokens),
                            1,
                        )
                    )

                    # Exact normalized containment is acceptable only when
                    # the alias has at least 4 characters. Very short aliases
                    # (e.g. "id", "bu") are too ambiguous for fuzzy matching.
                    if (
                        len(alias) >= 4
                        and (
                            alias in normalized_name
                            or normalized_name in alias
                        )
                    ):
                        score = max(
                            score,
                            0.70,
                        )

                    if score > best_score:
                        best_score = score
                        best_candidate = original_name

            if (
                best_candidate is not None
                and best_score >= 0.70
            ):
                row["Suggested_Source_Column"] = (
                    best_candidate
                )
                row["Confidence"] = "MEDIUM"
                used_sources.add(
                    best_candidate
                )

        return pd.DataFrame(rows)

    def mapping_dict(
        self,
        include_unfound: bool = False,
    ) -> dict:

        suggestions = self.suggestions()

        mapping = {}

        for _, row in suggestions.iterrows():

            source = row[
                "Suggested_Source_Column"
            ]

            # pandas converts missing values in DataFrames to NaN, not None.
            # Never turn a missing suggestion into a literal source column named "nan".
            if pd.isna(source):
                if include_unfound:
                    # Keep the target available to the caller without creating
                    # an invalid source-column mapping.
                    continue
                continue

            mapping[str(source)] = row[
                "VARIA_Field"
            ]

        return mapping

    def apply(
        self,
        mapping: dict,
        strict: bool = False,
    ) -> pd.DataFrame:

        if not isinstance(
            mapping,
            dict,
        ):
            raise TypeError(
                "mapping must be a dictionary of source_column -> VARIA_field."
            )

        source_columns = set(
            self.dataframe.columns
        )

        invalid_sources = [
            source
            for source in mapping
            if source not in source_columns
        ]

        if invalid_sources:
            raise ValueError(
                "Mapping contains source columns not present in the file: "
                + ", ".join(
                    map(
                        str,
                        invalid_sources,
                    )
                )
            )

        result = self.dataframe.rename(
            columns=mapping
        ).copy()

        return result

    def missing_required(
        self,
        data_type: str,
        mapping: dict | None = None,
    ) -> list[str]:

        mapped = (
            self.apply(mapping)
            if mapping
            else self.dataframe.copy()
        )

        required = (
            self.REQUIRED_BUDGET
            if data_type.lower() == "budget"
            else self.REQUIRED_ACTUALS
        )

        return [
            col
            for col in required
            if col not in mapped.columns
        ]
