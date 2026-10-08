
import pandas as pd
import numpy as np


class BudgetReallocationEngine:
    """
    Reallocate modeled full-year budget between forecast combinations.

    This changes only the modeled budget, never historical actuals or
    forecast spend.

    Example:
        Move ₹200,000 from:
            Category = "Digital Advertising"
        to:
            Category = "Cloud Infrastructure"

    Scope can use any combination of:
        Business_Unit
        Department
        Category

    The engine prevents a source reallocation that would push the source
    modeled budget below zero.
    """

    DIMENSIONS = [
        "Business_Unit",
        "Department",
        "Category",
    ]

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Forecast_At_Completion",
        "Estimated_Full_Year_Budget",
    ]

    def __init__(self, forecast_data: pd.DataFrame):

        if not isinstance(
            forecast_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "forecast_data must be a pandas DataFrame."
            )

        self.forecast_data = forecast_data.copy()
        self._validate()

    def _validate(self):

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.forecast_data.columns
        ]

        if missing:
            raise ValueError(
                "Forecast data is missing: "
                + ", ".join(missing)
            )

    def _mask(
        self,
        data: pd.DataFrame,
        scope: dict | None,
    ) -> pd.Series:

        mask = pd.Series(
            True,
            index=data.index,
        )

        if not scope:
            return mask

        for dimension, value in scope.items():

            if dimension not in self.DIMENSIONS:
                raise ValueError(
                    f"Unsupported dimension: {dimension}"
                )

            if value is None or value == "All":
                continue

            if isinstance(
                value,
                (list, tuple, set),
            ):
                values = [
                    str(v)
                    for v in value
                ]
                mask &= (
                    data[dimension]
                    .astype(str)
                    .isin(values)
                )
            else:
                mask &= (
                    data[dimension]
                    .astype(str)
                    == str(value)
                )

        return mask

    @staticmethod
    def _classify(
        variance: float,
    ) -> str:

        if variance > 0:
            return "Projected Overspend"

        if variance < 0:
            return "Projected Underspend"

        return "On Budget"

    def run(
        self,
        source_scope: dict,
        target_scope: dict,
        amount: float,
        scenario_name: str = "Budget Reallocation",
    ) -> pd.DataFrame:

        amount = float(amount)

        if amount <= 0:
            raise ValueError(
                "Reallocation amount must be greater than zero."
            )

        if not source_scope:
            raise ValueError(
                "source_scope is required."
            )

        if not target_scope:
            raise ValueError(
                "target_scope is required."
            )

        data = self.forecast_data.copy()

        for col in [
            "Forecast_At_Completion",
            "Estimated_Full_Year_Budget",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(0.0)

        source_mask = self._mask(
            data,
            source_scope,
        )

        target_mask = self._mask(
            data,
            target_scope,
        )

        source_count = int(
            source_mask.sum()
        )

        target_count = int(
            target_mask.sum()
        )

        if source_count == 0:
            raise ValueError(
                "Source scope does not match any forecast records."
            )

        if target_count == 0:
            raise ValueError(
                "Target scope does not match any forecast records."
            )

        # A row can be both source and target only when scopes overlap.
        # That is ambiguous for a transfer and therefore rejected.
        if bool((source_mask & target_mask).any()):
            raise ValueError(
                "Source and target scopes overlap. "
                "Choose non-overlapping scopes."
            )

        source_budget = float(
            data.loc[
                source_mask,
                "Estimated_Full_Year_Budget",
            ].sum()
        )

        if amount > source_budget:
            raise ValueError(
                "Reallocation amount exceeds the source budget."
            )

        result = data[
            [
                "Business_Unit",
                "Department",
                "Category",
                "Forecast_At_Completion",
                "Estimated_Full_Year_Budget",
            ]
        ].copy()

        result["Base_Budget"] = result[
            "Estimated_Full_Year_Budget"
        ]

        result["Base_Variance"] = (
            result["Forecast_At_Completion"]
            - result["Estimated_Full_Year_Budget"]
        )

        # Allocate proportional to the budget within each scope.
        # This keeps the transfer fair when multiple records are selected.
        source_weights = (
            result.loc[
                source_mask,
                "Base_Budget",
            ].clip(lower=0)
        )

        target_weights = (
            result.loc[
                target_mask,
                "Base_Budget",
            ].clip(lower=0)
        )

        source_weight_sum = float(
            source_weights.sum()
        )
        target_weight_sum = float(
            target_weights.sum()
        )

        if source_weight_sum <= 0:
            source_share = pd.Series(
                amount / source_count,
                index=result.index[source_mask],
            )
        else:
            source_share = (
                source_weights
                / source_weight_sum
                * amount
            )

        if target_weight_sum <= 0:
            target_share = pd.Series(
                amount / target_count,
                index=result.index[target_mask],
            )
        else:
            target_share = (
                target_weights
                / target_weight_sum
                * amount
            )

        result["Reallocation_In"] = 0.0
        result["Reallocation_Out"] = 0.0

        result.loc[
            source_mask,
            "Reallocation_Out",
        ] = source_share

        result.loc[
            target_mask,
            "Reallocation_In",
        ] = target_share

        result["Reallocation_Net"] = (
            result["Reallocation_In"]
            - result["Reallocation_Out"]
        )

        result["Scenario_Budget"] = (
            result["Base_Budget"]
            + result["Reallocation_Net"]
        )

        result["Scenario_Variance"] = (
            result["Forecast_At_Completion"]
            - result["Scenario_Budget"]
        )

        result["Variance_Improvement"] = (
            result["Base_Variance"]
            - result["Scenario_Variance"]
        )

        result["Scenario_Variance_Pct"] = np.where(
            result["Scenario_Budget"] != 0,
            result["Scenario_Variance"]
            / result["Scenario_Budget"]
            * 100.0,
            0.0,
        )

        result["Scope_Role"] = np.select(
            [
                source_mask,
                target_mask,
            ],
            [
                "Source",
                "Target",
            ],
            default="Unaffected",
        )

        result["Scenario_Name"] = scenario_name

        result["Scenario_Status"] = (
            result["Scenario_Variance"]
            .apply(self._classify)
        )

        return result

    def summary(
        self,
        result: pd.DataFrame,
    ) -> dict:

        if not isinstance(
            result,
            pd.DataFrame,
        ):
            raise TypeError(
                "result must be a pandas DataFrame."
            )

        return {
            "Scenario_Name": (
                result["Scenario_Name"].iloc[0]
                if len(result)
                else ""
            ),
            "Source_Budget_Change": float(
                result.loc[
                    result["Scope_Role"] == "Source",
                    "Reallocation_Net",
                ].sum()
            ),
            "Target_Budget_Change": float(
                result.loc[
                    result["Scope_Role"] == "Target",
                    "Reallocation_Net",
                ].sum()
            ),
            "Total_Budget_Change": float(
                result["Reallocation_Net"].sum()
            ),
            "Base_Variance": float(
                result["Base_Variance"].sum()
            ),
            "Scenario_Variance": float(
                result["Scenario_Variance"].sum()
            ),
            "Variance_Improvement": float(
                result["Variance_Improvement"].sum()
            ),
            "Source_Records": int(
                (
                    result["Scope_Role"] == "Source"
                ).sum()
            ),
            "Target_Records": int(
                (
                    result["Scope_Role"] == "Target"
                ).sum()
            ),
        }
