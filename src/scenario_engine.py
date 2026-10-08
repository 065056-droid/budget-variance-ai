
import pandas as pd
import numpy as np


class ScenarioEngine:
    """
    Scenario / What-if analysis for FP&A forecast decisions.

    The engine operates at the forecast grain:
        Business_Unit × Department × Category

    Core logic:
        Base Forecast = YTD Actual + Forecast Remaining Spend

        Scenario Remaining Spend =
            Base Remaining Spend × (1 + future_spend_change_pct / 100)

        Scenario Full-Year Budget =
            Base Full-Year Budget × (1 + budget_change_pct / 100)

        Scenario Forecast-at-Completion =
            YTD Actual + Scenario Remaining Spend

        Scenario Projected Variance =
            Scenario Forecast - Scenario Full-Year Budget

    A negative future_spend_change_pct represents a cost reduction.
    Example:
        -10 means reduce remaining spend by 10%.

    Optional scope filters allow the user to apply the scenario only to:
        Business_Unit
        Department
        Category

    Example:
        Category="Digital Advertising", future_spend_change_pct=-10
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "YTD_Actual",
        "Forecast_Remaining_Spend",
        "Forecast_At_Completion",
        "Estimated_Full_Year_Budget",
        "Projected_Variance",
        "Projected_Variance_Pct",
    ]

    DIMENSIONS = [
        "Business_Unit",
        "Department",
        "Category",
    ]

    def __init__(self, forecast_data: pd.DataFrame):
        if not isinstance(forecast_data, pd.DataFrame):
            raise TypeError("forecast_data must be a pandas DataFrame.")

        self.forecast_data = forecast_data.copy()
        self._validate_columns()

    def _validate_columns(self):
        missing = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in self.forecast_data.columns
        ]

        if missing:
            raise ValueError(
                "Forecast data is missing required columns: "
                + ", ".join(missing)
            )

    def prepare_data(self) -> pd.DataFrame:
        data = self.forecast_data.copy()

        numeric_columns = [
            "YTD_Actual",
            "Forecast_Remaining_Spend",
            "Forecast_At_Completion",
            "Estimated_Full_Year_Budget",
            "Projected_Variance",
            "Projected_Variance_Pct",
        ]

        for col in numeric_columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(0.0)

        return data

    def _scope_mask(
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
                    f"Unsupported scenario dimension: {dimension}. "
                    f"Use one of {self.DIMENSIONS}."
                )

            if value is None or value == "All":
                continue

            if isinstance(value, (list, tuple, set)):
                mask &= data[dimension].astype(str).isin(
                    [str(v) for v in value]
                )
            else:
                mask &= (
                    data[dimension].astype(str)
                    == str(value)
                )

        return mask

    @staticmethod
    def _classify(
        projected_variance: float,
    ) -> str:

        if projected_variance > 0:
            return "Projected Overspend"

        if projected_variance < 0:
            return "Projected Underspend"

        return "On Budget"

    def run(
        self,
        future_spend_change_pct: float = 0.0,
        budget_change_pct: float = 0.0,
        scope: dict | None = None,
        cap_forecast_at_budget: bool = False,
        scenario_name: str = "Base Scenario",
    ) -> pd.DataFrame:

        data = self.prepare_data()

        scope_mask = self._scope_mask(
            data,
            scope,
        )

        result = data[
            [
                "Business_Unit",
                "Department",
                "Category",
                "YTD_Actual",
                "Forecast_Remaining_Spend",
                "Forecast_At_Completion",
                "Estimated_Full_Year_Budget",
                "Projected_Variance",
                "Projected_Variance_Pct",
            ]
        ].copy()

        result["Scenario_Name"] = scenario_name

        # Preserve base values.
        result["Base_Forecast_At_Completion"] = (
            result["Forecast_At_Completion"]
        )

        result["Base_Full_Year_Budget"] = (
            result["Estimated_Full_Year_Budget"]
        )

        result["Base_Projected_Variance"] = (
            result["Projected_Variance"]
        )

        # Scenario assumptions only affect the selected scope.
        result["Scenario_Spend_Change_Pct"] = 0.0
        result["Scenario_Budget_Change_Pct"] = 0.0

        result.loc[
            scope_mask,
            "Scenario_Spend_Change_Pct",
        ] = float(future_spend_change_pct)

        result.loc[
            scope_mask,
            "Scenario_Budget_Change_Pct",
        ] = float(budget_change_pct)

        # Remaining spend changes because the scenario is applied
        # to future spend rather than rewriting historical actuals.
        result["Scenario_Remaining_Spend"] = (
            result["Forecast_Remaining_Spend"]
            * (
                1.0
                + result["Scenario_Spend_Change_Pct"]
                / 100.0
            )
        )

        # Avoid negative projected future spend.
        result["Scenario_Remaining_Spend"] = (
            result["Scenario_Remaining_Spend"]
            .clip(lower=0)
        )

        result["Scenario_Forecast_At_Completion"] = (
            result["YTD_Actual"]
            + result["Scenario_Remaining_Spend"]
        )

        result["Scenario_Full_Year_Budget"] = (
            result["Estimated_Full_Year_Budget"]
            * (
                1.0
                + result["Scenario_Budget_Change_Pct"]
                / 100.0
            )
        )

        if cap_forecast_at_budget:
            result["Scenario_Forecast_At_Completion"] = np.minimum(
                result["Scenario_Forecast_At_Completion"],
                result["Scenario_Full_Year_Budget"],
            )

        result["Scenario_Projected_Variance"] = (
            result["Scenario_Forecast_At_Completion"]
            - result["Scenario_Full_Year_Budget"]
        )

        result["Scenario_Projected_Variance_Pct"] = np.where(
            result["Scenario_Full_Year_Budget"] != 0,
            (
                result["Scenario_Projected_Variance"]
                / result["Scenario_Full_Year_Budget"]
                * 100.0
            ),
            0.0,
        )

        # Positive = improvement versus base projected variance.
        # Example: base overspend +10L, scenario overspend +6L
        # -> improvement +4L.
        result["Variance_Improvement"] = (
            result["Base_Projected_Variance"]
            - result["Scenario_Projected_Variance"]
        )

        result["Forecast_Savings"] = (
            result["Base_Forecast_At_Completion"]
            - result["Scenario_Forecast_At_Completion"]
        )

        result["Budget_Headroom_Change"] = (
            result["Scenario_Full_Year_Budget"]
            - result["Base_Full_Year_Budget"]
        )

        result["Scenario_Status"] = (
            result["Scenario_Projected_Variance"]
            .apply(self._classify)
        )

        # Make the output useful for UI, exports and audit.
        result["Scenario_Applied"] = scope_mask.values

        return result

    def summary(
        self,
        scenario_result: pd.DataFrame,
    ) -> dict:

        if not isinstance(
            scenario_result,
            pd.DataFrame,
        ):
            raise TypeError(
                "scenario_result must be a pandas DataFrame."
            )

        base_variance = (
            scenario_result[
                "Base_Projected_Variance"
            ].sum()
        )

        scenario_variance = (
            scenario_result[
                "Scenario_Projected_Variance"
            ].sum()
        )

        return {
            "Scenario_Records": len(
                scenario_result
            ),
            "Applied_Records": int(
                scenario_result[
                    "Scenario_Applied"
                ].sum()
            ),
            "Base_Projected_Variance": base_variance,
            "Scenario_Projected_Variance": scenario_variance,
            "Variance_Improvement": (
                base_variance
                - scenario_variance
            ),
            "Forecast_Savings": (
                scenario_result[
                    "Forecast_Savings"
                ].sum()
            ),
            "Budget_Headroom_Change": (
                scenario_result[
                    "Budget_Headroom_Change"
                ].sum()
            ),
            "Projected_Overspend_Records": int(
                (
                    scenario_result[
                        "Scenario_Status"
                    ]
                    == "Projected Overspend"
                ).sum()
            ),
            "Projected_Underspend_Records": int(
                (
                    scenario_result[
                        "Scenario_Status"
                    ]
                    == "Projected Underspend"
                ).sum()
            ),
            "On_Budget_Records": int(
                (
                    scenario_result[
                        "Scenario_Status"
                    ]
                    == "On Budget"
                ).sum()
            ),
        }

    def sensitivity_analysis(
        self,
        spend_change_pcts: list[float],
        scope: dict | None = None,
        budget_change_pct: float = 0.0,
    ) -> pd.DataFrame:
        """
        Run multiple spend assumptions for a sensitivity table.

        Example:
            [-20, -15, -10, -5, 0, 5]
        """

        rows = []

        for change in spend_change_pcts:

            scenario = self.run(
                future_spend_change_pct=change,
                budget_change_pct=budget_change_pct,
                scope=scope,
                scenario_name=f"Spend {change:+.1f}%",
            )

            summary = self.summary(
                scenario
            )

            rows.append(
                {
                    "Spend_Change_Pct": change,
                    **summary,
                }
            )

        return pd.DataFrame(rows)

    def multi_scenario(
        self,
        scenarios: list[dict],
    ) -> pd.DataFrame:
        """
        Run named scenarios.

        Each scenario dictionary can contain:
            name
            future_spend_change_pct
            budget_change_pct
            scope
            cap_forecast_at_budget

        Returns one summary row per scenario.
        """

        rows = []

        for scenario in scenarios:

            name = scenario.get(
                "name",
                "Unnamed Scenario",
            )

            result = self.run(
                future_spend_change_pct=scenario.get(
                    "future_spend_change_pct",
                    0.0,
                ),
                budget_change_pct=scenario.get(
                    "budget_change_pct",
                    0.0,
                ),
                scope=scenario.get(
                    "scope"
                ),
                cap_forecast_at_budget=scenario.get(
                    "cap_forecast_at_budget",
                    False,
                ),
                scenario_name=name,
            )

            rows.append(
                {
                    "Scenario_Name": name,
                    **self.summary(result),
                }
            )

        return pd.DataFrame(rows)
