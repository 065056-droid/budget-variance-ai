
import pandas as pd
import numpy as np


class SensitivityEngine:
    """
    Driver-level sensitivity analysis using a transparent spend-reduction
    assumption.

    For every selected driver (by Category, Department or Business Unit),
    apply the same percentage reduction to future remaining spend and
    calculate the resulting improvement in projected variance.

    This produces the data needed for a tornado-style ranking.
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
        "YTD_Actual",
        "Forecast_Remaining_Spend",
        "Forecast_At_Completion",
        "Estimated_Full_Year_Budget",
        "Projected_Variance",
    ]

    def __init__(self, forecast_data: pd.DataFrame):

        if not isinstance(
            forecast_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "forecast_data must be a pandas DataFrame."
            )

        self.data = forecast_data.copy()
        self._validate()

    def _validate(self):

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.data.columns
        ]

        if missing:
            raise ValueError(
                "Forecast data is missing: "
                + ", ".join(missing)
            )

    def run(
        self,
        dimension: str = "Category",
        spend_reduction_pct: float = 10.0,
    ) -> pd.DataFrame:

        if dimension not in self.DIMENSIONS:
            raise ValueError(
                f"Unsupported dimension: {dimension}. "
                f"Use one of {self.DIMENSIONS}."
            )

        reduction = float(
            spend_reduction_pct
        )

        if reduction < 0 or reduction > 100:
            raise ValueError(
                "spend_reduction_pct must be between 0 and 100."
            )

        data = self.data.copy()

        for col in [
            "YTD_Actual",
            "Forecast_Remaining_Spend",
            "Forecast_At_Completion",
            "Estimated_Full_Year_Budget",
            "Projected_Variance",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(0.0)

        rows = []

        for driver_value, group in data.groupby(
            dimension,
            dropna=False,
        ):

            base_variance = float(
                group[
                    "Projected_Variance"
                ].sum()
            )

            remaining = float(
                group[
                    "Forecast_Remaining_Spend"
                ].sum()
            )

            savings = (
                remaining
                * reduction
                / 100.0
            )

            scenario_forecast = float(
                group[
                    "Forecast_At_Completion"
                ].sum()
                - savings
            )

            budget = float(
                group[
                    "Estimated_Full_Year_Budget"
                ].sum()
            )

            scenario_variance = (
                scenario_forecast
                - budget
            )

            rows.append(
                {
                    "Dimension": dimension,
                    "Driver": driver_value,
                    "Base_Projected_Variance": base_variance,
                    "Scenario_Projected_Variance": scenario_variance,
                    "Variance_Improvement": (
                        base_variance
                        - scenario_variance
                    ),
                    "Forecast_Savings": savings,
                    "Current_Forecast": float(
                        group[
                            "Forecast_At_Completion"
                        ].sum()
                    ),
                    "Scenario_Forecast": scenario_forecast,
                    "Full_Year_Budget": budget,
                    "Spend_Reduction_Pct": reduction,
                    "Records": len(group),
                }
            )

        result = pd.DataFrame(rows)

        if result.empty:
            return result

        total_improvement = result[
            "Variance_Improvement"
        ].sum()

        result["Impact_Share_Pct"] = np.where(
            total_improvement != 0,
            result[
                "Variance_Improvement"
            ]
            / total_improvement
            * 100.0,
            0.0,
        )

        result = result.sort_values(
            "Variance_Improvement",
            ascending=False,
        ).reset_index(drop=True)

        result["Impact_Rank"] = (
            np.arange(len(result)) + 1
        )

        return result

    def tornado_data(
        self,
        dimension: str = "Category",
        spend_reduction_pct: float = 10.0,
        top_n: int = 10,
    ) -> pd.DataFrame:

        result = self.run(
            dimension=dimension,
            spend_reduction_pct=spend_reduction_pct,
        )

        return result.head(
            int(top_n)
        )
