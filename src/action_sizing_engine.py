
import numpy as np
import pandas as pd


class ActionSizingEngine:
    """
    Translate forecast gaps into quantified management actions.

    For projected overspend:
        Required_Future_Spend_Reduction =
            max(Projected_Variance, 0)

        Required_Reduction_Pct =
            Required_Future_Spend_Reduction
            / Forecast_Remaining_Spend * 100

    This answers a CFO question that simple variance reporting cannot:
        "How much of the remaining spend must change to get back to budget?"

    For projected underspend:
        Potential_Budget_Release =
            abs(Projected_Variance)

    The engine does not modify any source financial data.
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Forecast_At_Completion",
        "Forecast_Remaining_Spend",
        "Estimated_Full_Year_Budget",
        "Projected_Variance",
        "Projected_Variance_Pct",
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

    def run(self) -> pd.DataFrame:

        data = self.data.copy()

        for col in [
            "Forecast_At_Completion",
            "Forecast_Remaining_Spend",
            "Estimated_Full_Year_Budget",
            "Projected_Variance",
            "Projected_Variance_Pct",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(0.0)

        data["Required_Cost_Control"] = (
            data["Projected_Variance"]
            .clip(lower=0)
        )

        data["Future_Spend_Base"] = (
            data["Forecast_Remaining_Spend"]
            .clip(lower=0)
        )

        data["Required_Future_Spend_Reduction_Pct"] = np.where(
            data["Future_Spend_Base"] > 0,
            data["Required_Cost_Control"]
            / data["Future_Spend_Base"]
            * 100.0,
            np.where(
                data["Required_Cost_Control"] > 0,
                100.0,
                0.0,
            ),
        )

        data[
            "Required_Future_Spend_Reduction_Pct"
        ] = data[
            "Required_Future_Spend_Reduction_Pct"
        ].clip(
            lower=0,
            upper=100,
        )

        # An alternative is to increase the budget while maintaining the
        # projected spend trajectory.
        data["Budget_Increase_Required"] = (
            data["Projected_Variance"]
            .clip(lower=0)
        )

        # For underspend, quantify potential budget that might be released.
        data["Potential_Budget_Release"] = (
            (-data["Projected_Variance"])
            .clip(lower=0)
        )

        data["Action_Sizing"] = np.select(
            [
                data["Projected_Variance"] > 0,
                data["Projected_Variance"] < 0,
            ],
            [
                "Reduce future spend or increase budget",
                "Validate saving / timing and consider budget release",
            ],
            default="Maintain current trajectory",
        )

        data["Spend_Control_Feasibility"] = np.select(
            [
                (
                    data["Projected_Variance"] > 0
                )
                & (
                    data["Forecast_Remaining_Spend"] > 0
                )
                & (
                    data[
                        "Required_Future_Spend_Reduction_Pct"
                    ] <= 100
                ),
                (
                    data["Projected_Variance"] > 0
                )
                & (
                    data["Forecast_Remaining_Spend"] <= 0
                ),
            ],
            [
                "Quantifiable from remaining spend",
                "No remaining spend base — budget/action decision required",
            ],
            default="Not required",
        )

        return data.sort_values(
            [
                "Required_Cost_Control",
                "Required_Future_Spend_Reduction_Pct",
            ],
            ascending=False,
        ).reset_index(drop=True)

    def priority_actions(
        self,
        action_data: pd.DataFrame | None = None,
        top_n: int = 25,
    ) -> pd.DataFrame:

        data = (
            self.run()
            if action_data is None
            else action_data.copy()
        )

        return data.head(int(top_n))

    def summary(
        self,
        action_data: pd.DataFrame | None = None,
    ) -> dict:

        data = (
            self.run()
            if action_data is None
            else action_data.copy()
        )

        overspend = data[
            data["Projected_Variance"] > 0
        ]

        underspend = data[
            data["Projected_Variance"] < 0
        ]

        return {
            "Records": len(data),
            "Projected_Overspend_Records": len(
                overspend
            ),
            "Projected_Underspend_Records": len(
                underspend
            ),
            "Total_Cost_Control_Required": float(
                overspend[
                    "Required_Cost_Control"
                ].sum()
            ),
            "Total_Budget_Increase_Required": float(
                overspend[
                    "Budget_Increase_Required"
                ].sum()
            ),
            "Potential_Budget_Release": float(
                underspend[
                    "Potential_Budget_Release"
                ].sum()
            ),
            "Average_Required_Reduction_Pct": float(
                overspend[
                    "Required_Future_Spend_Reduction_Pct"
                ].mean()
            )
            if len(overspend)
            else 0.0,
        }
