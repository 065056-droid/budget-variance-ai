
import numpy as np
import pandas as pd


class AdvancedForecastEngine:
    """
    Ensemble forecast layer for monthly actual spend.

    Methods:
      1. Run-rate forecast
      2. Weighted recent-month forecast
      3. Linear-trend forecast

    The ensemble combines the methods and exposes the spread between them
    as a simple uncertainty indicator.

    This layer is intentionally transparent:
    management can see the component forecasts instead of receiving a
    black-box prediction.
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Month",
        "Fiscal_Year",
        "Actual_Amount",
        "Budget_Amount",
    ]

    def __init__(self, variance_data: pd.DataFrame):

        if not isinstance(
            variance_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "variance_data must be a pandas DataFrame."
            )

        self.data = variance_data.copy()
        self._validate()

    def _validate(self):

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.data.columns
        ]

        if missing:
            raise ValueError(
                "Variance data is missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _fy_month(month):

        return (
            month.month
            - 3
            if month.month >= 4
            else month.month + 9
        )

    @staticmethod
    def _forecast_linear(values, remaining):

        values = np.asarray(
            values,
            dtype=float,
        )

        if len(values) < 2:
            return float(
                values[-1]
                if len(values)
                else 0
            ) * remaining

        x = np.arange(
            1,
            len(values) + 1,
            dtype=float,
        )

        slope, intercept = np.polyfit(
            x,
            values,
            1,
        )

        future_x = np.arange(
            len(values) + 1,
            len(values) + remaining + 1,
            dtype=float,
        )

        predictions = (
            intercept
            + slope * future_x
        )

        return float(
            np.clip(
                predictions,
                0,
                None,
            ).sum()
        )

    @staticmethod
    def _forecast_weighted(values, remaining):

        values = np.asarray(
            values,
            dtype=float,
        )

        if len(values) == 0:
            return 0.0

        recent = values[-min(3, len(values)):]

        weights = np.arange(
            1,
            len(recent) + 1,
            dtype=float,
        )

        weighted_average = np.average(
            recent,
            weights=weights,
        )

        return float(
            max(
                weighted_average,
                0,
            )
            * remaining
        )

    def run(
        self,
        forecast_periods: int | None = None,
    ) -> pd.DataFrame:

        data = self.data.copy()

        data["Month"] = pd.to_datetime(
            data["Month"],
            errors="coerce",
        )

        data["Actual_Amount"] = pd.to_numeric(
            data["Actual_Amount"],
            errors="coerce",
        ).fillna(0.0)

        data["Budget_Amount"] = pd.to_numeric(
            data["Budget_Amount"],
            errors="coerce",
        ).fillna(0.0)

        valid_months = data["Month"].dropna()

        if valid_months.empty:
            raise ValueError(
                "No valid Month values available for forecasting."
            )

        # Determine latest fiscal year represented in data.
        latest_month = valid_months.max()

        latest_fy_start = (
            latest_month.year
            if latest_month.month >= 4
            else latest_month.year - 1
        )

        latest_fy = (
            f"FY{latest_fy_start % 100:02d}-"
            f"{(latest_fy_start + 1) % 100:02d}"
        )

        fy_data = data[
            data["Fiscal_Year"].astype(str)
            == latest_fy
        ].copy()

        if fy_data.empty:
            raise ValueError(
                f"No data found for latest fiscal year {latest_fy}."
            )

        latest_elapsed = (
            fy_data["Month"]
            .dt.to_period("M")
            .nunique()
        )

        if forecast_periods is None:
            remaining = max(
                12 - latest_elapsed,
                0,
            )
        else:
            remaining = max(
                int(forecast_periods),
                0,
            )

        keys = [
            "Business_Unit",
            "Department",
            "Category",
        ]

        rows = []

        for key_values, group in fy_data.groupby(
            keys,
            dropna=False,
        ):

            group = (
                group.groupby(
                    "Month",
                    as_index=False,
                )
                .agg(
                    Actual_Amount=(
                        "Actual_Amount",
                        "sum",
                    ),
                    Budget_Amount=(
                        "Budget_Amount",
                        "sum",
                    ),
                )
                .sort_values("Month")
            )

            monthly_actual = group[
                "Actual_Amount"
            ].to_numpy(
                dtype=float
            )

            ytd_actual = float(
                monthly_actual.sum()
            )

            annualized_run_rate = (
                ytd_actual
                / latest_elapsed
                * remaining
                if latest_elapsed
                else 0
            )

            weighted_forecast = (
                self._forecast_weighted(
                    monthly_actual,
                    remaining,
                )
            )

            trend_forecast = (
                self._forecast_linear(
                    monthly_actual,
                    remaining,
                )
            )

            ensemble_remaining = (
                0.40 * annualized_run_rate
                + 0.30 * weighted_forecast
                + 0.30 * trend_forecast
            )

            forecast_at_completion = (
                ytd_actual
                + ensemble_remaining
            )

            # Component spread gives a transparent uncertainty proxy.
            component_values = np.array(
                [
                    annualized_run_rate,
                    weighted_forecast,
                    trend_forecast,
                ],
                dtype=float,
            )

            low_forecast = float(
                ytd_actual
                + component_values.min()
            )

            high_forecast = float(
                ytd_actual
                + component_values.max()
            )

            component_mean = (
                component_values.mean()
            )

            component_std = (
                component_values.std()
            )

            if component_mean > 0:
                uncertainty_pct = (
                    component_std
                    / component_mean
                    * 100
                )
            else:
                uncertainty_pct = 0.0

            confidence = (
                "HIGH"
                if uncertainty_pct <= 5
                else (
                    "MEDIUM"
                    if uncertainty_pct <= 12
                    else "LOW"
                )
            )

            full_year_budget = float(
                group[
                    "Budget_Amount"
                ].sum()
            )

            projected_variance = (
                forecast_at_completion
                - full_year_budget
            )

            projected_pct = (
                projected_variance
                / full_year_budget
                * 100
                if full_year_budget
                else 0.0
            )

            rows.append(
                {
                    "Business_Unit": key_values[0],
                    "Department": key_values[1],
                    "Category": key_values[2],
                    "Fiscal_Year": latest_fy,
                    "Months_Elapsed": latest_elapsed,
                    "Months_Remaining": remaining,
                    "YTD_Actual": ytd_actual,
                    "Full_Year_Budget_YTD_Available": full_year_budget,
                    "Run_Rate_Remaining": annualized_run_rate,
                    "Weighted_Remaining": weighted_forecast,
                    "Trend_Remaining": trend_forecast,
                    "Ensemble_Remaining": ensemble_remaining,
                    "Forecast_At_Completion": forecast_at_completion,
                    "Forecast_Low": low_forecast,
                    "Forecast_High": high_forecast,
                    "Forecast_Uncertainty_Pct": uncertainty_pct,
                    "Forecast_Confidence": confidence,
                    "Projected_Variance": projected_variance,
                    "Projected_Variance_Pct": projected_pct,
                }
            )

        result = pd.DataFrame(rows)

        if result.empty:
            return result

        result["Forecast_Method"] = (
            "40% Run Rate + 30% Weighted Recent + 30% Trend"
        )

        return result.sort_values(
            "Projected_Variance",
            ascending=False,
        ).reset_index(drop=True)

    def summary(
        self,
        forecast_data: pd.DataFrame | None = None,
    ) -> dict:

        data = (
            self.run()
            if forecast_data is None
            else forecast_data.copy()
        )

        return {
            "Records": len(data),
            "Average_Confidence": (
                data["Forecast_Confidence"]
                .value_counts()
                .index[0]
                if len(data)
                else "N/A"
            ),
            "High_Confidence": int(
                (
                    data["Forecast_Confidence"]
                    == "HIGH"
                ).sum()
            ),
            "Medium_Confidence": int(
                (
                    data["Forecast_Confidence"]
                    == "MEDIUM"
                ).sum()
            ),
            "Low_Confidence": int(
                (
                    data["Forecast_Confidence"]
                    == "LOW"
                ).sum()
            ),
            "Total_Projected_Variance": float(
                data["Projected_Variance"].sum()
            ),
        }
