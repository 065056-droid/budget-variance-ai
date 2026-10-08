
import numpy as np
import pandas as pd


class AnomalyEngine:
    """
    Detect financial anomalies at the budget-vs-actual planning grain.

    Signals:
    1. Statistical outlier:
       IQR rule on absolute variance.
    2. Z-score:
       standardized absolute variance.
    3. Unfavorable variance:
       positive current variance.
    4. Large variance percentage:
       absolute variance percentage.
    5. Persistence:
       repeated unfavorable observations for the same
       Department + Category across months.

    The engine flags anomalies; it never deletes or changes financial data.
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Month",
        "Budget_Amount",
        "Actual_Amount",
        "Variance",
        "Variance_Pct",
    ]

    def __init__(self, variance_data: pd.DataFrame):
        if not isinstance(variance_data, pd.DataFrame):
            raise TypeError("variance_data must be a pandas DataFrame.")

        self.variance_data = variance_data.copy()
        self._validate_columns()

    def _validate_columns(self):
        missing = [
            c for c in self.REQUIRED_COLUMNS
            if c not in self.variance_data.columns
        ]
        if missing:
            raise ValueError(
                "Variance data is missing required columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _safe_zscore(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)

        if pd.isna(std) or std == 0:
            return pd.Series(
                np.zeros(len(series)),
                index=series.index,
            )

        return (series - mean) / std

    def run(
        self,
        z_threshold: float = 3.0,
        variance_pct_threshold: float = 20.0,
        persistence_threshold: int = 3,
    ) -> pd.DataFrame:

        data = self.variance_data.copy()

        data["Month"] = pd.to_datetime(
            data["Month"],
            errors="coerce",
        )

        data["Variance"] = pd.to_numeric(
            data["Variance"],
            errors="coerce",
        ).fillna(0.0)

        data["Variance_Pct"] = pd.to_numeric(
            data["Variance_Pct"],
            errors="coerce",
        ).fillna(0.0)

        data["Absolute_Variance"] = data["Variance"].abs()
        data["Absolute_Variance_Pct"] = data["Variance_Pct"].abs()

        q1 = data["Absolute_Variance"].quantile(0.25)
        q3 = data["Absolute_Variance"].quantile(0.75)
        iqr = q3 - q1

        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr

        data["IQR_Outlier"] = (
            data["Absolute_Variance"] < lower
        ) | (
            data["Absolute_Variance"] > upper
        )

        data["Variance_Z_Score"] = self._safe_zscore(
            data["Absolute_Variance"]
        )

        data["Z_Score_Outlier"] = (
            data["Variance_Z_Score"].abs()
            >= z_threshold
        )

        data["Large_Variance_Pct"] = (
            data["Absolute_Variance_Pct"]
            >= variance_pct_threshold
        )

        data["Unfavorable"] = (
            data["Variance"] > 0
        )

        # Persistence is measured at Department + Category level.
        # This is intentionally independent of Business Unit so that
        # a recurring cost driver across BUs is visible.
        persistence = (
            data.groupby(
                ["Department", "Category"],
                dropna=False,
            )["Unfavorable"]
            .sum()
            .rename("Unfavorable_Month_Count")
            .reset_index()
        )

        data = data.merge(
            persistence,
            on=["Department", "Category"],
            how="left",
        )

        data["Recurring_Driver"] = (
            data["Unfavorable_Month_Count"]
            >= persistence_threshold
        )

        data["Anomaly_Signal_Count"] = (
            data["IQR_Outlier"].astype(int)
            + data["Z_Score_Outlier"].astype(int)
            + data["Large_Variance_Pct"].astype(int)
            + data["Recurring_Driver"].astype(int)
        )

        def severity(row):
            score = int(row["Anomaly_Signal_Count"])
            if score >= 3:
                return "CRITICAL"
            if score == 2:
                return "HIGH"
            if score == 1:
                return "REVIEW"
            return "NORMAL"

        data["Anomaly_Level"] = data.apply(
            severity,
            axis=1,
        )

        data["Requires_Investigation"] = (
            data["Anomaly_Level"].isin(
                ["CRITICAL", "HIGH", "REVIEW"]
            )
        )

        return data

    def anomaly_queue(
        self,
        anomaly_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        data = (
            self.run()
            if anomaly_data is None
            else anomaly_data.copy()
        )

        return (
            data[
                data["Requires_Investigation"]
            ]
            .sort_values(
                [
                    "Anomaly_Signal_Count",
                    "Absolute_Variance",
                ],
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def summary(
        self,
        anomaly_data: pd.DataFrame | None = None,
    ) -> dict:

        data = (
            self.run()
            if anomaly_data is None
            else anomaly_data.copy()
        )

        return {
            "Total_Records": len(data),
            "Investigation_Records": int(
                data["Requires_Investigation"].sum()
            ),
            "Critical_Records": int(
                (data["Anomaly_Level"] == "CRITICAL").sum()
            ),
            "High_Records": int(
                (data["Anomaly_Level"] == "HIGH").sum()
            ),
            "Review_Records": int(
                (data["Anomaly_Level"] == "REVIEW").sum()
            ),
            "Recurring_Driver_Records": int(
                data["Recurring_Driver"].sum()
            ),
            "Outlier_Records": int(
                data["IQR_Outlier"].sum()
            ),
            "Large_Variance_Pct_Records": int(
                data["Large_Variance_Pct"].sum()
            ),
        }
