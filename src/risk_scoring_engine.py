
import numpy as np
import pandas as pd


class RiskScoringEngine:
    """
    Composite FP&A financial risk score.

    Score components:
      - Projected variance magnitude: up to 35 points
      - Projected variance percentage: up to 25 points
      - Early warning: up to 20 points
      - Current materiality / severity: up to 10 points
      - Recurring driver signal: up to 10 points

    Final score is bounded to 0-100.

    The score is designed for prioritization, not for replacing
    accounting controls or management judgement.

    Important dependency rule:
      Risk scoring is upstream of the CFO decision queue and therefore
      does not require Decision_Priority as an input.
    """

    REQUIRED_MANAGEMENT = [
        "Business_Unit",
        "Department",
        "Category",
        "Projected_Variance",
        "Projected_Variance_Pct",
        "Early_Warning",
    ]

    REQUIRED_MATERIALITY = [
        "Business_Unit",
        "Department",
        "Category",
        "Absolute_Variance",
        "Severity",
        "Is_Material",
    ]

    def __init__(
        self,
        management_data: pd.DataFrame,
        materiality_data: pd.DataFrame,
        root_cause_data: pd.DataFrame | None = None,
    ):

        self.management = management_data.copy()
        self.materiality = materiality_data.copy()
        self.root_cause = (
            None
            if root_cause_data is None
            else root_cause_data.copy()
        )

        self._validate()

    def _validate(self):

        missing_m = [
            c
            for c in self.REQUIRED_MANAGEMENT
            if c not in self.management.columns
        ]

        if missing_m:
            raise ValueError(
                "Management data is missing: "
                + ", ".join(missing_m)
            )

        missing_mat = [
            c
            for c in self.REQUIRED_MATERIALITY
            if c not in self.materiality.columns
        ]

        if missing_mat:
            raise ValueError(
                "Materiality data is missing: "
                + ", ".join(missing_mat)
            )

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        max_value = series.abs().max()

        if not max_value or pd.isna(max_value):
            return pd.Series(
                np.zeros(len(series)),
                index=series.index,
            )

        return series.abs() / max_value

    def run(self) -> pd.DataFrame:

        data = self.management.copy()

        for col in [
            "Projected_Variance",
            "Projected_Variance_Pct",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(0.0)

        # Aggregate materiality to the same decision grain.
        mat = (
            self.materiality
            .groupby(
                [
                    "Business_Unit",
                    "Department",
                    "Category",
                ],
                as_index=False,
            )
            .agg(
                Current_Absolute_Variance=(
                    "Absolute_Variance",
                    "sum",
                ),
                Material_Record_Count=(
                    "Is_Material",
                    "sum",
                ),
                High_Critical_Count=(
                    "Severity",
                    lambda s: int(
                        s.isin(
                            [
                                "HIGH",
                                "CRITICAL",
                            ]
                        ).sum()
                    ),
                ),
            )
        )

        data = data.merge(
            mat,
            on=[
                "Business_Unit",
                "Department",
                "Category",
            ],
            how="left",
        )

        if self.root_cause is not None and not self.root_cause.empty:

            rc = self.root_cause.copy()

            # Recurring driver outputs normally contain Department
            # and Category. If those fields are absent, we skip the
            # recurrence component instead of inventing a mapping.
            if {
                "Department",
                "Category",
                "Occurrences",
            }.issubset(rc.columns):

                rc_small = (
                    rc[
                        [
                            "Department",
                            "Category",
                            "Occurrences",
                        ]
                    ]
                    .groupby(
                        [
                            "Department",
                            "Category",
                        ],
                        as_index=False,
                    )["Occurrences"]
                    .max()
                )

                data = data.merge(
                    rc_small,
                    on=[
                        "Department",
                        "Category",
                    ],
                    how="left",
                )

            else:
                data["Occurrences"] = 0

        else:
            data["Occurrences"] = 0

        data["Current_Absolute_Variance"] = (
            pd.to_numeric(
                data[
                    "Current_Absolute_Variance"
                ],
                errors="coerce",
            ).fillna(0.0)
        )

        data["Material_Record_Count"] = (
            pd.to_numeric(
                data[
                    "Material_Record_Count"
                ],
                errors="coerce",
            ).fillna(0.0)
        )

        data["High_Critical_Count"] = (
            pd.to_numeric(
                data[
                    "High_Critical_Count"
                ],
                errors="coerce",
            ).fillna(0.0)
        )

        data["Occurrences"] = (
            pd.to_numeric(
                data["Occurrences"],
                errors="coerce",
            ).fillna(0.0)
        )

        # Components
        data["Risk_Project_Variance"] = (
            self._normalize(
                data["Projected_Variance"]
            ) * 35.0
        )

        data["Risk_Project_Pct"] = (
            self._normalize(
                data["Projected_Variance_Pct"]
            ) * 25.0
        )

        warning_points = {
            "RED": 20.0,
            "AMBER": 10.0,
            "GREEN": 0.0,
        }

        data["Risk_Warning"] = (
            data["Early_Warning"]
            .map(warning_points)
            .fillna(0.0)
        )

        data["Risk_Materiality"] = (
            np.minimum(
                data["High_Critical_Count"] * 2.5,
                10.0,
            )
        )

        data["Risk_Recurrence"] = (
            np.minimum(
                data["Occurrences"] * 1.0,
                10.0,
            )
        )

        data["Risk_Score"] = (
            data["Risk_Project_Variance"]
            + data["Risk_Project_Pct"]
            + data["Risk_Warning"]
            + data["Risk_Materiality"]
            + data["Risk_Recurrence"]
        ).clip(
            lower=0,
            upper=100,
        )

        def band(score):
            if score >= 75:
                return "CRITICAL"
            if score >= 55:
                return "HIGH"
            if score >= 30:
                return "MODERATE"
            return "LOW"

        data["Risk_Band"] = data[
            "Risk_Score"
        ].apply(band)

        data["Risk_Rank"] = (
            data["Risk_Score"]
            .rank(
                method="dense",
                ascending=False,
            )
            .astype(int)
        )

        return data.sort_values(
            [
                "Risk_Score",
                "Projected_Variance",
            ],
            ascending=False,
        ).reset_index(drop=True)

    def summary(
        self,
        risk_data: pd.DataFrame | None = None,
    ) -> dict:

        data = (
            self.run()
            if risk_data is None
            else risk_data.copy()
        )

        return {
            "Total_Records": len(data),
            "Average_Risk_Score": float(
                data["Risk_Score"].mean()
            ),
            "Maximum_Risk_Score": float(
                data["Risk_Score"].max()
            ),
            "Critical": int(
                (data["Risk_Band"] == "CRITICAL").sum()
            ),
            "High": int(
                (data["Risk_Band"] == "HIGH").sum()
            ),
            "Moderate": int(
                (data["Risk_Band"] == "MODERATE").sum()
            ),
            "Low": int(
                (data["Risk_Band"] == "LOW").sum()
            ),
        }
