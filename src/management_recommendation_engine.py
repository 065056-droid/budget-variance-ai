
import pandas as pd
import numpy as np


class ManagementRecommendationEngine:
    """
    Rule-based management recommendation layer.

    Recommendations are generated from measurable financial signals:
      - forecast direction
      - projected variance %
      - early-warning status
      - current severity
      - anomaly
      - recurrence

    The engine does not alter numbers. It translates detected patterns into
    management actions that can later be handed to an LLM for richer wording.
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Projected_Variance",
        "Projected_Variance_Pct",
        "Early_Warning",
    ]

    def __init__(
        self,
        management_data: pd.DataFrame,
        risk_data: pd.DataFrame | None = None,
        anomaly_data: pd.DataFrame | None = None,
        interpretation_data: pd.DataFrame | None = None,
    ):

        self.management = management_data.copy()
        self.risk = (
            None
            if risk_data is None
            else risk_data.copy()
        )
        self.anomaly = (
            None
            if anomaly_data is None
            else anomaly_data.copy()
        )
        self.interpretation = (
            None
            if interpretation_data is None
            else interpretation_data.copy()
        )

        self._validate()

    def _validate(self):

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.management.columns
        ]

        if missing:
            raise ValueError(
                "Management data is missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _scope_key(row):

        return (
            str(row["Business_Unit"]),
            str(row["Department"]),
            str(row["Category"]),
        )

    def _lookup_risk(self):

        if self.risk is None or self.risk.empty:
            return {}

        required = {
            "Business_Unit",
            "Department",
            "Category",
            "Risk_Score",
            "Risk_Band",
        }

        if not required.issubset(
            self.risk.columns
        ):
            return {}

        return {
            (
                str(row["Business_Unit"]),
                str(row["Department"]),
                str(row["Category"]),
            ): (
                float(row["Risk_Score"]),
                str(row["Risk_Band"]),
            )
            for _, row in self.risk.iterrows()
        }

    def _lookup_anomaly(self):

        if self.anomaly is None or self.anomaly.empty:
            return {}

        required = {
            "Business_Unit",
            "Department",
            "Category",
            "Anomaly_Level",
            "Requires_Investigation",
        }

        if not required.issubset(
            self.anomaly.columns
        ):
            return {}

        # Multiple monthly anomaly records can exist for a key.
        grouped = (
            self.anomaly.groupby(
                [
                    "Business_Unit",
                    "Department",
                    "Category",
                ],
                as_index=False,
            )
            .agg(
                Anomaly_Level=(
                    "Anomaly_Level",
                    lambda s: (
                        "CRITICAL"
                        if "CRITICAL" in set(s)
                        else (
                            "HIGH"
                            if "HIGH" in set(s)
                            else (
                                "REVIEW"
                                if "REVIEW" in set(s)
                                else "NORMAL"
                            )
                        )
                    ),
                ),
                Anomaly_Investigations=(
                    "Requires_Investigation",
                    "sum",
                ),
            )
        )

        return {
            (
                str(row["Business_Unit"]),
                str(row["Department"]),
                str(row["Category"]),
            ): (
                str(row["Anomaly_Level"]),
                int(row["Anomaly_Investigations"]),
            )
            for _, row in grouped.iterrows()
        }

    def _lookup_interpretation(self):

        if (
            self.interpretation is None
            or self.interpretation.empty
        ):
            return {}

        if not {
            "Business_Unit",
            "Department",
            "Category",
            "Variance_Type",
        }.issubset(
            self.interpretation.columns
        ):
            return {}

        grouped = (
            self.interpretation.groupby(
                [
                    "Business_Unit",
                    "Department",
                    "Category",
                ],
                as_index=False,
            )["Variance_Type"]
            .agg(
                lambda s: (
                    "Reversal/Credit"
                    if "Reversal/Credit" in set(s)
                    else (
                        "Overspend"
                        if "Overspend" in set(s)
                        else (
                            "Underspend"
                            if "Underspend" in set(s)
                            else "On Budget"
                        )
                    )
                )
            )
        )

        return {
            (
                str(row["Business_Unit"]),
                str(row["Department"]),
                str(row["Category"]),
            ): str(row["Variance_Type"])
            for _, row in grouped.iterrows()
        }

    @staticmethod
    def _issue_type(
        projected_variance: float,
    ) -> str:

        if projected_variance > 0:
            return "Projected Overspend"
        if projected_variance < 0:
            return "Projected Underspend"
        return "On Budget"

    @staticmethod
    def _recommendation(
        issue_type: str,
        warning: str,
        projected_pct: float,
        risk_band: str,
        anomaly_level: str,
        recurring: bool,
        interpretation: str,
    ) -> tuple[str, str]:

        if interpretation == "Reversal/Credit":
            return (
                "Accounting validation",
                "Validate the reversal, credit, refund or accounting adjustment before using the amount for operational conclusions.",
            )

        if issue_type == "Projected Overspend":

            if (
                warning == "RED"
                or risk_band in ["CRITICAL", "HIGH"]
                or anomaly_level in ["CRITICAL", "HIGH"]
            ):

                if recurring:
                    return (
                        "Immediate cost intervention",
                        "Validate the recurring driver, challenge the run-rate, identify avoidable spend and evaluate corrective action or controlled budget reallocation.",
                    )

                return (
                    "Immediate variance investigation",
                    "Validate the driver, confirm business need, identify exceptional items and implement a corrective action plan before year-end.",
                )

            if abs(projected_pct) >= 10:
                return (
                    "Management review",
                    "Review the projected overspend, challenge remaining commitments and identify practical cost-control actions.",
                )

            return (
                "Monitor and control",
                "Monitor the remaining spend trajectory and intervene if the forecast continues to move unfavorably.",
            )

        if issue_type == "Projected Underspend":

            if recurring:
                return (
                    "Validate budget utilization",
                    "Determine whether the underspend reflects genuine savings, delayed activity, cancellations or an unrealistic original budget.",
                )

            return (
                "Review utilization",
                "Confirm whether the underspend is timing-related or represents a sustainable saving that should be reflected in the forecast.",
            )

        return (
            "Maintain control",
            "Continue monitoring performance and update assumptions if business conditions change.",
        )

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

        risk_lookup = self._lookup_risk()
        anomaly_lookup = self._lookup_anomaly()
        interpretation_lookup = (
            self._lookup_interpretation()
        )

        # Occurrence-based recurrence is available directly if passed from
        # an upstream recurring-driver table.
        if "Occurrences" in data.columns:
            occurrences = pd.to_numeric(
                data["Occurrences"],
                errors="coerce",
            ).fillna(0)
        else:
            occurrences = pd.Series(
                0,
                index=data.index,
            )

        risk_scores = []
        risk_bands = []
        anomaly_levels = []
        anomaly_counts = []
        interpretations = []
        issue_types = []
        recurring_flags = []
        action_types = []
        recommendations = []

        for idx, row in data.iterrows():

            key = self._scope_key(row)

            risk_score, risk_band = risk_lookup.get(
                key,
                (0.0, "LOW"),
            )

            anomaly_level, anomaly_count = (
                anomaly_lookup.get(
                    key,
                    ("NORMAL", 0),
                )
            )

            interpretation = (
                interpretation_lookup.get(
                    key,
                    "On Budget",
                )
            )

            projected_variance = float(
                row["Projected_Variance"]
            )

            projected_pct = float(
                row["Projected_Variance_Pct"]
            )

            warning = str(
                row["Early_Warning"]
            )

            issue_type = self._issue_type(
                projected_variance
            )

            recurring = (
                int(occurrences.loc[idx])
                >= 3
            )

            action_type, recommendation = (
                self._recommendation(
                    issue_type,
                    warning,
                    projected_pct,
                    risk_band,
                    anomaly_level,
                    recurring,
                    interpretation,
                )
            )

            risk_scores.append(
                risk_score
            )

            risk_bands.append(
                risk_band
            )

            anomaly_levels.append(
                anomaly_level
            )

            anomaly_counts.append(
                anomaly_count
            )

            interpretations.append(
                interpretation
            )

            issue_types.append(
                issue_type
            )

            recurring_flags.append(
                recurring
            )

            action_types.append(
                action_type
            )

            recommendations.append(
                recommendation
            )

        data["AI_Issue_Type"] = issue_types
        data["Risk_Score"] = risk_scores
        data["Risk_Band"] = risk_bands
        data["Anomaly_Level"] = anomaly_levels
        data["Anomaly_Investigation_Count"] = (
            anomaly_counts
        )
        data["Variance_Interpretation"] = (
            interpretations
        )
        data["Recurring_Driver"] = (
            recurring_flags
        )
        data["Management_Action_Type"] = (
            action_types
        )
        data["Management_Recommendation"] = (
            recommendations
        )

        def urgency(row):

            if (
                row["Risk_Band"]
                == "CRITICAL"
                or row["Anomaly_Level"]
                == "CRITICAL"
            ):
                return "1 — Immediate"

            if (
                row["Early_Warning"]
                == "RED"
                or row["Risk_Band"]
                == "HIGH"
                or row["Anomaly_Level"]
                == "HIGH"
            ):
                return "2 — High"

            if (
                row["Early_Warning"]
                == "AMBER"
                or row["Risk_Band"]
                == "MODERATE"
                or row["Anomaly_Level"]
                == "REVIEW"
            ):
                return "3 — Review"

            return "4 — Monitor"

        data["Action_Urgency"] = data.apply(
            urgency,
            axis=1,
        )

        return data.sort_values(
            [
                "Action_Urgency",
                "Risk_Score",
                "Projected_Variance",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        ).reset_index(drop=True)

    def priority_queue(
        self,
        recommendation_data: pd.DataFrame | None = None,
        top_n: int = 25,
    ) -> pd.DataFrame:

        data = (
            self.run()
            if recommendation_data is None
            else recommendation_data.copy()
        )

        return data.head(
            int(top_n)
        )

    def summary(
        self,
        recommendation_data: pd.DataFrame | None = None,
    ) -> dict:

        data = (
            self.run()
            if recommendation_data is None
            else recommendation_data.copy()
        )

        return {
            "Records": len(data),
            "Immediate": int(
                (
                    data["Action_Urgency"]
                    == "1 — Immediate"
                ).sum()
            ),
            "High": int(
                (
                    data["Action_Urgency"]
                    == "2 — High"
                ).sum()
            ),
            "Review": int(
                (
                    data["Action_Urgency"]
                    == "3 — Review"
                ).sum()
            ),
            "Monitor": int(
                (
                    data["Action_Urgency"]
                    == "4 — Monitor"
                ).sum()
            ),
            "Projected_Overspend": int(
                (
                    data["AI_Issue_Type"]
                    == "Projected Overspend"
                ).sum()
            ),
            "Projected_Underspend": int(
                (
                    data["AI_Issue_Type"]
                    == "Projected Underspend"
                ).sum()
            ),
        }
