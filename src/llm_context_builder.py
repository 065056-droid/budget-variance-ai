
import json
import pandas as pd


class LLMContextBuilder:
    """
    Build grounded, structured context for an LLM.

    The builder does not ask the model to calculate financial numbers.
    It prepares verified analytical facts for later use with Ollama/Qwen.

    This separation is intentional:

        VARIA analytical engines -> calculate
        LLMContextBuilder         -> packages evidence
        LLM                      -> explains / communicates
    """

    REQUIRED_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "Current_Variance",
        "Projected_Variance",
        "Projected_Variance_Pct",
        "Early_Warning",
        "Decision_Priority",
    ]

    def __init__(
        self,
        decision_queue: pd.DataFrame,
        recommendation_data: pd.DataFrame | None = None,
        risk_data: pd.DataFrame | None = None,
        scenario_data: pd.DataFrame | None = None,
    ):

        self.queue = decision_queue.copy()
        self.recommendations = (
            None
            if recommendation_data is None
            else recommendation_data.copy()
        )
        self.risk = (
            None
            if risk_data is None
            else risk_data.copy()
        )
        self.scenario = (
            None
            if scenario_data is None
            else scenario_data.copy()
        )

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.queue.columns
        ]

        if missing:
            raise ValueError(
                "Decision queue is missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _records(
        dataframe: pd.DataFrame,
        columns: list[str],
        top_n: int,
    ) -> list[dict]:

        available = [
            c for c in columns
            if c in dataframe.columns
        ]

        return (
            dataframe[available]
            .head(int(top_n))
            .to_dict(orient="records")
        )

    def build_issue_context(
        self,
        row: pd.Series,
    ) -> dict:

        context = {
            "scope": {
                "business_unit": row[
                    "Business_Unit"
                ],
                "department": row[
                    "Department"
                ],
                "category": row[
                    "Category"
                ],
            },
            "financial_position": {
                "current_variance": float(
                    row["Current_Variance"]
                ),
                "projected_variance": float(
                    row["Projected_Variance"]
                ),
                "projected_variance_pct": float(
                    row["Projected_Variance_Pct"]
                ),
            },
            "early_warning": row[
                "Early_Warning"
            ],
            "decision_priority": row[
                "Decision_Priority"
            ],
        }

        if self.recommendations is not None:

            matches = self.recommendations[
                (
                    self.recommendations[
                        "Business_Unit"
                    ].astype(str)
                    == str(
                        row["Business_Unit"]
                    )
                )
                &
                (
                    self.recommendations[
                        "Department"
                    ].astype(str)
                    == str(
                        row["Department"]
                    )
                )
                &
                (
                    self.recommendations[
                        "Category"
                    ].astype(str)
                    == str(
                        row["Category"]
                    )
                )
            ]

            if len(matches):

                recommendation = matches.iloc[
                    0
                ]

                for field in [
                    "Risk_Score",
                    "Risk_Band",
                    "Anomaly_Level",
                    "Variance_Interpretation",
                    "Management_Action_Type",
                    "Management_Recommendation",
                    "Action_Urgency",
                ]:
                    if field in recommendation:
                        value = recommendation[
                            field
                        ]

                        context.setdefault(
                            "management_signals",
                            {},
                        )[field] = (
                            value.item()
                            if hasattr(
                                value,
                                "item",
                            )
                            else value
                        )

        return context

    def top_issues(
        self,
        top_n: int = 10,
    ) -> list[dict]:

        queue = self.queue.copy()

        queue["Abs_Projected_Variance"] = (
            queue["Projected_Variance"]
            .abs()
        )

        queue = queue.sort_values(
            "Abs_Projected_Variance",
            ascending=False,
        )

        return [
            self.build_issue_context(
                row
            )
            for _, row in queue.head(
                int(top_n)
            ).iterrows()
        ]

    def build_prompt_payload(
        self,
        top_n: int = 10,
        instruction: str = (
            "Explain the financial issues using only the supplied evidence. "
            "Do not invent causes, figures or facts. Clearly distinguish "
            "observed facts from management recommendations."
        ),
    ) -> dict:

        payload = {
            "system_instruction": instruction,
            "model_role": (
                "You are a grounded FP&A management commentary assistant."
            ),
            "issues": self.top_issues(
                top_n=top_n
            ),
        }

        if self.scenario is not None and not self.scenario.empty:
            payload["scenario_evidence"] = (
                self._records(
                    self.scenario,
                    [
                        "Scenario_Name",
                        "Scenario_Projected_Variance",
                        "Variance_Improvement",
                        "Forecast_Savings",
                    ],
                    top_n=10,
                )
            )

        return payload

    def to_json(
        self,
        top_n: int = 10,
    ) -> str:

        return json.dumps(
            self.build_prompt_payload(
                top_n=top_n
            ),
            indent=2,
            default=str,
        )
