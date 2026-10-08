import pandas as pd


class CFODecisionQueue:
    """
    Integrates current variance, materiality, forecast,
    early-warning, and management-action signals into
    a single CFO decision queue.
    """

    def __init__(
        self,
        variance_data: pd.DataFrame,
        materiality_data: pd.DataFrame,
        management_data: pd.DataFrame,
    ):
        self.variance_data = variance_data.copy()
        self.materiality_data = materiality_data.copy()
        self.management_data = management_data.copy()

        self.decision_queue = None

    # ---------------------------------------------------------
    # BUILD QUEUE
    # ---------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Build the integrated CFO decision queue.
        """

        # -----------------------------------------------------
        # REQUIRED MANAGEMENT COLUMNS
        # -----------------------------------------------------

        management_required = [
            "Business_Unit",
            "Department",
            "Category",
            "YTD_Budget",
            "YTD_Actual",
            "Estimated_Full_Year_Budget",
            "Forecast_At_Completion",
            "Projected_Variance",
            "Projected_Variance_Pct",
            "Early_Warning",
            "Variance_Type",
            "Management_Priority",
            "Management_Action",
            "Management_Rationale",
        ]

        missing_management = [
            column
            for column in management_required
            if column not in self.management_data.columns
        ]

        if missing_management:
            raise ValueError(
                "Management data is missing required columns: "
                f"{missing_management}"
            )

        # -----------------------------------------------------
        # MATERIALITY DATA
        # -----------------------------------------------------

        materiality_columns = [
            "Business_Unit",
            "Department",
            "Category",
            "Variance",
            "Variance_Pct",
            "Absolute_Variance",
            "Absolute_Variance_Pct",
            "Is_Material",
            "Severity",
            "Management_Review",
        ]

        missing_materiality = [
            column
            for column in materiality_columns
            if column not in self.materiality_data.columns
        ]

        if missing_materiality:
            raise ValueError(
                "Materiality data is missing required columns: "
                f"{missing_materiality}"
            )

        # -----------------------------------------------------
        # AGGREGATE CURRENT MATERIALITY
        #
        # Forecast data is at:
        # BU × Department × Category
        #
        # Materiality data is monthly.
        #
        # Therefore aggregate current variance information
        # to the same BU × Department × Category grain.
        # -----------------------------------------------------

        materiality = (
            self.materiality_data
            .groupby(
                [
                    "Business_Unit",
                    "Department",
                    "Category",
                ],
                dropna=False,
            )
            .agg(
                Current_Variance=(
                    "Variance",
                    "sum",
                ),
                Current_Variance_Pct=(
                    "Variance_Pct",
                    "mean",
                ),
                Absolute_Current_Variance=(
                    "Absolute_Variance",
                    "sum",
                ),
                Material_Variance_Count=(
                    "Is_Material",
                    "sum",
                ),
                High_Critical_Count=(
                    "Severity",
                    lambda x: (
                        x.isin(
                            ["HIGH", "CRITICAL"]
                        ).sum()
                    ),
                ),
            )
            .reset_index()
        )

        # -----------------------------------------------------
        # MERGE
        # -----------------------------------------------------

        queue = self.management_data.merge(
            materiality,
            on=[
                "Business_Unit",
                "Department",
                "Category",
            ],
            how="left",
        )

        # -----------------------------------------------------
        # FILL NUMERIC NULLS
        # -----------------------------------------------------

        numeric_columns = [
            "Current_Variance",
            "Current_Variance_Pct",
            "Absolute_Current_Variance",
            "Material_Variance_Count",
            "High_Critical_Count",
        ]

        for column in numeric_columns:
            queue[column] = (
                queue[column]
                .fillna(0)
            )

        # -----------------------------------------------------
        # DECISION SCORE
        #
        # Higher score = higher management attention.
        # -----------------------------------------------------

        queue["Decision_Score"] = 0

        # RED
        queue.loc[
            queue["Early_Warning"] == "RED",
            "Decision_Score",
        ] += 3

        # AMBER
        queue.loc[
            queue["Early_Warning"] == "AMBER",
            "Decision_Score",
        ] += 2

        # Material current variance
        queue.loc[
            queue["Material_Variance_Count"] > 0,
            "Decision_Score",
        ] += 1

        # HIGH / CRITICAL current variance
        queue.loc[
            queue["High_Critical_Count"] > 0,
            "Decision_Score",
        ] += 2

        # -----------------------------------------------------
        # PRIORITY BAND
        # -----------------------------------------------------

        queue["Decision_Priority"] = "MONITOR"

        queue.loc[
            queue["Decision_Score"] >= 2,
            "Decision_Priority",
        ] = "REVIEW"

        queue.loc[
            queue["Decision_Score"] >= 4,
            "Decision_Priority",
        ] = "HIGH PRIORITY"

        queue.loc[
            queue["Decision_Score"] >= 6,
            "Decision_Priority",
        ] = "CRITICAL"

        # -----------------------------------------------------
        # SORT
        # -----------------------------------------------------

        queue = (
            queue
            .sort_values(
                [
                    "Decision_Score",
                    "Projected_Variance",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        # -----------------------------------------------------
        # MANAGEMENT VIEW
        # -----------------------------------------------------

        queue["Management_View"] = (
            queue["Decision_Priority"]
            + " | "
            + queue["Variance_Type"]
            + " | "
            + queue["Early_Warning"]
        )

        self.decision_queue = queue

        return queue

    # ---------------------------------------------------------
    # TOP DECISIONS
    # ---------------------------------------------------------

    def top_decisions(
        self,
        n: int = 15,
    ) -> pd.DataFrame:
        """
        Return the highest-priority management decisions.
        """

        if self.decision_queue is None:
            self.build()

        return (
            self.decision_queue
            .head(n)
            .copy()
        )

    # ---------------------------------------------------------
    # CRITICAL QUEUE
    # ---------------------------------------------------------

    def critical_queue(self) -> pd.DataFrame:
        """
        Return only CRITICAL decisions.
        """

        if self.decision_queue is None:
            self.build()

        return self.decision_queue[
            self.decision_queue["Decision_Priority"]
            == "CRITICAL"
        ].copy()

    # ---------------------------------------------------------
    # HIGH PRIORITY QUEUE
    # ---------------------------------------------------------

    def high_priority_queue(self) -> pd.DataFrame:
        """
        Return HIGH PRIORITY and CRITICAL decisions.
        """

        if self.decision_queue is None:
            self.build()

        return self.decision_queue[
            self.decision_queue[
                "Decision_Priority"
            ].isin(
                [
                    "HIGH PRIORITY",
                    "CRITICAL",
                ]
            )
        ].copy()

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    def summary(self) -> dict:
        """
        Return CFO-level decision summary.
        """

        if self.decision_queue is None:
            self.build()

        df = self.decision_queue

        return {
            "Total_Decision_Records": len(df),

            "Critical": (
                df["Decision_Priority"]
                == "CRITICAL"
            ).sum(),

            "High_Priority": (
                df["Decision_Priority"]
                == "HIGH PRIORITY"
            ).sum(),

            "Review": (
                df["Decision_Priority"]
                == "REVIEW"
            ).sum(),

            "Monitor": (
                df["Decision_Priority"]
                == "MONITOR"
            ).sum(),

            "Total_Projected_Variance": (
                df["Projected_Variance"]
                .sum()
            ),

            "Total_Current_Variance": (
                df["Current_Variance"]
                .sum()
            ),
        }