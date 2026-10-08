import pandas as pd


class ManagementActionEngine:
    """
    Converts forecast and early-warning outputs into
    management-oriented priorities and recommended actions.
    """

    def __init__(
        self,
        variance_data: pd.DataFrame,
        materiality_data: pd.DataFrame,
        root_cause_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        warning_data: pd.DataFrame,
    ):
        self.variance_data = variance_data.copy()
        self.materiality_data = materiality_data.copy()
        self.root_cause_data = root_cause_data.copy()
        self.forecast_data = forecast_data.copy()
        self.warning_data = warning_data.copy()

        self.management_data = None

    def prepare_data(self) -> pd.DataFrame:
        """Prepare management-level forecast interpretation."""

        df = self.warning_data.copy()

        required_columns = [
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
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "ManagementActionEngine is missing "
                f"required columns: {missing_columns}"
            )

        # -----------------------------------------------------
        # PROJECTED VARIANCE TYPE
        # -----------------------------------------------------

        df["Variance_Type"] = "On Budget"

        df.loc[
            df["Projected_Variance"] > 0,
            "Variance_Type",
        ] = "Projected Overspend"

        df.loc[
            df["Projected_Variance"] < 0,
            "Variance_Type",
        ] = "Projected Underspend"

        # -----------------------------------------------------
        # MANAGEMENT PRIORITY
        # -----------------------------------------------------

        df["Management_Priority"] = "Monitor"

        df.loc[
            df["Early_Warning"] == "AMBER",
            "Management_Priority",
        ] = "Review"

        df.loc[
            df["Early_Warning"] == "RED",
            "Management_Priority",
        ] = "Immediate Attention"

        # -----------------------------------------------------
        # DEFAULT ACTION
        # -----------------------------------------------------

        df["Management_Action"] = (
            "Continue monitoring performance against "
            "the projected full-year budget."
        )

        # -----------------------------------------------------
        # RED OVERSPEND
        # -----------------------------------------------------

        red_overspend = (
            (df["Early_Warning"] == "RED")
            & (df["Variance_Type"] == "Projected Overspend")
        )

        df.loc[
            red_overspend,
            "Management_Action",
        ] = (
            "Investigate the spending drivers, validate "
            "business need, and assess corrective action "
            "or budget reallocation."
        )

        # -----------------------------------------------------
        # RED UNDERSPEND
        # -----------------------------------------------------

        red_underspend = (
            (df["Early_Warning"] == "RED")
            & (df["Variance_Type"] == "Projected Underspend")
        )

        df.loc[
            red_underspend,
            "Management_Action",
        ] = (
            "Investigate whether the underspend reflects "
            "genuine savings, delayed activity, "
            "cancellations, or an unrealistic budget."
        )

        # -----------------------------------------------------
        # AMBER OVERSPEND
        # -----------------------------------------------------

        amber_overspend = (
            (df["Early_Warning"] == "AMBER")
            & (df["Variance_Type"] == "Projected Overspend")
        )

        df.loc[
            amber_overspend,
            "Management_Action",
        ] = (
            "Monitor spending closely and identify the "
            "primary driver before the variance becomes "
            "material."
        )

        # -----------------------------------------------------
        # AMBER UNDERSPEND
        # -----------------------------------------------------

        amber_underspend = (
            (df["Early_Warning"] == "AMBER")
            & (df["Variance_Type"] == "Projected Underspend")
        )

        df.loc[
            amber_underspend,
            "Management_Action",
        ] = (
            "Review whether spending is delayed or whether "
            "the projected savings are sustainable."
        )

        # -----------------------------------------------------
        # MANAGEMENT RATIONALE
        # -----------------------------------------------------

        df["Management_Rationale"] = (
            "Projected performance is within the "
            "normal monitoring range."
        )

        df.loc[
            red_overspend,
            "Management_Rationale",
        ] = (
            "Projected full-year spend is materially above "
            "the estimated full-year budget."
        )

        df.loc[
            red_underspend,
            "Management_Rationale",
        ] = (
            "Projected full-year spend is materially below "
            "the estimated full-year budget."
        )

        df.loc[
            amber_overspend,
            "Management_Rationale",
        ] = (
            "Projected full-year spend is moderately above "
            "the estimated full-year budget."
        )

        df.loc[
            amber_underspend,
            "Management_Rationale",
        ] = (
            "Projected full-year spend is moderately below "
            "the estimated full-year budget."
        )

        self.management_data = df

        return df

    def summary(self) -> dict:
        """Return management-level summary statistics."""

        if self.management_data is None:
            self.prepare_data()

        df = self.management_data

        return {
            "Total_Records": len(df),
            "Immediate_Attention": (
                df["Management_Priority"]
                == "Immediate Attention"
            ).sum(),
            "Review": (
                df["Management_Priority"]
                == "Review"
            ).sum(),
            "Monitor": (
                df["Management_Priority"]
                == "Monitor"
            ).sum(),
            "Projected_Overspend_Records": (
                df["Variance_Type"]
                == "Projected Overspend"
            ).sum(),
            "Projected_Underspend_Records": (
                df["Variance_Type"]
                == "Projected Underspend"
            ).sum(),
            "Net_Projected_Variance": (
                df["Projected_Variance"].sum()
            ),
        }

    def priority_queue(
        self,
        priority: str = "Immediate Attention",
    ) -> pd.DataFrame:
        """Return records for a specific management priority."""

        if self.management_data is None:
            self.prepare_data()

        return (
            self.management_data[
                self.management_data["Management_Priority"]
                == priority
            ]
            .sort_values(
                "Projected_Variance",
                key=lambda x: x.abs(),
                ascending=False,
            )
            .copy()
        )

    def overspend_queue(self) -> pd.DataFrame:
        """Return projected overspend records."""

        if self.management_data is None:
            self.prepare_data()

        return (
            self.management_data[
                self.management_data["Variance_Type"]
                == "Projected Overspend"
            ]
            .sort_values(
                "Projected_Variance",
                ascending=False,
            )
            .copy()
        )

    def underspend_queue(self) -> pd.DataFrame:
        """Return projected underspend records."""

        if self.management_data is None:
            self.prepare_data()

        return (
            self.management_data[
                self.management_data["Variance_Type"]
                == "Projected Underspend"
            ]
            .sort_values(
                "Projected_Variance",
                ascending=True,
            )
            .copy()
        )

    def action_queue(self) -> pd.DataFrame:
        """Return all records requiring action or review."""

        if self.management_data is None:
            self.prepare_data()

        return (
            self.management_data[
                self.management_data["Management_Priority"]
                != "Monitor"
            ]
            .sort_values(
                "Projected_Variance",
                key=lambda x: x.abs(),
                ascending=False,
            )
            .copy()
        )
