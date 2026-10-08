import pandas as pd


class DriverAnalysis:
    """
    Driver and Pareto analysis for budget variance.

    Identifies the dimensions that contribute most to
    unfavorable budget variance.

    Primary analysis:
        Positive variance = unfavorable / overspend

    Negative variance is excluded from the primary
    unfavorable-driver analysis because it represents
    favorable variance / underspend.
    """

    def __init__(
        self,
        variance: pd.DataFrame,
    ):
        self.variance = variance.copy()

    # =========================================================
    # 1. PREPARE DATA
    # =========================================================

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare variance data for driver analysis.
        """

        df = self.variance.copy()

        required_columns = [
            "Variance",
            "Business_Unit",
            "Department",
            "Category",
            "Month",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Variance data is missing required columns: "
                f"{missing_columns}"
            )

        # -----------------------------------------------------
        # Unfavorable variance
        #
        # Positive variance = overspend
        # Negative variance = favorable
        # -----------------------------------------------------

        df["Unfavorable_Variance"] = (
            df["Variance"].clip(lower=0)
        )

        return df

    # =========================================================
    # 2. ANALYSE DIMENSION
    # =========================================================

    def by_dimension(
        self,
        dimension: str,
    ) -> pd.DataFrame:
        """
        Analyse unfavorable variance by a selected dimension.

        Supported dimensions:
            Business_Unit
            Department
            Category
            Month
        """

        supported_dimensions = [
            "Business_Unit",
            "Department",
            "Category",
            "Month",
        ]

        if dimension not in supported_dimensions:
            raise ValueError(
                f"Unsupported dimension '{dimension}'. "
                f"Choose from: {supported_dimensions}"
            )

        df = self.prepare_data()

        # -----------------------------------------------------
        # Aggregate unfavorable variance
        # -----------------------------------------------------

        analysis = (
            df.groupby(dimension, dropna=False)
            .agg(
                Unfavorable_Variance=(
                    "Unfavorable_Variance",
                    "sum",
                ),
                Variance_Record_Count=(
                    "Variance",
                    "count",    
                ),
            )
            .reset_index()
        )

        # -----------------------------------------------------
        # Rank drivers
        # -----------------------------------------------------

        analysis = analysis.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        ).reset_index(drop=True)

        # -----------------------------------------------------
        # Total unfavorable variance
        # -----------------------------------------------------

        total_unfavorable = (
            analysis["Unfavorable_Variance"].sum()
        )

        # -----------------------------------------------------
        # Contribution %
        # -----------------------------------------------------

        if total_unfavorable > 0:

            analysis["Contribution_Pct"] = (
                analysis["Unfavorable_Variance"]
                / total_unfavorable
                * 100
            )

        else:

            analysis["Contribution_Pct"] = 0.0

        # -----------------------------------------------------
        # Cumulative contribution
        # -----------------------------------------------------

        analysis["Cumulative_Contribution_Pct"] = (
            analysis["Contribution_Pct"]
            .cumsum()
        )

        # -----------------------------------------------------
        # Pareto flag
        #
        # Drivers contributing to the first 80% of
        # unfavorable variance are identified as Pareto drivers.
        # -----------------------------------------------------

        analysis["Pareto_Driver"] = (
            analysis["Cumulative_Contribution_Pct"]
            <= 80
        )

        # Include the first driver that crosses 80%.
        if len(analysis) > 0:

            first_above_80 = (
                analysis[
                    analysis[
                        "Cumulative_Contribution_Pct"
                    ] > 80
                ]
            )

            if not first_above_80.empty:

                first_index = first_above_80.index[0]

                analysis.loc[
                    first_index,
                    "Pareto_Driver",
                ] = True

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        analysis.insert(
            0,
            "Rank",
            range(1, len(analysis) + 1),
        )

        return analysis

    # =========================================================
    # 3. BUSINESS UNIT ANALYSIS
    # =========================================================

    def business_unit_analysis(self) -> pd.DataFrame:
        """Return unfavorable variance by business unit."""

        return self.by_dimension(
            "Business_Unit"
        )

    # =========================================================
    # 4. DEPARTMENT ANALYSIS
    # =========================================================

    def department_analysis(self) -> pd.DataFrame:
        """Return unfavorable variance by department."""

        return self.by_dimension(
            "Department"
        )

    # =========================================================
    # 5. CATEGORY ANALYSIS
    # =========================================================

    def category_analysis(self) -> pd.DataFrame:
        """Return unfavorable variance by category."""

        return self.by_dimension(
            "Category"
        )

    # =========================================================
    # 6. MONTH ANALYSIS
    # =========================================================

    def month_analysis(self) -> pd.DataFrame:
        """Return unfavorable variance by month."""

        return self.by_dimension(
            "Month"
        )

    # =========================================================
    # 7. TOP DRIVERS
    # =========================================================

    def top_drivers(
        self,
        dimension: str,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the top N unfavorable variance drivers.
        """

        analysis = self.by_dimension(
            dimension
        )

        return analysis.head(n)

    # =========================================================
    # 8. PARETO DRIVERS
    # =========================================================

    def pareto_drivers(
        self,
        dimension: str,
    ) -> pd.DataFrame:
        """
        Return drivers contributing to approximately
        the first 80% of unfavorable variance.
        """

        analysis = self.by_dimension(
            dimension
        )

        return analysis[
            analysis["Pareto_Driver"]
        ].copy()

    # =========================================================
    # 9. SUMMARY
    # =========================================================

    def summary(self) -> dict:
        """
        Return overall unfavorable variance summary.
        """

        df = self.prepare_data()

        total_variance = df["Variance"].sum()

        total_unfavorable = (
            df["Unfavorable_Variance"].sum()
        )

        total_favorable = (
            df.loc[
                df["Variance"] < 0,
                "Variance",
            ].sum()
        )

        return {
            "Total_Variance": total_variance,
            "Total_Unfavorable_Variance": (
                total_unfavorable
            ),
            "Total_Favorable_Variance": (
                total_favorable
            ),
            "Unfavorable_Record_Count": int(
                (df["Variance"] > 0).sum()
            ),
            "Favorable_Record_Count": int(
                (df["Variance"] < 0).sum()
            ),
            "On_Budget_Record_Count": int(
                (df["Variance"] == 0).sum()
            ),
        }