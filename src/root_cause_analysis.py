import pandas as pd


class RootCauseAnalysis:
    """
    Drill-down and recurring-driver analysis for budget variance.

    This module identifies where unfavorable variance is concentrated
    across Category, Department, Business Unit, and Month.

    It does not claim causal root cause. It identifies financial
    drivers and recurring patterns that require investigation.
    """

    def __init__(self, variance: pd.DataFrame):
        self.variance = variance.copy()

    # =========================================================
    # 1. PREPARE DATA
    # =========================================================

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare variance data for drill-down analysis.
        """

        df = self.variance.copy()

        required_columns = [
            "Month",
            "Business_Unit",
            "Department",
            "Category",
            "Budget_Amount",
            "Actual_Amount",
            "Variance",
            "Variance_Pct",
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

        # Positive variance represents unfavorable variance.
        df["Unfavorable_Variance"] = (
            df["Variance"].clip(lower=0)
        )

        # Flag unfavorable records.
        df["Is_Unfavorable"] = (
            df["Variance"] > 0
        )

        return df

    # =========================================================
    # 2. FILTER CATEGORY
    # =========================================================

    def by_category(
        self,
        category: str,
    ) -> pd.DataFrame:
        """
        Return all variance records for a selected category.
        """

        df = self.prepare_data()

        result = df[
            df["Category"] == category
        ].copy()

        return result.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        )

    # =========================================================
    # 3. DRILL DOWN
    # =========================================================

    def drill_down(
        self,
        category: str | None = None,
        department: str | None = None,
        business_unit: str | None = None,
        month=None,
    ) -> pd.DataFrame:
        """
        Drill down into a specific variance driver.

        Filters can be combined.

        Example:
            drill_down(
                category="Digital Advertising",
                department="Marketing",
            )
        """

        df = self.prepare_data()

        # -----------------------------------------------------
        # Category filter
        # -----------------------------------------------------

        if category is not None:

            df = df[
                df["Category"] == category
            ]

        # -----------------------------------------------------
        # Department filter
        # -----------------------------------------------------

        if department is not None:

            df = df[
                df["Department"] == department
            ]

        # -----------------------------------------------------
        # Business unit filter
        # -----------------------------------------------------

        if business_unit is not None:

            df = df[
                df["Business_Unit"] == business_unit
            ]

        # -----------------------------------------------------
        # Month filter
        # -----------------------------------------------------

        if month is not None:

            month_value = pd.to_datetime(
                month
            )

            df = df[
                pd.to_datetime(
                    df["Month"]
                ) == month_value
            ]

        return df.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        ).reset_index(drop=True)

    # =========================================================
    # 4. CATEGORY → DEPARTMENT DRILL DOWN
    # =========================================================

    def category_department(
        self,
        category: str,
    ) -> pd.DataFrame:
        """
        Show how a category's unfavorable variance is
        distributed across departments.
        """

        df = self.by_category(
            category
        )

        result = (
            df.groupby(
                "Department",
                dropna=False,
            )
            .agg(
                Budget_Amount=(
                    "Budget_Amount",
                    "sum",
                ),
                Actual_Amount=(
                    "Actual_Amount",
                    "sum",
                ),
                Variance=(
                    "Variance",
                    "sum",
                ),
                Unfavorable_Variance=(
                    "Unfavorable_Variance",
                    "sum",
                ),
                Record_Count=(
                    "Variance",
                    "count",
                ),
            )
            .reset_index()
        )

        result = result.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        ).reset_index(drop=True)

        total = result[
            "Unfavorable_Variance"
        ].sum()

        if total > 0:

            result["Contribution_Pct"] = (
                result["Unfavorable_Variance"]
                / total
                * 100
            )

        else:

            result["Contribution_Pct"] = 0.0

        return result

    # =========================================================
    # 5. CATEGORY → BUSINESS UNIT DRILL DOWN
    # =========================================================

    def category_business_unit(
        self,
        category: str,
    ) -> pd.DataFrame:
        """
        Show how a category's unfavorable variance is
        distributed across business units.
        """

        df = self.by_category(
            category
        )

        result = (
            df.groupby(
                "Business_Unit",
                dropna=False,
            )
            .agg(
                Budget_Amount=(
                    "Budget_Amount",
                    "sum",
                ),
                Actual_Amount=(
                    "Actual_Amount",
                    "sum",
                ),
                Variance=(
                    "Variance",
                    "sum",
                ),
                Unfavorable_Variance=(
                    "Unfavorable_Variance",
                    "sum",
                ),
                Record_Count=(
                    "Variance",
                    "count",
                ),
            )
            .reset_index()
        )

        result = result.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        ).reset_index(drop=True)

        total = result[
            "Unfavorable_Variance"
        ].sum()

        if total > 0:

            result["Contribution_Pct"] = (
                result["Unfavorable_Variance"]
                / total
                * 100
            )

        else:

            result["Contribution_Pct"] = 0.0

        return result

    # =========================================================
    # 6. CATEGORY → MONTH DRILL DOWN
    # =========================================================

    def category_month(
        self,
        category: str,
    ) -> pd.DataFrame:
        """
        Show how a category's unfavorable variance
        changes over time.
        """

        df = self.by_category(
            category
        )

        result = (
            df.groupby(
                "Month",
                dropna=False,
            )
            .agg(
                Budget_Amount=(
                    "Budget_Amount",
                    "sum",
                ),
                Actual_Amount=(
                    "Actual_Amount",
                    "sum",
                ),
                Variance=(
                    "Variance",
                    "sum",
                ),
                Unfavorable_Variance=(
                    "Unfavorable_Variance",
                    "sum",
                ),
                Record_Count=(
                    "Variance",
                    "count",
                ),
            )
            .reset_index()
        )

        result = result.sort_values(
            "Month"
        ).reset_index(drop=True)

        return result

    # =========================================================
    # 7. RECURRING DRIVER ANALYSIS
    # =========================================================

    def recurring_drivers(
        self,
        min_occurrences: int = 3,
    ) -> pd.DataFrame:
        """
        Identify recurring unfavorable drivers using the
        combination of Department and Category.

        A driver is considered recurring when it has at least
        min_occurrences unfavorable variance records.
        """

        df = self.prepare_data()

        unfavorable = df[
            df["Is_Unfavorable"]
        ].copy()

        result = (
            unfavorable.groupby(
                [
                    "Department",
                    "Category",
                ],
                dropna=False,
            )
            .agg(
                Occurrences=(
                    "Variance",
                    "count",
                ),
                Unfavorable_Variance=(
                    "Unfavorable_Variance",
                    "sum",
                ),
                Average_Variance=(
                    "Variance",
                    "mean",
                ),
                Average_Variance_Pct=(
                    "Variance_Pct",
                    "mean",
                ),
                Business_Units_Affected=(
                    "Business_Unit",
                    "nunique",
                ),
                Months_Affected=(
                    "Month",
                    "nunique",
                ),
            )
            .reset_index()
        )

        result = result[
            result["Occurrences"]
            >= min_occurrences
        ].copy()

        # -----------------------------------------------------
        # Recurrence classification
        # -----------------------------------------------------

        result["Recurring_Flag"] = True

        # -----------------------------------------------------
        # Sort by financial impact
        # -----------------------------------------------------

        result = result.sort_values(
            "Unfavorable_Variance",
            ascending=False,
        ).reset_index(drop=True)

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        result.insert(
            0,
            "Rank",
            range(1, len(result) + 1),
        )

        return result

    # =========================================================
    # 8. DRIVER MATRIX
    # =========================================================

    def driver_matrix(self) -> pd.DataFrame:
        """
        Create a Department × Category matrix of unfavorable
        variance.

        Useful for identifying hotspots.
        """

        df = self.prepare_data()

        matrix = pd.pivot_table(
            df,
            values="Unfavorable_Variance",
            index="Department",
            columns="Category",
            aggfunc="sum",
            fill_value=0,
        )

        return matrix

    # =========================================================
    # 9. TOP DRIVER
    # =========================================================

    def top_driver(
        self,
    ) -> pd.Series | None:
        """
        Return the highest-value Department × Category driver.
        """

        recurring = self.recurring_drivers(
            min_occurrences=1
        )

        if recurring.empty:
            return None

        return recurring.iloc[0]

    # =========================================================
    # 10. SUMMARY
    # =========================================================

    def summary(self) -> dict:
        """
        Return high-level root-cause / driver statistics.
        """

        df = self.prepare_data()

        unfavorable = df[
            df["Is_Unfavorable"]
        ]

        return {
            "Unfavorable_Record_Count": int(
                len(unfavorable)
            ),
            "Total_Unfavorable_Variance": float(
                unfavorable[
                    "Unfavorable_Variance"
                ].sum()
            ),
            "Unique_Categories": int(
                unfavorable[
                    "Category"
                ].nunique()
            ),
            "Unique_Departments": int(
                unfavorable[
                    "Department"
                ].nunique()
            ),
            "Unique_Business_Units": int(
                unfavorable[
                    "Business_Unit"
                ].nunique()
            ),
            "Unique_Months": int(
                unfavorable[
                    "Month"
                ].nunique()
            ),
        }