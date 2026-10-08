import pandas as pd


class VarianceInterpretation:
    """
    Interprets budget variance from a financial perspective.

    Distinguishes between:
        - Overspend
        - Underspend
        - Reversal / Credit
        - On Budget

    Important:
        A negative variance is normally favorable, but a negative
        Actual_Amount is treated separately as a potential
        reversal / credit rather than ordinary underspending.
    """

    def __init__(self, variance: pd.DataFrame):
        self.variance = variance.copy()
        self.interpreted_variance = None

    # =========================================================
    # 1. VALIDATE INPUT
    # =========================================================

    def validate_input(self) -> None:
        """Validate required variance columns."""

        required_columns = [
            "Budget_Amount",
            "Actual_Amount",
            "Variance",
            "Variance_Pct",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.variance.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Variance data is missing required columns: "
                f"{missing_columns}"
            )

    # =========================================================
    # 2. CLASSIFY VARIANCE TYPE
    # =========================================================

    def classify_variance_type(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify each variance record.

        Rules:

        Actual < 0
            → Reversal/Credit

        Variance > 0
            → Overspend

        Variance < 0 and Actual >= 0
            → Underspend

        Variance == 0
            → On Budget
        """

        df = df.copy()

        df["Variance_Type"] = "On Budget"

        # -----------------------------------------------------
        # Reversal / Credit
        #
        # Negative actual expense is not treated as ordinary
        # underspending.
        # -----------------------------------------------------

        reversal_mask = (
            df["Actual_Amount"] < 0
        )

        df.loc[
            reversal_mask,
            "Variance_Type",
        ] = "Reversal/Credit"

        # -----------------------------------------------------
        # Overspend
        #
        # Only classify positive variance as overspend when
        # actual amount itself is not negative.
        # -----------------------------------------------------

        overspend_mask = (
            (df["Variance"] > 0)
            &
            (df["Actual_Amount"] >= 0)
        )

        df.loc[
            overspend_mask,
            "Variance_Type",
        ] = "Overspend"

        # -----------------------------------------------------
        # Underspend
        # -----------------------------------------------------

        underspend_mask = (
            (df["Variance"] < 0)
            &
            (df["Actual_Amount"] >= 0)
        )

        df.loc[
            underspend_mask,
            "Variance_Type",
        ] = "Underspend"

        return df

    # =========================================================
    # 3. BUSINESS INTERPRETATION
    # =========================================================

    def add_business_interpretation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add a management-oriented interpretation.
        """

        df = df.copy()

        interpretation_map = {
            "Overspend": (
                "Actual spending exceeded budget; "
                "investigate cost drivers."
            ),
            "Underspend": (
                "Actual spending was below budget; "
                "assess whether savings are sustainable."
            ),
            "Reversal/Credit": (
                "Actual amount is negative; "
                "investigate reversal, credit, refund, "
                "or accounting adjustment."
            ),
            "On Budget": (
                "Actual spending is aligned with budget."
            ),
        }

        df["Business_Interpretation"] = (
            df["Variance_Type"]
            .map(interpretation_map)
        )

        return df

    # =========================================================
    # 4. RUN
    # =========================================================

    def run(self) -> pd.DataFrame:
        """Run variance interpretation."""

        self.validate_input()

        df = self.variance.copy()

        df = self.classify_variance_type(
            df
        )

        df = self.add_business_interpretation(
            df
        )

        # -----------------------------------------------------
        # Investigation flag
        #
        # Reversals require investigation because they should
        # not automatically be interpreted as operating savings.
        # -----------------------------------------------------

        df["Requires_Investigation"] = (
            df["Variance_Type"]
            == "Reversal/Credit"
        )

        self.interpreted_variance = df

        return df

    # =========================================================
    # 5. FILTER BY TYPE
    # =========================================================

    def by_type(
        self,
        variance_type: str,
    ) -> pd.DataFrame:
        """Return records belonging to a variance type."""

        if self.interpreted_variance is None:
            self.run()

        valid_types = [
            "Overspend",
            "Underspend",
            "Reversal/Credit",
            "On Budget",
        ]

        if variance_type not in valid_types:
            raise ValueError(
                f"Invalid variance type '{variance_type}'. "
                f"Choose from: {valid_types}"
            )

        return self.interpreted_variance[
            self.interpreted_variance["Variance_Type"]
            == variance_type
        ].copy()

    # =========================================================
    # 6. INVESTIGATION QUEUE
    # =========================================================

    def investigation_queue(self) -> pd.DataFrame:
        """
        Return records requiring investigation.

        Currently this includes negative actual / reversal
        records.
        """

        if self.interpreted_variance is None:
            self.run()

        return self.interpreted_variance[
            self.interpreted_variance[
                "Requires_Investigation"
            ]
        ].copy()

    # =========================================================
    # 7. TYPE SUMMARY
    # =========================================================

    def type_summary(self) -> pd.DataFrame:
        """
        Summarize records and financial impact by variance type.
        """

        if self.interpreted_variance is None:
            self.run()

        df = self.interpreted_variance

        summary = (
            df.groupby(
                "Variance_Type",
                dropna=False,
            )
            .agg(
                Record_Count=(
                    "Variance",
                    "count",
                ),
                Total_Variance=(
                    "Variance",
                    "sum",
                ),
                Absolute_Variance=(
                    "Variance",
                    lambda x: x.abs().sum(),
                ),
                Total_Budget=(
                    "Budget_Amount",
                    "sum",
                ),
                Total_Actual=(
                    "Actual_Amount",
                    "sum",
                ),
            )
            .reset_index()
        )

        return summary

    # =========================================================
    # 8. SUMMARY
    # =========================================================

    def summary(self) -> dict:
        """Return high-level interpretation summary."""

        if self.interpreted_variance is None:
            self.run()

        df = self.interpreted_variance

        return {
            "Overspend_Records": int(
                (
                    df["Variance_Type"]
                    == "Overspend"
                ).sum()
            ),
            "Underspend_Records": int(
                (
                    df["Variance_Type"]
                    == "Underspend"
                ).sum()
            ),
            "Reversal_Credit_Records": int(
                (
                    df["Variance_Type"]
                    == "Reversal/Credit"
                ).sum()
            ),
            "On_Budget_Records": int(
                (
                    df["Variance_Type"]
                    == "On Budget"
                ).sum()
            ),
            "Investigation_Records": int(
                df["Requires_Investigation"].sum()
            ),
        }