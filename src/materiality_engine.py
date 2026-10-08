import pandas as pd


class MaterialityEngine:
    """
    Identifies financially material budget variances
    and assigns management-review severity.
    """

    def __init__(
        self,
        variance: pd.DataFrame,
        amount_threshold: float = 100_000,
        percentage_threshold: float = 10.0,
        high_amount: float = 500_000,
        high_percentage: float = 20.0,
        critical_amount: float = 1_000_000,
        critical_percentage: float = 30.0,
    ):
        self.variance = variance.copy()

        self.amount_threshold = amount_threshold
        self.percentage_threshold = percentage_threshold

        self.high_amount = high_amount
        self.high_percentage = high_percentage

        self.critical_amount = critical_amount
        self.critical_percentage = critical_percentage

        self.materiality = None

    # ---------------------------------------------------------
    # RUN MATERIALITY ENGINE
    # ---------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Calculate variance materiality and severity."""

        df = self.variance.copy()

        # -----------------------------------------------------
        # 1. ABSOLUTE VARIANCE
        # -----------------------------------------------------

        df["Absolute_Variance"] = (
            df["Variance"].abs()
        )

        df["Absolute_Variance_Pct"] = (
            df["Variance_Pct"].abs()
        )

        # -----------------------------------------------------
        # 2. MATERIALITY
        #
        # Material if either the absolute amount threshold
        # OR percentage threshold is exceeded.
        # -----------------------------------------------------

        df["Is_Material"] = (
            (
                df["Absolute_Variance"]
                >= self.amount_threshold
            )
            |
            (
                df["Absolute_Variance_Pct"]
                >= self.percentage_threshold
            )
        )

        # -----------------------------------------------------
        # 3. DEFAULT SEVERITY
        # -----------------------------------------------------

        df["Severity"] = "LOW"

        # -----------------------------------------------------
        # 4. MEDIUM
        # -----------------------------------------------------

        medium_mask = df["Is_Material"]

        df.loc[
            medium_mask,
            "Severity",
        ] = "MEDIUM"

        # -----------------------------------------------------
        # 5. HIGH
        #
        # HIGH if either amount OR percentage threshold
        # is crossed.
        # -----------------------------------------------------

        high_mask = (
            (
                df["Absolute_Variance"]
                >= self.high_amount
            )
            |
            (
                df["Absolute_Variance_Pct"]
                >= self.high_percentage
            )
        )

        df.loc[
            high_mask,
            "Severity",
        ] = "HIGH"

        # -----------------------------------------------------
        # 6. CRITICAL
        #
        # CRITICAL only when BOTH thresholds are crossed.
        # -----------------------------------------------------

        critical_mask = (
            (
                df["Absolute_Variance"]
                >= self.critical_amount
            )
            &
            (
                df["Absolute_Variance_Pct"]
                >= self.critical_percentage
            )
        )

        df.loc[
            critical_mask,
            "Severity",
        ] = "CRITICAL"

        # -----------------------------------------------------
        # 7. MANAGEMENT REVIEW
        # -----------------------------------------------------

        df["Management_Review"] = (
            df["Is_Material"]
        )

        self.materiality = df

        return df

    # ---------------------------------------------------------
    # MATERIAL VARIANCES
    # ---------------------------------------------------------

    def material_variances(self) -> pd.DataFrame:
        """Return only financially material variances."""

        if self.materiality is None:
            self.run()

        return self.materiality[
            self.materiality["Is_Material"]
        ].copy()

    # ---------------------------------------------------------
    # MANAGEMENT REVIEW QUEUE
    # ---------------------------------------------------------

    def management_review_queue(self) -> pd.DataFrame:
        """Return material variances requiring management review."""

        if self.materiality is None:
            self.run()

        return self.materiality[
            self.materiality["Management_Review"]
        ].copy()

    # ---------------------------------------------------------
    # SEVERITY SUMMARY
    # ---------------------------------------------------------

    def severity_summary(self) -> pd.Series:
        """Return count of records by severity."""

        if self.materiality is None:
            self.run()

        return (
            self.materiality["Severity"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                ],
                fill_value=0,
            )
        )