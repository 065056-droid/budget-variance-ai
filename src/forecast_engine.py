import pandas as pd


class ForecastEngine:
    """
    Forecast and early-warning engine for VARIA.

    Builds fiscal-year level financial data from the
    monthly variance dataset and calculates forecasts
    for the latest incomplete fiscal year.
    """

    def __init__(self, variance_data: pd.DataFrame):
        self.data = variance_data.copy()

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare monthly variance data for forecasting.
        """

        df = self.data.copy()

        df["Month"] = pd.to_datetime(df["Month"])

        if "Fiscal_Year" not in df.columns:
            month = df["Month"].dt.month
            year = df["Month"].dt.year

            fiscal_start_year = year.where(
                month >= 4,
                year - 1,
            )

            df["Fiscal_Year"] = (
                "FY"
                + (fiscal_start_year % 100)
                .astype(str)
                .str.zfill(2)
                + "-"
                + ((fiscal_start_year + 1) % 100)
                .astype(str)
                .str.zfill(2)
            )

        # April = 1, May = 2, ..., March = 12
        df["Fiscal_Month_Number"] = (
            (df["Month"].dt.month - 4) % 12
        ) + 1

        df = df.sort_values(
            [
                "Fiscal_Year",
                "Business_Unit",
                "Department",
                "Category",
                "Month",
            ]
        ).reset_index(drop=True)

        return df

    def annual_summary(self) -> pd.DataFrame:
        """
        Aggregate monthly budget and actuals to fiscal-year level.
        """

        df = self.prepare_data()

        group_columns = [
            "Fiscal_Year",
            "Business_Unit",
            "Department",
            "Category",
        ]

        annual = (
            df.groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                Annual_Budget=(
                    "Budget_Amount",
                    "sum",
                ),
                Annual_Actual=(
                    "Actual_Amount",
                    "sum",
                ),
                Months_Available=(
                    "Month",
                    "nunique",
                ),
            )
        )

        annual["Annual_Variance"] = (
            annual["Annual_Actual"]
            - annual["Annual_Budget"]
        )

        annual["Annual_Variance_Pct"] = (
            annual["Annual_Variance"]
            / annual["Annual_Budget"]
            * 100
        )

        return annual

    def fiscal_year_summary(self) -> pd.DataFrame:
        """
        Return high-level fiscal-year totals.
        """

        df = self.prepare_data()

        summary = (
            df.groupby(
                "Fiscal_Year",
                as_index=False,
            )
            .agg(
                Budget=(
                    "Budget_Amount",
                    "sum",
                ),
                Actual=(
                    "Actual_Amount",
                    "sum",
                ),
                Months_Available=(
                    "Month",
                    "nunique",
                ),
            )
        )

        summary["Variance"] = (
            summary["Actual"]
            - summary["Budget"]
        )

        summary["Variance_Pct"] = (
            summary["Variance"]
            / summary["Budget"]
            * 100
        )

        return summary

    def latest_fiscal_year(self) -> str:
        """
        Identify the latest fiscal year in the dataset.
        """

        df = self.prepare_data()

        return df["Fiscal_Year"].max()

    def estimate_full_year_budget(self) -> pd.DataFrame:
        """
        Estimate the full-year budget for the latest incomplete
        fiscal year.

        Method:
        1. Use FY25-26 as the latest completed fiscal year.
        2. Calculate annual budget growth from FY24-25 to FY25-26
           at Business Unit + Department + Category level.
        3. Apply that growth rate to the corresponding
           Jan-Mar FY25-26 monthly budgets.
        4. Use actual FY26-27 budgets for months already available.
        """

        df = self.prepare_data()

        latest_fy = self.latest_fiscal_year()

        historical_fys = sorted(
            df["Fiscal_Year"].unique()
        )

        if "FY24-25" not in historical_fys:
            raise ValueError(
                "FY24-25 is required for budget growth calculation."
            )

        if "FY25-26" not in historical_fys:
            raise ValueError(
                "FY25-26 is required as the latest completed fiscal year."
            )

        previous_fy = "FY24-25"
        completed_fy = "FY25-26"

        group_columns = [
            "Business_Unit",
            "Department",
            "Category",
        ]

        # -----------------------------------------------------
        # HISTORICAL ANNUAL BUDGETS
        # -----------------------------------------------------

        previous_annual = (
            df[
                df["Fiscal_Year"] == previous_fy
            ]
            .groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                Previous_Annual_Budget=(
                    "Budget_Amount",
                    "sum",
                )
            )
        )

        completed_annual = (
            df[
                df["Fiscal_Year"] == completed_fy
            ]
            .groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                Completed_Annual_Budget=(
                    "Budget_Amount",
                    "sum",
                )
            )
        )

        growth = previous_annual.merge(
            completed_annual,
            on=group_columns,
            how="inner",
        )

        # Avoid division by zero.
        growth["Budget_Growth_Rate"] = (
            growth["Completed_Annual_Budget"]
            / growth["Previous_Annual_Budget"]
            - 1
        ).where(
            growth["Previous_Annual_Budget"] != 0
        )

        # -----------------------------------------------------
        # CURRENT FY DATA
        # -----------------------------------------------------

        current = df[
            df["Fiscal_Year"] == latest_fy
        ].copy()

        latest_available_month = (
            current["Fiscal_Month_Number"].max()
        )

        current_grouped = (
            current.groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                YTD_Budget=(
                    "Budget_Amount",
                    "sum",
                ),
                YTD_Actual=(
                    "Actual_Amount",
                    "sum",
                ),
            )
        )

        # -----------------------------------------------------
        # MISSING JAN-MAR BUDGET
        # -----------------------------------------------------

        missing_months = list(
            range(
                latest_available_month + 1,
                13,
            )
        )

        historical_remaining = df[
            (
                df["Fiscal_Year"]
                == completed_fy
            )
            &
            (
                df["Fiscal_Month_Number"].isin(
                    missing_months
                )
            )
        ].copy()

        historical_remaining = historical_remaining.merge(
            growth[
                group_columns
                + ["Budget_Growth_Rate"]
            ],
            on=group_columns,
            how="left",
        )

        historical_remaining[
            "Estimated_Budget"
        ] = (
            historical_remaining[
                "Budget_Amount"
            ]
            * (
                1
                + historical_remaining[
                    "Budget_Growth_Rate"
                ].fillna(0)
            )
        )

        # -----------------------------------------------------
        # ESTIMATED REMAINING BUDGET
        # -----------------------------------------------------

        remaining_budget = (
            historical_remaining
            .groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                Estimated_Remaining_Budget=(
                    "Estimated_Budget",
                    "sum",
                )
            )
        )

        result = current_grouped.merge(
            remaining_budget,
            on=group_columns,
            how="left",
        )

        result[
            "Estimated_Remaining_Budget"
        ] = result[
            "Estimated_Remaining_Budget"
        ].fillna(0)

        result["Estimated_Full_Year_Budget"] = (
            result["YTD_Budget"]
            + result["Estimated_Remaining_Budget"]
        )

        result["Latest_Fiscal_Year"] = latest_fy

        result["Months_Elapsed"] = (
            latest_available_month
        )

        result["Remaining_Months"] = (
            12 - latest_available_month
        )

        return result

    def ytd_forecast(self) -> pd.DataFrame:
        """
        Calculate YTD position and forecast at completion.
        """

        result = self.estimate_full_year_budget()

        result["Monthly_Run_Rate"] = (
            result["YTD_Actual"]
            / result["Months_Elapsed"]
        )

        result["Forecast_Remaining_Spend"] = (
            result["Monthly_Run_Rate"]
            * result["Remaining_Months"]
        )

        result["Forecast_At_Completion"] = (
            result["YTD_Actual"]
            + result["Forecast_Remaining_Spend"]
        )

        result["Projected_Variance"] = (
            result["Forecast_At_Completion"]
            - result["Estimated_Full_Year_Budget"]
        )

        result["Projected_Variance_Pct"] = (
            result["Projected_Variance"]
            / result["Estimated_Full_Year_Budget"]
            * 100
        )

        return result

    def classify_early_warning(
        self,
        forecast_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Classify projected financial risk.

        GREEN:
            Absolute projected variance < 5%

        AMBER:
            Absolute projected variance >= 5% and < 10%

        RED:
            Absolute projected variance >= 10%
        """

        if forecast_data is None:
            forecast_data = self.ytd_forecast()

        df = forecast_data.copy()

        df["Early_Warning"] = "GREEN"

        absolute_variance_pct = (
            df["Projected_Variance_Pct"].abs()
        )

        df.loc[
            absolute_variance_pct >= 5,
            "Early_Warning",
        ] = "AMBER"

        df.loc[
            absolute_variance_pct >= 10,
            "Early_Warning",
        ] = "RED"

        return df

    def early_warning_summary(
        self,
        forecast_data: pd.DataFrame | None = None,
    ) -> dict:
        """
        Summarize forecast exposure by warning level.
        """

        if forecast_data is None:
            forecast_data = (
                self.classify_early_warning()
            )

        df = forecast_data.copy()

        red = df[
            df["Early_Warning"] == "RED"
        ]

        amber = df[
            df["Early_Warning"] == "AMBER"
        ]

        green = df[
            df["Early_Warning"] == "GREEN"
        ]

        return {
            "Total_Records": len(df),
            "RED_Records": len(red),
            "AMBER_Records": len(amber),
            "GREEN_Records": len(green),
            "RED_Projected_Variance": red[
                "Projected_Variance"
            ].sum(),
            "AMBER_Projected_Variance": amber[
                "Projected_Variance"
            ].sum(),
            "Total_Projected_Variance": df[
                "Projected_Variance"
            ].sum(),
        }
        
    def summary(self) -> dict:
        """
        Return high-level forecast data summary.
        """

        annual = self.annual_summary()

        return {
            "Fiscal_Years": annual[
                "Fiscal_Year"
            ].nunique(),
            "Annual_Combinations": len(annual),
            "Total_Annual_Budget": annual[
                "Annual_Budget"
            ].sum(),
            "Total_Annual_Actual": annual[
                "Annual_Actual"
            ].sum(),
        }