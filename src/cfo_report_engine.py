
from datetime import datetime
import pandas as pd


class CFOReportEngine:
    """
    Generate an auditable, management-readable CFO brief from VARIA outputs.

    The report is Markdown so it can be saved, reviewed and later transformed
    into PDF / PowerPoint / email formats.
    """

    def __init__(
        self,
        variance_data: pd.DataFrame,
        warning_data: pd.DataFrame,
        cfo_queue: pd.DataFrame,
        recommendation_data: pd.DataFrame | None = None,
        scenario_summary: dict | None = None,
    ):

        self.variance = variance_data.copy()
        self.warning = warning_data.copy()
        self.cfo = cfo_queue.copy()
        self.recommendations = (
            None
            if recommendation_data is None
            else recommendation_data.copy()
        )
        self.scenario = scenario_summary

    @staticmethod
    def _money(value: float) -> str:

        value = float(value)
        sign = "-" if value < 0 else ""
        value = abs(value)

        if value >= 1e7:
            return f"{sign}₹{value / 1e7:.2f} Cr"

        if value >= 1e5:
            return f"{sign}₹{value / 1e5:.2f} L"

        if value >= 1e3:
            return f"{sign}₹{value / 1e3:.1f} K"

        return f"{sign}₹{value:,.0f}"

    @staticmethod
    def _pct(value: float) -> str:
        return f"{float(value):.2f}%"

    def generate(
        self,
        report_title: str = "VARIA CFO Brief",
    ) -> str:

        budget = self.variance[
            "Budget_Amount"
        ].sum()

        actual = self.variance[
            "Actual_Amount"
        ].sum()

        variance = self.variance[
            "Variance"
        ].sum()

        variance_pct = (
            variance / budget * 100
            if budget
            else 0
        )

        unfavorable = self.variance.loc[
            self.variance["Variance"] > 0,
            "Variance",
        ].sum()

        red = int(
            (
                self.warning["Early_Warning"]
                == "RED"
            ).sum()
        )

        amber = int(
            (
                self.warning["Early_Warning"]
                == "AMBER"
            ).sum()
        )

        green = int(
            (
                self.warning["Early_Warning"]
                == "GREEN"
            ).sum()
        )

        high_priority = self.cfo[
            self.cfo[
                "Decision_Priority"
            ].isin(
                [
                    "CRITICAL",
                    "HIGH PRIORITY",
                ]
            )
        ].copy()

        top_issues = (
            high_priority
            .sort_values(
                "Projected_Variance",
                ascending=False,
            )
            .head(5)
        )

        top_categories = (
            self.variance.assign(
                Unfavorable=(
                    self.variance["Variance"]
                    .clip(lower=0)
                )
            )
            .groupby(
                "Category",
                as_index=False,
            )["Unfavorable"]
            .sum()
            .sort_values(
                "Unfavorable",
                ascending=False,
            )
            .head(5)
        )

        generated = datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

        lines = []

        lines.append(
            f"# {report_title}"
        )
        lines.append("")
        lines.append(
            f"_Generated: {generated}_"
        )
        lines.append("")

        lines.append(
            "## 1. Executive Position"
        )
        lines.append("")
        lines.append(
            f"- Budget: **{self._money(budget)}**"
        )
        lines.append(
            f"- Actual: **{self._money(actual)}**"
        )
        lines.append(
            f"- Net variance: **{self._money(variance)} "
            f"({self._pct(variance_pct)})**"
        )
        lines.append(
            f"- Unfavorable exposure: **{self._money(unfavorable)}**"
        )
        lines.append(
            f"- Early warning: **{red} RED / {amber} AMBER / {green} GREEN**"
        )
        lines.append("")

        lines.append(
            "## 2. Largest Unfavorable Drivers"
        )
        lines.append("")

        if top_categories.empty:
            lines.append(
                "No unfavorable categories identified."
            )
        else:
            for _, row in top_categories.iterrows():
                lines.append(
                    f"- **{row['Category']}** — "
                    f"{self._money(row['Unfavorable'])}"
                )

        lines.append("")

        lines.append(
            "## 3. Top CFO Issues"
        )
        lines.append("")

        if top_issues.empty:
            lines.append(
                "No Critical / High Priority issues in the selected scope."
            )
        else:
            for _, row in top_issues.iterrows():

                lines.append(
                    f"- **{row['Business_Unit']} / "
                    f"{row['Department']} / "
                    f"{row['Category']}** — "
                    f"current {self._money(row['Current_Variance'])}; "
                    f"projected {self._money(row['Projected_Variance'])} "
                    f"({self._pct(row['Projected_Variance_Pct'])}); "
                    f"{row['Early_Warning']} / "
                    f"{row['Decision_Priority']}."
                )

        lines.append("")

        if self.recommendations is not None and not self.recommendations.empty:

            lines.append(
                "## 4. Recommended Management Actions"
            )
            lines.append("")

            priority = self.recommendations[
                self.recommendations[
                    "Action_Urgency"
                ].isin(
                    [
                        "1 — Immediate",
                        "2 — High",
                    ]
                )
            ].head(8)

            if priority.empty:
                lines.append(
                    "No immediate or high-priority recommendations."
                )
            else:
                for _, row in priority.iterrows():

                    lines.append(
                        f"- **{row['Business_Unit']} / "
                        f"{row['Department']} / "
                        f"{row['Category']}** — "
                        f"{row['Management_Action_Type']}: "
                        f"{row['Management_Recommendation']}"
                    )

            lines.append("")

        if self.scenario:

            lines.append(
                "## 5. Scenario Opportunity"
            )
            lines.append("")

            lines.append(
                f"- Base projected variance: "
                f"**{self._money(self.scenario.get('Base_Projected_Variance', 0))}**"
            )

            lines.append(
                f"- Scenario projected variance: "
                f"**{self._money(self.scenario.get('Scenario_Projected_Variance', 0))}**"
            )

            lines.append(
                f"- Variance improvement: "
                f"**{self._money(self.scenario.get('Variance_Improvement', 0))}**"
            )

            lines.append(
                f"- Forecast savings: "
                f"**{self._money(self.scenario.get('Forecast_Savings', 0))}**"
            )

            lines.append("")

        lines.append(
            "## 6. Management Interpretation"
        )
        lines.append("")

        if variance > 0:
            lines.append(
                "Overall spending is above the modeled budget position. "
                "Management attention should focus first on the largest "
                "unfavorable drivers and forecast RED issues."
            )
        elif variance < 0:
            lines.append(
                "Overall spending is below the modeled budget position. "
                "Management should distinguish sustainable savings from "
                "timing-related or delayed activity."
            )
        else:
            lines.append(
                "Overall spending is on budget at the selected scope."
            )

        return "\n".join(lines)

    def save_markdown(
        self,
        path: str,
        report_title: str = "VARIA CFO Brief",
    ) -> None:

        markdown = self.generate(
            report_title
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(markdown)
