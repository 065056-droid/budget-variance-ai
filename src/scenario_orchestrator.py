from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.action_sizing_engine import ActionSizingEngine
from src.budget_reallocation_engine import BudgetReallocationEngine
from src.scenario_engine import ScenarioEngine
from src.sensitivity_engine import SensitivityEngine


@dataclass(frozen=True)
class ScenarioRunConfig:
    name: str
    future_spend_change_pct: float = 0.0
    budget_change_pct: float = 0.0
    scope: dict[str, Any] | None = None
    cap_forecast_at_budget: bool = False


class ScenarioOrchestrator:
    """
    Decision layer that coordinates VARIA's existing what-if engines.

    It does not alter historical actuals or source data. It combines:
      - ScenarioEngine: spend/budget what-if scenarios
      - SensitivityEngine: driver-level sensitivity ranking
      - BudgetReallocationEngine: budget transfer between scopes
      - ActionSizingEngine: quantify the size of required intervention

    The orchestrator produces comparable outputs so the UI can present a
    single decision simulator instead of exposing four disconnected engines.
    """

    REQUIRED_FORECAST_COLUMNS = [
        "Business_Unit",
        "Department",
        "Category",
        "YTD_Actual",
        "Forecast_Remaining_Spend",
        "Forecast_At_Completion",
        "Estimated_Full_Year_Budget",
        "Projected_Variance",
        "Projected_Variance_Pct",
    ]

    def __init__(self, forecast_data: pd.DataFrame):
        if not isinstance(forecast_data, pd.DataFrame):
            raise TypeError("forecast_data must be a pandas DataFrame.")

        missing = [
            c for c in self.REQUIRED_FORECAST_COLUMNS
            if c not in forecast_data.columns
        ]
        if missing:
            raise ValueError(
                "Forecast data is missing required columns: "
                + ", ".join(missing)
            )

        self.forecast_data = forecast_data.copy()
        self.scenario_engine = ScenarioEngine(self.forecast_data)
        self.sensitivity_engine = SensitivityEngine(self.forecast_data)
        self.reallocation_engine = BudgetReallocationEngine(self.forecast_data)
        self.action_sizing_engine = ActionSizingEngine(self.forecast_data)

    @staticmethod
    def _validate_scenarios(
        scenarios: list[ScenarioRunConfig | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        for index, scenario in enumerate(scenarios, start=1):
            if isinstance(scenario, ScenarioRunConfig):
                payload = {
                    "name": scenario.name,
                    "future_spend_change_pct": scenario.future_spend_change_pct,
                    "budget_change_pct": scenario.budget_change_pct,
                    "scope": scenario.scope,
                    "cap_forecast_at_budget": scenario.cap_forecast_at_budget,
                }
            elif isinstance(scenario, dict):
                payload = dict(scenario)
            else:
                raise TypeError(
                    f"Scenario {index} must be a dict or ScenarioRunConfig."
                )

            name = str(payload.get("name", f"Scenario {index}"))
            spend = float(payload.get("future_spend_change_pct", 0.0))
            budget = float(payload.get("budget_change_pct", 0.0))

            if spend < -100:
                raise ValueError(
                    f"Scenario '{name}' cannot reduce future spend by more than 100%."
                )

            normalized.append(
                {
                    "name": name,
                    "future_spend_change_pct": spend,
                    "budget_change_pct": budget,
                    "scope": payload.get("scope"),
                    "cap_forecast_at_budget": bool(
                        payload.get("cap_forecast_at_budget", False)
                    ),
                }
            )

        if not normalized:
            raise ValueError("At least one scenario is required.")

        return normalized

    def run_scenario_library(
        self,
        scenarios: list[ScenarioRunConfig | dict[str, Any]],
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
        """Run named scenarios and identify the best variance outcome."""
        configs = self._validate_scenarios(scenarios)
        comparison = self.scenario_engine.multi_scenario(configs)

        if comparison.empty:
            raise ValueError("Scenario library produced no results.")

        # Lower projected variance is financially better.
        ranked = comparison.sort_values(
            ["Scenario_Projected_Variance", "Variance_Improvement"],
            ascending=[True, False],
        ).reset_index(drop=True)
        ranked["Outcome_Rank"] = range(1, len(ranked) + 1)

        best = ranked.iloc[0].to_dict()

        return comparison, best, ranked

    def run_sensitivity(
        self,
        dimension: str = "Category",
        spend_reduction_pct: float = 10.0,
        top_n: int = 10,
    ) -> pd.DataFrame:
        return self.sensitivity_engine.tornado_data(
            dimension=dimension,
            spend_reduction_pct=spend_reduction_pct,
            top_n=top_n,
        )

    def run_reallocation(
        self,
        source_scope: dict[str, Any],
        target_scope: dict[str, Any],
        amount: float,
        scenario_name: str = "Budget Reallocation",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        result = self.reallocation_engine.run(
            source_scope=source_scope,
            target_scope=target_scope,
            amount=amount,
            scenario_name=scenario_name,
        )
        return result, self.reallocation_engine.summary(result)

    def action_sizing(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        result = self.action_sizing_engine.run()
        return result, self.action_sizing_engine.summary(result)

    def decision_pack(
        self,
        scenarios: list[ScenarioRunConfig | dict[str, Any]],
        sensitivity_dimension: str = "Category",
        sensitivity_reduction_pct: float = 10.0,
        sensitivity_top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Create one structured decision pack for the dashboard/report layer.
        """
        comparison, best, ranked = self.run_scenario_library(scenarios)
        sensitivity = self.run_sensitivity(
            dimension=sensitivity_dimension,
            spend_reduction_pct=sensitivity_reduction_pct,
            top_n=sensitivity_top_n,
        )
        action_data, action_summary = self.action_sizing()

        base_variance = float(
            self.forecast_data["Projected_Variance"].sum()
        )

        return {
            "base": {
                "Projected_Variance": base_variance,
                "Projected_Overrun": max(base_variance, 0.0),
                "Projected_Underspend": abs(min(base_variance, 0.0)),
            },
            "scenario_comparison": comparison,
            "scenario_ranking": ranked,
            "recommended_scenario": best,
            "sensitivity": sensitivity,
            "action_sizing": action_data,
            "action_summary": action_summary,
        }

    def compare_scenario_to_base(
        self,
        scenario: ScenarioRunConfig | dict[str, Any],
    ) -> dict[str, Any]:
        comparison, best, _ = self.run_scenario_library([scenario])
        row = comparison.iloc[0]
        return {
            "Scenario_Name": row["Scenario_Name"],
            "Base_Projected_Variance": float(
                row["Base_Projected_Variance"]
            ),
            "Scenario_Projected_Variance": float(
                row["Scenario_Projected_Variance"]
            ),
            "Variance_Improvement": float(
                row["Variance_Improvement"]
            ),
            "Forecast_Savings": float(row["Forecast_Savings"]),
            "Budget_Headroom_Change": float(
                row["Budget_Headroom_Change"]
            ),
            "Scenario_Status": (
                "Improved"
                if best["Variance_Improvement"] > 0
                else "No Improvement"
            ),
        }
