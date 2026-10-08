from __future__ import annotations

import json
from typing import Any

import pandas as pd


class CFOEvidencePackEngine:
    """
    Build a compact, deterministic evidence pack for VARIA's grounded LLM.

    The pack deliberately carries semantic metadata so the LLM does not have
    to infer whether a financial number is an overspend, underspend, forecast,
    savings amount, or reconciliation exception.
    """

    KEY_COLS = ["Business_Unit", "Department", "Category"]

    def __init__(
        self,
        outputs: dict[str, Any],
        top_driver_n: int = 5,
        top_issue_n: int = 5,
    ) -> None:
        self.outputs = outputs
        self.top_driver_n = max(1, int(top_driver_n))
        self.top_issue_n = max(1, int(top_issue_n))

    @staticmethod
    def _frame(outputs: dict[str, Any], key: str) -> pd.DataFrame:
        value = outputs.get(key)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

    @staticmethod
    def _num(value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return str(value)

    @staticmethod
    def _direction_for_variance(value: Any) -> str:
        number = CFOEvidencePackEngine._num(value)
        if number is None or abs(number) < 1e-12:
            return "ON_BUDGET"
        return "UNFAVORABLE" if number > 0 else "FAVORABLE"

    @classmethod
    def _ev(
        cls,
        eid: str,
        section: str,
        field: str,
        value: Any,
        source: str,
        semantic_type: str | None = None,
        direction: str | None = None,
        supports_cause: bool = False,
        supports_trend: bool = False,
    ) -> dict[str, Any]:
        """Create one auditable evidence record with explicit semantics."""
        semantic = semantic_type or "FACT"
        resolved_direction = direction or "NOT_APPLICABLE"
        normalized_field = str(field).replace("_", " ").strip().lower()

        # Financial semantics are determined by VARIA, not by the LLM.
        # Normalize underscores because scenario DataFrame columns use names
        # such as Projected_Variance while executive fields use display names.
        if normalized_field in {"current variance", "current variance %"}:
            semantic = (
                "CURRENT_VARIANCE"
                if normalized_field == "current variance"
                else "CURRENT_VARIANCE_PCT"
            )
            resolved_direction = cls._direction_for_variance(value)
        elif normalized_field == "unfavorable variance":
            semantic = "UNFAVORABLE_OVERSPEND"
            resolved_direction = "UNFAVORABLE"
        elif normalized_field == "projected variance":
            semantic = "PROJECTED_VARIANCE"
            resolved_direction = cls._direction_for_variance(value)
        elif normalized_field == "scenario projected variance":
            semantic = "SCENARIO_PROJECTED_VARIANCE"
            resolved_direction = cls._direction_for_variance(value)
        elif normalized_field == "forecast savings":
            semantic = "SCENARIO_SAVINGS"
            resolved_direction = "SAVINGS"
        elif normalized_field == "variance improvement":
            semantic = "SCENARIO_IMPROVEMENT"
            resolved_direction = "IMPROVEMENT"
        elif normalized_field == "projected overspend records":
            semantic = "PROJECTED_OVERSPEND_RECORD_COUNT"
            resolved_direction = "UNFAVORABLE"
        elif normalized_field == "projected underspend records":
            semantic = "PROJECTED_UNDERSPEND_RECORD_COUNT"
            resolved_direction = "FAVORABLE"
        elif normalized_field == "on budget records":
            semantic = "ON_BUDGET_RECORD_COUNT"
            resolved_direction = "ON_BUDGET"
        elif normalized_field == "exception amount":
            semantic = "PLANNING_EXCEPTION_AMOUNT"
            resolved_direction = "NOT_APPLICABLE"
        elif normalized_field == "exception transactions":
            semantic = "PLANNING_EXCEPTION_COUNT"
            resolved_direction = "NOT_APPLICABLE"
        elif normalized_field == "exception types":
            semantic = "PLANNING_EXCEPTION_TYPE"
            resolved_direction = "NOT_APPLICABLE"

        return {
            "id": eid,
            "section": section,
            "field": field,
            "value": value,
            "source": source,
            "semantic_type": semantic,
            "direction": resolved_direction,
            "supports_cause": bool(supports_cause),
            "supports_trend": bool(supports_trend),
        }

    def _executive(self, variance: pd.DataFrame):
        required = {"Budget_Amount", "Actual_Amount", "Variance"}
        if variance.empty or not required.issubset(variance.columns):
            return (
                {
                    "status": "UNAVAILABLE",
                    "reason": "Variance output is unavailable or incomplete.",
                },
                [],
            )

        budget = float(variance["Budget_Amount"].sum())
        actual = float(variance["Actual_Amount"].sum())
        current_variance = float(variance["Variance"].sum())
        pct = current_variance / budget * 100 if budget else 0.0

        position = {
            "budget_total": budget,
            "actual_total": actual,
            "current_variance": current_variance,
            "current_variance_pct": pct,
        }

        materiality = self._frame(self.outputs, "materiality")
        forecast = self._frame(self.outputs, "forecast")
        warning = self._frame(self.outputs, "warning")
        risk = self._frame(self.outputs, "risk")

        if "Is_Material" in materiality:
            position["material_variance_count"] = int(
                materiality["Is_Material"].fillna(False).sum()
            )
        if "Projected_Variance" in forecast:
            position["total_projected_variance"] = float(
                forecast["Projected_Variance"].sum()
            )
        if "Early_Warning" in warning:
            position["early_warning_counts"] = (
                warning["Early_Warning"]
                .value_counts(dropna=False)
                .astype(int)
                .to_dict()
            )
        if "Risk_Band" in risk:
            position["risk_band_counts"] = (
                risk["Risk_Band"]
                .value_counts(dropna=False)
                .astype(int)
                .to_dict()
            )

        evidence = [
            self._ev(
                "EXEC-001",
                "Executive Position",
                "Budget Total",
                budget,
                "variance.Budget_Amount.sum",
                semantic_type="BUDGET_TOTAL",
            ),
            self._ev(
                "EXEC-002",
                "Executive Position",
                "Actual Total",
                actual,
                "variance.Actual_Amount.sum",
                semantic_type="ACTUAL_TOTAL",
            ),
            self._ev(
                "EXEC-003",
                "Executive Position",
                "Current Variance",
                current_variance,
                "variance.Variance.sum",
            ),
            self._ev(
                "EXEC-004",
                "Executive Position",
                "Current Variance %",
                pct,
                "derived from EXEC-001 and EXEC-003",
            ),
        ]
        return position, evidence

    def _drivers(self, variance: pd.DataFrame):
        required = {"Category", "Variance"}
        if variance.empty or not required.issubset(variance.columns):
            return [], []

        x = variance.copy()
        x["Unfavorable_Variance"] = x["Variance"].clip(lower=0)
        grouped = (
            x.groupby("Category", dropna=False, as_index=False)
            .agg(
                Unfavorable_Variance=("Unfavorable_Variance", "sum"),
                Affected_Records=(
                    "Unfavorable_Variance",
                    lambda s: int((s > 0).sum()),
                ),
            )
            .query("Unfavorable_Variance > 0")
            .sort_values("Unfavorable_Variance", ascending=False)
            .head(self.top_driver_n)
            .reset_index(drop=True)
        )

        total = float(grouped["Unfavorable_Variance"].sum())
        output: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        for index, row in grouped.iterrows():
            rank = index + 1
            category = self._text(row["Category"])
            amount = float(row["Unfavorable_Variance"])
            share = amount / total * 100 if total else 0.0
            affected = int(row["Affected_Records"])

            output.append(
                {
                    "rank": rank,
                    "category": category,
                    "unfavorable_variance": amount,
                    "contribution_pct_within_top_set": share,
                    "affected_records": affected,
                }
            )

            evidence.extend(
                [
                    self._ev(
                        f"DRV-{rank:03d}-A",
                        "Top Unfavorable Drivers",
                        "Category",
                        category,
                        "variance.Category",
                        semantic_type="DRIVER_CATEGORY",
                    ),
                    self._ev(
                        f"DRV-{rank:03d}-B",
                        "Top Unfavorable Drivers",
                        "Unfavorable Variance",
                        amount,
                        "variance.Variance.clip(lower=0).groupby(Category).sum",
                    ),
                    self._ev(
                        f"DRV-{rank:03d}-C",
                        "Top Unfavorable Drivers",
                        "Affected Records",
                        affected,
                        "derived from variance",
                        semantic_type="DRIVER_AFFECTED_RECORDS",
                    ),
                ]
            )

        return output, evidence

    def _lookup(self, frame: pd.DataFrame, row: pd.Series) -> pd.Series | None:
        if frame.empty or not set(self.KEY_COLS).issubset(frame.columns):
            return None

        mask = pd.Series(True, index=frame.index)
        for column in self.KEY_COLS:
            value = row.get(column)
            if pd.isna(value):
                mask &= frame[column].isna()
            else:
                mask &= frame[column].eq(value)

        matches = frame.loc[mask]
        return matches.iloc[0] if not matches.empty else None

    def _issues(self):
        forecast = self._frame(self.outputs, "forecast")
        if forecast.empty or not set(self.KEY_COLS).issubset(forecast.columns):
            return [], []

        x = forecast.copy()
        x["Issue_Amount"] = (
            x["Projected_Variance"].abs()
            if "Projected_Variance" in x
            else 0
        )

        rank_map = {"RED": 3, "AMBER": 2, "GREEN": 1}
        x["Warning_Rank"] = (
            x["Early_Warning"].map(rank_map).fillna(0)
            if "Early_Warning" in x
            else 0
        )
        x = (
            x.sort_values(
                ["Warning_Rank", "Issue_Amount"],
                ascending=[False, False],
            )
            .head(self.top_issue_n)
            .reset_index(drop=True)
        )

        risk = self._frame(self.outputs, "risk")
        recommendations = self._frame(self.outputs, "recommendations")
        action = self._frame(self.outputs, "action_sizing")

        output: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        for index, row in x.iterrows():
            rank = index + 1
            scope = {c: self._text(row.get(c)) for c in self.KEY_COLS}
            item: dict[str, Any] = {
                "rank": rank,
                "scope": scope,
                "projected_variance": self._num(row.get("Projected_Variance")),
                "projected_variance_pct": self._num(
                    row.get("Projected_Variance_Pct")
                ),
                "early_warning": self._text(row.get("Early_Warning")),
            }

            risk_row = self._lookup(risk, row)
            rec_row = self._lookup(recommendations, row)
            action_row = self._lookup(action, row)

            if risk_row is not None:
                item["risk_score"] = self._num(risk_row.get("Risk_Score"))
                item["risk_band"] = self._text(risk_row.get("Risk_Band"))
            if rec_row is not None:
                item["management_recommendation"] = self._text(
                    rec_row.get("Management_Recommendation")
                )
            if action_row is not None and "Required_Cost_Control" in action:
                item["required_cost_control"] = self._num(
                    action_row.get("Required_Cost_Control")
                )

            output.append(item)

            evidence.extend(
                [
                    self._ev(
                        f"ISS-{rank:03d}-A",
                        "Top CFO Issues",
                        "Scope",
                        scope,
                        "forecast planning grain",
                        semantic_type="ISSUE_SCOPE",
                    ),
                    self._ev(
                        f"ISS-{rank:03d}-B",
                        "Top CFO Issues",
                        "Projected Variance",
                        item["projected_variance"],
                        "forecast.Projected_Variance",
                    ),
                    self._ev(
                        f"ISS-{rank:03d}-C",
                        "Top CFO Issues",
                        "Projected Variance %",
                        item["projected_variance_pct"],
                        "forecast.Projected_Variance_Pct",
                    ),
                    self._ev(
                        f"ISS-{rank:03d}-D",
                        "Top CFO Issues",
                        "Early Warning",
                        item["early_warning"],
                        "forecast.Early_Warning",
                        semantic_type="EARLY_WARNING",
                    ),
                ]
            )

            if "risk_score" in item:
                evidence.append(
                    self._ev(
                        f"ISS-{rank:03d}-E",
                        "Top CFO Issues",
                        "Risk Score",
                        item["risk_score"],
                        "risk.Risk_Score",
                        semantic_type="RISK_SCORE",
                    )
                )
            if "risk_band" in item:
                evidence.append(
                    self._ev(
                        f"ISS-{rank:03d}-F",
                        "Top CFO Issues",
                        "Risk Band",
                        item["risk_band"],
                        "risk.Risk_Band",
                        semantic_type="RISK_BAND",
                    )
                )
            if item.get("management_recommendation"):
                evidence.append(
                    self._ev(
                        f"ISS-{rank:03d}-G",
                        "Top CFO Issues",
                        "Management Recommendation",
                        item["management_recommendation"],
                        "management_recommendation_engine",
                        semantic_type="MANAGEMENT_RECOMMENDATION",
                    )
                )

        return output, evidence

    def _scenarios(self):
        ranking = self._frame(self.outputs, "scenario_ranking")
        if ranking.empty:
            return [], []

        columns = [
            "Scenario_Name",
            "Base_Projected_Variance",
            "Scenario_Projected_Variance",
            "Variance_Improvement",
            "Forecast_Savings",
            "Budget_Headroom_Change",
            "Projected_Overspend_Records",
            "Projected_Underspend_Records",
            "On_Budget_Records",
            "Outcome_Rank",
        ]

        output: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        for index, row in ranking.iterrows():
            rank = index + 1
            item: dict[str, Any] = {}
            for column in columns:
                if column not in ranking.columns:
                    continue

                value = row[column]
                if column == "Scenario_Name":
                    item[column] = self._text(value)
                elif column in {
                    "Projected_Overspend_Records",
                    "Projected_Underspend_Records",
                    "On_Budget_Records",
                    "Outcome_Rank",
                }:
                    number = self._num(value)
                    item[column] = int(number) if number is not None else None
                else:
                    item[column] = self._num(value)

                evidence.append(
                    self._ev(
                        f"SCN-{rank:03d}-{column}",
                        "Scenario Results",
                        column,
                        item[column],
                        "scenario_ranking",
                    )
                )

            output.append(item)

        recommended = self.outputs.get("recommended_scenario")
        if isinstance(recommended, dict) and recommended.get("Scenario_Name"):
            name = self._text(recommended["Scenario_Name"])
            output.insert(0, {"Recommended_Scenario": name})
            evidence.insert(
                0,
                self._ev(
                    "SCN-REC-001",
                    "Scenario Results",
                    "Recommended Scenario",
                    name,
                    "recommended_scenario",
                    semantic_type="RECOMMENDED_SCENARIO",
                ),
            )

        return output, evidence

    def _planning(self):
        summary = self.outputs.get("planning_summary")
        if not isinstance(summary, dict):
            return {"status": "UNAVAILABLE"}, []

        evidence = [
            self._ev(
                "PLAN-001",
                "Planning Exceptions",
                "Exception Transactions",
                summary.get("Exception Transactions"),
                "planning_summary",
            ),
            self._ev(
                "PLAN-002",
                "Planning Exceptions",
                "Exception Amount",
                summary.get("Exception Amount"),
                "planning_summary",
            ),
            self._ev(
                "PLAN-003",
                "Planning Exceptions",
                "Exception Types",
                summary.get("Exception Types"),
                "planning_summary",
            ),
        ]
        return summary, evidence

    def build(self) -> dict[str, Any]:
        variance = self._frame(self.outputs, "variance")
        executive, executive_evidence = self._executive(variance)
        drivers, driver_evidence = self._drivers(variance)
        issues, issue_evidence = self._issues()
        scenarios, scenario_evidence = self._scenarios()
        planning, planning_evidence = self._planning()

        ledger = (
            executive_evidence
            + driver_evidence
            + issue_evidence
            + scenario_evidence
            + planning_evidence
        )

        pack = {
            "pack_version": "2.0",
            "purpose": "Compact verified evidence for grounded CFO commentary.",
            "principle": (
                "VARIA decides WHAT matters; the LLM only communicates the supplied evidence."
            ),
            "verified_financial_rules": {
                "expense_direction_rule": (
                    "positive variance = unfavorable / overspend; "
                    "negative variance = favorable / underspend"
                ),
                "planning_exception_rule": (
                    "a planning exception amount is reconciliation exposure and "
                    "must not be treated as operating overspend without explicit evidence"
                ),
                "cause_rule": (
                    "causes are not established by the evidence pack unless an evidence record explicitly supports them"
                ),
                "trend_rule": (
                    "trend/continuation claims require evidence records with supports_trend=true"
                ),
            },
            "executive_position": executive,
            "top_unfavorable_drivers": drivers,
            "top_cfo_issues": issues,
            "scenario_results": scenarios,
            "planning_exceptions": planning,
            "evidence_ledger": ledger,
            "llm_rules": [
                "Use only supplied evidence.",
                "Cite factual statements with evidence IDs.",
                "Do not invent causes.",
                "Do not calculate new financial figures.",
                "Do not reverse overspend versus underspend.",
                "Do not treat planning exceptions as overspend.",
                "Recommendations must be labeled as recommendations.",
                "When cause is unknown, state that the cause cannot be established from the available evidence.",
            ],
        }

        # Serialization check: fail immediately if the evidence pack itself is
        # not JSON serializable, because it is intended for a JSON-grounded LLM.
        json.dumps(pack, default=str)
        return pack

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.build(), indent=indent, default=str)
