import numpy as np

import pandas as pd

from src.data_cleaner import DataCleaner
from src.variance_engine import VarianceEngine
from src.planning_reconciliation import PlanningReconciliation
from src.materiality_engine import MaterialityEngine
from src.forecast_engine import ForecastEngine
from src.root_cause_analysis import RootCauseAnalysis
from src.management_action_engine import ManagementActionEngine
from src.cfo_decision_queue import CFODecisionQueue
from src.variance_ai_engine import VarianceAIEngine

from src.scenario_engine import ScenarioEngine
from src.scenario_orchestrator import ScenarioOrchestrator
from src.anomaly_engine import AnomalyEngine
from src.advanced_forecast_engine import AdvancedForecastEngine
from src.risk_scoring_engine import RiskScoringEngine
from src.management_recommendation_engine import (
    ManagementRecommendationEngine,
)
from src.cfo_report_engine import CFOReportEngine

from src.audit_trail import AuditTrail
from src.variance_interpretation import VarianceInterpretation
from src.action_sizing_engine import ActionSizingEngine
from src.model_governance import ModelGovernance
from src.llm_context_builder import LLMContextBuilder
from src.cfo_evidence_pack_engine import CFOEvidencePackEngine
from src.grounded_ai_engine import VARIAGroundedAIEngine


class DecisionIntelligencePipeline:
    """
    Full VARIA analytical pipeline.

    Baseline analytical chain:
        cleaning
        -> planning reconciliation
        -> variance
        -> materiality
        -> root cause
        -> forecast
        -> early warning
        -> management action
        -> CFO queue
        -> AI explanation

    Advanced decision chain:
        anomaly
        -> advanced forecast
        -> composite risk
        -> variance interpretation
        -> management recommendations
        -> action sizing
        -> scenario analysis
        -> CFO report
        -> grounded LLM context
        -> governance manifest

    The pipeline exposes every intermediate dataframe so each calculation
    remains auditable and reusable by the UI.
    """

    def __init__(
        self,
        budget: pd.DataFrame,
        actuals: pd.DataFrame,
    ):

        if not isinstance(
            budget,
            pd.DataFrame,
        ):
            raise TypeError(
                "budget must be a pandas DataFrame."
            )

        if not isinstance(
            actuals,
            pd.DataFrame,
        ):
            raise TypeError(
                "actuals must be a pandas DataFrame."
            )

        self.budget = budget.copy()
        self.actuals = actuals.copy()

        self.audit = AuditTrail()

    def run(
        self,
        category_mapping: dict | None = None,
        enable_local_llm: bool = False,
        llm_temperature: float = 0.0,
    ) -> dict:

        mapping = (
            category_mapping
            if category_mapping is not None
            else {
                "Digital Ads":
                "Digital Advertising"
            }
        )

        # ----------------------------------------------------
        # Cleaning
        # ----------------------------------------------------

        cleaner = DataCleaner(
            actuals=self.actuals,
            category_mapping=mapping,
        )

        cleaned_actuals = cleaner.cleaned

        self.audit.record(
            "Data Cleaning",
            input_records=len(self.actuals),
            output_records=len(cleaned_actuals),
            input_amount=float(
                self.actuals[
                    "Actual_Amount"
                ].sum()
            )
            if "Actual_Amount"
            in self.actuals.columns
            else None,
            output_amount=float(
                cleaned_actuals[
                    "Actual_Amount"
                ].sum()
            )
            if "Actual_Amount"
            in cleaned_actuals.columns
            else None,
            status="PASS",
            metadata={
                "records_requiring_review": int(
                    cleaned_actuals[
                        "Requires_Review"
                    ].sum()
                )
                if "Requires_Review"
                in cleaned_actuals.columns
                else 0
            },
        )

        # ----------------------------------------------------
        # Planning reconciliation control
        # ----------------------------------------------------
        # Reconcile every cleaned transaction to a real budget planning
        # line BEFORE variance analysis. This prevents unmatched actuals
        # from silently disappearing during the budget/actual merge.

        planning_reconciliation_engine = PlanningReconciliation(
            budget=self.budget,
            actuals=cleaned_actuals,
        )

        planning_reconciled = (
            planning_reconciliation_engine.run()
        )

        matched_actuals = (
            planning_reconciliation_engine.matched_actuals()
        )

        planning_exceptions = (
            planning_reconciliation_engine.exception_queue()
        )

        planning_summary = (
            planning_reconciliation_engine.summary()
        )

        planning_status = (
            "PASS"
            if planning_summary["Exception Transactions"] == 0
            else "REVIEW"
        )

        self.audit.record(
            "Planning Reconciliation",
            input_records=len(cleaned_actuals),
            output_records=len(matched_actuals),
            input_amount=float(
                pd.to_numeric(
                    cleaned_actuals["Actual_Amount"],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
            output_amount=float(
                planning_summary["Matched Amount"]
            ),
            status=planning_status,
            message=(
                "All actuals matched to budget planning lines."
                if planning_status == "PASS"
                else "One or more actual transactions require planning reconciliation review."
            ),
            metadata={
                "exception_transactions": planning_summary[
                    "Exception Transactions"
                ],
                "exception_amount": planning_summary[
                    "Exception Amount"
                ],
                "gross_exception_exposure": planning_summary[
                    "Gross Exception Exposure"
                ],
                "exception_types": planning_summary[
                    "Exception Types"
                ],
            },
        )

        # ----------------------------------------------------
        # Baseline analytical engine
        # ----------------------------------------------------

        # Only matched actuals enter variance analysis. Exception rows stay
        # available through the planning_exceptions output and audit trail.
        variance = VarianceEngine(
            budget=self.budget,
            actuals=matched_actuals,
        ).run()

        self.audit.record(
            "Variance Engine",
            input_records=len(matched_actuals),
            output_records=len(variance),
            input_amount=float(
                planning_summary["Matched Amount"]
            ),
            output_amount=float(
                variance["Actual_Amount"].sum()
            ),
            status="PASS",
        )

        materiality = MaterialityEngine(
            variance
        ).run()

        root_cause = RootCauseAnalysis(
            variance
        ).recurring_drivers(
            min_occurrences=3
        )

        # ----------------------------------------------------
        # Forecast layer
        #
        # Primary method = existing VARIA ForecastEngine.
        # If the dataset does not contain enough historical fiscal
        # years for its historical budget-growth calculation, fall
        # back to the transparent AdvancedForecastEngine.
        #
        # This makes the full pipeline usable with shorter/newer
        # datasets instead of failing just because FY24-25 history
        # is unavailable.
        # ----------------------------------------------------

        forecast_engine = ForecastEngine(
            variance
        )

        forecast_method = "VARIA historical-growth forecast"

        try:

            forecast = forecast_engine.ytd_forecast()

            warning = (
                forecast_engine
                .classify_early_warning(
                    forecast
                )
            )

        except ValueError as forecast_error:

            error_text = str(
                forecast_error
            )

            history_related = (
                "FY24-25" in error_text
                or "budget growth calculation"
                in error_text.lower()
                or "historical" in error_text.lower()
            )

            if not history_related:
                raise

            advanced_engine = AdvancedForecastEngine(
                variance
            )

            advanced_forecast = (
                advanced_engine.run()
            )

            # Adapt advanced forecast to the same contract expected
            # by downstream VARIA management / CFO engines.
            forecast = advanced_forecast.copy()

            # Adapt the advanced forecast to the exact contract expected
            # by the existing downstream management/CFO engines.
            #
            # AdvancedForecastEngine reports the budget available in the
            # observed portion of the fiscal year. For short-history data,
            # convert that amount into an estimated full-year budget using
            # the same transparent annualization principle used for spend:
            #
            #   Estimated Full-Year Budget =
            #       observed budget / months elapsed * 12
            #
            # If there are already 12 months, this equals the observed
            # full-year budget.
            if "Full_Year_Budget_YTD_Available" in forecast.columns:

                forecast["YTD_Budget"] = pd.to_numeric(
                    forecast[
                        "Full_Year_Budget_YTD_Available"
                    ],
                    errors="coerce",
                ).fillna(0.0)

            else:

                forecast["YTD_Budget"] = 0.0

            elapsed = pd.to_numeric(
                forecast.get(
                    "Months_Elapsed",
                    pd.Series(
                        12,
                        index=forecast.index,
                    ),
                ),
                errors="coerce",
            ).replace(
                0,
                np.nan,
            ).fillna(12.0)

            forecast["Estimated_Full_Year_Budget"] = (
                forecast["YTD_Budget"]
                / elapsed
                * 12.0
            )

            # Keep the naming expected by downstream engines.
            if "Ensemble_Remaining" in forecast.columns:

                forecast["Forecast_Remaining_Spend"] = (
                    pd.to_numeric(
                        forecast[
                            "Ensemble_Remaining"
                        ],
                        errors="coerce",
                    ).fillna(0.0)
                )

            forecast_method = (
                "Adaptive ensemble forecast "
                "(short-history fallback)"
            )

            # Transparent warning rules for short-history datasets.
            # The primary ForecastEngine uses its own established
            # thresholds; the fallback uses forecast variance %.
            abs_pct = (
                pd.to_numeric(
                    forecast[
                        "Projected_Variance_Pct"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .abs()
            )

            warning = forecast.copy()

            warning["Early_Warning"] = np.select(
                [
                    abs_pct >= 10.0,
                    abs_pct >= 5.0,
                ],
                [
                    "RED",
                    "AMBER",
                ],
                default="GREEN",
            )

            warning["Forecast_Method"] = (
                forecast_method
            )

        # Keep the method visible for downstream audit/reporting.
        forecast["Forecast_Method"] = forecast_method
        warning["Forecast_Method"] = forecast_method

        # Final downstream contract check. This gives a much more useful
        # diagnostic than allowing a later engine to fail with an opaque
        # missing-column error.
        required_forecast_columns = [
            "Business_Unit",
            "Department",
            "Category",
            "YTD_Budget",
            "YTD_Actual",
            "Estimated_Full_Year_Budget",
            "Forecast_At_Completion",
            "Projected_Variance",
            "Projected_Variance_Pct",
        ]

        missing_forecast_columns = [
            col
            for col in required_forecast_columns
            if col not in forecast.columns
        ]

        if missing_forecast_columns:
            raise ValueError(
                "Forecast layer could not provide the downstream "
                "management contract. Missing columns: "
                + ", ".join(missing_forecast_columns)
            )

        management = ManagementActionEngine(
            variance,
            materiality,
            root_cause,
            forecast,
            warning,
        ).prepare_data()

        cfo = CFODecisionQueue(
            variance_data=variance,
            materiality_data=materiality,
            management_data=management,
        ).build()

        ai = VarianceAIEngine(
            decision_queue=cfo,
            root_cause_data=root_cause,
            forecast_data=forecast,
        ).run()

        # ----------------------------------------------------
        # Advanced analytical layers
        # ----------------------------------------------------

        anomaly_engine = AnomalyEngine(
            variance
        )

        anomaly = anomaly_engine.run()

        advanced_forecast_engine = (
            AdvancedForecastEngine(
                variance
            )
        )

        advanced_forecast = (
            advanced_forecast_engine.run()
        )

        risk_engine = RiskScoringEngine(
            management_data=management,
            materiality_data=materiality,
            root_cause_data=root_cause,
        )

        risk = risk_engine.run()

        interpretation = VarianceInterpretation(
            variance
        ).run()

        recommendation_engine = (
            ManagementRecommendationEngine(
                management_data=management,
                risk_data=risk,
                anomaly_data=anomaly,
                interpretation_data=interpretation,
            )
        )

        recommendations = (
            recommendation_engine.run()
        )

        # Quantify exactly how much corrective action would be required
        # to close projected overspend, or how much budget may be released
        # from projected underspend.
        action_sizing = ActionSizingEngine(
            forecast
        ).run()

        scenario_engine = ScenarioEngine(
            forecast
        )

        # Base scenario is always available.
        base_scenario = scenario_engine.run(
            future_spend_change_pct=0,
            budget_change_pct=0,
            scope=None,
            scenario_name="Base Scenario",
        )

        scenario_summary = (
            scenario_engine.summary(
                base_scenario
            )
        )

        # ----------------------------------------------------
        # Scenario Orchestration layer
        #
        # Coordinates the existing scenario, sensitivity, and action-sizing
        # engines into one decision pack for the UI/reporting layer.
        # Historical actuals are never modified by these scenarios.
        # ----------------------------------------------------

        scenario_orchestrator = ScenarioOrchestrator(
            forecast
        )

        scenario_library = [
            {
                "name": "Base Scenario",
                "future_spend_change_pct": 0.0,
                "budget_change_pct": 0.0,
                "scope": None,
            },
            {
                "name": "5% Future Spend Control",
                "future_spend_change_pct": -5.0,
                "budget_change_pct": 0.0,
                "scope": None,
            },
            {
                "name": "10% Future Spend Control",
                "future_spend_change_pct": -10.0,
                "budget_change_pct": 0.0,
                "scope": None,
            },
            {
                "name": "15% Future Spend Control",
                "future_spend_change_pct": -15.0,
                "budget_change_pct": 0.0,
                "scope": None,
            },
        ]

        scenario_decision_pack = (
            scenario_orchestrator.decision_pack(
                scenarios=scenario_library,
                sensitivity_dimension="Category",
                sensitivity_reduction_pct=10.0,
                sensitivity_top_n=10,
            )
        )

        # Grounded context payload for the future Ollama / Qwen layer.
        llm_context_builder = LLMContextBuilder(
            decision_queue=cfo,
            recommendation_data=recommendations,
            risk_data=risk,
            scenario_data=base_scenario,
        )

        llm_context = (
            llm_context_builder
            .build_prompt_payload(
                top_n=10
            )
        )

        # Compact, deterministic evidence selected by VARIA itself.
        # The LLM receives this pack instead of the broader raw context.
        cfo_evidence_pack = CFOEvidencePackEngine(
            outputs={
                "variance": variance,
                "materiality": materiality,
                "forecast": forecast,
                "risk": risk,
                "recommendations": recommendations,
                "action_sizing": action_sizing,
                "scenario_ranking": scenario_decision_pack[
                    "scenario_ranking"
                ],
                "recommended_scenario": scenario_decision_pack[
                    "recommended_scenario"
                ],
                "planning_summary": planning_summary,
            }
        ).build()

        # Optional local LLM layer. Disabled by default so the analytical
        # regression suite remains deterministic and does not require Ollama.
        # When enabled, Qwen receives only the verified VARIA context and its
        # response is validated before being exposed as grounded commentary.
        grounded_ai = {
            "status": "DISABLED",
            "grounded": False,
            "response": None,
            "validation": None,
            "attempts": [],
        }

        if enable_local_llm:
            grounded_ai = VARIAGroundedAIEngine().generate(
                context_payload=cfo_evidence_pack,
                temperature=llm_temperature,
            )

            self.audit.record(
                "Grounded AI",
                input_records=len(llm_context.get("issues", [])),
                output_records=1,
                status=(
                    "PASS"
                    if grounded_ai.get("grounded")
                    else "REVIEW"
                ),
                metadata={
                    "model": "qwen2.5:7b",
                    "attempts": len(
                        grounded_ai.get("attempts", [])
                    ),
                    "grounding_status": grounded_ai.get(
                        "status"
                    ),
                },
            )

        # Model governance / reproducibility manifest.
        governance = ModelGovernance(
            assumptions={
                "Fiscal_Year_Start_Month": 4,
                "Scenario_Historical_Actuals_Modified": False,
                "Forecast_Fallback_Allowed": True,
                "Planning_Exception_Rows_Excluded_From_Variance": True,
                "Unmapped_Budget_Lines_Are_Not_Auto_Created": True,
            },
            thresholds={
                "Materiality": 100000,
                "Materiality_Pct": 10.0,
                "Forecast_RED_Pct": 10.0,
                "Forecast_AMBER_Pct": 5.0,
            },
        )

        governance.record_stage(
            "Data Cleaning",
            "PASS",
            input_records=len(self.actuals),
            output_records=len(cleaned_actuals),
        )

        governance.record_stage(
            "Planning Reconciliation",
            planning_status,
            input_records=len(cleaned_actuals),
            output_records=len(matched_actuals),
            message=(
                "No planning exceptions."
                if planning_status == "PASS"
                else f"{planning_summary['Exception Transactions']} transaction(s) require reconciliation review."
            ),
        )

        governance.record_stage(
            "Variance",
            "PASS",
            input_records=len(matched_actuals),
            output_records=len(variance),
        )

        governance.record_stage(
            "Decision Intelligence",
            "PASS",
            input_records=len(variance),
            output_records=len(recommendations),
        )

        governance_final_status = (
            "PASS"
            if planning_status == "PASS"
            else "REVIEW"
        )

        governance_manifest = governance.build_manifest(
            budget=self.budget,
            actuals=self.actuals,
            final_status=governance_final_status,
        )

        report_engine = CFOReportEngine(
            variance_data=variance,
            warning_data=warning,
            cfo_queue=cfo,
            recommendation_data=recommendations,
            scenario_summary=scenario_summary,
        )

        report = report_engine.generate()

        self.audit.record(
            "Advanced Analytics",
            input_records=len(variance),
            output_records=len(recommendations),
            status="PASS",
            metadata={
                "anomaly_records": len(anomaly),
                "advanced_forecast_records": len(
                    advanced_forecast
                ),
                "risk_records": len(risk),
                "recommendation_records": len(
                    recommendations
                ),
                "action_sizing_records": len(
                    action_sizing
                ),
            },
        )

        self.audit.record(
            "Scenario Engine",
            input_records=len(forecast),
            output_records=len(base_scenario),
            status="PASS",
            metadata={
                "scenario_name": "Base Scenario",
            },
        )

        self.audit.record(
            "Governance",
            input_records=len(variance),
            output_records=len(governance_manifest["Stages"]),
            status="PASS",
        )

        return {
            "budget": self.budget,
            "actuals_raw": self.actuals,
            "actuals_cleaned": cleaned_actuals,
            "planning_reconciliation": planning_reconciled,
            "matched_actuals": matched_actuals,
            "planning_exceptions": planning_exceptions,
            "planning_summary": planning_summary,
            "variance": variance,
            "materiality": materiality,
            "root_cause": root_cause,
            "forecast": forecast,
            "warning": warning,
            "management": management,
            "cfo_queue": cfo,
            "ai_output": ai,
            "interpretation": interpretation,
            "anomaly": anomaly,
            "advanced_forecast": advanced_forecast,
            "risk": risk,
            "recommendations": recommendations,
            "action_sizing": action_sizing,
            "scenario_engine": scenario_engine,
            "base_scenario": base_scenario,
            "scenario_summary": scenario_summary,
            "scenario_orchestrator": scenario_orchestrator,
            "scenario_decision_pack": scenario_decision_pack,
            "scenario_comparison": scenario_decision_pack["scenario_comparison"],
            "scenario_ranking": scenario_decision_pack["scenario_ranking"],
            "recommended_scenario": scenario_decision_pack["recommended_scenario"],
            "sensitivity": scenario_decision_pack["sensitivity"],
            "cfo_report": report,
            "llm_context": llm_context,
            "cfo_evidence_pack": cfo_evidence_pack,
            "grounded_ai": grounded_ai,
            "governance_manifest": governance_manifest,
            "audit": self.audit.to_dataframe(),
            "audit_summary": self.audit.summary(),
        }
