import pandas as pd

from src.data_ingestion_engine import DataIngestionEngine


class VARIAInputGateway:
    """
    Universal entry point for VARIA raw financial data.

    Supported upload patterns:
      1. Separate raw budget + raw actuals DataFrames
      2. One combined DataFrame containing both budget and actual amounts

    Responsibilities:
      - profile raw inputs
      - detect / validate dataset type
      - suggest and apply column mappings
      - standardize budget and actuals into VARIA contracts
      - expose ingestion metadata
      - hand standardized data to DecisionIntelligencePipeline

    Financial amounts are never invented or altered beyond deterministic
    aggregation required to reach VARIA's planning grain for budgets.
    """

    def __init__(
        self,
        budget: pd.DataFrame | None = None,
        actuals: pd.DataFrame | None = None,
        combined: pd.DataFrame | None = None,
    ):
        supplied = sum(
            x is not None
            for x in (budget, actuals, combined)
        )

        if supplied == 0:
            raise ValueError(
                "Provide either budget + actuals or a combined dataset."
            )

        if combined is not None and (
            budget is not None or actuals is not None
        ):
            raise ValueError(
                "Use either combined or separate budget/actuals inputs, not both."
            )

        if (budget is None) != (actuals is None):
            raise ValueError(
                "When using separate inputs, both budget and actuals are required."
            )

        self.raw_budget = (
            budget.copy() if budget is not None else None
        )
        self.raw_actuals = (
            actuals.copy() if actuals is not None else None
        )
        self.raw_combined = (
            combined.copy() if combined is not None else None
        )

        self._standardized_budget: pd.DataFrame | None = None
        self._standardized_actuals: pd.DataFrame | None = None
        self._ingestion_metadata: dict | None = None

    @property
    def source_type(self) -> str:
        if self.raw_combined is not None:
            return "combined"
        return "separate"

    def _engine(
        self,
        dataframe: pd.DataFrame,
        dataset_type: str,
    ) -> DataIngestionEngine:
        return DataIngestionEngine(
            dataframe=dataframe,
            dataset_type=dataset_type,
        )

    def profile(self) -> dict:
        if self.raw_combined is not None:
            engine = self._engine(
                self.raw_combined,
                "combined",
            )
            return {
                "Source_Type": self.source_type,
                "Combined": engine.profile(),
            }

        budget_engine = self._engine(
            self.raw_budget,
            "budget",
        )
        actual_engine = self._engine(
            self.raw_actuals,
            "actuals",
        )

        return {
            "Source_Type": self.source_type,
            "Budget": budget_engine.profile(),
            "Actuals": actual_engine.profile(),
        }

    def mapping_suggestions(self) -> dict:
        if self.raw_combined is not None:
            engine = self._engine(
                self.raw_combined,
                "combined",
            )
            return {
                "Combined": engine.mapping_suggestions(),
            }

        budget_engine = self._engine(
            self.raw_budget,
            "budget",
        )
        actual_engine = self._engine(
            self.raw_actuals,
            "actuals",
        )

        return {
            "Budget": budget_engine.mapping_suggestions(),
            "Actuals": actual_engine.mapping_suggestions(),
        }

    def readiness_report(
        self,
        budget_mapping: dict | None = None,
        actuals_mapping: dict | None = None,
        combined_mapping: dict | None = None,
    ) -> dict:
        if self.raw_combined is not None:
            engine = self._engine(
                self.raw_combined,
                "combined",
            )
            return {
                "Source_Type": self.source_type,
                "Combined": engine.readiness_report(
                    mapping=combined_mapping
                ),
            }

        budget_engine = self._engine(
            self.raw_budget,
            "budget",
        )
        actual_engine = self._engine(
            self.raw_actuals,
            "actuals",
        )

        return {
            "Source_Type": self.source_type,
            "Budget": budget_engine.readiness_report(
                mapping=budget_mapping
            ),
            "Actuals": actual_engine.readiness_report(
                mapping=actuals_mapping
            ),
        }

    def build(
        self,
        budget_mapping: dict | None = None,
        actuals_mapping: dict | None = None,
        combined_mapping: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize raw inputs into VARIA budget and actuals contracts.
        """
        if self.raw_combined is not None:
            engine = self._engine(
                self.raw_combined,
                "combined",
            )
            budget, actuals = engine.build(
                mapping=combined_mapping
            )
        else:
            budget_engine = self._engine(
                self.raw_budget,
                "budget",
            )
            actual_engine = self._engine(
                self.raw_actuals,
                "actuals",
            )

            budget = budget_engine.build(
                mapping=budget_mapping
            )
            actuals = actual_engine.build(
                mapping=actuals_mapping
            )

        self._standardized_budget = budget.copy()
        self._standardized_actuals = actuals.copy()

        self._ingestion_metadata = {
            "Source_Type": self.source_type,
            "Raw_Budget_Rows": (
                int(len(self.raw_budget))
                if self.raw_budget is not None
                else None
            ),
            "Raw_Actuals_Rows": (
                int(len(self.raw_actuals))
                if self.raw_actuals is not None
                else None
            ),
            "Raw_Combined_Rows": (
                int(len(self.raw_combined))
                if self.raw_combined is not None
                else None
            ),
            "Standardized_Budget_Rows": int(len(budget)),
            "Standardized_Actuals_Rows": int(len(actuals)),
            "Budget_Total": float(
                pd.to_numeric(
                    budget["Budget_Amount"],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
            "Actuals_Total": float(
                pd.to_numeric(
                    actuals["Actual_Amount"],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
        }

        return budget.copy(), actuals.copy()

    def ingestion_metadata(self) -> dict:
        if self._ingestion_metadata is None:
            raise RuntimeError(
                "Call build() before requesting ingestion metadata."
            )
        return dict(self._ingestion_metadata)

    def run(
        self,
        category_mapping: dict | None = None,
        budget_mapping: dict | None = None,
        actuals_mapping: dict | None = None,
        combined_mapping: dict | None = None,
    ) -> dict:
        """
        Full product entry point: ingest raw data, standardize it, then run
        the complete VARIA decision-intelligence pipeline.
        """
        budget, actuals = self.build(
            budget_mapping=budget_mapping,
            actuals_mapping=actuals_mapping,
            combined_mapping=combined_mapping,
        )

        # Import lazily so the ingestion gateway remains independently
        # testable and can be used for schema/profile workflows before the
        # heavier decision-intelligence stack is loaded.
        from src.decision_intelligence_pipeline import DecisionIntelligencePipeline

        pipeline = DecisionIntelligencePipeline(
            budget=budget,
            actuals=actuals,
        )

        result = pipeline.run(
            category_mapping=category_mapping
        )

        result["ingestion"] = {
            "profile": self.profile(),
            "readiness": self.readiness_report(
                budget_mapping=budget_mapping,
                actuals_mapping=actuals_mapping,
                combined_mapping=combined_mapping,
            ),
            "metadata": self.ingestion_metadata(),
        }

        return result

    @property
    def standardized_budget(self) -> pd.DataFrame:
        if self._standardized_budget is None:
            raise RuntimeError(
                "Call build() or run() before requesting standardized budget."
            )
        return self._standardized_budget.copy()

    @property
    def standardized_actuals(self) -> pd.DataFrame:
        if self._standardized_actuals is None:
            raise RuntimeError(
                "Call build() or run() before requesting standardized actuals."
            )
        return self._standardized_actuals.copy()
