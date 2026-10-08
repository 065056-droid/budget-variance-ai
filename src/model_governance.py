
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


class ModelGovernance:
    """
    Lightweight governance / reproducibility manifest for a VARIA run.

    The manifest records:
      - VARIA model version
      - run identifier
      - UTC timestamp
      - source row/column counts
      - source fingerprints
      - model assumptions
      - threshold configuration
      - pipeline stages
      - final status

    This creates an auditable explanation of which data and assumptions
    produced a run, without storing the financial dataset itself.
    """

    MODEL_VERSION = "VARIA-1.0"

    def __init__(
        self,
        assumptions: dict | None = None,
        thresholds: dict | None = None,
    ):

        self.assumptions = assumptions or {}
        self.thresholds = thresholds or {}

        self.created_at = datetime.now(
            timezone.utc
        ).isoformat()

        self.stages: list[dict] = []

    @staticmethod
    def dataframe_fingerprint(
        dataframe: pd.DataFrame,
    ) -> str:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        # Stable hash of column order + values.
        payload = pd.util.hash_pandas_object(
            dataframe,
            index=True,
        ).values.tobytes()

        schema = "|".join(
            map(
                str,
                dataframe.columns,
            )
        ).encode("utf-8")

        return hashlib.sha256(
            schema + payload
        ).hexdigest()

    def record_stage(
        self,
        stage: str,
        status: str = "PASS",
        input_records: int | None = None,
        output_records: int | None = None,
        message: str = "",
    ) -> None:

        self.stages.append(
            {
                "Stage": stage,
                "Status": status,
                "Input_Records": input_records,
                "Output_Records": output_records,
                "Message": message,
            }
        )

    def build_manifest(
        self,
        budget: pd.DataFrame | None = None,
        actuals: pd.DataFrame | None = None,
        final_status: str = "PASS",
    ) -> dict:

        manifest = {
            "Model_Version": self.MODEL_VERSION,
            "Generated_At_UTC": self.created_at,
            "Final_Status": final_status,
            "Assumptions": self.assumptions,
            "Thresholds": self.thresholds,
            "Stages": self.stages,
        }

        if budget is not None:
            manifest["Budget_Source"] = {
                "Rows": len(budget),
                "Columns": len(budget.columns),
                "Fingerprint": self.dataframe_fingerprint(
                    budget
                ),
            }

        if actuals is not None:
            manifest["Actuals_Source"] = {
                "Rows": len(actuals),
                "Columns": len(actuals.columns),
                "Fingerprint": self.dataframe_fingerprint(
                    actuals
                ),
            }

        return manifest

    def to_json(
        self,
        budget: pd.DataFrame | None = None,
        actuals: pd.DataFrame | None = None,
        final_status: str = "PASS",
    ) -> str:

        return json.dumps(
            self.build_manifest(
                budget=budget,
                actuals=actuals,
                final_status=final_status,
            ),
            indent=2,
            default=str,
        )

    def save_json(
        self,
        path: str,
        budget: pd.DataFrame | None = None,
        actuals: pd.DataFrame | None = None,
        final_status: str = "PASS",
    ) -> None:

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                self.to_json(
                    budget=budget,
                    actuals=actuals,
                    final_status=final_status,
                )
            )

    def summary(self) -> dict:

        return {
            "Model_Version": self.MODEL_VERSION,
            "Stages_Recorded": len(
                self.stages
            ),
            "Pass_Stages": sum(
                1
                for stage in self.stages
                if stage["Status"] == "PASS"
            ),
            "Review_Stages": sum(
                1
                for stage in self.stages
                if stage["Status"] == "REVIEW"
            ),
            "Fail_Stages": sum(
                1
                for stage in self.stages
                if stage["Status"] == "FAIL"
            ),
        }
