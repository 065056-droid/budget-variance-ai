
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class AuditTrail:
    """
    Lightweight audit log for the VARIA pipeline.

    Each stage records:
      - timestamp
      - stage name
      - input record count
      - output record count
      - input amount
      - output amount
      - difference
      - status
      - message / metadata

    The audit trail is append-only in memory and can be exported to CSV.
    """

    def __init__(self):
        self._events: list[dict[str, Any]] = []

    def record(
        self,
        stage: str,
        input_records: int | None = None,
        output_records: int | None = None,
        input_amount: float | None = None,
        output_amount: float | None = None,
        status: str = "PASS",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:

        difference = None

        if (
            input_amount is not None
            and output_amount is not None
        ):
            difference = (
                float(output_amount)
                - float(input_amount)
            )

        self._events.append(
            {
                "Timestamp_UTC": datetime.now(
                    timezone.utc
                ).isoformat(),
                "Stage": stage,
                "Input_Records": input_records,
                "Output_Records": output_records,
                "Input_Amount": input_amount,
                "Output_Amount": output_amount,
                "Amount_Difference": difference,
                "Status": status,
                "Message": message,
                "Metadata": (
                    repr(metadata)
                    if metadata
                    else ""
                ),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            self._events,
            columns=[
                "Timestamp_UTC",
                "Stage",
                "Input_Records",
                "Output_Records",
                "Input_Amount",
                "Output_Amount",
                "Amount_Difference",
                "Status",
                "Message",
                "Metadata",
            ],
        )

    def clear(self) -> None:
        self._events.clear()

    def summary(self) -> dict:

        data = self.to_dataframe()

        if data.empty:
            return {
                "Stages": 0,
                "Pass": 0,
                "Review": 0,
                "Fail": 0,
            }

        return {
            "Stages": len(data),
            "Pass": int(
                (data["Status"] == "PASS").sum()
            ),
            "Review": int(
                (data["Status"] == "REVIEW").sum()
            ),
            "Fail": int(
                (data["Status"] == "FAIL").sum()
            ),
        }

    def export_csv(
        self,
        path: str,
    ) -> None:

        self.to_dataframe().to_csv(
            path,
            index=False,
        )
