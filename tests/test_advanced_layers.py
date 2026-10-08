
import pandas as pd

from src.anomaly_engine import AnomalyEngine
from src.advanced_forecast_engine import AdvancedForecastEngine
from src.risk_scoring_engine import RiskScoringEngine
from src.column_mapping_engine import ColumnMappingEngine
from src.audit_trail import AuditTrail


def main():

    print("\n" + "=" * 80)
    print("VARIA ADVANCED LAYERS TEST")
    print("=" * 80)

    months = pd.date_range(
        "2025-04-01",
        "2026-03-01",
        freq="MS",
    )

    rows = []

    for month_index, month in enumerate(months, start=1):

        base = 100000 + month_index * 2500

        rows.append(
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Month": month,
                "Fiscal_Year": "FY25-26",
                "Budget_Amount": base,
                "Actual_Amount": base * 1.12,
                "Variance": base * 0.12,
                "Variance_Pct": 12.0,
            }
        )

        rows.append(
            {
                "Business_Unit": "Enterprise",
                "Department": "Operations",
                "Category": "Utilities",
                "Month": month,
                "Fiscal_Year": "FY25-26",
                "Budget_Amount": base,
                "Actual_Amount": base * 0.96,
                "Variance": -base * 0.04,
                "Variance_Pct": -4.0,
            }
        )

    variance = pd.DataFrame(rows)

    # Anomaly layer
    anomaly_engine = AnomalyEngine(
        variance
    )
    anomaly = anomaly_engine.run()

    assert "Anomaly_Level" in anomaly.columns
    assert "Requires_Investigation" in anomaly.columns

    print("\nANOMALY LAYER: PASS")
    print(
        anomaly_engine.summary(
            anomaly
        )
    )

    # Advanced forecast
    forecast_engine = AdvancedForecastEngine(
        variance
    )
    forecast = forecast_engine.run()

    assert len(forecast) == 2
    assert "Ensemble_Remaining" in forecast.columns
    assert "Forecast_Confidence" in forecast.columns

    print("\nADVANCED FORECAST: PASS")
    print(
        forecast[
            [
                "Business_Unit",
                "Department",
                "Category",
                "Forecast_At_Completion",
                "Forecast_Low",
                "Forecast_High",
                "Forecast_Confidence",
                "Projected_Variance",
            ]
        ].to_string(index=False)
    )

    # Risk layer
    management = forecast[
        [
            "Business_Unit",
            "Department",
            "Category",
            "Projected_Variance",
            "Projected_Variance_Pct",
        ]
    ].copy()

    management["Early_Warning"] = "RED"
    management["Decision_Priority"] = "HIGH PRIORITY"

    materiality = variance[
        [
            "Business_Unit",
            "Department",
            "Category",
            "Variance",
        ]
    ].copy()

    materiality["Absolute_Variance"] = (
        materiality["Variance"].abs()
    )
    materiality["Severity"] = "MEDIUM"
    materiality["Is_Material"] = True

    root_cause = pd.DataFrame(
        [
            {
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Occurrences": 12,
            },
            {
                "Department": "Operations",
                "Category": "Utilities",
                "Occurrences": 4,
            },
        ]
    )

    risk_engine = RiskScoringEngine(
        management_data=management,
        materiality_data=materiality,
        root_cause_data=root_cause,
    )

    risk = risk_engine.run()

    assert "Risk_Score" in risk.columns
    assert risk["Risk_Score"].between(
        0,
        100,
    ).all()

    print("\nRISK LAYER: PASS")
    print(
        risk[
            [
                "Business_Unit",
                "Department",
                "Category",
                "Risk_Score",
                "Risk_Band",
                "Risk_Rank",
            ]
        ].to_string(index=False)
    )

    # Mapping layer
    upload = pd.DataFrame(
        {
            "Date": ["2026-04-01"],
            "BU": ["Consumer"],
            "Dept": ["Marketing"],
            "Expense Head": ["Digital Advertising"],
            "Plan": [100000],
            "Actual": [115000],
        }
    )

    mapper = ColumnMappingEngine(
        upload
    )

    suggestions = mapper.suggestions()

    assert (
        suggestions.loc[
            suggestions["VARIA_Field"]
            == "Department",
            "Confidence",
        ].iloc[0]
        == "HIGH"
    )

    mapping = mapper.mapping_dict()
    mapped = mapper.apply(mapping)

    assert "Department" in mapped.columns
    assert "Actual_Amount" in mapped.columns
    assert "Budget_Amount" in mapped.columns

    print("\nCOLUMN MAPPING: PASS")

    # Audit layer
    audit = AuditTrail()

    audit.record(
        "Data Ingestion",
        input_records=1,
        output_records=1,
        status="PASS",
        message="Test ingestion",
    )

    audit.record(
        "Variance Engine",
        input_records=1,
        output_records=1,
        input_amount=100000,
        output_amount=115000,
        status="PASS",
        message="Test variance calculation",
    )

    audit_df = audit.to_dataframe()

    assert len(audit_df) == 2
    assert audit.summary()["Pass"] == 2

    print("\nAUDIT TRAIL: PASS")

    print("\n" + "=" * 80)
    print("ALL ADVANCED LAYERS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
