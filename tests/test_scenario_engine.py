
import pandas as pd

from src.scenario_engine import ScenarioEngine


def main():

    print("\n" + "=" * 80)
    print("VARIA SCENARIO ENGINE TEST")
    print("=" * 80)

    forecast = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "YTD_Actual": 8500000.0,
                "Forecast_Remaining_Spend": 1500000.0,
                "Forecast_At_Completion": 10000000.0,
                "Estimated_Full_Year_Budget": 8700000.0,
                "Projected_Variance": 1300000.0,
                "Projected_Variance_Pct": 14.9425,
            },
            {
                "Business_Unit": "Enterprise",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "YTD_Actual": 7500000.0,
                "Forecast_Remaining_Spend": 1400000.0,
                "Forecast_At_Completion": 8900000.0,
                "Estimated_Full_Year_Budget": 8200000.0,
                "Projected_Variance": 700000.0,
                "Projected_Variance_Pct": 8.5366,
            },
            {
                "Business_Unit": "Consumer",
                "Department": "Operations",
                "Category": "Utilities",
                "YTD_Actual": 5000000.0,
                "Forecast_Remaining_Spend": 900000.0,
                "Forecast_At_Completion": 5900000.0,
                "Estimated_Full_Year_Budget": 6100000.0,
                "Projected_Variance": -200000.0,
                "Projected_Variance_Pct": -3.2787,
            },
        ]
    )

    engine = ScenarioEngine(forecast)

    # 10% reduction only for Digital Advertising.
    result = engine.run(
        future_spend_change_pct=-10.0,
        scope={
            "Category": "Digital Advertising"
        },
        scenario_name="10% Digital Advertising Reduction",
    )

    summary = engine.summary(result)

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Validate that only the two Digital Advertising rows changed.
    assert (
        int(result["Scenario_Applied"].sum())
        == 2
    )

    applied = result[
        result["Scenario_Applied"]
    ]

    not_applied = result[
        ~result["Scenario_Applied"]
    ]

    assert np_close(
        applied.iloc[0]["Scenario_Remaining_Spend"],
        1350000.0,
    )

    assert np_close(
        not_applied.iloc[0]["Scenario_Remaining_Spend"],
        900000.0,
    )

    # The first row should improve by 150,000.
    assert np_close(
        applied.iloc[0]["Variance_Improvement"],
        150000.0,
    )

    # Sensitivity should create six scenarios.
    sensitivity = engine.sensitivity_analysis(
        [-20, -15, -10, -5, 0, 5],
        scope={
            "Category": "Digital Advertising"
        },
    )

    assert len(sensitivity) == 6
    assert list(
        sensitivity["Spend_Change_Pct"]
    ) == [-20, -15, -10, -5, 0, 5]

    # Multi-scenario test.
    scenarios = engine.multi_scenario(
        [
            {
                "name": "Base",
                "future_spend_change_pct": 0,
                "scope": {
                    "Category":
                    "Digital Advertising"
                },
            },
            {
                "name": "10% Cut",
                "future_spend_change_pct": -10,
                "scope": {
                    "Category":
                    "Digital Advertising"
                },
            },
        ]
    )

    assert len(scenarios) == 2

    print("\nTEST STATUS: PASS")

    print("\nSENSITIVITY")
    print(
        sensitivity.to_string(
            index=False
        )
    )

    print("\nMULTI-SCENARIO")
    print(
        scenarios.to_string(
            index=False
        )
    )


def np_close(a, b, tolerance=1e-6):
    return abs(float(a) - float(b)) <= tolerance


if __name__ == "__main__":
    main()
