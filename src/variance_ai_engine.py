import pandas as pd


class VarianceAIEngine:

    def __init__(
        self,
        decision_queue,
        driver_data=None,
        root_cause_data=None,
        interpretation_data=None,
        forecast_data=None,
    ):
        self.decision_queue = decision_queue.copy()
        self.driver_data = driver_data
        self.root_cause_data = root_cause_data
        self.interpretation_data = interpretation_data
        self.forecast_data = forecast_data

    def prepare_data(self):
        required_columns = [
            "Business_Unit",
            "Department",
            "Category",
            "Current_Variance",
            "Projected_Variance",
            "Projected_Variance_Pct",
            "Early_Warning",
            "Decision_Priority",
            "Management_Action",
        ]

        missing = [
            col for col in required_columns
            if col not in self.decision_queue.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns in decision queue: {missing}"
            )

        df = self.decision_queue.copy()

        df["Current_Variance"] = pd.to_numeric(
            df["Current_Variance"], errors="coerce"
        )

        df["Projected_Variance"] = pd.to_numeric(
            df["Projected_Variance"], errors="coerce"
        )

        df["Projected_Variance_Pct"] = pd.to_numeric(
            df["Projected_Variance_Pct"], errors="coerce"
        )

        return df

    def classify_issue(self, row):

        projected_variance = row["Projected_Variance"]

        if projected_variance > 0:
            return "Projected Overspend"

        elif projected_variance < 0:
            return "Projected Underspend"

        return "On Budget"

    def generate_explanation(self, row):

        issue_type = self.classify_issue(row)

        business_unit = row["Business_Unit"]
        department = row["Department"]
        category = row["Category"]

        projected_variance = row["Projected_Variance"]
        projected_pct = row["Projected_Variance_Pct"]

        if issue_type == "Projected Overspend":

            explanation = (
                f"{business_unit} {department} spending on {category} "
                f"is projected to exceed budget by "
                f"₹{projected_variance:,.2f} "
                f"({projected_pct:.2f}%)."
            )

        elif issue_type == "Projected Underspend":

            explanation = (
                f"{business_unit} {department} spending on {category} "
                f"is projected to remain below budget by "
                f"₹{abs(projected_variance):,.2f} "
                f"({abs(projected_pct):.2f}%)."
            )

        else:

            explanation = (
                f"{business_unit} {department} spending on {category} "
                f"is projected to remain broadly on budget."
            )

        return explanation

    def generate_management_context(self, row):

        warning = row["Early_Warning"]
        priority = row["Decision_Priority"]

        if warning == "RED":
            warning_text = (
                "The forecast indicates a significant deviation from "
                "the expected full-year budget and requires immediate attention."
            )

        elif warning == "AMBER":
            warning_text = (
                "The forecast indicates a moderate deviation and "
                "should be reviewed by management."
            )

        else:
            warning_text = (
                "The forecast remains within the normal monitoring range."
            )

        priority_text = (
            f"The current decision priority is {priority}."
        )

        return f"{warning_text} {priority_text}"

    def generate_action(self, row):

        action = row["Management_Action"]

        return (
            f"Recommended management action: {action}"
        )

    def run(self):

        df = self.prepare_data()

        df["AI_Issue_Type"] = df.apply(
            self.classify_issue,
            axis=1
        )

        df["AI_Explanation"] = df.apply(
            self.generate_explanation,
            axis=1
        )

        df["AI_Management_Context"] = df.apply(
            self.generate_management_context,
            axis=1
        )

        df["AI_Recommended_Action"] = df.apply(
            self.generate_action,
            axis=1
        )

        df["AI_Executive_Summary"] = (
            df["AI_Explanation"]
            + " "
            + df["AI_Management_Context"]
            + " "
            + df["AI_Recommended_Action"]
        )

        self.result = df

        return df

    def top_explanations(self, n=10):

        if not hasattr(self, "result"):
            self.run()

        return self.result.head(n)

    def priority_explanations(self):

        if not hasattr(self, "result"):
            self.run()

        return self.result[
            self.result["Decision_Priority"].isin(
                ["CRITICAL", "HIGH PRIORITY"]
            )
        ]

    def summary(self):

        if not hasattr(self, "result"):
            self.run()

        return {
            "Total_Records": len(self.result),
            "Overspend_Records": int(
                (self.result["AI_Issue_Type"] == "Projected Overspend").sum()
            ),
            "Underspend_Records": int(
                (self.result["AI_Issue_Type"] == "Projected Underspend").sum()
            ),
            "On_Budget_Records": int(
                (self.result["AI_Issue_Type"] == "On Budget").sum()
            ),
            "Red_Warnings": int(
                (self.result["Early_Warning"] == "RED").sum()
            ),
            "High_Priority": int(
                (
                    self.result["Decision_Priority"]
                    == "HIGH PRIORITY"
                ).sum()
            ),
        }