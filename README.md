# VARIA — Budget Variance & Financial Decision Intelligence System

VARIA is an AI-powered financial decision-support system designed to transform budget and actual expenditure data into actionable management insights.

Instead of simply reporting whether spending is above or below budget, VARIA helps finance teams understand:

- What went wrong?
- How material is the variance?
- What are the key drivers?
- Which issues require management attention?
- Is the variance recurring?
- What could happen next?
- What actions should management consider?

## Key Capabilities

### 1. Data Ingestion & Validation
- Budget and actual expenditure ingestion
- Flexible column mapping
- Data quality validation
- Missing-value detection
- Duplicate transaction detection
- Invalid category detection
- Negative expense detection

### 2. Budget Variance Analysis
- Budget vs. actual comparison
- Absolute variance calculation
- Variance percentage
- Favorable / unfavorable classification
- Department, category, business-unit and regional analysis

### 3. Reconciliation
- Transaction-level reconciliation
- Matched and unmatched transaction identification
- Exception amount analysis
- Planning-to-actual reconciliation

### 4. Materiality & Risk Prioritization
VARIA evaluates the significance of variances using materiality and severity logic to prioritize issues for management review.

The system categorizes variance issues into:
- LOW
- MEDIUM
- CRITICAL

### 5. Driver & Root-Cause Analysis
The system investigates the underlying contributors to significant variances rather than stopping at the variance number.

### 6. Forecasting & Anomaly Detection
VARIA includes analytical layers for:
- Forecasting
- Anomaly detection
- Recurring variance identification
- Risk scoring
- Sensitivity analysis

### 7. AI-Powered Financial Investigation
VARIA incorporates grounded AI capabilities to investigate significant financial variances and generate evidence-based explanations.

### 8. Management Decision Support
The system translates financial findings into management-oriented outputs including:
- CFO decision queues
- Management recommendations
- Action sizing
- Budget reallocation analysis
- Scenario analysis
- What-if analysis

## Architecture

```text
                    ┌─────────────────────┐
                    │   Budget & Actuals  │
                    │       Data          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Ingestion &    │
                    │ Validation Layer   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Variance &          │
                    │ Reconciliation      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌───────────┐   ┌────────────┐   ┌────────────┐
        │ Materiality│   │ Root Cause │   │ Forecasting│
        │ & Risk    │   │ & Drivers  │   │ & Anomaly  │
        └─────┬─────┘   └──────┬─────┘   └──────┬─────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ AI Investigation &  │
                    │ Decision Intelligence│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ CFO / Management    │
                    │ Decision Support    │
                    └─────────────────────┘