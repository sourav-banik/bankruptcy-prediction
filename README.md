# Bankruptcy Early-Warning Prediction Model

## Team 3

Yutong Chen · Shuyi Chen · Sourav Banik · Hao Yang · Khizer Shabbir

## Project Overview

This project develops an early-warning screening model that estimates the probability that a publicly traded U.S. non-financial company will file for Chapter 7 or Chapter 11 bankruptcy within the following 12 months.

The model is intended to help investors, creditors, and analysts prioritize companies for additional review. It is a screening and research-prioritization tool, not an automatic investment recommendation and not a replacement for analyst judgment.

## Business Problem

Investors and credit analysts must monitor a large universe of companies using financial statements, market data, and other publicly available information. Reviewing every company with the same level of attention is inefficient. A probability-based risk score can help identify firms with elevated financial-distress risk and direct limited research time toward the companies most likely to require deeper analysis.

Potential use cases include:

- Corporate-bond screening before investment
- Monitoring existing portfolio holdings
- Credit-watch-list construction
- Prioritizing analyst research
- Identifying potential distressed-investment opportunities
- Supporting analysis of capital structure, asset values, restructuring options, and potential recovery

## Research Question

Can publicly disclosed financial information and market data predict whether a publicly traded U.S. non-financial company will file for bankruptcy within the next 12 months?

## Data and Unit of Observation

The unit of observation is a **company-year**. Each row represents one eligible company observed at a specific annual prediction date.

### Primary data sources

1. **Florida-UCLA-LoPucki Bankruptcy Research Database (BRD)**
   - Provides bankruptcy filing information and identifiable filing dates.
   - Chapter 7 and Chapter 11 filings are used to define positive events.

2. **Bloomberg historical company and market data**
   - Provides financial-statement fundamentals, market capitalization, stock returns, volatility, and other market variables.

### Company matching

The data sources will be linked using company identifiers in the following order where available:

1. CIK
2. CUSIP
3. Carefully reviewed company-name matching

Uncertain matches will be flagged and reviewed rather than accepted automatically.

## Target Variable

The primary target is:

```text
bankruptcy_12m = 1
```

when a company files Chapter 7 or Chapter 11 bankruptcy within the 12 months following the prediction date, and:

```text
bankruptcy_12m = 0
```

when no qualifying filing is observed during that period.

The project will use only information available before each prediction date. Observations after bankruptcy and information that would not have been available at the prediction date will be excluded.

## Candidate Features

The feature set will focus on indicators that can be compared across non-financial industries.

### Profitability

- Return on assets (ROA)
- Operating margin
- EBIT/assets
- Net income/assets

### Leverage

- Total liabilities/assets
- Debt/assets
- Debt/equity

### Liquidity

- Current ratio
- Quick ratio
- Cash/assets
- Working capital/assets

### Debt-service capacity

- Interest coverage
- Debt/EBITDA

### Cash flow

- Operating cash flow/assets
- Free cash flow/assets

### Growth and firm characteristics

- Revenue growth
- Asset growth
- Earnings growth
- Total assets
- Log market capitalization
- Company age, where available

### Market indicators

- Prior-year stock return
- Stock-price volatility
- Market-to-book ratio
- Drawdown measures

### Industry controls

- Broad industry or sector indicators
- Industry-year-relative financial ratios

Lagged and trend features will be constructed only from prior observations. Industry-relative features may measure how far a company’s leverage, liquidity, or profitability differs from the median for comparable firms in the same industry and year.

## Methodology

The project follows the CRISP-DM process and includes two related data-science tasks.

### 1. Exploratory and visual analysis

The analysis will examine:

- Bankruptcy counts and rates over time
- Differences between bankrupt and non-bankrupt companies
- Industry-level differences
- Missingness and data coverage
- Financial-ratio distributions
- Risk segments and predicted-probability distributions

### 2. Predictive modeling

Candidate models include:

- Majority-class or simple-ratio benchmark
- Weighted logistic regression
- Regularized logistic regression
- Tree-based model, such as random forest or gradient boosting, if appropriate

The final model will be selected based on predictive performance, probability calibration, interpretability, and usefulness for screening.

## Validation Strategy

Validation will be time-based rather than randomly split across all years:

- Earlier years: training data
- Subsequent years: validation data
- Later years: out-of-time test data

This design better reflects how the model would be used in practice and helps reduce look-ahead bias.

Class imbalance will be addressed using model weights where possible. If the full panel is too large, stratified negative sampling may be used by calendar year, industry, and potentially firm size. Because sampling can distort predicted probabilities, probabilities will be recalibrated using an appropriate validation procedure.

## Evaluation Metrics

The project will report metrics that are appropriate for a rare-event screening problem, including:

- ROC-AUC
- Precision-recall AUC
- Recall for bankruptcy cases
- Precision among high-risk companies
- F1 score, where useful
- Probability calibration
- Confusion matrix at a validation-selected threshold
- Lift and capture rate among the highest-risk companies

The business-value analysis will include a screening simulation measuring how many future bankruptcy cases are captured when analysts review the highest-risk group, such as the top 1%, 5%, or 10% of companies.

## Business Value and Limitations

A high predicted probability identifies a company for additional investigation. It does not automatically mean that the company should be avoided or that its securities should be sold. Analysts should review liquidity, debt structure, maturity schedules, restructuring options, asset values, and potential recovery value before making an investment or credit decision.

The project will report scenario-based benefits rather than claiming a precise return on investment because portfolio exposures, research costs, recovery rates, and transaction outcomes may not be available.

Important limitations may include:

- Incomplete or inconsistent historical data
- Limited number of bankruptcy events
- Selection limitations in the bankruptcy database
- Imperfect company matching across sources
- Missing data that may not be random
- Differences in accounting practices across industries
- Potential changes in market conditions over time
- False positives and false negatives
- Reduced generalizability to financial institutions or private companies

## Repository Structure

```text
team3-bankruptcy-prediction/
├── README.md
├── data/
│   ├── raw/                  # Original files; do not modify
│   ├── interim/              # Intermediate cleaned and matched files
│   ├── processed/            # Final model-ready data
│   └── data_dictionary.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_results.ipynb
├── src/
│   ├── 01_clean_data.py
│   ├── 02_create_labels.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   └── 05_evaluation.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── model_results/
├── report/
├── presentation/
├── requirements.txt
└── .gitignore
```

## Reproducibility Guidelines

1. Keep raw data unchanged and store it outside the public repository when access or licensing restrictions apply.
2. Use scripts for repeatable cleaning, label construction, feature engineering, modeling, and evaluation.
3. Keep exploratory notebooks in the `notebooks/` folder and document their purpose.
4. Record data dates, variable definitions, transformations, and exclusions in the data dictionary.
5. Do not use future information when constructing features or labels.
6. Do not use the final test period to select features, tune models, or choose thresholds.
7. Save random seeds and model settings where applicable.
8. Verify that another team member can run the pipeline and reproduce the main tables and figures.
9. Use branches and pull requests for code changes; review changes before merging them into the main branch.

## Team Work Allocation

| Member | Primary responsibility |
|---|---|
| Yutong Chen | Data acquisition, company matching, bankruptcy labels, and reproducibility audit |
| Shuyi Chen | Exploratory analysis, visualizations, charts, and tables |
| Sourav Banik | Data cleaning, feature engineering, and look-ahead-bias review |
| Hao Yang | Predictive modeling, model comparison, and evaluation metrics |
| Khizer Shabbir | Business interpretation, deployment, report integration, and presentation coordination |

All team members contribute to the analysis, final write-up, code review, and presentation.

## Project Timeline

| Period | Focus | Main output |
|---|---|---|
| 09/15/2026–09/27/2026 | Data integration, initial cleaning, exploratory analysis, baseline model | Project Update / Deliverable #2 |
| 09/28/2026–10/04/2026 | Final feature engineering, predictive modeling, evaluation, and business-value simulation | Complete report draft |
| 10/05/2026–10/08/2026 | Reproducibility checks, report editing, slides, and rehearsal | Final report, data, code, slides, and presentation |

## Final Deliverables

### Deliverable #2 — Project Update

The update will summarize data-cleaning progress, preliminary results, issues encountered, selected industry, and next steps.

### Deliverable #3 — Final Submission

The final submission will include:

- Final write-up of no more than 10 double-spaced pages
- Data or documented data extract
- Well-documented code
- Presentation slides in PPT or PDF format
- Citations and bibliography
- Appendix describing each team member’s contribution

### Deliverable #4 — Presentation

The presentation will communicate the project as an executive or investor pitch. It will cover the business problem, data, key patterns, modeling approach, performance, screening value, limitations, and recommended use.

## Academic Context

This project is structured using the CRISP-DM data-mining process and is being completed for the team term project. The proposal and term-project instructions define the business problem, required analytical tasks, report structure, and submission requirements.

