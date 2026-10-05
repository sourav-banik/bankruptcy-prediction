# Bankruptcy Early-Warning Prediction Model

## Project overview

This repository contains a 22-notebook workflow for building and evaluating an event-based bankruptcy early-warning model for publicly traded U.S. companies. It combines bankruptcy filing records with Bloomberg company, market, and macroeconomic data.

Each observation is one company at one prediction date. Bankrupt-company prediction dates are set 90 days before filing. Non-bankrupt firms receive candidate pseudo-event dates based on bankrupt-company prediction dates and are retained as negative examples only when their full 12-month outcome window ends on or before the verified bankruptcy-label cutoff of **2022-12-31**. Bloomberg market-data availability is used to establish historical feature coverage before a prediction date; it is not evidence that a company avoided bankruptcy.

The matched case-control sample is designed for comparing relative risk. Its class balance does not represent real-world bankruptcy prevalence, so model scores should not be interpreted as population bankruptcy probabilities.

## Workflow

Run notebooks in numerical order. Notebooks 01 and 02 prepare Bloomberg requests; the Bloomberg Excel Add-In must populate the generated workbooks before cleaning begins. The populated workbooks are read from `data/processed/` by notebook 03.

| Notebooks | Purpose | Main outputs |
|---|---|---|
| 01–02 | Create the Bloomberg security list and request workbooks. Security requests are split into sheets of up to 250 companies; macro requests are separate. | `data/intermediate/bloomberg_security_list.csv`, `data/intermediate/security/`, `data/intermediate/macro/` |
| 03–04 | Clean Bloomberg exports and inspect company coverage, missingness, distributions, and macro data. | `clean_company_long.csv`, `clean_macro_long.csv`, `company_availability.csv`; reports and figures under `artifacts/04_profile_and_visualize_clean_data/` |
| 05–06 | Create bankruptcy and eligible non-bankrupt candidate events, then match controls by year, Bloomberg industry, size, and sector preference. | Event CSVs and matched samples in `data/processed/` (1:2 primary; 1:1 and 1:3 robustness) |
| 07–12 | Align information to each prediction date and create original, derived, trend, market, and macro features. | Stage-specific CSVs in `data/processed/` |
| 13–15 | Combine the feature families, validate the modeling dataset, and visualize event-aligned data. | `modeling_dataset_unprocessed.csv`; validation reports and visualization figures under `artifacts/` |
| 16–18 | Create chronological grouped splits, fit preprocessing on training data, and produce training-only feature diagnostics. | Split IDs and matrices in `data/processed/`; pipeline in `models/`; diagnostics under `artifacts/18_feature_independence_and_selection/` |
| 19–20 | Tune and evaluate Elastic Net logistic regression and Random Forest using the same time-based test sample. | Models in `models/`; metrics, predictions, tuning results, and reports under `artifacts/19_model_elastic_net_logistic/` and `artifacts/20_model_random_forest/` |
| 21–22 | Run robustness analyses and calibrate scores using validation predictions. | Robustness results in `reports/`; calibration report and metrics in `reports/`; comparison and calibration figures in `figures/` |

Notebook 02 creates formula-based workbooks for use in Excel with the Bloomberg Add-In. After retrieval, place the populated company and macro workbooks under `data/processed/` so notebook 03 can clean them. Bloomberg data use must comply with the applicable license; do not redistribute raw exports unless permitted.

## Event and feature design

- A bankrupt event uses `prediction_date = bankruptcy_date - 90 days` and a 12-month outcome window.
- A non-bankrupt candidate uses a bankrupt-company prediction date as its pseudo-event date. Its outcome window must end by the verified label cutoff, and known filings within that window disqualify it.
- Matching exports include a pre-specified 1:2 primary sample and 1:1/1:3 robustness samples, with control-company reuse capped at three. Matching-variable balance is reported with standardized mean differences.
- Financial statements use the latest observation at least 90 days before prediction. Market and macro observations are aligned as of each event date. Market-history eligibility requires 13 monthly prices to form 12 monthly returns.
- Feature generation includes original financial and market fields, financial ratios, historical changes and trends, market measures, and macroeconomic measures. Missing values are not imputed until notebook 17; preprocessing is fit using training data only.
- Notebook 16 uses chronological splits, purges outcome windows that cross split boundaries, and keeps company/security groups together. Cutoffs adjust to meet configured event-count minimums; random splitting is not the primary design.
- Elastic Net logistic regression is the primary benchmark; Random Forest is the nonlinear comparison. Model selection uses validation data, with the held-out test period reserved for final evaluation.

## Repository layout

```text
.
├── data/
│   ├── raw/                  # Supplied bankrupt and non-bankrupt company files
│   ├── intermediate/         # Security list and Bloomberg request workbooks
│   └── processed/            # Clean, event, feature, split, and matrix CSVs
├── notebooks/                # Ordered workflow notebooks 01–22
├── artifacts/                # Reports and figures grouped by notebook/stage
├── reports/                  # Robustness and calibration reports
├── figures/                  # Robustness and calibration figures
├── models/                   # Saved preprocessing and model objects
├── docs/                     # Project documentation
├── notebook_utils.py         # Shared notebook helpers and project paths
├── project_config.py         # Bloomberg mappings and shared feature schemas
├── requirements.txt          # Python dependencies
└── README.md
```

Shared field mappings and feature schemas—including financial and market columns, ratio formulas, trend columns, and macro definitions—are maintained in `project_config.py`. Common functions for file hashing, date/security normalization, safe division, and evaluation are in `notebook_utils.py`. Notebooks import these modules instead of installing packages individually.

## Setup and execution

From the project root, install the declared dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks from this repository. Run them in order, pausing after notebook 02 to retrieve Bloomberg data and place the completed workbooks in `data/processed/`. Later notebooks consume outputs from earlier stages. Re-run upstream notebooks after changing their inputs or configuration so downstream tables stay consistent.

## Main generated files

The principal processed tables include:

- `data/processed/clean_company_long.csv`
- `data/processed/clean_macro_long.csv`
- `data/processed/company_availability.csv`
- `data/processed/bankruptcy_events.csv`
- `data/processed/nonbankruptcy_candidate_events.csv`
- `data/processed/matched_event_sample_1to1.csv` (robustness), `matched_event_sample_1to2.csv` (primary), and `matched_event_sample_1to3.csv` (robustness)
- `data/processed/event_aligned_company_data.csv` and `event_aligned_macro_data.csv`
- `data/processed/original_features.csv`, `derived_features.csv`, `trend_features.csv`, `market_features.csv`, and `macro_features.csv`
- `data/processed/modeling_dataset_unprocessed.csv`
- `data/processed/train_matrix.csv`, `validation_matrix.csv`, and `test_matrix.csv`

Generated reports and figures are grouped under `artifacts/<notebook_name>/` where the notebook defines a stage-specific directory. Notebooks 21 and 22 currently write their report files to `reports/` and figures to `figures/`.

## Limitations and interpretation

The matched sample is not population-representative, and therefore its raw scores and calibration do not establish real-world bankruptcy probabilities. Bloomberg coverage, company identifiers, and bankruptcy records may be incomplete. A company with no known filing in the verified observation window is labeled non-bankrupt for that window only; it is not a claim that the company never failed or remained solvent afterward. Results are research and screening outputs, not investment recommendations.
