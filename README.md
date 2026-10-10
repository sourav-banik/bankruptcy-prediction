# Bankruptcy Early-Warning Prediction Model

## Project overview

This repository contains a 22-notebook workflow for preparing Bloomberg data and building an event-based bankruptcy early-warning model for publicly traded U.S. companies. The workflow combines bankruptcy filing records with company financial statements, market data, and macroeconomic series.

Each observation represents one company at one prediction date. A bankrupt firm's prediction date is 90 days before its filing. Non-bankrupt firms receive candidate pseudo-event dates based on bankrupt firms' prediction dates. A negative observation is eligible only when its 12-month outcome window ends on or before the verified bankruptcy-label cutoff of **2022-12-31**. Bloomberg data availability establishes feature coverage; it does not establish that a firm remained non-bankrupt.

The matched case-control sample supports comparisons of relative bankruptcy risk. Its constructed class balance does not represent real-world bankruptcy prevalence, so model scores should not be interpreted as population probabilities.

## Workflow

Run notebooks in numerical order. Notebook 02 prepares Bloomberg request workbooks; the Bloomberg Excel Add-In must populate them before notebook 03 can clean the exports. Place the completed company and macro workbooks in `data/processed/`.

| Notebooks | Purpose | Main outputs |
|---|---|---|
| 01-02 | Prepare Bloomberg identifiers and generate company and macro request workbooks. Company requests use sheets of up to 250 companies. | `data/intermediate/bloomberg_security_list.csv`, `data/intermediate/security/`, `data/intermediate/macro/` |
| 03-04 | Clean Bloomberg exports and profile company coverage, missingness, financial distributions, and macro data. | Clean long tables and availability data in `data/processed/`; stage 01 reports and figures |
| 05-06 | Create bankruptcy and eligible non-bankrupt candidate events, then match controls by year, industry, size, and sector preference. | Event and matched-sample CSVs in `data/processed/`; 1:2 primary and 1:1/1:3 sensitivity samples |
| 07-12 | Align financial, market, and macro data to event dates; create original, derived, trend, market, and macro features. | Stage-specific feature CSVs in `data/processed/` |
| 13-15 | Combine feature families, validate the modeling dataset, and visualize event-aligned data. | `modeling_dataset_unprocessed.csv`; stage 02 and 03 reports and figures |
| 16-18 | Create chronological grouped splits, fit preprocessing using training data, and run training-only feature diagnostics. | Split IDs and matrices in `data/processed/`; preprocessing pipeline in `models/`; stage 04 reports |
| 19-20 | Tune and evaluate Elastic Net logistic regression and Random Forest on the same time-based test sample. | Model objects in `models/`; metrics, predictions, reports, and figures in stage 05 and 06 folders |
| 21-22 | Compare robustness scenarios and report relative-score calibration diagnostics. | Stage 07 and 08 reports and figures |

Notebook 19 first compares compact and full feature specifications using walk-forward out-of-fold PR-AUC. It selects compact when the full model improves by no more than 0.02; otherwise it selects full. Elastic Net tuning and threshold selection follow, and only then does notebook 19 evaluate the frozen model on the final test split. Notebook 21 reports validation robustness without using test results to select the feature specification.

Bloomberg data use must comply with the applicable license. Do not redistribute raw Bloomberg exports unless permitted.

## Event and feature design

- Bankrupt events use `prediction_date = bankruptcy_date - 90 days` and a 12-month prediction horizon.
- Non-bankrupt candidate events use bankrupt-company prediction dates as pseudo-events. Their outcome windows must end by the verified label cutoff; known filings during the window disqualify the observation.
- The primary matched sample is pre-specified at 1:2. The 1:1 and 1:3 samples support sensitivity analysis. Control reuse is capped at three, and matching balance is summarized with standardized mean differences.
- Financial statements are aligned to the latest report available at least 90 days before prediction. Market and macro data are aligned as of each prediction date. Market eligibility requires 13 monthly prices to calculate 12 monthly returns.
- Feature generation retains original financial and market fields and adds ratios, historical changes and trends, market measures, and macroeconomic measures. Missing values are not imputed until notebook 17; preprocessing is fitted on training data only.
- Notebook 16 creates chronological, grouped splits and prevents outcome windows from crossing split boundaries. Random splitting is not the primary design.
- Elastic Net logistic regression is the primary benchmark. Random Forest is a nonlinear comparison. Model and threshold selection use validation data; the held-out test period is reserved for final evaluation.

## Repository layout

```text
.
|-- data/
|   |-- raw/                  # Supplied bankrupt and non-bankrupt company files
|   |-- intermediate/         # Security list and Bloomberg request workbooks
|   `-- processed/            # Clean, event, feature, split, and matrix data
|-- notebooks/                # Ordered project workflow, notebooks 01-22
|-- analysis/                 # Flat, stage-numbered analysis scripts 01-08
|-- app/                      # Streamlit project page and model sandbox
|-- reports/                  # Generated reports grouped by analysis stage
|-- figures/                  # Generated figures grouped by analysis stage
|-- models/                   # Saved preprocessing and model objects
|-- docs/                     # Project documentation
|-- notebook_utils.py         # Shared helpers and standard project paths
|-- project_config.py         # Bloomberg mappings and shared feature schemas
|-- requirements.txt          # Python dependencies
`-- README.md
```

Analysis scripts are kept flat in `analysis/` and are invoked by their corresponding numbered notebooks:

| Script | Analysis stage |
|---|---|
| `01_data_profile.py` | Clean-data profile and coverage |
| `02_modeling_data_validation.py` | Modeling-data, matching, and leakage validation |
| `03_event_aligned_visualization.py` | Event-aligned distributions, trajectories, and balance |
| `04_feature_diagnostics.py` | Training-only feature diagnostics and compact selection |
| `05_elastic_net_reporting.py` | Elastic Net validation and test reporting |
| `06_random_forest_reporting.py` | Random Forest and Elastic Net comparison |
| `07_robustness_visualization.py` | Robustness scenario results |
| `08_probability_calibration.py` | Relative-risk calibration diagnostics |

Outputs are grouped under matching stage directories in `reports/` and `figures/`; some stages produce reports and tables without standalone figures.

## Interactive project page

**Live demo:** [bankruptcy-lab.streamlit.app](https://bankruptcy-lab.streamlit.app)

The Streamlit page in `app/streamlit_app.py` has two tabs: Project and Explore companies. Visitors can select a company and historical prediction date from `data/processed/modeling_dataset_unprocessed.csv`; the app displays the event's financial, market, and macro metrics, summarizes one-, two-, and three-year trends, and scores the selected row with Elastic Net, Random Forest, or the Gradient Boosting sensitivity model. The Project tab includes phase-based report and figure selectors, with HTML reports rendered inline. The app labels outputs as relative bankruptcy risk scores and ranks them against walk-forward validation scores for that model; they are not population bankruptcy probabilities. The app does not require raw Bloomberg workbooks. Confirm that publishing the processed dataset and fitted models is allowed by the applicable Bloomberg data license.

Run it locally from the repository root:

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py
```

For deployment or maintenance, use Streamlit Community Cloud with `app/streamlit_app.py` as the entrypoint. The processed modeling dataset and saved model artifacts must be available in the deployed repository; rerun the upstream notebooks if they are missing. Follow [Streamlit's deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). Do not publish raw Bloomberg exports, and verify the relevant data license before sharing processed data or model artifacts.

Shared Bloomberg mappings and feature schemas, such as financial and market columns, ratio formulas, trend columns, and macro definitions, are maintained in `project_config.py`. Common functions for file hashing, date and security normalization, safe division, and evaluation are in `notebook_utils.py`. Notebooks import these modules and use the dependencies declared in `requirements.txt`.

## Setup and execution

From the project root, install dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks from this repository and run them in numerical order. Pause after notebook 02 to retrieve the Bloomberg data, then put the completed workbooks in `data/processed/` before continuing with notebook 03. Downstream stages consume the files created upstream; rerun affected downstream notebooks when an input or shared configuration changes.

## Main generated files

- `data/processed/clean_company_long.csv`
- `data/processed/clean_macro_long.csv`
- `data/processed/company_availability.csv`
- `data/processed/bankruptcy_events.csv`
- `data/processed/nonbankruptcy_candidate_events.csv`
- `data/processed/matched_event_sample_1to1.csv`, `matched_event_sample_1to2.csv` (primary), and `matched_event_sample_1to3.csv`
- `data/processed/event_aligned_company_data.csv` and `event_aligned_macro_data.csv`
- `data/processed/original_features.csv`, `derived_features.csv`, `trend_features.csv`, `market_features.csv`, and `macro_features.csv`
- `data/processed/modeling_dataset_unprocessed.csv`
- `data/processed/train_matrix.csv`, `validation_matrix.csv`, and `test_matrix.csv`

The analysis scripts and notebooks write reports to `reports/` and figures to `figures/`. These directories hold presentation and diagnostic outputs; `data/` and `models/` hold pipeline inputs and outputs.

## Limitations and interpretation

The matched sample is not population-representative, so its scores and calibration diagnostics do not establish real-world bankruptcy probabilities. Bloomberg coverage, company identifiers, and bankruptcy records may be incomplete. A company with no known filing during its verified outcome window is labeled non-bankrupt for that window only; this does not assert that the company never failed or remained solvent afterward. Results are research and screening outputs, not investment recommendations.
