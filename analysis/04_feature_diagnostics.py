"""Run training-only redundancy, selection, and missingness diagnostics."""

import base64
import html
import io
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils import file_hash, project_paths  # noqa: E402
from project_config import COMPACT_FEATURE_COLUMNS  # noqa: E402

PROCESSED = project_paths(ROOT).processed
REPORTS = project_paths(ROOT).reports / "04_feature_diagnostics"
REVIEW_CORRELATION = 0.80
NEAR_DUPLICATE_CORRELATION = 0.9999
AFFINE_TOLERANCE = 1e-9
MIN_RAW_PAIRS = 30
MIN_BINARY_LEVEL_COUNT = 10
VIF_REVIEW_THRESHOLD = 10.0
MI_RANDOM_STATE = 42
input_hashes = {}


train_path = PROCESSED / "train_matrix.csv"
ids_path = PROCESSED / "train_ids.csv"
artifact_path = ROOT / "models/preprocessing_pipeline.joblib"
raw_path = PROCESSED / "modeling_dataset_unprocessed.csv"
for path in [train_path, ids_path, artifact_path, raw_path]:
    input_hashes[path] = file_hash(path)
pipeline = joblib.load(artifact_path)
schema = pipeline.project_schema_
if schema["fit_split"] != "train":
    raise ValueError("Preprocessing must have been fitted on training")
train = pd.read_csv(
    train_path, dtype={"company_id": "string", "metadata_event_id": "string"}
)
train_ids = pd.read_csv(
    ids_path, dtype={"company_id": "string", "metadata_event_id": "string"}
)
if (
    input_hashes[ids_path] != schema["split_file_sha256"]["train"]
    or input_hashes[raw_path] != schema["source_sha256"]
):
    raise ValueError("Source/split changed; rerun notebook 17 before diagnostics")
if train.metadata_event_id.tolist() != schema["training_event_ids"]:
    raise ValueError("Training event membership/order changed")
if (
    train.empty
    or train.duplicated(["company_id", "prediction_date"]).any()
    or not train.target.isin([0, 1]).all()
):
    raise ValueError("Invalid training rows")
train_before = train.copy(deep=True)
FEATURES = schema["matrix_feature_columns"]
if train.columns.tolist() != schema["matrix_metadata_columns"] + FEATURES:
    raise ValueError("Matrix schema mismatch")
X = train[FEATURES].astype(float).copy()
y = train.target.to_numpy(dtype=int)
groups = train.company_group_id.to_numpy()
if len(np.unique(y)) != 2 or not np.isfinite(X.to_numpy()).all():
    raise ValueError("Need two classes and finite predictors")

# Read raw rows, retaining only exact training keys before any diagnostic statistic.
KEYS = ["company_id", "prediction_date"]
training_keys = set(pd.MultiIndex.from_frame(train_ids[KEYS]).tolist())
raw_columns = list(dict.fromkeys(KEYS + ["target"] + schema["input_feature_columns"]))
raw_chunks = []
for chunk in pd.read_csv(
    raw_path, usecols=raw_columns, dtype={"company_id": "string"}, chunksize=256
):
    keys = pd.MultiIndex.from_frame(chunk[KEYS])
    selected = chunk.loc[keys.isin(training_keys)].copy()
    if len(selected):
        raw_chunks.append(selected)
raw_train = pd.concat(raw_chunks, ignore_index=True)
if raw_train.duplicated(KEYS).any():
    raise ValueError("Duplicate raw training events")
raw_train = (
    raw_train.set_index(KEYS).reindex(pd.MultiIndex.from_frame(train_ids[KEYS])).copy()
)
raw_train.index = train.index
if raw_train.target.isna().any() or not raw_train.target.eq(train.target).all():
    raise ValueError("Raw training membership/labels disagree")
pipeline_state = joblib.hash(pipeline)

# Families describe actual feature provenance, not ID/target metadata.
financial_base = {
    "total_assets",
    "total_liabilities",
    "total_debt",
    "short_term_debt",
    "long_term_debt",
    "cash_short_term_investments",
    "current_assets",
    "current_liabilities",
    "shareholders_equity",
    "accounts_receivable",
    "inventory",
    "working_capital",
    "revenue",
    "ebit",
    "ebitda",
    "net_income",
    "interest_expense",
    "depreciation_amortization",
    "operating_cash_flow",
    "capital_expenditures",
    "free_cash_flow",
}
market_base = {
    "adjusted_price",
    "market_cap",
    "shares_outstanding",
    "volume",
    "turnover",
}


def feature_details(feature):
    branch, source = feature.split("__", 1)
    if branch == "categorical":
        field = "sector" if source.startswith("sector_") else "industry"
        return {
            "branch": branch,
            "source_column": field,
            "family": "categorical_" + field,
            "block": "categorical_" + field,
            "discrete": True,
            "effect_unit": "0 to 1 category indicator",
        }
    if branch == "missing":
        source = source.removeprefix("missingindicator_")
    if source.startswith("original_"):
        base = source.removeprefix("original_")
        family = (
            "original_financial"
            if base in financial_base
            else "original_market_history"
            if base.startswith("market_monthly_price_")
            else "original_market"
        )
        block = family
    elif source.startswith("derived_"):
        family = block = "derived_ratios"
    elif source.startswith("trend_"):
        body = source.removeprefix("trend_")
        base, kind, period = body.rsplit("_", 2)
        family = (
            "trend_financial"
            if base in financial_base
            else "trend_market"
            if base in market_base
            else "trend_ratios"
        )
        block = family + "_" + kind + "_" + period
    elif source.startswith("market_"):
        family = block = "market_features"
    elif source.startswith("macro_"):
        family = block = "macro_features"
    else:
        raise ValueError(f"Unrecognized feature provenance: {feature}")
    if branch == "missing":
        family = "missing_" + family
        block = "missing_" + block
    return {
        "branch": branch,
        "source_column": source,
        "family": family,
        "block": block,
        "discrete": branch == "missing",
        "effect_unit": "0 to 1 missingness indicator"
        if branch == "missing"
        else "one training SD of median-imputed numeric value",
    }


feature_info = pd.DataFrame(
    [{"feature": f, **feature_details(f)} for f in FEATURES]
).set_index("feature")
constant_mask = X.nunique().le(1)
variable_features = constant_mask.index[~constant_mask].tolist()
variable_X = X[variable_features]

# Compute both dependence measures without collapsing financial/market families.
pearson = variable_X.corr(method="pearson")
spearman = variable_X.corr(method="spearman")
numeric_sources = schema["retained_numeric_columns"]
raw_numeric = raw_train[numeric_sources]
raw_pearson = raw_numeric.corr(method="pearson", min_periods=MIN_RAW_PAIRS)
raw_spearman = raw_numeric.corr(method="spearman", min_periods=MIN_RAW_PAIRS)
observed_numeric = raw_numeric.notna().astype("int32")
raw_pair_counts = observed_numeric.T.dot(observed_numeric)

# Pair-level findings include exact/affine duplicates, monotonic relationships,
# and threshold flags; these are review candidates, never automatic deletions.
A = variable_X.to_numpy(dtype=float)
standardized = (A - A.mean(axis=0)) / A.std(axis=0, ddof=0)
rows, cols = np.triu_indices(len(variable_features), k=1)
p_values = pearson.to_numpy()[rows, cols]
s_values = spearman.to_numpy()[rows, cols]
selected_pairs = (np.abs(p_values) >= REVIEW_CORRELATION) | (
    np.abs(s_values) >= REVIEW_CORRELATION
)
redundancy_rows = []
for left, right, p, s in zip(
    rows[selected_pairs],
    cols[selected_pairs],
    p_values[selected_pairs],
    s_values[selected_pairs],
):
    a, b = variable_features[left], variable_features[right]
    exact = np.array_equal(A[:, left], A[:, right])
    difference = standardized[:, left] - (1 if p >= 0 else -1) * standardized[:, right]
    max_difference = float(np.max(np.abs(difference)))
    affine = max_difference <= AFFINE_TOLERANCE
    near = abs(p) >= NEAR_DUPLICATE_CORRELATION
    monotonic = abs(s) >= NEAR_DUPLICATE_CORRELATION
    info_a, info_b = feature_info.loc[a], feature_info.loc[b]
    raw_p = raw_s = np.nan
    raw_n = np.nan
    if info_a.branch == "numeric" and info_b.branch == "numeric":
        source_a, source_b = info_a.source_column, info_b.source_column
        raw_p = raw_pearson.loc[source_a, source_b]
        raw_s = raw_spearman.loc[source_a, source_b]
        raw_n = int(raw_pair_counts.loc[source_a, source_b])
    relation = (
        "exact_duplicate"
        if exact
        else "affine_duplicate"
        if affine
        else "near_affine_duplicate"
        if near
        else "near_monotonic"
        if monotonic
        else "high_correlation"
    )
    missing_pattern = (
        info_a.branch == "missing" and info_b.branch == "missing" and (exact or affine)
    )
    redundancy_rows.append(
        {
            "record_type": "feature_pair",
            "feature_a": a,
            "feature_b": b,
            "family_a": info_a.family,
            "family_b": info_b.family,
            "relation": relation,
            "training_n": len(train),
            "pearson": p,
            "spearman": s,
            "observed_raw_pearson": raw_p,
            "observed_raw_spearman": raw_s,
            "observed_raw_pair_n": raw_n,
            "exact_duplicate": exact,
            "affine_duplicate": affine,
            "near_affine_duplicate": near,
            "near_monotonic": monotonic,
            "duplicate_missing_pattern": missing_pattern,
            "max_standardized_difference": max_difference,
            "constant_value": np.nan,
            "recommendation": "Review shared missingness; equality may change in future"
            if missing_pattern
            else "Review definition/alias before choosing a representative"
            if exact or affine
            else "Retain for Elastic Net consideration; do not remove automatically",
        }
    )
for feature in constant_mask.index[constant_mask]:
    info = feature_info.loc[feature]
    redundancy_rows.append(
        {
            "record_type": "single_feature",
            "feature_a": feature,
            "feature_b": "",
            "family_a": info.family,
            "family_b": "",
            "relation": "constant_in_training",
            "training_n": len(train),
            "pearson": np.nan,
            "spearman": np.nan,
            "observed_raw_pearson": np.nan,
            "observed_raw_spearman": np.nan,
            "observed_raw_pair_n": np.nan,
            "exact_duplicate": False,
            "affine_duplicate": False,
            "near_affine_duplicate": False,
            "near_monotonic": False,
            "duplicate_missing_pattern": False,
            "max_standardized_difference": np.nan,
            "constant_value": float(X[feature].iloc[0]),
            "recommendation": "Review training-constant signal; reserved missing/novel-missing indicators may activate later",
        }
    )
redundancy = pd.DataFrame(redundancy_rows)
redundancy = redundancy.sort_values(
    ["record_type", "relation", "feature_a", "feature_b"], kind="stable"
).reset_index(drop=True)


# Compute classical VIF while distinguishing exact dependency from near collinearity.
# For centered unit-length X, null-space participation implies R_j^2=1 and VIF=inf.
# Finite VIF is diagonal inverse-Gram only when the feature is individually identifiable.
def vif_svd(frame):
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not len(frame.columns):
        raise ValueError("VIF needs finite nonempty predictors")
    centered = values - values.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if (norms == 0).any():
        raise ValueError("Exclude constant predictors before VIF")
    standardized = centered / norms
    U, singular_values, Vt = np.linalg.svd(standardized, full_matrices=False)
    tolerance = singular_values[0] * max(standardized.shape) * np.finfo(float).eps
    rank = int((singular_values > tolerance).sum())
    row_basis = Vt[:rank]
    null_participation = np.clip(1 - np.sum(row_basis**2, axis=0), 0, 1)
    exact_dependent = null_participation > 1e-8
    diagonal_inverse = np.sum((row_basis / singular_values[:rank, None]) ** 2, axis=0)
    vif = np.where(exact_dependent, np.inf, np.maximum(diagonal_inverse, 1.0))
    table = pd.DataFrame(
        {
            "vif": vif,
            "exact_linear_dependency": exact_dependent,
            "null_space_participation": null_participation,
        },
        index=frame.columns,
    )
    return table, {
        "rows": len(frame),
        "predictors": len(frame.columns),
        "rank": rank,
        "nullity": len(frame.columns) - rank,
        "svd_tolerance": tolerance,
    }


full_vif, full_design = vif_svd(variable_X)

# Smaller blocks are supplementary diagnostics. Remove within-block affine copies
# only for the diagnostic calculation; keep every original matrix predictor.
affine_pairs = redundancy.loc[
    redundancy.record_type.eq("feature_pair") & redundancy.affine_duplicate
]
affine_neighbors = {feature: set() for feature in variable_features}
for row in affine_pairs.itertuples(index=False):
    affine_neighbors[row.feature_a].add(row.feature_b)
    affine_neighbors[row.feature_b].add(row.feature_a)
block_vif_rows = []
block_design_rows = []
for block, information in feature_info.loc[variable_features].groupby(
    "block", sort=True
):
    candidates = information.index.tolist()
    reference = None
    if block.startswith("categorical_"):
        reference = X[candidates].mean().idxmax()
    representatives = []
    exclusions = {}
    for feature in candidates:
        if feature == reference:
            exclusions[feature] = "reference dummy for diagnostic intercept"
        else:
            alias = next(
                (
                    retained
                    for retained in representatives
                    if retained in affine_neighbors[feature]
                ),
                None,
            )
            if alias is not None:
                exclusions[feature] = "affine duplicate of " + alias
            else:
                representatives.append(feature)
    if representatives:
        table, design = vif_svd(X[representatives])
        block_design_rows.append(
            {
                "block": block,
                "candidate_features": len(candidates),
                "diagnostic_features": len(representatives),
                "rank": design["rank"],
                "nullity": design["nullity"],
                "reference_feature": reference,
                "infinite_vif_features": int(np.isinf(table.vif).sum()),
                "vif_over_10_features": int(table.vif.gt(VIF_REVIEW_THRESHOLD).sum()),
            }
        )
        for feature in representatives:
            block_vif_rows.append(
                {
                    "feature": feature,
                    "diagnostic_vif": table.loc[feature, "vif"],
                    "diagnostic_vif_block": block,
                    "diagnostic_vif_note": "included in supplementary block",
                }
            )
    for feature, reason in exclusions.items():
        block_vif_rows.append(
            {
                "feature": feature,
                "diagnostic_vif": np.nan,
                "diagnostic_vif_block": block,
                "diagnostic_vif_note": reason,
            }
        )
block_vif = pd.DataFrame(block_vif_rows).set_index("feature")
block_designs = pd.DataFrame(block_design_rows)

# Mutual information uses training labels and the correct discrete/continuous mask.
# Numerical imputed observations are continuous; missingness/dummy indicators discrete.
mi = pd.Series(0.0, index=FEATURES, name="mutual_information_nats")
discrete_mask = feature_info.loc[variable_features, "discrete"].to_numpy(dtype=bool)
mi.loc[variable_features] = mutual_info_classif(
    variable_X.to_numpy(),
    y,
    discrete_features=discrete_mask,
    n_neighbors=3,
    random_state=MI_RANDOM_STATE,
    n_jobs=1,
)


# Univariate MLE is not finite under complete/quasi separation. Detect disjoint
# class supports (including touching endpoints) before reporting Wald statistics.
def separated_support(values, labels):
    a, b = values[labels == 0], values[labels == 1]
    return bool(a.max() <= b.min() or b.max() <= a.min())


def finite_exponential(value):
    if not np.isfinite(value):
        return np.nan
    if value > np.log(np.finfo(float).max):
        return np.inf
    return float(np.exp(value))


result_rows = []
for feature in FEATURES:
    info = feature_info.loc[feature]
    values = X[feature].to_numpy(dtype=float)
    source_missing = raw_train[info.source_column].isna()
    unique, counts = np.unique(values, return_counts=True)
    binary = len(unique) <= 2 and np.isin(unique, [0.0, 1.0]).all()
    rare_binary = bool(
        binary and len(unique) == 2 and counts.min() < MIN_BINARY_LEVEL_COUNT
    )
    record = {
        "feature": feature,
        "family": info.family,
        "branch": info.branch,
        "source_column": info.source_column,
        "effect_unit": info.effect_unit,
        "training_n": len(train),
        "company_groups": len(np.unique(groups)),
        "bankrupt_n": int(y.sum()),
        "nonbankrupt_n": int((y == 0).sum()),
        "unique_values": len(unique),
        "constant_in_training": bool(constant_mask[feature]),
        "rare_binary_level": rare_binary,
        "smallest_binary_level_count": int(counts.min())
        if binary and len(unique) == 2
        else np.nan,
        "source_missing_n": int(source_missing.sum()),
        "source_missing_pct_bankrupt": float(
            source_missing.loc[train.target.eq(1)].mean() * 100
        ),
        "source_missing_pct_nonbankrupt": float(
            source_missing.loc[train.target.eq(0)].mean() * 100
        ),
        "mutual_information_nats": float(mi[feature]),
        "logistic_status": "not_fitted",
        "converged": False,
        "coef_log_odds": np.nan,
        "cluster_robust_se": np.nan,
        "odds_ratio": np.nan,
        "odds_ratio_ci_low": np.nan,
        "odds_ratio_ci_high": np.nan,
        "p_value": np.nan,
        "q_value_bh": np.nan,
        "training_auc": np.nan,
        "training_log_likelihood": np.nan,
        "fit_warnings": "",
        "full_design_vif": full_vif.loc[feature, "vif"]
        if feature in full_vif.index
        else np.nan,
        "diagnostic_vif": block_vif.loc[feature, "diagnostic_vif"]
        if feature in block_vif.index
        else np.nan,
        "diagnostic_vif_block": info.block,
        "diagnostic_vif_note": block_vif.loc[feature, "diagnostic_vif_note"]
        if feature in block_vif.index
        else "constant; VIF undefined",
    }
    if constant_mask[feature]:
        record["logistic_status"] = "constant_predictor"
    elif separated_support(values, y):
        record["logistic_status"] = "complete_or_quasi_separation"
        record["fit_warnings"] = (
            "Class supports are disjoint/touching; unpenalized MLE is not finite"
        )
    else:
        try:
            design = np.column_stack([np.ones(len(values)), values])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(
                    maxiter=100,
                    tol=1e-8,
                    cov_type="cluster",
                    cov_kwds={"groups": groups},
                    disp=0,
                )
            warning_text = "; ".join(sorted({str(w.message) for w in captured}))
            record["fit_warnings"] = warning_text
            record["converged"] = bool(fit.converged)
            separation_warning = any(
                issubclass(w.category, PerfectSeparationWarning) for w in captured
            )
            convergence_warning = any(
                issubclass(w.category, ConvergenceWarning) for w in captured
            )
            coefficients_valid = (
                np.isfinite(fit.params).all()
                and np.isfinite(fit.bse).all()
                and np.isfinite(fit.pvalues).all()
            )
            if separation_warning:
                record["logistic_status"] = "complete_or_quasi_separation"
            elif not fit.converged or convergence_warning:
                record["logistic_status"] = "not_converged"
            elif captured:
                record["logistic_status"] = "fit_warning_review"
            elif not coefficients_valid:
                record["logistic_status"] = "nonfinite_inference"
            else:
                coef = float(fit.params[1])
                se = float(fit.bse[1])
                limits = fit.conf_int(alpha=0.05)[1]
                record.update(
                    logistic_status="ok",
                    coef_log_odds=coef,
                    cluster_robust_se=se,
                    odds_ratio=finite_exponential(coef),
                    odds_ratio_ci_low=finite_exponential(limits[0]),
                    odds_ratio_ci_high=finite_exponential(limits[1]),
                    p_value=float(fit.pvalues[1]),
                    training_auc=float(roc_auc_score(y, fit.predict(design))),
                    training_log_likelihood=float(fit.llf),
                )
        except (
            PerfectSeparationError,
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ) as exc:
            record["logistic_status"] = "fit_error"
            record["fit_warnings"] = type(exc).__name__ + ": " + str(exc)
    record["source_missing_difference_pp"] = (
        record["source_missing_pct_bankrupt"] - record["source_missing_pct_nonbankrupt"]
    )
    result_rows.append(record)
feature_results = pd.DataFrame(result_rows)
valid_p = feature_results.logistic_status.eq("ok") & feature_results.p_value.notna()
if valid_p.any():
    feature_results.loc[valid_p, "q_value_bh"] = multipletests(
        feature_results.loc[valid_p, "p_value"], method="fdr_bh"
    )[1]
feature_results = feature_results.sort_values(
    ["mutual_information_nats", "feature"], ascending=[False, True], kind="stable"
).reset_index(drop=True)

# Select from the pre-specified compact economic feature list using training rows only.
# Variance and correlation rules filter candidates; MI, univariate fits, and VIF are reported.
COMPACT_CORRELATION_THRESHOLD = 0.95
COMPACT_VIF_THRESHOLD = VIF_REVIEW_THRESHOLD
compact_candidates = ["numeric__" + name for name in COMPACT_FEATURE_COLUMNS]
compact_results = []
compact_available = []

for priority, feature in enumerate(compact_candidates, start=1):
    row = {
        "economic_priority": priority,
        "source_feature": feature.removeprefix("numeric__"),
        "feature": feature,
        "mutual_information_nats": np.nan,
        "mutual_information_rank": np.nan,
        "univariate_q_value": np.nan,
        "univariate_training_auc": np.nan,
        "max_abs_correlation": np.nan,
        "correlated_with": "",
        "vif_at_removal_or_final": np.nan,
        "selected": False,
        "selection_status": "excluded",
        "selection_reason": "",
    }
    if feature not in X.columns:
        row["selection_reason"] = "not available in training matrix"
    else:
        row["mutual_information_nats"] = float(mi.get(feature, np.nan))
        mi_rank = mi.rank(ascending=False, method="min").get(feature, np.nan)
        row["mutual_information_rank"] = int(mi_rank) if pd.notna(mi_rank) else np.nan
        univariate = feature_results.loc[feature_results.feature.eq(feature)]
        if not univariate.empty:
            row["univariate_q_value"] = univariate.iloc[0].q_value_bh
            row["univariate_training_auc"] = univariate.iloc[0].training_auc
        if bool(constant_mask[feature]):
            row["selection_reason"] = "zero variance in training data"
        else:
            row["selection_status"] = "eligible"
            compact_available.append(feature)
    compact_results.append(row)

# Greedy correlation filter follows the frozen economic-priority order.
compact_after_correlation = []
for feature in compact_available:
    if compact_after_correlation:
        correlations = (
            variable_X[compact_after_correlation].corrwith(variable_X[feature]).abs()
        )
        maximum = float(correlations.max())
        closest = correlations[correlations.eq(maximum)].index[0]
        row = next(item for item in compact_results if item["feature"] == feature)
        row["max_abs_correlation"] = maximum
        if maximum >= COMPACT_CORRELATION_THRESHOLD:
            row["correlated_with"] = closest
            row["selection_reason"] = (
                f"correlation filter: |r| >= {COMPACT_CORRELATION_THRESHOLD:.2f} with {closest}"
            )
            row["selection_status"] = "excluded_correlation"
            continue
    compact_after_correlation.append(feature)

# Iteratively remove the highest-VIF member until the training design is below threshold.
# Ties remove the later economic-priority feature, preserving the earlier listed variable.
compact_vif_removed = {}
while len(compact_after_correlation) > 1:
    compact_vif_table, _ = vif_svd(variable_X[compact_after_correlation])
    excessive = compact_vif_table.vif > COMPACT_VIF_THRESHOLD
    if not excessive.any():
        break
    worst = float(compact_vif_table.vif.max())
    tied = [
        feature
        for feature in compact_after_correlation
        if np.isclose(compact_vif_table.loc[feature, "vif"], worst, equal_nan=False)
    ]
    removed = tied[-1]
    compact_vif_removed[removed] = float(compact_vif_table.loc[removed, "vif"])
    compact_after_correlation.remove(removed)

if compact_after_correlation:
    compact_vif_table, compact_vif_design = vif_svd(
        variable_X[compact_after_correlation]
    )
else:
    compact_vif_table = pd.DataFrame(columns=["vif"])
    compact_vif_design = {"rank": 0, "nullity": 0}
for row in compact_results:
    feature = row["feature"]
    if feature in compact_vif_table.index:
        row["vif_at_removal_or_final"] = float(compact_vif_table.loc[feature, "vif"])
    if feature in compact_after_correlation:
        row["selected"] = True
        row["selection_status"] = "selected"
        row["selection_reason"] = (
            "pre-specified compact feature; nonconstant; passed correlation and VIF filters"
        )
    elif feature in compact_vif_removed:
        row["vif_at_removal_or_final"] = compact_vif_removed[feature]
        row["selection_status"] = "excluded_vif"
        row["selection_reason"] = (
            f"VIF filter: training VIF > {COMPACT_VIF_THRESHOLD:g}"
        )

compact_selection = (
    pd.DataFrame(compact_results)
    .sort_values("economic_priority")
    .reset_index(drop=True)
)
compact_selected_columns = compact_selection.loc[
    compact_selection.selected, "feature"
].tolist()
compact_selection_summary = {
    "source": "project_config.COMPACT_FEATURE_COLUMNS",
    "training_event_count": len(train),
    "selection_rules": {
        "variance": "exclude features constant in training data",
        "correlation_threshold": COMPACT_CORRELATION_THRESHOLD,
        "correlation_tie_break": "retain the earlier feature in compact configuration order",
        "vif_threshold": COMPACT_VIF_THRESHOLD,
        "vif_tie_break": "exclude the later feature in compact configuration order",
        "mutual_information_and_univariate": "training-only diagnostics; rank and q-value reported, not standalone exclusion rules",
        "test_data_used_for_selection": False,
    },
    "candidate_features": compact_candidates,
    "selected_features": compact_selected_columns,
    "selected_feature_count": len(compact_selected_columns),
    "vif_design_rank": compact_vif_design["rank"],
    "vif_design_nullity": compact_vif_design["nullity"],
}

# Missingness audits use raw training masks, including training-empty dropped fields.
missing_rows = []
raw_sources = schema["input_feature_columns"]
for column in raw_sources:
    missing = raw_train[column].isna()
    numeric = (
        column in schema["retained_numeric_columns"]
        or column in schema["dropped_training_empty_numeric_columns"]
    )
    missing_rows.append(
        {
            "source_column": column,
            "kind": "numeric" if numeric else "categorical",
            "training_n": len(train),
            "observed_n": int((~missing).sum()),
            "missing_n": int(missing.sum()),
            "overall_missing_pct": float(missing.mean() * 100),
            "bankrupt_missing_pct": float(missing.loc[train.target.eq(1)].mean() * 100),
            "nonbankrupt_missing_pct": float(
                missing.loc[train.target.eq(0)].mean() * 100
            ),
            "difference_pp": float(
                (
                    missing.loc[train.target.eq(1)].mean()
                    - missing.loc[train.target.eq(0)].mean()
                )
                * 100
            ),
            "training_empty_exclusion": column
            in schema["dropped_training_empty_numeric_columns"],
        }
    )
missingness = pd.DataFrame(missing_rows).sort_values(
    ["overall_missing_pct", "source_column"], ascending=[False, True]
)

# Connected affine groups expose alias/pattern redundancy but do not select predictors.
parents = {feature: feature for feature in variable_features}


def find_group(feature):
    while parents[feature] != feature:
        parents[feature] = parents[parents[feature]]
        feature = parents[feature]
    return feature


for row in affine_pairs.itertuples(index=False):
    a, b = find_group(row.feature_a), find_group(row.feature_b)
    if a != b:
        parents[max(a, b)] = min(a, b)
components = {}
for feature in variable_features:
    components.setdefault(find_group(feature), []).append(feature)
duplicate_groups = sorted(
    [members for members in components.values() if len(members) > 1],
    key=lambda members: (-len(members), members[0]),
)
group_of = {
    feature: f"affine_group_{i:03d}"
    for i, members in enumerate(duplicate_groups, 1)
    for feature in members
}
feature_results["affine_group_id"] = feature_results.feature.map(group_of)
redundancy["affine_group_a"] = redundancy.feature_a.map(group_of)
redundancy["affine_group_b"] = redundancy.feature_b.map(group_of)
duplicate_group_table = pd.DataFrame(
    [
        {
            "group": f"affine_group_{i:03d}",
            "features": len(members),
            "families": ", ".join(sorted(feature_info.loc[members, "family"].unique())),
            "members": json.dumps(members),
        }
        for i, members in enumerate(duplicate_groups, 1)
    ]
)

family_summary = feature_results.groupby("family").agg(
    features=("feature", "size"),
    constants=("constant_in_training", "sum"),
    median_mi_nats=("mutual_information_nats", "median"),
    max_mi_nats=("mutual_information_nats", "max"),
    median_source_missing_pp=("source_missing_difference_pp", "median"),
    valid_univariate_fits=("logistic_status", lambda s: int(s.eq("ok").sum())),
)
family_summary["nonconstant_features"] = (
    family_summary.features - family_summary.constants
)
pair_findings = redundancy.loc[redundancy.record_type.eq("feature_pair")].copy()
family_pair_rows = []
for families, table in pair_findings.assign(
    family_pair=pair_findings.apply(
        lambda r: tuple(sorted([r.family_a, r.family_b])), axis=1
    )
).groupby("family_pair", sort=True):
    a, b = families
    count_a, count_b = (
        int(family_summary.loc[a, "nonconstant_features"]),
        int(family_summary.loc[b, "nonconstant_features"]),
    )
    possible = count_a * (count_a - 1) // 2 if a == b else count_a * count_b
    family_pair_rows.append(
        {
            "family_a": a,
            "family_b": b,
            "possible_nonconstant_pairs": possible,
            "flagged_pairs": len(table),
            "flagged_pair_pct": 100 * len(table) / possible if possible else np.nan,
            "exact_duplicates": int(table.exact_duplicate.sum()),
            "affine_duplicates": int(table.affine_duplicate.sum()),
            "near_affine_pairs": int(table.near_affine_duplicate.sum()),
            "duplicate_missing_patterns": int(table.duplicate_missing_pattern.sum()),
        }
    )
family_pair_summary = pd.DataFrame(family_pair_rows).sort_values(
    "flagged_pairs", ascending=False
)

# Embed compact family heatmaps and diagnostic rankings in the HTML report only.
# No unreadable 896-feature heatmap and no extra standalone output files.
plt.rcParams.update(
    {"font.size": 9, "figure.facecolor": "white", "savefig.facecolor": "white"}
)


def table_html(frame, index=False):
    return (
        '<div class="table-wrap">'
        + frame.to_html(
            index=index,
            escape=True,
            border=0,
            na_rep="Unavailable",
            float_format=lambda value: "Infinity"
            if np.isinf(value)
            else f"{value:,.5g}",
            classes="data-table",
        )
        + "</div>"
    )


def figure_html(fig, caption):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def short_label(feature):
    source = feature_info.loc[feature, "source_column"]
    for prefix in ["original_", "derived_", "trend_", "market_", "macro_"]:
        if source.startswith(prefix):
            source = source.removeprefix(prefix)
            break
    return (
        source.replace("cash_short_term_investments", "cash/ST investments")
        .replace("operating_cash_flow", "operating CF")
        .replace("depreciation_amortization", "deprec/amort")
        .replace("capital_expenditures", "capex")
        .replace("_", " ")
    )


heatmaps = []
heatmap_blocks = [
    "original_financial",
    "derived_ratios",
    "trend_financial_growth_1y",
    "market_features",
]
for block in heatmap_blocks:
    columns = (
        feature_info.loc[variable_features]
        .index[feature_info.loc[variable_features, "block"].eq(block)]
        .tolist()
    )
    if not columns:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    for ax, correlation, title in zip(
        axes, [pearson, spearman], ["Pearson", "Spearman"]
    ):
        image = ax.imshow(
            correlation.loc[columns, columns],
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        labels = [short_label(column) for column in columns]
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(title + " | " + block.replace("_", " "))
        fig.colorbar(image, ax=ax, shrink=0.7, label="Correlation")
    fig.tight_layout()
    heatmaps.append(
        figure_html(
            fig,
            f"Training median-imputed matrix, n={len(train)}. Family panels exclude constants; observed-value comparisons and sample sizes appear in the redundancy CSV.",
        )
    )

fig, ax = plt.subplots(figsize=(12, 10))
top_mi = feature_results.nlargest(25, "mutual_information_nats").iloc[::-1]
ax.barh(range(len(top_mi)), top_mi.mutual_information_nats, color="#3979a8")
ax.set_yticks(
    range(len(top_mi)),
    [
        f"{row.branch}: {short_label(row.feature)}"
        for row in top_mi.itertuples(index=False)
    ],
)
ax.set(
    title="Top training mutual-information estimates",
    xlabel="Mutual information (nats)",
)
ax.grid(axis="x", alpha=0.2)
mi_chart = figure_html(
    fig,
    "Fixed-seed training estimates; not causal importance or out-of-sample performance. Small differences, especially among tied/duplicate variables, need not be meaningful.",
)

fig, ax = plt.subplots(figsize=(12, 9))
top_missing = (
    missingness.loc[~missingness.training_empty_exclusion]
    .assign(abs_gap=lambda f: f.difference_pp.abs())
    .nlargest(25, "abs_gap")
    .iloc[::-1]
)
ypos = np.arange(len(top_missing))
ax.barh(
    ypos - 0.18,
    top_missing.nonbankrupt_missing_pct,
    height=0.35,
    label="Non-bankrupt",
    color="#3979a8",
)
ax.barh(
    ypos + 0.18,
    top_missing.bankrupt_missing_pct,
    height=0.35,
    label="Bankrupt",
    color="#b35a46",
)
ax.set_yticks(ypos, top_missing.source_column.str.replace("_", " "))
ax.set(
    title="Largest raw training missingness differences", xlabel="Missing percentage"
)
ax.legend()
ax.grid(axis="x", alpha=0.2)
missing_chart = figure_html(
    fig,
    "Raw training missing masks, including financial/market/macro unavailable observations. Missingness can reflect extraction/reporting differences; it is not proof of financial distress.",
)

# Review recommendations preserve correlated groups for downstream Elastic Net.
selection_policy = pd.DataFrame(
    [
        {
            "finding": "High Pearson/Spearman correlation",
            "review_action": "Retain correlated groups for Elastic Net consideration; do not remove every pair above a threshold",
        },
        {
            "finding": "Exact/affine duplicate values",
            "review_action": "Review field definitions and future behavior before choosing an alias representative",
        },
        {
            "finding": "Repeated missingness pattern",
            "review_action": "Training equality may not persist when new observations become missing; do not silently collapse indicators",
        },
        {
            "finding": "Training-constant dummy/indicator",
            "review_action": "No current training variation; reserved missing/unseen behavior may activate later, so any omission must be explicit",
        },
        {
            "finding": "Full-design VIF infinity",
            "review_action": "The unpenalized multivariate design is not identifiable. Regularization/structured reduction is required; block VIF is supplementary",
        },
        {
            "finding": "Univariate separation/fit warning",
            "review_action": "Unpenalized MLE inference is withheld. Consider regularized modeling later rather than interpreting an extreme odds ratio",
        },
        {
            "finding": "MI or univariate significance",
            "review_action": "Training ranking only; do not use it as independent validation or tune decisions on test results",
        },
    ]
)

CSS = """body{font:15px/1.6 system-ui,sans-serif;background:#f4f6f9;color:#202b39;margin:0}
main{max-width:1240px;margin:24px auto;padding:30px;background:white;border-radius:12px}h1{font-size:28px}h2{margin-top:32px;color:#193f5c}
.notice{padding:14px 18px;background:#fff2d8;border-left:5px solid #b47914}.scope{padding:14px;background:#e8f4ec;border-left:5px solid #2b794b}
.table-wrap{overflow:auto;margin:15px 0;max-height:650px}.data-table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #e0e5eb;text-align:left;vertical-align:top}th{background:#edf2f7;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}
img{max-width:100%;height:auto}figure{margin:24px 0}figcaption,small{font-size:13px;color:#536273}details{margin:18px 0}summary{font-weight:600;cursor:pointer}
code{background:#edf2f7;padding:2px 5px}"""

rank_table_columns = [
    "feature",
    "family",
    "mutual_information_nats",
    "coef_log_odds",
    "odds_ratio",
    "p_value",
    "q_value_bh",
    "training_auc",
    "logistic_status",
    "rare_binary_level",
]
body = f'<p class="scope"><strong>Training only:</strong> {len(train)} company-events, {int(y.sum())} bankrupt and {int((y == 0).sum())} non-bankrupt; {len(np.unique(groups))} company groups. No validation/test matrices or IDs were opened.</p>'
body += '<p class="notice"><strong>Diagnostic report, not automatic selection.</strong> No predictors were removed and no matrix/preprocessing artifact was modified. Correlated groups may be retained for Elastic Net. Original matching/outcome-surveillance/vintage limitations remain unresolved by these diagnostics.</p>'
body += "<h2>Diagnostic scope</h2><p>Pearson/Spearman cover median-imputed training predictors. Raw observed-value correlations use pairwise complete numerical observations with minimum n=30; compare the two views in feature_redundancy.csv. Categorical/missingness features are discrete for mutual information. Zero correlation does not establish independence.</p>"
body += table_html(
    pd.DataFrame(
        [
            {
                "training_rows": len(train),
                "matrix_features": len(FEATURES),
                "nonconstant_features": len(variable_features),
                "constant_features": int(constant_mask.sum()),
                "flagged_pairs": len(pair_findings),
                "affine_groups": len(duplicate_groups),
                "full_design_rank": full_design["rank"],
                "full_design_nullity": full_design["nullity"],
            }
        ]
    )
)
body += "<h2>Selection/review policy</h2>" + table_html(selection_policy)
body += (
    "<h2>Compact-model feature selection</h2><p>The compact candidate list is fixed in project_config.py. Selection uses training-only variance, Pearson correlation, and VIF filters in economic-priority order. Mutual information and univariate results are training-only diagnostics shown for review; they do not independently remove candidates. The full-model path retains nonconstant features for Elastic Net shrinkage. No validation or test results determine this list.</p>"
    + table_html(compact_selection)
)
body += (
    "<h2>Feature-family redundancy</h2>"
    + table_html(family_summary.reset_index())
    + table_html(family_pair_summary)
)
body += "<h2>Correlation heatmaps by family</h2>" + "".join(heatmaps)
body += "<h2>Exact, affine, and near duplicates</h2><p>Normalized difference compares centered/unit-SD values, including a sign flip for negative correlations. Training-only affine groups do not prove permanent equivalence.</p>"
body += table_html(
    pair_findings.loc[
        pair_findings.affine_duplicate | pair_findings.near_affine_duplicate
    ]
    .sort_values(
        ["affine_duplicate", "max_standardized_difference"], ascending=[False, True]
    )
    .head(100)
)
body += (
    "<details><summary>Affine groups and all members</summary>"
    + table_html(duplicate_group_table)
    + "</details>"
)
body += (
    "<details><summary>Training-constant predictors</summary>"
    + table_html(
        feature_results.loc[
            feature_results.constant_in_training,
            ["feature", "family", "logistic_status"],
        ]
    )
    + "</details>"
)
body += "<h2>Variance inflation factors</h2><p>Full-design VIF includes all nonconstant predictors and an intercept through centering. Null-space participation implies exact dependency and infinity. Finite values are reported only for individually identifiable features. Block VIFs use smaller, explicitly different designs; alias/reference omissions are diagnostic only.</p>"
body += table_html(pd.DataFrame([full_design])) + table_html(block_designs)
body += table_html(
    feature_results[
        [
            "feature",
            "full_design_vif",
            "diagnostic_vif",
            "diagnostic_vif_block",
            "diagnostic_vif_note",
        ]
    ]
    .sort_values("diagnostic_vif", ascending=False)
    .head(80)
)
body += "<h2>Mutual information and univariate logistic regressions</h2>" + mi_chart
body += "<p>GLMs have one predictor and an intercept, with company-group clustered standard errors. Numeric coefficients use one training SD; dummy/missing coefficients compare 0 to 1. Support separation, convergence problems, warnings, and nonfinite inference withhold MLE statistics. Rare binary levels are flagged. BH correction covers valid fits only. Training AUC is resubstitution performance, never a held-out result.</p>"
body += table_html(
    feature_results.logistic_status.value_counts()
    .rename_axis("status")
    .rename("features")
    .reset_index()
)
body += table_html(feature_results[rank_table_columns].head(60))
body += (
    "<details><summary>Separated, rare, or problematic predictors</summary>"
    + table_html(
        feature_results.loc[
            ~feature_results.logistic_status.eq("ok")
            | feature_results.rare_binary_level,
            [
                "feature",
                "logistic_status",
                "rare_binary_level",
                "smallest_binary_level_count",
                "fit_warnings",
            ],
        ]
    )
    + "</details>"
)
body += (
    "<h2>Raw training missingness by target</h2>"
    + missing_chart
    + table_html(missingness)
)
body += "<h2>Provenance</h2>" + table_html(
    pd.DataFrame(
        [
            {"input": str(path.relative_to(ROOT)), "sha256": digest}
            for path, digest in input_hashes.items()
        ]
    )
)
body += (
    "<p><small>Versions: Python "
    + html.escape(platform.python_version())
    + ", sklearn "
    + html.escape(sklearn.__version__)
    + ", statsmodels "
    + html.escape(statsmodels.__version__)
    + ", scipy "
    + html.escape(scipy.__version__)
    + ". Review thresholds: |correlation| >= .80, near affine >= .9999; MI seed 42. No classifier or selected modeling dataset was saved.</small></p>"
)
report_html = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Training feature independence and redundancy</title><style>'
    + CSS
    + "</style></head><body><main><h1>Training feature independence and redundancy</h1>"
    + body
    + "</main></body></html>"
)

# Verify training provenance, diagnostic coverage, and untouched inputs before export.
ok = feature_results.logistic_status.eq("ok")
REPORTS.mkdir(parents=True, exist_ok=True)
redundancy_path = REPORTS / "feature_redundancy.csv"
univariate_path = REPORTS / "univariate_feature_results.csv"
report_path = REPORTS / "correlation_report.html"
redundancy.to_csv(redundancy_path, index=False)
feature_results.to_csv(univariate_path, index=False)
report_path.write_text(report_html, encoding="utf-8")
compact_selection.to_csv(REPORTS / "compact_feature_selection.csv", index=False)
(REPORTS / "compact_feature_selection.json").write_text(
    json.dumps(compact_selection_summary, indent=2), encoding="utf-8"
)
