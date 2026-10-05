"""Evaluate relative-risk calibration by split, time period, and sector."""

import html
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils import project_paths  # noqa: E402

PATHS = project_paths(ROOT)
REPORT_DIR = PATHS.reports / "08_probability_calibration"
FIGURE_DIR = PATHS.figures / "08_probability_calibration"
PREDICTIONS_PATH = (
    PATHS.reports / "05_elastic_net_reporting" / "elastic_net_predictions.csv"
)
DATASET_PATH = PATHS.processed / "modeling_dataset_unprocessed.csv"

# Load the saved walk-forward scores and attach sector metadata for subgroup diagnostics.
predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"company_id": "string"})
required = {
    "company_id",
    "prediction_date",
    "target",
    "prediction_split",
    "prediction_probability",
}
missing = required - set(predictions.columns)
if missing:
    raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
predictions["prediction_date"] = pd.to_datetime(
    predictions["prediction_date"], errors="coerce"
)
predictions["target"] = pd.to_numeric(predictions["target"], errors="coerce").astype(
    "Int64"
)
predictions["raw_relative_risk_score"] = pd.to_numeric(
    predictions["prediction_probability"], errors="coerce"
)
predictions = predictions.dropna(
    subset=["prediction_date", "target", "raw_relative_risk_score"]
).copy()
predictions["target"] = predictions["target"].astype(int)
predictions["evaluation_split"] = predictions["prediction_split"]
predictions = predictions.loc[
    predictions["evaluation_split"].isin(["walk_forward_oof_validation", "final_test"])
].copy()
if not predictions["raw_relative_risk_score"].between(0, 1).all():
    raise ValueError("Saved model scores must be between zero and one")
keys = ["company_id", "prediction_date"]
if predictions.duplicated(keys).any():
    raise ValueError("Prediction rows must be unique company-events")

metadata = pd.read_csv(
    DATASET_PATH,
    usecols=["company_id", "prediction_date", "sector"],
    dtype={"company_id": "string", "sector": "string"},
)
metadata["prediction_date"] = pd.to_datetime(
    metadata["prediction_date"], errors="coerce"
)
metadata = metadata.drop_duplicates(keys)
predictions = predictions.merge(metadata, on=keys, how="left", validate="one_to_one")
predictions["sector"] = predictions["sector"].fillna("Missing / unavailable")
predictions["prediction_year"] = predictions["prediction_date"].dt.year
predictions["time_period"] = (
    predictions["fold"].astype("string") if "fold" in predictions else pd.NA
)
predictions.loc[predictions["evaluation_split"].eq("final_test"), "time_period"] = (
    "test_"
    + predictions.loc[
        predictions["evaluation_split"].eq("final_test"), "prediction_year"
    ].astype(str)
)
predictions.loc[
    predictions["evaluation_split"].eq("walk_forward_oof_validation"), "time_period"
] = predictions.loc[
    predictions["evaluation_split"].eq("walk_forward_oof_validation"), "time_period"
].fillna("walk_forward_oof_validation")
validation = predictions.loc[
    predictions["evaluation_split"].eq("walk_forward_oof_validation")
].copy()
test = predictions.loc[predictions["evaluation_split"].eq("final_test")].copy()
if validation["target"].nunique() != 2 or test["target"].nunique() != 2:
    raise ValueError("Both outcome classes are required in out-of-fold and test rows")


# Fit the diagnostic Platt transformation on out-of-fold scores only.
def clipped_logit(scores):
    values = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
platt.fit(clipped_logit(validation["raw_relative_risk_score"]), validation["target"])
predictions["platt_relative_risk_score"] = platt.predict_proba(
    clipped_logit(predictions["raw_relative_risk_score"])
)[:, 1]

platt_parameters = pd.DataFrame(
    [
        {
            "method": "Platt scaling diagnostic on walk-forward OOF scores",
            "regularization_C": 1.0,
            "intercept": float(platt.intercept_[0]),
            "slope": float(platt.coef_[0, 0]),
            "fit_rows": len(validation),
            "fit_bankruptcies": int(validation["target"].sum()),
            "fit_matched_sample_prevalence": float(validation["target"].mean()),
            "interpretation": "relative risk score; not a population probability",
        }
    ]
)


# Summarize calibration diagnostics and ranking/capture performance.
def calibration_fit(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) != 2:
        return np.nan, np.nan
    design = sm.add_constant(clipped_logit(scores), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(labels, design, family=sm.families.Binomial()).fit()
        intercept, slope = float(fit.params[0]), float(fit.params[1])
        if (
            not np.isfinite(intercept)
            or not np.isfinite(slope)
            or max(abs(intercept), abs(slope)) > 20
        ):
            return np.nan, np.nan
        return intercept, slope
    except Exception:
        return np.nan, np.nan


def capture_metrics(frame, score_col):
    ordered = frame.sort_values(score_col, ascending=False)
    n_events = len(ordered)
    n_cases = int(ordered["target"].sum())
    result = {}
    for pct in (1, 5, 10):
        take = max(1, int(np.ceil(n_events * pct / 100)))
        top = ordered.head(take)
        result[f"capture_top_{pct}pct"] = (
            float(top["target"].sum() / n_cases) if n_cases else np.nan
        )
        result[f"precision_top_{pct}pct"] = float(top["target"].mean())
    return result


def summarize(frame, group_type, group_name, score_col):
    labels = frame["target"].to_numpy(dtype=int)
    scores = frame[score_col].to_numpy(dtype=float)
    intercept, slope = calibration_fit(labels, scores)
    row = {
        "evaluation_split": frame["evaluation_split"].iloc[0] if len(frame) else "",
        "group_type": group_type,
        "group": str(group_name),
        "score": score_col,
        "events": len(frame),
        "bankrupt_events": int(labels.sum()),
        "matched_sample_prevalence": float(labels.mean()) if len(labels) else np.nan,
        "brier_score": float(brier_score_loss(labels, scores))
        if len(labels)
        else np.nan,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1]))
        if len(labels)
        else np.nan,
        "roc_auc": float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else np.nan,
        "pr_auc_average_precision": float(average_precision_score(labels, scores))
        if len(np.unique(labels)) == 2
        else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_stable": bool(
            len(frame) >= 50
            and labels.sum() >= 10
            and (labels == 0).sum() >= 10
            and np.isfinite(intercept)
            and np.isfinite(slope)
            and 0.25 <= slope <= 4
            and abs(intercept) <= 2
        ),
        "calibration_interpretation": "matched-cohort diagnostic only",
    }
    row.update(capture_metrics(frame, score_col))
    return row


score_columns = ["raw_relative_risk_score", "platt_relative_risk_score"]
summary_rows = []
for (split_name,), group in predictions.groupby(["evaluation_split"], sort=False):
    for score_col in score_columns:
        summary_rows.append(summarize(group, "overall", split_name, score_col))
for group_col, group_type in (("time_period", "time_period"), ("sector", "sector")):
    for (split_name, group_name), group in predictions.groupby(
        ["evaluation_split", group_col], dropna=False
    ):
        for score_col in score_columns:
            summary_rows.append(summarize(group, group_type, group_name, score_col))
calibration_results = pd.DataFrame(summary_rows)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
calibration_results.to_csv(
    REPORT_DIR / "probability_calibration_metrics.csv", index=False
)
platt_parameters.to_csv(REPORT_DIR / "probability_platt_parameters.csv", index=False)


# Create pooled, time-period, and sector reliability curves and save bin-level data.
def reliability_bins(frame, score_col, bins=6):
    scores = frame[score_col].to_numpy(dtype=float)
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    labels = (
        np.searchsorted(edges[1:-1], scores)
        if len(edges) > 1
        else np.zeros(len(scores), dtype=int)
    )
    temp = pd.DataFrame(
        {"target": frame["target"].to_numpy(), "score": scores, "bin": labels}
    )
    result = (
        temp.groupby("bin", sort=True)
        .agg(
            events=("target", "size"),
            mean_score=("score", "mean"),
            observed_bankruptcy_fraction=("target", "mean"),
        )
        .reset_index()
    )
    result["score"] = score_col
    result["evaluation_split"] = frame["evaluation_split"].iloc[0]
    return result


reliability_parts = []
for (split_name,), group in predictions.groupby(["evaluation_split"], sort=False):
    for score_col in score_columns:
        reliability_parts.append(reliability_bins(group, score_col))
reliability = pd.concat(reliability_parts, ignore_index=True)
reliability.to_csv(REPORT_DIR / "probability_reliability_bins.csv", index=False)


def plot_reliability(frames, title, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference line")
    for frame, score_col, label, color, marker in frames:
        bins = reliability_bins(frame, score_col)
        ax.plot(
            bins["mean_score"],
            bins["observed_bankruptcy_fraction"],
            marker=marker,
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean relative risk score",
        ylabel="Observed bankruptcy fraction in matched sample",
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


validation_scored = predictions.loc[
    predictions["evaluation_split"].eq("walk_forward_oof_validation")
].copy()
plot_reliability(
    [
        (validation_scored, "raw_relative_risk_score", "OOF raw", "#1769aa", "o"),
        (
            validation_scored,
            "platt_relative_risk_score",
            "OOF Platt diagnostic",
            "#d46c00",
            "s",
        ),
    ],
    "Walk-forward OOF reliability diagnostics",
    FIGURE_DIR / "probability_reliability_diagram.png",
)


# Draw Platt diagnostic reliability by fold and final test period.
def plot_group_reliability(group_col, output_name, title, minimum=15):
    groups = []
    for (split_name, key), frame in predictions.groupby(
        ["evaluation_split", group_col], dropna=False
    ):
        if len(frame) >= minimum and frame["target"].nunique() == 2:
            groups.append((f"{split_name} / {key}", frame))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference")
    for name, frame in groups:
        bins = reliability_bins(frame, "platt_relative_risk_score", bins=5)
        ax.plot(
            bins["mean_score"],
            bins["observed_bankruptcy_fraction"],
            marker="o",
            linewidth=1.3,
            label=f"{name} (n={len(frame)})",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean Platt relative risk score",
        ylabel="Observed bankruptcy fraction in matched sample",
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(FIGURE_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


plot_group_reliability(
    "time_period",
    "probability_calibration_by_time.png",
    "Platt diagnostic reliability by walk-forward period and test period",
)
plot_group_reliability(
    "sector",
    "probability_calibration_by_sector.png",
    "Platt diagnostic reliability by sector",
)

# Report calibration instability alongside rank-based screening measures.
test_overall = calibration_results.loc[
    calibration_results["group_type"].eq("overall")
    & calibration_results["evaluation_split"].eq("final_test")
    & calibration_results["score"].eq("raw_relative_risk_score")
].iloc[0]
calibration_status = (
    "stable enough for descriptive diagnostics"
    if test_overall["calibration_stable"]
    else "unstable or too small for probability-like interpretation"
)


def html_table(frame):
    return frame.to_html(
        index=False, border=0, float_format=lambda value: f"{value:.4f}", na_rep="?"
    )


time_table = calibration_results.loc[
    calibration_results["group_type"].eq("time_period")
]
sector_table = calibration_results.loc[calibration_results["group_type"].eq("sector")]
overall_table = calibration_results.loc[calibration_results["group_type"].eq("overall")]
ranking_table = overall_table.loc[
    overall_table["evaluation_split"].eq("final_test")
    & overall_table["score"].eq("raw_relative_risk_score"),
    [
        "events",
        "bankrupt_events",
        "roc_auc",
        "pr_auc_average_precision",
        "capture_top_1pct",
        "capture_top_5pct",
        "capture_top_10pct",
        "precision_top_1pct",
        "precision_top_5pct",
        "precision_top_10pct",
    ],
]
report = f"""<!doctype html><html><head><meta charset="utf-8"><title>Relative risk score diagnostics</title>
<style>body{{font:15px Arial,sans-serif;max-width:1180px;margin:30px auto;line-height:1.5;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 24px}}th,td{{border:1px solid #ccc;padding:6px;text-align:right}}th{{background:#eee}}
figure img{{max-width:100%}}.notice{{padding:14px;background:#fff4d6;border-left:4px solid #d49a00}}</style></head><body>
<h1>Relative bankruptcy risk score diagnostics</h1>
<p class="notice"><strong>Interpretation:</strong> This matched sample has artificial bankruptcy prevalence. Model outputs and Platt-scaled outputs are <strong>relative bankruptcy risk scores</strong>, not real-world bankruptcy probabilities. Population probabilities require a representative population sample or valid external prevalence information.</p>
<h2>Calibration status</h2><p>Final-test calibration status: <strong>{html.escape(calibration_status)}</strong>. Calibration is considered stable for descriptive reporting only when the group has at least 50 observations, 10 cases and 10 non-cases, finite slope/intercept, slope between 0.25 and 4, and absolute intercept no greater than 2. Even a stable result is a matched-cohort diagnostic, not population calibration. If calibration is unstable, emphasize ROC-AUC, PR-AUC, top-risk capture and precision rather than interpreting score values as probabilities.</p>
<h2>Method</h2><p>Platt scaling was fitted on pooled walk-forward out-of-fold scores only; the final test rows were transformed after the fit was fixed. The stability status above evaluates the raw model score; calibration of both raw and Platt scores is shown in the tables. Brier score, log loss, calibration slope/intercept, and reliability curves describe this matched cohort. The calibration-by-time and calibration-by-sector tables include subgroup sample sizes and stability flags; small subgroups should not be read as precise estimates.</p>
<h2>Held-out test ranking and screening metrics</h2>{html_table(ranking_table)}
<h2>Overall calibration metrics</h2>{html_table(overall_table)}
<h2>Reliability</h2><figure><img src="../../figures/08_probability_calibration/probability_reliability_diagram.png" alt="Walk-forward reliability diagnostics"></figure>
<h2>Calibration by time period</h2>{html_table(time_table)}
<figure><img src="../../figures/08_probability_calibration/probability_calibration_by_time.png" alt="Calibration by time period"></figure>
<h2>Calibration by sector</h2>{html_table(sector_table)}
<figure><img src="../../figures/08_probability_calibration/probability_calibration_by_sector.png" alt="Calibration by sector"></figure>
<h2>Platt parameters</h2>{html_table(platt_parameters)}
<p>Subgroup capture-rate and precision columns are retained in the metrics CSV. See <code>probability_calibration_metrics.csv</code> for complete machine-readable results.</p>
</body></html>"""
(REPORT_DIR / "probability_calibration.html").write_text(report, encoding="utf-8")
print(f"Wrote relative-risk calibration diagnostics to {REPORT_DIR} and {FIGURE_DIR}.")
