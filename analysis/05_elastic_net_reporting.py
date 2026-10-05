"""Create Elastic Net performance plots and the model report."""

import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils import calibration_table, project_paths  # noqa: E402

REPORTS = project_paths(ROOT).reports / "05_elastic_net_reporting"
FIGURES = project_paths(ROOT).figures / "05_elastic_net_reporting"


def generate_report(context):
    """Export Elastic Net validation results and generate report figures and HTML."""
    fold_tuning = context["fold_tuning"]
    tuning = context["tuning"]
    oof_predictions = context["oof_predictions"]
    metrics = context["metrics"]
    coefficients = context["coefficients"]
    predictions = context["predictions"]
    threshold_rows = context["threshold_rows"]
    model_configuration = context["model_configuration"]
    y_oof = context["y_oof"]
    p_oof = context["p_oof"]
    y_test = context["y_test"]
    p_test = context["p_test"]
    best = context["best"]
    OPERATING_THRESHOLD = context["OPERATING_THRESHOLD"]
    fold_names = context["fold_names"]

    # Save model settings, event predictions, and tuning results for review.
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for frame, filename in [
        (fold_tuning, "elastic_net_fold_tuning_results.csv"),
        (tuning, "elastic_net_tuning_results.csv"),
        (oof_predictions, "elastic_net_oof_validation_predictions.csv"),
        (metrics, "elastic_net_metrics.csv"),
        (coefficients, "elastic_net_coefficients.csv"),
        (predictions, "elastic_net_predictions.csv"),
        (threshold_rows, "elastic_net_oof_threshold_candidates.csv"),
    ]:
        frame.to_csv(REPORTS / filename, index=False)
    configuration_export = {
        key: value
        for key, value in model_configuration.items()
        if key
        not in {"training_event_ids", "oof_validation_event_ids", "test_event_ids"}
    }
    (REPORTS / "elastic_net_model_configuration.json").write_text(
        json.dumps(configuration_export, indent=2, default=str), encoding="utf-8"
    )

    # Plot discrimination and calibration for OOF validation and final test.
    performance_data = [
        (y_oof, p_oof, "Walk-forward OOF validation", "#d98300"),
        (y_test, p_test, "Final test", "#1769aa"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for labels, probabilities, name, color in performance_data:
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[0].plot(
            false_positive,
            true_positive,
            color=color,
            label=f"{name}: {roc_auc_score(labels, probabilities):.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            label=f"{name}: {average_precision_score(labels, probabilities):.3f}",
        )
    axes[0].plot([0, 1], [0, 1], "--", color="black")
    axes[0].set(
        xlabel="False positive rate",
        ylabel="Recall",
        title="ROC",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle("Elastic Net selected by mean walk-forward PR-AUC")
    performance_figure = FIGURES / "elastic_net_walk_forward_performance.png"
    fig.savefig(performance_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compare observed event rates with scores for the validation and test samples.
    calibration_tables = []
    fig, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for labels, probabilities, name, color in performance_data:
        table = calibration_table(labels, probabilities)
        table["split"] = name
        calibration_tables.append(table)
        axis.plot(
            table.mean_predicted_probability,
            table.observed_bankruptcy_rate,
            marker="o",
            color=color,
            label=name,
        )
    axis.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axis.set(
        xlabel="Mean predicted relative-risk score",
        ylabel="Observed bankruptcy rate",
        title="Reliability diagram",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    calibration_figure = FIGURES / "elastic_net_walk_forward_calibration.png"
    fig.savefig(calibration_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    calibration_bins = pd.concat(calibration_tables, ignore_index=True)
    # Assemble the model-selection, performance, and calibration report.
    report_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Elastic Net walk-forward benchmark</title><style>
    body{{font:15px/1.55 system-ui,sans-serif;margin:30px auto;padding:0 22px;max-width:1200px;color:#203142}}h1,h2{{color:#17425e}}.wrap{{overflow:auto;max-height:600px;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #d9e1e8;text-align:left;white-space:nowrap}}th{{background:#edf3f7;position:sticky;top:0}}tr:nth-child(even){{background:#f8fafb}}img{{max-width:100%;height:auto}}figure{{margin:22px 0}}figcaption{{font-size:13px;color:#52636e}}.note{{padding:12px 15px;background:#edf6ef;border-left:4px solid #398253}}
    </style></head><body><h1>Elastic Net logistic regression benchmark</h1>
    <p class="note">Hyperparameters are selected by mean average precision (PR-AUC) across walk-forward validation folds. The operating threshold is selected from pooled out-of-fold validation predictions by balanced accuracy, with sensitivity and specificity tie-breakers. The chronological test is opened only after both decisions are fixed. This matched sample supports relative-risk scores, not population prevalence.</p>
    <h2>Selected settings</h2><div class="wrap">{pd.DataFrame([model_configuration]).drop(columns=["training_event_ids", "oof_validation_event_ids", "test_event_ids", "training_constant_predictors"], errors="ignore").to_html(index=False, escape=True, border=0)}</div>
    <p>Selected mean validation PR-AUC: {best.mean_validation_pr_auc:.4f}; frozen threshold: {OPERATING_THRESHOLD:.6g}; folds: {html.escape(", ".join(fold_names))}.</p>
    <h2>Performance at frozen threshold</h2><div class="wrap">{metrics.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", na_rep="unavailable", border=0)}</div>
    <p>Top-risk capture is the share of bankrupt events in the top-scored 1%, 5%, or 10% of rows. Calibration slope and intercept are from logistic recalibration on the logit scores; calibration is descriptive only.</p>
    <figure><img src="../../figures/05_elastic_net_reporting/{performance_figure.name}" alt="ROC and precision-recall curves"><figcaption>Walk-forward OOF validation and final test discrimination.</figcaption></figure>
    <figure><img src="../../figures/05_elastic_net_reporting/{calibration_figure.name}" alt="Reliability diagram"><figcaption>Reliability by quantile bins; small bins are noisy.</figcaption></figure>
    <h2>Hyperparameter results</h2><div class="wrap">{tuning.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", border=0)}</div><div class="wrap">{fold_tuning.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", border=0)}</div>
    <h2>Calibration bins</h2><div class="wrap">{calibration_bins.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", border=0)}</div>
    <p>Each fold used its own training-fitted preprocessing and variance filter. The final model was fitted on the static training split. Test results were not used for feature handling, hyperparameter selection, or threshold choice.</p></body></html>"""

    (REPORTS / "elastic_net_logistic_report.html").write_text(
        report_html, encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(
        "Call generate_report(context) from the corresponding modeling notebook."
    )
