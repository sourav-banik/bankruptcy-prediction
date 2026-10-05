"""Create Random Forest validation plots and the model report."""

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

from notebook_utils import project_paths  # noqa: E402

REPORTS = project_paths(ROOT).reports / "06_random_forest_reporting"
FIGURES = project_paths(ROOT).figures / "06_random_forest_reporting"


def generate_report(context):
    """Export Random Forest diagnostics and generate its comparison report."""
    fold_tuning = context["fold_tuning"]
    tuning = context["tuning"]
    gb_fold_tuning = context["gb_fold_tuning"]
    gb_tuning = context["gb_tuning"]
    gb_predictions = context["gb_predictions"]
    gb_threshold_rows = context["gb_threshold_rows"]
    rf_predictions = context["rf_predictions"]
    oof = context["oof"]
    metrics = context["metrics"]
    training_diagnostic = context["training_diagnostic"]
    comparison = context["comparison"]
    permutation_by_fold = context["permutation_by_fold"]
    bootstrap_by_fold = context["bootstrap_by_fold"]
    permutation_importance = context["permutation_importance"]
    gb_configuration = context["gb_configuration"]
    configuration = context["configuration"]
    y_oof = context["y_oof"]
    p_oof = context["p_oof"]
    y_test = context["y_test"]
    p_test = context["p_test"]
    elastic_oof = context["elastic_oof"]
    elastic_test = context["elastic_test"]
    gb_y_oof = context["gb_y_oof"]
    gb_p_oof = context["gb_p_oof"]
    gb_p_test = context["gb_p_test"]

    # Preserve tuning, prediction, and validation-importance tables for audit.
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for frame, filename in [
        (fold_tuning, "random_forest_fold_tuning_results.csv"),
        (tuning, "random_forest_tuning_results.csv"),
        (gb_fold_tuning, "gradient_boosting_fold_tuning.csv"),
        (gb_tuning, "gradient_boosting_tuning.csv"),
        (gb_predictions, "gradient_boosting_oof_and_test_predictions.csv"),
        (gb_threshold_rows, "gradient_boosting_oof_threshold_candidates.csv"),
        (rf_predictions, "random_forest_predictions.csv"),
        (oof, "random_forest_oof_validation_predictions.csv"),
        (metrics, "random_forest_metrics.csv"),
        (pd.DataFrame([training_diagnostic]), "random_forest_training_diagnostic.csv"),
        (comparison, "random_forest_vs_elastic_net.csv"),
        (permutation_by_fold, "random_forest_validation_permutation_by_fold.csv"),
        (
            bootstrap_by_fold,
            "random_forest_validation_importance_bootstrap_by_fold.csv",
        ),
        (permutation_importance, "random_forest_validation_importance_stability.csv"),
    ]:
        frame.to_csv(REPORTS / filename, index=False)
    (REPORTS / "gradient_boosting_configuration.json").write_text(
        json.dumps(gb_configuration, indent=2, default=str), encoding="utf-8"
    )
    configuration_export = {
        k: v
        for k, v in configuration.items()
        if k not in {"training_event_ids", "test_event_ids"}
    }
    (REPORTS / "random_forest_configuration.json").write_text(
        json.dumps(configuration_export, indent=2, default=str), encoding="utf-8"
    )

    # Plot validation-only permutation importance and matched-cohort performance.
    top_features = permutation_importance.head(25).iloc[::-1]
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.barh(
        range(len(top_features)),
        top_features.mean_permutation_importance,
        color="#386f91",
    )
    feature_labels = [
        name.removeprefix("numeric__").removeprefix("derived__").replace("_", " ")
        for name in top_features.feature
    ]
    axis.set_yticks(range(len(top_features)), feature_labels)
    axis.set(
        xlabel="Decrease in validation average precision",
        title="Random Forest validation permutation importance",
    )
    axis.grid(axis="x", alpha=0.2)
    importance_figure = FIGURES / "random_forest_validation_permutation_importance.png"
    fig.savefig(importance_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison_scores = [
        (y_oof, p_oof, "OOF validation RF", "#d46c00"),
        (y_test, p_test, "Final test RF", "#1769aa"),
        (
            elastic_oof.target.to_numpy(dtype=int),
            elastic_oof.prediction_probability.to_numpy(dtype=float),
            "OOF validation Elastic Net",
            "#8a5aa6",
        ),
        (
            elastic_test.target.to_numpy(dtype=int),
            elastic_test.prediction_probability.to_numpy(dtype=float),
            "Final test Elastic Net",
            "#40865a",
        ),
        (gb_y_oof, gb_p_oof, "OOF Gradient Boosting sensitivity", "#7c59a5"),
        (y_test, gb_p_test, "Test Gradient Boosting sensitivity", "#a88bc4"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for labels, probabilities, name, color in comparison_scores:
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
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    performance_figure = FIGURES / "random_forest_vs_elastic_net.png"
    fig.savefig(performance_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Assemble the frozen-model comparison and validation-importance report.
    report_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Regularized Random Forest</title><style>
    body{{font:15px/1.55 system-ui,sans-serif;margin:28px auto;padding:0 22px;max-width:1200px;color:#203142}}h1,h2{{color:#17425e}}.wrap{{overflow:auto;max-height:600px;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #d9e1e8;text-align:left;white-space:nowrap}}th{{background:#edf3f7;position:sticky;top:0}}tr:nth-child(even){{background:#f8fafb}}img{{max-width:100%;height:auto}}figure{{margin:22px 0}}figcaption{{font-size:13px;color:#52636e}}.note{{padding:12px 15px;background:#edf6ef;border-left:4px solid #398253}}
    </style></head><body><h1>Regularized Random Forest benchmark</h1>
    <p class="note">Tree complexity, leaf/split sizes, feature subsampling, row subsampling, and class weights were selected using mean walk-forward validation PR-AUC. The classification threshold was selected from pooled OOF validation predictions. The test set was opened after all settings were frozen. Random Forest is a nonlinear sensitivity benchmark; Elastic Net remains primary.</p>
    <h2>Selected Random Forest configuration</h2><div class="wrap">{pd.DataFrame([configuration]).drop(columns=["training_event_ids", "test_event_ids"], errors="ignore").to_html(index=False, escape=True, border=0)}</div>
    <h2>Performance on validation and final test</h2><div class="wrap">{comparison.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", na_rep="unavailable", border=0)}</div>
    <h2>Training fit diagnostic</h2><div class="wrap">{pd.DataFrame([training_diagnostic]).to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", na_rep="unavailable", border=0)}</div><p>Training performance is in-sample and is reported only to diagnose overfitting. It is not used to select hyperparameters, threshold, or features.</p>
    <p>Top-risk capture is the fraction of bankrupt observations in the highest-risk 1%, 5%, or 10%. Importance is measured as validation average-precision decrease after feature permutation. Bootstrap summaries resample validation cases and controls within fold; top-10 frequency describes ranking stability, not causal importance.</p>
    <figure><img src="../../figures/06_random_forest_reporting/{performance_figure.name}" alt="Random Forest and Elastic Net validation and test curves"><figcaption>OOF validation and final test performance.</figcaption></figure>
    <figure><img src="../../figures/06_random_forest_reporting/{importance_figure.name}" alt="Validation permutation importance"><figcaption>Top validation permutation importance. No training impurity importance is used.</figcaption></figure>
    <h2>Importance and bootstrap stability</h2><div class="wrap">{permutation_importance.head(100).to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", na_rep="unavailable", border=0)}</div>
    <h2>Walk-forward tuning</h2><div class="wrap">{tuning.to_html(index=False, escape=True, float_format=lambda v: f"{v:.5g}", border=0)}</div>
    <p>Hyperparameter ranking uses mean validation PR-AUC; validation ROC-AUC and Brier are tie-breakers only. No test metric influenced tuning, the threshold, or importance analysis. The sampled matched cohort does not estimate population bankruptcy prevalence.</p></body></html>"""

    (REPORTS / "random_forest_report.html").write_text(report_html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(
        "Call generate_report(context) from the corresponding modeling notebook."
    )
