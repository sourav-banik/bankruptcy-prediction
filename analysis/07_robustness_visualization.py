"""Plot validation performance and save robustness-analysis summaries."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils import project_paths  # noqa: E402

PATHS = project_paths(ROOT)
REPORTS = PATHS.reports / "07_robustness_visualization"
FIGURES = PATHS.figures / "07_robustness_visualization"


def main(context):
    """Save validation-based robustness comparisons and primary-model selection."""
    oof_results = context["oof_results"]
    primary_scenario = context["primary_scenario"]
    robustness_results = context["robustness_results"]
    fold_validation_results = context["fold_validation_results"]
    primary_candidates = context["primary_candidates"]
    primary_choice = context["primary_choice"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Rank scenarios using pooled out-of-fold validation PR-AUC only.
    validation_results = oof_results.sort_values(
        "pooled_oof_pr_auc", ascending=True, kind="stable"
    )
    colors = np.where(
        validation_results.scenario.eq(primary_scenario), "#c75b39", "#397da1"
    )

    fig, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.barh(
        np.arange(len(validation_results)),
        validation_results.pooled_oof_pr_auc,
        color=colors,
    )
    axis.set_yticks(np.arange(len(validation_results)), validation_results.scenario)
    axis.set_xlabel("Pooled out-of-fold validation average precision")
    axis.set_title(
        f"Walk-forward validation robustness; primary specification: {primary_scenario}"
    )
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(
        FIGURES / "model_performance_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # These exports contain walk-forward validation results only; notebook 19 reports final test metrics.
    robustness_results.to_csv(REPORTS / "robustness_results.csv", index=False)
    fold_validation_results.to_csv(
        REPORTS / "robustness_fold_validation_results.csv", index=False
    )
    selection_path = PATHS.models / "primary_feature_specification.json"
    if selection_path.is_file():
        primary_record = json.loads(selection_path.read_text(encoding="utf-8"))
        primary_record["selected_primary_scenario"] = primary_scenario
        primary_record["selection_basis"] = (
            "pooled walk-forward OOF PR-AUC with compact-feature tolerance"
        )
    else:
        primary_record = {
            "selected_primary_scenario": primary_scenario,
            "selection_basis": "pooled out-of-fold validation PR-AUC",
            "compact_pooled_oof_pr_auc": float(
                primary_candidates.loc[
                    primary_candidates.scenario.eq("compact_features"),
                    "pooled_oof_pr_auc",
                ].iloc[0]
            ),
            "full_pooled_oof_pr_auc": float(
                primary_candidates.loc[
                    primary_candidates.scenario.eq("full_features"),
                    "pooled_oof_pr_auc",
                ].iloc[0]
            ),
            "selected_C": float(primary_choice.selected_C),
            "test_used_for_selection": False,
        }
    (REPORTS / "primary_model_selection.json").write_text(
        json.dumps(primary_record, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit("Call main(context) from notebook 21.")
