"""Shared helpers used by the project notebooks."""
from __future__ import annotations

import ast
import base64
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    intermediate: Path
    processed: Path
    reports: Path
    figures: Path
    models: Path
    artifacts: Path


def project_paths(root: str | Path | None = None) -> ProjectPaths:
    """Return the standard project directories for a project root or notebooks folder."""
    base = Path.cwd() if root is None else Path(root)
    base = base.resolve()
    if base.name.lower() == "notebooks":
        base = base.parent
    return ProjectPaths(
        root=base,
        raw=base / "data" / "raw",
        intermediate=base / "data" / "intermediate",
        processed=base / "data" / "processed",
        reports=base / "reports",
        figures=base / "figures",
        models=base / "models",
        artifacts=base / "artifacts",
    )


def file_hash(path: str | Path) -> str:
    """Return a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(split, processed, schema, id_columns, features, input_hashes):
    """Load a saved split after checking its membership and feature schema."""
    processed = Path(processed)
    matrix_path = processed / f"{split}_matrix.csv"
    ids_path = processed / f"{split}_ids.csv"
    input_hashes[matrix_path] = file_hash(matrix_path)
    input_hashes[ids_path] = file_hash(ids_path)
    if input_hashes[ids_path] != schema["split_file_sha256"][split]:
        raise ValueError(f"{split} membership changed: rerun notebook 17")
    dtype = {"company_id": "string", "metadata_event_id": "string"}
    frame = pd.read_csv(matrix_path, dtype=dtype)
    pd.read_csv(ids_path, dtype=dtype)
    if frame.columns.tolist() != list(id_columns) + list(features):
        raise ValueError("Matrix schema mismatch")
    return frame


def safe_divide(numerator, denominator):
    """Divide while treating zero denominators and non-finite results as missing."""
    nonzero = denominator.where(denominator.ne(0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = numerator / nonzero
    return result.replace([np.inf, -np.inf], np.nan)


def canonical_security(values):
    """Normalize Bloomberg security strings for consistent joins."""
    return values.astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def notebook_literal(filename: str, variable: str, root: str | Path | None = None):
    """Read a literal assignment from a code cell in a project notebook."""
    base = Path.cwd() if root is None else Path(root)
    base = base.resolve()
    if base.name.lower() == "notebooks":
        base = base.parent
    if not (base / "notebooks").is_dir() and base.parent.joinpath("notebooks").is_dir():
        base = base.parent
    path = base / "notebooks" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise ValueError(f"Literal definition {variable} missing from {filename}")


def probability_metrics(labels, probabilities, threshold=0.5):
    """Calculate the shared binary-classification metrics used by model notebooks."""
    from sklearn.metrics import (
        auc, average_precision_score, balanced_accuracy_score, brier_score_loss,
        confusion_matrix, f1_score, precision_recall_curve, roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability vector")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities outside [0,1]")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "n": len(labels), "bankrupt_n": int(labels.sum()),
        "prevalence": float(labels.mean()), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_table(labels, probabilities, n_bins=8):
    """Summarize observed event rates across quantile-binned predictions."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    bins = np.zeros(len(probabilities), dtype=int) if len(edges) < 2 else np.searchsorted(edges[1:-1], probabilities)
    data = pd.DataFrame({"bin": bins, "probability": probabilities, "target": labels})
    return data.groupby("bin", sort=True).agg(
        n=("target", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_bankruptcy_rate=("target", "mean"),
    ).reset_index()


def embedded_image(path: str | Path, caption: str) -> str:
    """Return a compact HTML figure containing a locally saved PNG."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" '
        f'alt="{html.escape(caption)}" style="max-width:100%;height:auto">'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
    )
