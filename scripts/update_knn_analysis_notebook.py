#!/usr/bin/env python3
"""Add an idempotent KNN-versus-competitor analysis section to the notebook.

The numerical work remains in ``build_knn_competitor_analysis.py``.  This
script only adds lightweight presentation cells that read its versioned JSON
export, so the notebook and web explorer cannot silently implement different
metric formulas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
SECTION_TAG = "knn-competitor-analysis"
KNN10 = "Clean PLM + KNN k=10 (weighted_support)"
METHODS = ("S2F", "TALE", "ATGO", "PANDA2", KNN10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=S2F_ROOT)
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/competitors_performance.ipynb"))
    return parser.parse_args()


def source_lines(text: str):
    return text.splitlines(keepends=True)


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [SECTION_TAG]},
        "source": source_lines(text.rstrip() + "\n"),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [SECTION_TAG]},
        "outputs": [],
        "source": source_lines(text.rstrip() + "\n"),
    }


def metric_value(payload, organism: str, method: str, metric: str) -> float:
    row = next(
        row for row in payload["metric_rows"]
        if row["organism"] == organism
        and row["method"] == method
        and row["scope"] == "per-gene"
        and row["metric"] == metric
    )
    return float(row["value"])


def snapshot_markdown(payload) -> str:
    lines = [
        "## Why can a simple KNN obtain a strong Fmax?",
        "",
        f"Generated from the post-blacklist diagnostics at `{payload['generated_at_utc']}`. ",
        "The code cells below load the same JSON used by the explorer. Archived pre-blacklist rows are not used.",
        "",
        "### Current per-gene benchmark snapshot",
        "",
        "| Organism | Method | Fmax | AUPR | AUROC | Smin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for organism in payload["metadata"]["organisms"]:
        for method in METHODS:
            label = method.replace("Clean PLM + ", "")
            values = [metric_value(payload, organism, method, metric) for metric in ("F_max", "AUPR", "AUC", "smin")]
            lines.append(
                f"| {organism} | {label} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[3]:.4f} |"
            )
    lines.extend(
        [
            "",
            "The reported per-gene Fmax is an **oracle average**: F1 is maximized independently for every protein, then averaged. "
            "The threshold diagnostics below contrast it with one shared threshold and repeated held-out threshold selection. "
            "AUPR instead scores the complete ranking, so it exposes poorly ordered false positives that a single best F1 point can hide.",
            "",
            "> Smin warning: historical competitor Smin rows used full-organism information content, while Clean PLM used the shared evaluation set. "
            "Use `harmonized_value` for the fair same-IC sensitivity analysis.",
        ]
    )
    return "\n".join(lines)


def analysis_cells(payload):
    return [
        markdown_cell(snapshot_markdown(payload)),
        code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import display

KNN10 = 'Clean PLM + KNN k=10 (weighted_support)'
ANALYSIS_PATH = next(
    candidate / 'notebooks/esm_go_explorer/data/knn_competitor_analysis.json'
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / 'notebooks/esm_go_explorer/data/knn_competitor_analysis.json').is_file()
)
KNN_ANALYSIS = json.loads(ANALYSIS_PATH.read_text())
METRIC_DF = pd.DataFrame(KNN_ANALYSIS['metric_rows'])
THRESHOLD_DF = pd.DataFrame(KNN_ANALYSIS['threshold_summary'])
SHARED_THRESHOLD_DF = pd.DataFrame(KNN_ANALYSIS['shared_threshold_summary'])
CV_THRESHOLD_DF = pd.DataFrame(KNN_ANALYSIS['cross_validation_summary'])
SCORE_DF = pd.DataFrame(KNN_ANALYSIS['score_summary'])
ONTOLOGY_DF = pd.DataFrame(KNN_ANALYSIS['ontology_rows'])
FREQUENCY_DF = pd.DataFrame(KNN_ANALYSIS['frequency_rows'])
NEIGHBOR_AUDIT_DF = pd.DataFrame(KNN_ANALYSIS['neighbor_audit'])
NEIGHBOR_PROFILE_DF = pd.DataFrame(KNN_ANALYSIS['neighbor_profiles'])
NEIGHBOR_CORRELATION_DF = pd.DataFrame(KNN_ANALYSIS['neighbor_correlations'])
EXAMPLE_DF = pd.DataFrame(KNN_ANALYSIS['examples'])
print('Loaded', ANALYSIS_PATH)
print('Generated', KNN_ANALYSIS['generated_at_utc'])"""
        ),
        markdown_cell(
            """### Metric rankings and Smin comparability

Use the first table to reproduce the benchmark values. The second table puts every method on the same shared-evaluation-set information-content scale for Smin."""
        ),
        code_cell(
            """PRIMARY = KNN_ANALYSIS['metadata']['primary_methods']
display(
    METRIC_DF.query("scope == 'per-gene' and method in @PRIMARY")
    .pivot(index=['organism', 'method'], columns='metric', values='value')
    .reset_index()
)
display(
    METRIC_DF.query("scope == 'per-gene' and metric == 'smin' and method in @PRIMARY")
    [['organism', 'method', 'value', 'harmonized_value', 'saved_information_content_scope']]
    .rename(columns={'value': 'saved_smin', 'harmonized_value': 'shared_set_ic_smin'})
)"""
        ),
        markdown_cell(
            """### Is KNN's Fmax advantage threshold-specific?

`oracle_fmax` averages a separately optimized F1 for each protein. `shared_threshold_f1` uses one best threshold for the whole organism. `heldout_f1_mean` learns thresholds without the held-out proteins."""
        ),
        code_cell(
            """threshold_comparison = (
    THRESHOLD_DF.query("scope == 'per-gene' and method in @PRIMARY")
    [['organism', 'method', 'oracle_fmax', 'optimal_threshold_median', 'precision_at_oracle_mean', 'recall_at_oracle_mean']]
    .merge(
        SHARED_THRESHOLD_DF.query("scope == 'per-gene' and method in @PRIMARY")
        [['organism', 'method', 'shared_threshold', 'shared_threshold_f1']],
        on=['organism', 'method'],
    )
    .merge(
        CV_THRESHOLD_DF.query("method in @PRIMARY")
        [['organism', 'method', 'threshold_mean', 'heldout_precision_mean', 'heldout_recall_mean', 'heldout_f1_mean']],
        on=['organism', 'method'],
    )
)
display(threshold_comparison)"""
        ),
        markdown_cell("### Score sparsity, repeated values, ontology, and GO-term frequency"),
        code_cell(
            """display(
    SCORE_DF.query('method in @PRIMARY')
    [['organism', 'method', 'nonzero_predictions', 'zero_fraction', 'unique_nonzero_scores',
      'most_repeated_score_count', 'score_median_nonzero', 'mean_predictions_per_protein']]
)
display(
    ONTOLOGY_DF.query("scope == 'per-gene' and ontology_roots == 'excluded' and metric in ['F_max', 'AUPR'] and method in @PRIMARY")
    .pivot(index=['organism', 'ontology', 'method'], columns='metric', values='value')
    .reset_index()
)
display(
    FREQUENCY_DF.query("scope == 'per-term' and metric in ['F_max', 'AUPR'] and method in @PRIMARY")
    .pivot(index=['organism', 'frequency_bin', 'method'], columns='metric', values='value')
    .reset_index()
)"""
        ),
        markdown_cell("### Post-blacklist neighbor evidence and paired protein examples"),
        code_cell(
            """display(NEIGHBOR_AUDIT_DF)
display(NEIGHBOR_CORRELATION_DF)
display(
    EXAMPLE_DF[['organism', 'direction', 'rank', 'protein_id', 'best_advanced_method',
                'knn_fmax', 'advanced_fmax', 'fmax_delta', 'knn_aupr', 'advanced_aupr']]
    .sort_values(['organism', 'direction', 'rank'])
)
print('The full machine-readable tables and bootstrap intervals are in notebooks/exports/knn_competitor_analysis/.')"""
        ),
    ]


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    notebook_path = args.notebook if args.notebook.is_absolute() else project_root / args.notebook
    analysis_path = project_root / "notebooks" / "esm_go_explorer" / "data" / "knn_competitor_analysis.json"
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if SECTION_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    notebook["cells"].extend(analysis_cells(payload))
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(notebook_path), "cells": len(notebook["cells"])}, indent=2))


if __name__ == "__main__":
    main()
