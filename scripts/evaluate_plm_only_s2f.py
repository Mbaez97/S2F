#!/usr/bin/env python3
"""Evaluate completed PLM-only S2F NPZ runs on the shared protein sets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT))
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_knn_competitor_analysis as analysis  # noqa: E402
import clean_plm_benchmark as benchmark  # noqa: E402


ALL_METRICS = [
    "overall::AUC", "overall::AUPR", "overall::Precision at 0.2 Recall",
    "overall::F_max", "overall::NDCG", "overall::Jaccard", "overall::smin",
    "AUC per-gene", "AUPR per-gene", "Precision at 0.2 Recall per-gene",
    "F_max per-gene", "NDCG per-gene", "Jaccard per-gene", "smin per-gene",
    "AUC per-term", "AUPR per-term", "Precision at 0.2 Recall per-term",
    "F_max per-term", "NDCG per-term", "Jaccard per-term", "smin per-term",
]
ORGANISMS = ("83333", "1111708")
RUN_ALIASES = {
    ("83333", "knn"): "83333_plm_knn_k160",
    ("83333", "kde"): "83333_plm_kde",
    ("1111708", "knn"): "1111708_plm_knn_k80",
    ("1111708", "kde"): "1111708_plm_kde",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--installation-dir",
        type=Path,
        default=Path("/run/media/marcelo_baez/HD_Disc1/.S2F"),
    )
    parser.add_argument(
        "--goa",
        type=Path,
        default=Path("/run/media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=S2F_ROOT / "notebooks" / "exports" / "plm_only_s2f",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def load_original_metrics() -> pd.DataFrame:
    original_metrics = pd.read_csv(
        S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
        / "competitor_context_metrics.csv"
    )
    original_metrics["organism"] = original_metrics["organism"].astype(str)
    return original_metrics[
        original_metrics["model"].astype(str) == "S2F"
    ].drop_duplicates(
        "organism", keep="first"
    )


def prediction_table(run_dir: Path, evaluation: set[str]) -> pd.DataFrame:
    prediction_path = run_dir / "prediction.npz"
    proteins_path = run_dir / "proteins.df"
    terms_path = run_dir / "terms.df"
    for path in (prediction_path, proteins_path, terms_path):
        if not path.is_file():
            raise RuntimeError(f"Missing completed-run artifact: {path}")
    proteins = pd.read_pickle(proteins_path)
    terms = pd.read_pickle(terms_path)
    selected = proteins[proteins.index.astype(str).isin(evaluation)]
    matrix = sparse.load_npz(prediction_path).tocsr()
    if matrix.shape != (len(proteins), len(terms)):
        raise RuntimeError(
            f"Prediction shape {matrix.shape} does not match indexes "
            f"{(len(proteins), len(terms))}: {prediction_path}"
        )
    row_ids = selected["protein idx"].to_numpy(dtype=int)
    subset = matrix[row_ids].tocoo()
    protein_by_local_row = selected.index.astype(str).to_numpy()
    term_by_column = np.empty(len(terms), dtype=object)
    term_by_column[terms["term idx"].to_numpy(dtype=int)] = terms.index.astype(str)
    result = pd.DataFrame(
        {
            "protein_id": protein_by_local_row[subset.row],
            "term_id": term_by_column[subset.col],
            "score": subset.data.astype(float),
        }
    )
    return result[result["score"] > 0].copy()


def matrix_for_bootstrap(
    organism: str,
    model: str,
    predictions: pd.DataFrame,
    original: analysis.MethodMatrix,
) -> analysis.MethodMatrix:
    protein_index = {protein: index for index, protein in enumerate(original.proteins)}
    term_index = {term: index for index, term in enumerate(original.terms)}
    matrix = np.zeros_like(original.prediction, dtype=float)
    rows = predictions[
        predictions["protein_id"].isin(protein_index)
        & predictions["term_id"].isin(term_index)
    ]
    for row in rows.itertuples(index=False):
        matrix[protein_index[row.protein_id], term_index[row.term_id]] = float(row.score)
    return analysis.MethodMatrix(
        organism=organism,
        method=model,
        proteins=original.proteins,
        terms=original.terms,
        prediction=matrix,
        gold=original.gold,
        information_content=original.information_content,
        term_domains=original.term_domains,
        term_names=original.term_names,
        detail_path=Path("prediction.npz"),
    )


def bootstrap_deltas(
    organism: str,
    model: str,
    new_matrix: analysis.MethodMatrix,
    old_matrix: analysis.MethodMatrix,
    replicates: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    new_units = analysis.unit_metric_rows(new_matrix, axis=1)
    old_units = analysis.unit_metric_rows(old_matrix, axis=1)
    if [row["unit_id"] for row in new_units] != [row["unit_id"] for row in old_units]:
        raise RuntimeError("Paired bootstrap protein order differs.")
    count = len(new_units)
    samples = rng.integers(0, count, size=(replicates, count))
    rows = []
    for metric in analysis.METRICS:
        new_values = np.asarray([row[metric] for row in new_units], dtype=float)
        old_values = np.asarray([row[metric] for row in old_units], dtype=float)
        differences = (new_values - old_values)[samples].mean(axis=1)
        lower = float(np.quantile(differences, 0.025))
        upper = float(np.quantile(differences, 0.975))
        if lower > 0:
            conclusion = "worse" if metric == "smin" else "better"
        elif upper < 0:
            conclusion = "better" if metric == "smin" else "worse"
        else:
            conclusion = "unresolved"
        rows.append(
            {
                "organism": organism,
                "model": model,
                "scope": "per-gene",
                "metric": metric,
                "replicates": replicates,
                "delta_new_minus_original": float(np.mean(new_values - old_values)),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "conclusion": conclusion,
            }
        )
    return rows


def main():
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise RuntimeError("--bootstrap-replicates must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_sets, evaluation_summary = benchmark.build_evaluation_sets(args.goa)
    original_metrics = load_original_metrics()
    advanced_matrices, matrix_inputs = analysis.load_method_matrices(
        analysis.FRONTEND_DATA, list(ORGANISMS)
    )
    metrics_rows = []
    comparison_rows = []
    bootstrap_rows = []
    run_inputs = []

    for organism in ORGANISMS:
        annotations, ontology, organism_name = benchmark.prepare_ground_truth(
            args.goa, S2F_ROOT / "go.obo", organism, evaluation_sets[organism]
        )
        original_row = original_metrics[
            original_metrics["organism"] == organism
        ].iloc[0]
        for strategy in ("knn", "kde"):
            alias = RUN_ALIASES[(organism, strategy)]
            run_dir = args.installation_dir.expanduser().resolve() / "output" / alias
            predictions = prediction_table(run_dir, evaluation_sets[organism])
            model = f"PLM-only S2F {strategy.upper()}"
            row = benchmark.evaluate_prediction_table(
                model, organism, predictions, annotations, ontology,
                organism_name, len(evaluation_sets[organism]),
                source_note=f"Full S2F graph diffusion using only the production {strategy} PLM seed.",
                extra_metadata={"transfer_strategy": strategy, "run_alias": alias},
            )
            missing = [metric for metric in ALL_METRICS if metric not in row]
            if missing:
                raise RuntimeError(f"Evaluator omitted scalar metrics: {missing}")
            metrics_rows.append(row)
            for metric in ALL_METRICS:
                new_value = float(row[metric])
                old_value = float(original_row[metric])
                comparison_rows.append(
                    {
                        "organism": organism,
                        "model": model,
                        "metric": metric,
                        "plm_only_s2f": new_value,
                        "original_s2f": old_value,
                        "delta_new_minus_original": new_value - old_value,
                        "optimization": "lower" if "smin" in metric else "higher",
                    }
                )
            old_matrix = advanced_matrices[(organism, "S2F")]
            new_matrix = matrix_for_bootstrap(organism, model, predictions, old_matrix)
            bootstrap_rows.extend(
                bootstrap_deltas(
                    organism, model, new_matrix, old_matrix,
                    args.bootstrap_replicates,
                    args.seed + len(bootstrap_rows),
                )
            )
            run_inputs.append(str(run_dir))

    metrics_path = output_dir / "plm_only_s2f_metrics.csv"
    comparison_path = output_dir / "plm_only_vs_original_s2f.csv"
    bootstrap_path = output_dir / "paired_protein_bootstrap.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    evaluation_summary.to_csv(output_dir / "evaluation_protein_sets.csv", index=False)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "evaluation_sizes": {
            organism: len(evaluation_sets[organism]) for organism in ORGANISMS
        },
        "all_scalar_metrics": ALL_METRICS,
        "run_inputs": run_inputs,
        "saved_original_metrics": [
            str(S2F_ROOT / "notebooks" / "esm_go_explorer" / "data" / "competitor_context_metrics.csv"),
        ],
        "matrix_inputs": sorted(str(path) for path in matrix_inputs),
        "outputs": [str(metrics_path), str(comparison_path), str(bootstrap_path)],
        "bootstrap_note": (
            "Paired protein bootstrap covers per-gene AUC, AUPR, F_max, and smin "
            "for 83333 and 1111708. A confidence interval spanning zero is "
            "reported as unresolved, never as equal."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
