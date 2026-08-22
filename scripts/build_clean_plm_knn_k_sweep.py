#!/usr/bin/env python3
"""Evaluate a metrics-only KNN neighbourhood-size sweep for Clean PLM.

The active Clean PLM benchmark intentionally remains untouched.  This runner
reuses its cached embeddings, shared evaluation sets, donor exclusions, GO
transfer, and evaluator, but writes a separate sweep artifact for the explorer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT))
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import clean_plm_benchmark as bench  # noqa: E402
from GOTool import GeneOntology  # noqa: E402


DEFAULT_K_VALUES = [3, 5, 7, 10, 20, 40, 80, 160, 320, 640, 1024]
DEFAULT_SOURCE_OUTPUT_DIR = S2F_ROOT / "notebooks" / "exports" / "clean_plm_benchmark"
DEFAULT_OUTPUT_DIR = S2F_ROOT / "notebooks" / "exports" / "clean_plm_knn_k_sweep"
DEFAULT_FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
DEFAULT_GOA_PATH = Path("/run/media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa")
DEFAULT_TARGET_CACHE = Path(
    "/run/media/marcelo_baez/HD_Disc1/.S2F/data/PLM/embeddings/target_57e9174250d35f0d"
)
DEFAULT_BLACKLIST_DIR = Path(
    "/run/media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/blacklist"
)
DISPLAY_ORGANISMS = ["83333", "1111708"]
PRIMARY_METRIC = "F_max per-gene"
GUARDRAIL_METRIC = "AUPR per-gene"
# The legacy evaluator can select a neighboring tied threshold after minor
# floating-point ordering changes.  This remains far below report precision
# while rejecting any substantive reproduction drift.
METRIC_TOLERANCE = 2e-4
COMPATIBILITY_METRICS = [
    "overall::F_max",
    "overall::AUPR",
    "F_max per-gene",
    "AUPR per-gene",
]
ALL_SCALAR_METRICS = [
    "overall::AUC",
    "overall::AUPR",
    "overall::Precision at 0.2 Recall",
    "overall::F_max",
    "overall::NDCG",
    "overall::Jaccard",
    "overall::smin",
    "AUC per-gene",
    "AUPR per-gene",
    "Precision at 0.2 Recall per-gene",
    "F_max per-gene",
    "NDCG per-gene",
    "Jaccard per-gene",
    "smin per-gene",
    "AUC per-term",
    "AUPR per-term",
    "Precision at 0.2 Recall per-term",
    "F_max per-term",
    "NDCG per-term",
    "Jaccard per-term",
    "smin per-term",
]


def parse_k_values(values: Sequence[int]) -> List[int]:
    parsed = sorted({int(value) for value in values})
    if not parsed or parsed[0] < 1:
        raise ValueError("--k-values must contain one or more positive integers.")
    return parsed


def finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def json_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        clean: Dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (np.floating, float)):
                clean[key] = finite_float(value)
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def file_fingerprint(path: Path, include_sha256: bool = False) -> Dict[str, object]:
    stat = path.stat()
    payload: Dict[str, object] = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        payload["sha256"] = bench.file_sha256(path)
    return payload


def cache_fingerprint(cache_dir: Path) -> Dict[str, object]:
    return {
        "path": str(cache_dir.resolve()),
        "files": {
            name: file_fingerprint(cache_dir / name)
            for name in ("embeddings.npy", "ids.tsv", "meta.json")
            if (cache_dir / name).is_file()
        },
    }


def classify_k_sweep(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Summarise the peak and the first confirmed two-step Fmax decline."""
    ordered = sorted(rows, key=lambda row: int(float(row["k"])))
    if not ordered:
        raise ValueError("Cannot classify an empty K sweep.")
    primary = [finite_float(row.get(PRIMARY_METRIC)) for row in ordered]
    if any(value is None for value in primary):
        raise ValueError(f"K sweep is missing finite {PRIMARY_METRIC} values.")
    primary_values = [float(value) for value in primary if value is not None]
    peak_value = max(primary_values)
    peak_index = next(
        index
        for index, value in enumerate(primary_values)
        if math.isclose(value, peak_value, rel_tol=0.0, abs_tol=1e-12)
    )
    below_peak = [value < peak_value - 1e-12 for value in primary_values]
    first_decline_index = next(
        (index for index in range(peak_index + 1, len(ordered)) if below_peak[index]),
        None,
    )
    confirmation_index = next(
        (
            index
            for index in range(peak_index + 2, len(ordered))
            if below_peak[index - 1] and below_peak[index]
        ),
        None,
    )
    aupr_values = [finite_float(row.get(GUARDRAIL_METRIC)) for row in ordered]
    aupr_peak = max(value for value in aupr_values if value is not None)
    annotated_rows = []
    for index, row in enumerate(ordered):
        annotated = dict(row)
        annotated["is_primary_peak"] = index == peak_index
        annotated["is_below_primary_peak"] = below_peak[index]
        annotated["aupr_below_observed_peak"] = (
            aupr_values[index] is not None and float(aupr_values[index]) < aupr_peak - 1e-12
        )
        annotated_rows.append(annotated)
    return {
        "primary_metric": PRIMARY_METRIC,
        "guardrail_metric": GUARDRAIL_METRIC,
        "selection_rule": (
            "Lowest K tied for the maximum per-gene Fmax; a crash is confirmed when two "
            "consecutive larger K values are both below that peak."
        ),
        "best_k": int(float(ordered[peak_index]["k"])),
        "best_primary_value": peak_value,
        "first_decline_k": (
            int(float(ordered[first_decline_index]["k"])) if first_decline_index is not None else None
        ),
        "confirmed_crash_k": (
            int(float(ordered[confirmation_index]["k"])) if confirmation_index is not None else None
        ),
        "crash_confirmed": confirmation_index is not None,
        "best_aupr_value": aupr_peak,
        "rows": annotated_rows,
    }


def metric_optimization(metric: str) -> str:
    """Return the direction in which an evaluator metric improves."""
    return "minimize" if metric.endswith("smin") or metric.startswith("smin ") else "maximize"


def classify_metric_k_sweep(
    rows: Sequence[Mapping[str, object]], metric: str
) -> Dict[str, object]:
    """Classify an individual metric's optimum and two-step degradation point."""
    ordered = sorted(rows, key=lambda row: int(float(row["k"])))
    if not ordered:
        raise ValueError("Cannot classify an empty K sweep.")
    values = [finite_float(row.get(metric)) for row in ordered]
    if any(value is None for value in values):
        raise ValueError(f"K sweep is missing finite {metric} values.")
    numeric_values = [float(value) for value in values if value is not None]
    optimization = metric_optimization(metric)
    best_value = min(numeric_values) if optimization == "minimize" else max(numeric_values)
    peak_index = next(
        index
        for index, value in enumerate(numeric_values)
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
    )
    if optimization == "minimize":
        is_worse = [value > best_value + 1e-12 for value in numeric_values]
    else:
        is_worse = [value < best_value - 1e-12 for value in numeric_values]
    first_worse_index = next(
        (index for index in range(peak_index + 1, len(ordered)) if is_worse[index]),
        None,
    )
    confirmation_index = next(
        (
            index
            for index in range(peak_index + 2, len(ordered))
            if is_worse[index - 1] and is_worse[index]
        ),
        None,
    )
    annotated_rows = []
    for index, row in enumerate(ordered):
        annotated = {
            "k": int(float(row["k"])),
            "value": numeric_values[index],
            "is_best": index == peak_index,
            "is_worse_than_best": is_worse[index],
        }
        annotated_rows.append(annotated)
    return {
        "metric": metric,
        "optimization": optimization,
        "selection_rule": (
            "Lowest K tied for the optimum; degradation is confirmed when two consecutive "
            "larger K values are both worse than that optimum."
        ),
        "best_k": int(float(ordered[peak_index]["k"])),
        "best_value": best_value,
        "first_worse_k": (
            int(float(ordered[first_worse_index]["k"])) if first_worse_index is not None else None
        ),
        "confirmed_degradation_k": (
            int(float(ordered[confirmation_index]["k"])) if confirmation_index is not None else None
        ),
        "degradation_confirmed": confirmation_index is not None,
        "rows": annotated_rows,
    }


def write_all_metric_analysis(
    metrics: pd.DataFrame, output_dir: Path, k_values: Sequence[int]
) -> Dict[str, Path]:
    """Export every scalar evaluator metric in both compact and chart-ready forms."""
    missing = [metric for metric in ALL_SCALAR_METRICS if metric not in metrics.columns]
    if missing:
        raise RuntimeError(f"Completed metrics are missing evaluator columns: {', '.join(missing)}")

    summary_rows = []
    point_rows = []
    organism_payload: Dict[str, object] = {}
    for organism in bench.ORGANISMS:
        organism_rows = json_records(metrics[metrics["organism"].astype(str) == organism])
        classifications = {
            metric: classify_metric_k_sweep(organism_rows, metric)
            for metric in ALL_SCALAR_METRICS
        }
        organism_payload[organism] = {"metrics": classifications}
        for metric in ALL_SCALAR_METRICS:
            classification = classifications[metric]
            summary_rows.append(
                {
                    "organism": organism,
                    "metric": metric,
                    "optimization": classification["optimization"],
                    "best_k": classification["best_k"],
                    "best_value": classification["best_value"],
                    "first_worse_k": classification["first_worse_k"],
                    "confirmed_degradation_k": classification["confirmed_degradation_k"],
                    "degradation_confirmed": classification["degradation_confirmed"],
                }
            )
            for point in classification["rows"]:
                point_rows.append(
                    {
                        "organism": organism,
                        "k": point["k"],
                        "metric": metric,
                        "value": point["value"],
                        "optimization": classification["optimization"],
                        "is_best": point["is_best"],
                        "is_worse_than_best": point["is_worse_than_best"],
                    }
                )

    summary_path = output_dir / "clean_plm_knn_k_sweep_all_metric_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    points_path = output_dir / "clean_plm_knn_k_sweep_all_metrics_by_k.csv"
    pd.DataFrame(point_rows).to_csv(points_path, index=False)
    payload_path = output_dir / "knn_k_sweep_all_metrics.json"
    payload = {
        "schema_version": 1,
        "description": "Complete scalar evaluator-metric analysis for the Clean PLM KNN K sweep.",
        "candidate_k_values": [int(value) for value in k_values],
        "metric_definitions": [
            {"metric": metric, "optimization": metric_optimization(metric)}
            for metric in ALL_SCALAR_METRICS
        ],
        "organisms": organism_payload,
    }
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return {
        "all_metric_summary": summary_path,
        "all_metrics_by_k": points_path,
        "all_metrics_json": payload_path,
    }


def compatibility_check(metrics: pd.DataFrame, baseline_path: Path) -> Dict[str, object]:
    required_k = {3, 5, 7, 10}
    if not baseline_path.is_file():
        raise RuntimeError(f"Missing active benchmark metrics for compatibility check: {baseline_path}")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["transfer_strategy"].astype(str) == "knn"].copy()
    baseline["k"] = pd.to_numeric(baseline["k"], errors="coerce")
    sweep = metrics.copy()
    sweep["k"] = pd.to_numeric(sweep["k"], errors="coerce")
    comparisons = []
    for organism in bench.ORGANISMS:
        for k_value in sorted(required_k & set(sweep["k"].dropna().astype(int))):
            expected = baseline[(baseline["organism"].astype(str) == organism) & (baseline["k"] == k_value)]
            observed = sweep[(sweep["organism"].astype(str) == organism) & (sweep["k"] == k_value)]
            if len(expected) != 1 or len(observed) != 1:
                raise RuntimeError(f"Compatibility row missing for organism={organism}, k={k_value}.")
            for column in COMPATIBILITY_METRICS:
                delta = abs(float(observed.iloc[0][column]) - float(expected.iloc[0][column]))
                comparisons.append(
                    {
                        "organism": organism,
                        "k": int(k_value),
                        "metric": column,
                        "absolute_difference": delta,
                        "passes": delta <= METRIC_TOLERANCE,
                    }
                )
    passed = all(row["passes"] for row in comparisons)
    return {
        "status": "pass" if passed else "fail",
        "tolerance": METRIC_TOLERANCE,
        "metrics": COMPATIBILITY_METRICS,
        "baseline_path": str(baseline_path.resolve()),
        "rows": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--source-output-dir", type=Path, default=DEFAULT_SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontend-data", type=Path, default=DEFAULT_FRONTEND_DATA)
    parser.add_argument("--goa-path", type=Path, default=DEFAULT_GOA_PATH)
    parser.add_argument("--target-cache-dir", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--blacklist-dir", type=Path, default=DEFAULT_BLACKLIST_DIR)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument(
        "--publish-existing",
        action="store_true",
        help="Publish a completed metrics CSV after rerunning the compatibility gate, without recomputing embeddings or metrics.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Write complete all-metric result files from a completed metrics CSV without changing web assets.",
    )
    return parser.parse_args()


def load_cached_queries(source_output_dir: Path, evaluation_sets: Mapping[str, set[str]]) -> Dict[str, object]:
    caches = {}
    for organism in bench.ORGANISMS:
        cache_dir = source_output_dir / "embeddings" / f"query_{organism}"
        cache = bench.plm.load_embedding_cache(cache_dir, expected_metadata=None, validate=False)
        if cache is None:
            raise RuntimeError(f"Missing reusable query embedding cache: {cache_dir}")
        missing = set(evaluation_sets[organism]) - set(cache.ids)
        if missing:
            raise RuntimeError(
                f"Query cache {cache_dir} does not cover the shared evaluation set for {organism}; "
                f"missing {len(missing)} proteins."
            )
        caches[organism] = cache
    return caches


def write_frontend_payload(frontend_data: Path, payload: Dict[str, object]) -> Path:
    frontend_data.mkdir(parents=True, exist_ok=True)
    destination = frontend_data / "knn_k_sweep.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    index_path = frontend_data / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["knn_k_sweep"] = {
        "description": "Metrics-only Clean PLM KNN neighbourhood-size sweep with a per-organism Fmax failure horizon.",
        "path": "data/knn_k_sweep.json",
        "schema_version": 1,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return destination


def publish_completed_metrics(args: argparse.Namespace, k_values: Sequence[int]) -> None:
    output_dir = args.output_dir.resolve()
    source_output_dir = args.source_output_dir.resolve()
    frontend_data = args.frontend_data.resolve()
    metrics_path = output_dir / "clean_plm_knn_k_sweep_metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"No completed sweep metrics are available to publish: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics["organism"] = metrics["organism"].astype(str)
    metrics["k"] = pd.to_numeric(metrics["k"], errors="raise").astype(int)
    expected_rows = len(bench.ORGANISMS) * len(k_values)
    if len(metrics) != expected_rows or set(metrics["k"]) != set(k_values):
        raise RuntimeError("Completed metrics do not match the requested K grid; refusing to publish.")
    compatibility = compatibility_check(metrics, source_output_dir / "clean_plm_benchmark_metrics.csv")
    if compatibility["status"] != "pass":
        raise RuntimeError("Completed metrics do not satisfy the compatibility gate; refusing to publish.")
    all_metric_outputs = write_all_metric_analysis(metrics, output_dir, k_values)
    by_organism = {
        organism: classify_k_sweep(json_records(metrics[metrics["organism"] == organism]))
        for organism in bench.ORGANISMS
    }
    payload: Dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Clean PLM KNN neighbourhood-size sweep evaluated on the post-blacklist shared benchmark. "
            "The curve continues after a confirmed decline so later behaviour remains visible."
        ),
        "candidate_k_values": list(k_values),
        "display_organisms": DISPLAY_ORGANISMS,
        "excluded_from_frontend": {
            "223283": "Only three proteins remain in the shared evaluation set; retained in exports but omitted from the website.",
        },
        "primary_metric": PRIMARY_METRIC,
        "guardrail_metric": GUARDRAIL_METRIC,
        "organisms": by_organism,
        "compatibility": compatibility,
    }
    export_payload_path = output_dir / "knn_k_sweep.json"
    export_payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    frontend_payload_path = write_frontend_payload(frontend_data, payload)
    active_paths = [
        source_output_dir / "clean_plm_benchmark_metrics.csv",
        source_output_dir / "clean_plm_predictions.tsv",
        source_output_dir / "clean_plm_benchmark_summary.json",
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "candidate_k_values": list(k_values),
        "metrics_only": True,
        "published_from_completed_metrics": True,
        "active_benchmark_checksums": {
            str(path.resolve()): bench.file_sha256(path) for path in active_paths if path.is_file()
        },
        "outputs": {
            "metrics": file_fingerprint(metrics_path, include_sha256=True),
            **{
                label: file_fingerprint(path, include_sha256=True)
                for label, path in all_metric_outputs.items()
            },
            "export_json": file_fingerprint(export_payload_path, include_sha256=True),
            "frontend_json": file_fingerprint(frontend_payload_path, include_sha256=True),
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "frontend": str(frontend_payload_path), "manifest": str(manifest_path)}, indent=2))


def propagate_exactly_like_active_benchmark(go, direct_rows: List[Dict[str, object]], namespace: str) -> pd.DataFrame:
    """Use the active propagation path, then release its temporary namespace."""
    predictions = bench.propagate_predictions(go, direct_rows, namespace)
    for term in go.terms.values():
        term.annotations.pop(namespace, None)
    return predictions


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    if args.publish_existing and args.summarize_existing:
        raise ValueError("--publish-existing and --summarize-existing cannot be used together.")
    if args.publish_existing:
        publish_completed_metrics(args, k_values)
        return
    if args.summarize_existing:
        output_dir = args.output_dir.resolve()
        metrics_path = output_dir / "clean_plm_knn_k_sweep_metrics.csv"
        if not metrics_path.is_file():
            raise RuntimeError(f"No completed sweep metrics are available to summarise: {metrics_path}")
        metrics = pd.read_csv(metrics_path)
        metrics["organism"] = metrics["organism"].astype(str)
        metrics["k"] = pd.to_numeric(metrics["k"], errors="raise").astype(int)
        expected_rows = len(bench.ORGANISMS) * len(k_values)
        if len(metrics) != expected_rows or set(metrics["k"]) != set(k_values):
            raise RuntimeError("Completed metrics do not match the requested K grid; refusing to summarise.")
        outputs = write_all_metric_analysis(metrics, output_dir, k_values)
        print(json.dumps({label: str(path) for label, path in outputs.items()}, indent=2))
        return
    if args.query_chunk_size < 1:
        raise ValueError("--query-chunk-size must be positive.")
    output_dir = args.output_dir.resolve()
    source_output_dir = args.source_output_dir.resolve()
    frontend_data = args.frontend_data.resolve()
    goa_path = args.goa_path.resolve()
    target_cache_dir = args.target_cache_dir.resolve()
    blacklist_dir = args.blacklist_dir.resolve()
    for path, label in ((goa_path, "GOA file"), (target_cache_dir, "target cache"), (blacklist_dir, "blacklist directory")):
        if not path.exists():
            raise RuntimeError(f"Missing {label}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    active_paths = [
        source_output_dir / "clean_plm_benchmark_metrics.csv",
        source_output_dir / "clean_plm_predictions.tsv",
        source_output_dir / "clean_plm_benchmark_summary.json",
    ]
    active_checksums_before = {
        str(path.resolve()): bench.file_sha256(path) for path in active_paths if path.is_file()
    }

    plm_args = bench.ArgsForPlm(
        model_name="facebook/esm1b_t33_650M_UR50S",
        model_dir="",
        device="cpu",
        local_files_only=True,
        long_sequence_mode="sliding_mean",
        long_window_size=1022,
        long_overlap=128,
        batch_tokens=2048,
        query_chunk_size=args.query_chunk_size,
    )
    target_args = SimpleNamespace(
        target_fasta="",
        target_cache_dir=str(target_cache_dir),
        build_target_cache=False,
        device="cpu",
    )
    target_cache, loaded_target_cache_dir, _ = bench.load_or_build_target_cache(target_args, output_dir, plm_args)
    norm_path = Path(tempfile.gettempdir()) / f"clean_plm_target_norm_{loaded_target_cache_dir.name}.npy"
    target_norm = bench.normalized_target_memmap(target_cache.embeddings, norm_path)
    target_cache.embeddings = target_norm

    evaluation_sets, evaluation_summary = bench.build_evaluation_sets(goa_path)
    query_caches = load_cached_queries(source_output_dir, evaluation_sets)
    blacklists, blacklist_paths = bench.load_organism_blacklists(blacklist_dir)
    benchmark_exclusion_ids = set().union(*evaluation_sets.values())
    source_exclusions, blacklist_audit = bench.build_source_exclusions_by_organism(
        target_cache, benchmark_exclusion_ids, blacklists, blacklist_paths
    )
    audit_by_organism = {
        str(row["organism"]): row for row in blacklist_audit.to_dict(orient="records")
    }
    go = GeneOntology.GeneOntology(str(S2F_ROOT / "go.obo"), verbose=False)
    go.build_structure()
    # Match the active KNN+KDE benchmark: loading the full donor annotation
    # map keeps GO propagation independent of the maximum K requested here.
    print("[knn-k-sweep] Loading experimental GO terms for the full Swiss-Prot donor pool.", flush=True)
    accession_to_terms = bench.read_target_go_terms(go, goa_path, set(target_cache.ids))
    metrics_rows = []

    for organism in bench.ORGANISMS:
        query_cache = query_caches[organism]
        available = len(target_cache.ids) - len(source_exclusions[organism])
        if max(k_values) > available:
            raise RuntimeError(
                f"Requested max K={max(k_values)} exceeds {available} eligible donors for {organism}."
            )
        indices, scores, _excluded_indices = bench.top_neighbors_for_queries(
            query_cache.embeddings,
            target_norm,
            target_cache.ids,
            source_exclusions[organism],
            max(k_values),
            args.query_chunk_size,
            f"KNN sweep {organism}",
        )
        annotations, ontology, organism_name = bench.prepare_ground_truth(
            goa_path, S2F_ROOT / "go.obo", organism, evaluation_sets[organism]
        )
        for k_value in k_values:
            selected_indices = [neighbors[:k_value] for neighbors in indices]
            selected_scores = [neighbor_scores[:k_value] for neighbor_scores in scores]
            direct_rows = bench.direct_rows_for_neighbors(
                query_cache.ids,
                target_cache.ids,
                selected_indices,
                selected_scores,
                accession_to_terms,
                "weighted_support",
            )
            predictions = propagate_exactly_like_active_benchmark(
                go,
                direct_rows,
                f"clean_plm_knn_k_sweep_{organism}_{k_value}",
            )
            source_note = (
                "Metrics-only Clean PLM KNN neighbourhood-size sweep; reused cached query embeddings and "
                "the active target cache; related-taxon blacklist and complete shared benchmark exclusion "
                "were applied before neighbour selection and GO transfer."
            )
            row = bench.evaluate_prediction_table(
                f"Clean PLM + KNN k={k_value} (weighted_support)",
                organism,
                predictions,
                annotations,
                ontology,
                organism_name,
                evaluation_size=len(evaluation_sets[organism]),
                source_note=source_note,
                extra_metadata={
                    "transfer_strategy": "knn",
                    "k": int(k_value),
                    "score_mode": "weighted_support",
                    "score_mode_label": "weighted_support",
                    "blacklist_filter_enabled": True,
                    **audit_by_organism[organism],
                },
            )
            metrics_rows.append(row)
            print(
                f"[knn-k-sweep] {organism} k={k_value}: "
                f"{PRIMARY_METRIC}={row[PRIMARY_METRIC]:.6f} {GUARDRAIL_METRIC}={row[GUARDRAIL_METRIC]:.6f}",
                flush=True,
            )
            del direct_rows, predictions
            gc.collect()
        del indices, scores, annotations, ontology
        gc.collect()

    metrics = pd.DataFrame(metrics_rows).sort_values(["organism", "k"]).reset_index(drop=True)
    metrics_path = output_dir / "clean_plm_knn_k_sweep_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    all_metric_outputs = write_all_metric_analysis(metrics, output_dir, k_values)
    evaluation_summary.to_csv(output_dir / "evaluation_protein_sets.csv", index=False)
    blacklist_audit.to_csv(output_dir / "clean_plm_knn_k_sweep_blacklist_audit.csv", index=False)
    compatibility = compatibility_check(metrics, source_output_dir / "clean_plm_benchmark_metrics.csv")
    if compatibility["status"] != "pass":
        raise RuntimeError("Existing k=3,5,7,10 metrics did not reproduce within tolerance; refusing to publish sweep.")

    by_organism = {
        organism: classify_k_sweep(
            json_records(metrics[metrics["organism"].astype(str) == organism])
        )
        for organism in bench.ORGANISMS
    }
    payload: Dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Clean PLM KNN neighbourhood-size sweep evaluated on the post-blacklist shared benchmark. "
            "The curve continues after a confirmed decline so later behaviour remains visible."
        ),
        "candidate_k_values": k_values,
        "display_organisms": DISPLAY_ORGANISMS,
        "excluded_from_frontend": {
            "223283": "Only three proteins remain in the shared evaluation set; retained in exports but omitted from the website.",
        },
        "primary_metric": PRIMARY_METRIC,
        "guardrail_metric": GUARDRAIL_METRIC,
        "organisms": by_organism,
        "compatibility": compatibility,
    }
    export_payload_path = output_dir / "knn_k_sweep.json"
    export_payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    frontend_payload_path = write_frontend_payload(frontend_data, payload)
    active_checksums_after = {
        str(path.resolve()): bench.file_sha256(path) for path in active_paths if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "candidate_k_values": k_values,
        "metrics_only": True,
        "active_benchmark_artifacts_unchanged": active_checksums_before == active_checksums_after,
        "active_benchmark_checksums": active_checksums_after,
        "inputs": {
            "goa": file_fingerprint(goa_path),
            "target_cache": cache_fingerprint(loaded_target_cache_dir),
            "query_caches": {
                organism: cache_fingerprint(source_output_dir / "embeddings" / f"query_{organism}")
                for organism in bench.ORGANISMS
            },
            "blacklists": {
                organism: file_fingerprint(path, include_sha256=True)
                for organism, path in blacklist_paths.items()
            },
        },
        "outputs": {
            "metrics": file_fingerprint(metrics_path, include_sha256=True),
            **{
                label: file_fingerprint(path, include_sha256=True)
                for label, path in all_metric_outputs.items()
            },
            "export_json": file_fingerprint(export_payload_path, include_sha256=True),
            "frontend_json": file_fingerprint(frontend_payload_path, include_sha256=True),
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "frontend": str(frontend_payload_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
