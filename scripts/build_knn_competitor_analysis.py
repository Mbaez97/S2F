#!/usr/bin/env python3
"""Build reproducible diagnostics for KNN versus PFP benchmark methods.

The exporter intentionally consumes the already materialized, blacklist-aware
prediction-detail files used by the explorer.  It does not reload model
checkpoints or rerun S2F/competitor predictions.  Missing protein--GO pairs are
materialized as zero scores so the resulting matrices follow the same shared
evaluation universe as the benchmark metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from clean_plm_method_registry import ACTIVE_KDE_LABEL, ACTIVE_KDE_MODEL


S2F_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
DEFAULT_OUTPUT_DIR = S2F_ROOT / "notebooks" / "exports" / "knn_competitor_analysis"
DEFAULT_REPORT = S2F_ROOT / "doc" / "knn_vs_advanced_metrics_analysis.md"
DEFAULT_ORGANISMS = ("83333", "1111708")
PRIMARY_METHODS = (
    "S2F",
    "TALE",
    "ATGO",
    "PANDA2",
    "Clean PLM + KNN k=10 (weighted_support)",
)
SENSITIVITY_METHODS = (
    "Clean PLM + KNN k=3 (weighted_support)",
    "Clean PLM + KNN k=5 (weighted_support)",
    "Clean PLM + KNN k=7 (weighted_support)",
    ACTIVE_KDE_MODEL,
)
METHODS = PRIMARY_METHODS + SENSITIVITY_METHODS
ADVANCED_METHODS = ("S2F", "TALE", "ATGO", "PANDA2")
KNN10 = "Clean PLM + KNN k=10 (weighted_support)"
ONTOLOGY_LABELS = {
    "all": "All ontologies",
    "biological_process": "Biological Process",
    "molecular_function": "Molecular Function",
    "cellular_component": "Cellular Component",
}
ONTOLOGY_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
FREQUENCY_BINS = (
    ("singleton", 1, 1),
    ("rare_2_4", 2, 4),
    ("medium_5_9", 5, 9),
    ("common_10_plus", 10, None),
)
METRICS = ("AUC", "AUPR", "F_max", "smin")
SCOPES = ("overall", "per-gene", "per-term")
SCHEMA_VERSION = 2


@dataclass
class MethodMatrix:
    organism: str
    method: str
    proteins: List[str]
    terms: List[str]
    prediction: np.ndarray
    gold: np.ndarray
    information_content: np.ndarray
    term_domains: List[str]
    term_names: List[str]
    detail_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KNN-versus-competitor threshold, score, GO, and neighbor diagnostics.",
    )
    parser.add_argument("--project-root", type=Path, default=S2F_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--organisms", nargs="+", default=list(DEFAULT_ORGANISMS))
    parser.add_argument("--blacklist-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--skip-metric-validation", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative_or_name(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def finite_or_none(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return finite_or_none(value)


def dataframe_records(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [json_ready(dict(row)) for row in rows]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_none(row.get(key)) for key in fields})


def binary_curve(y_true: np.ndarray, y_score: np.ndarray):
    truth = np.asarray(y_true, dtype=np.int8).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()
    if truth.shape != scores.shape:
        raise ValueError("Truth and score arrays must have the same shape.")
    if truth.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, empty

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, sorted_truth.size - 1]
    tp = np.cumsum(sorted_truth, dtype=float)[threshold_indices]
    fp = (1 + threshold_indices).astype(float) - tp
    positives = float(np.sum(sorted_truth > 0))
    negatives = float(sorted_truth.size - positives)
    fn = positives - tp
    tn = negatives - fp
    thresholds = sorted_scores[threshold_indices]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return thresholds, precision, recall, f1, tp, fp, fn


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    truth = np.asarray(y_true, dtype=np.int8).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()
    positives = int(np.sum(truth > 0))
    negatives = int(truth.size - positives)
    unique_scores = np.unique(scores)
    if positives * negatives == 0 or unique_scores.size < 2:
        prevalence = positives / truth.size if truth.size else 0.0
        return {
            "AUC": 0.5,
            "AUPR": prevalence,
            "average_precision": prevalence,
            "F_max": 0.0,
            "best_threshold": float(unique_scores[-1]) if unique_scores.size else 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
            "tp_at_best": 0,
            "fp_at_best": 0,
            "fn_at_best": positives,
            "degenerate": True,
        }

    thresholds, precision, recall, f1, tp, fp, fn = binary_curve(truth, scores)
    max_f1 = float(np.max(f1))
    # Thresholds are descending; the first tied optimum is the most conservative.
    best_idx = int(np.flatnonzero(np.isclose(f1, max_f1, rtol=0.0, atol=1e-12))[0])
    fpr = fp / negatives
    tpr = tp / positives
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    auc = float(integrate(tpr, fpr))
    aupr = float(integrate(precision, recall))
    recall_delta = np.diff(np.r_[0.0, recall])
    average_precision = float(np.sum(recall_delta * precision))
    return {
        "AUC": auc,
        "AUPR": aupr,
        "average_precision": average_precision,
        "F_max": max_f1,
        "best_threshold": float(thresholds[best_idx]),
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx]),
        "tp_at_best": int(tp[best_idx]),
        "fp_at_best": int(fp[best_idx]),
        "fn_at_best": int(fn[best_idx]),
        "degenerate": False,
    }


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_score, dtype=float) >= float(threshold)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_scope_at_threshold(
    gold: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
    scope: str,
) -> Dict[str, float]:
    """Evaluate one deployable threshold using the requested aggregation rule."""
    if scope == "overall":
        return evaluate_at_threshold(gold, prediction, threshold)
    predicted = np.asarray(prediction, dtype=float) >= float(threshold)
    truth = np.asarray(gold, dtype=bool)
    axis = 1 if scope == "per-gene" else 0
    tp = np.sum(predicted & truth, axis=axis)
    fp = np.sum(predicted & ~truth, axis=axis)
    fn = np.sum(~predicted & truth, axis=axis)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    return {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "tp": int(np.sum(tp)),
        "fp": int(np.sum(fp)),
        "fn": int(np.sum(fn)),
    }


def smin_for_vector(prediction: np.ndarray, gold: np.ndarray, ic, denominator: int) -> float:
    scores = np.asarray(prediction, dtype=float)
    truth = np.asarray(gold, dtype=float)
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.0
    ic_values = np.asarray(ic, dtype=float)
    best = math.inf
    for threshold in thresholds:
        false_negative = (scores < threshold) & (truth > 0)
        false_positive = (scores >= threshold) & (truth < 1)
        ru = float(np.sum(false_negative * ic_values)) / max(int(denominator), 1)
        mi = float(np.sum(false_positive * ic_values)) / max(int(denominator), 1)
        best = min(best, math.sqrt(ru * ru + mi * mi))
    return float(best)


def unit_metric_rows(matrix: MethodMatrix, axis: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if axis == 1:
        for index, protein in enumerate(matrix.proteins):
            binary = evaluate_binary(matrix.gold[index, :], matrix.prediction[index, :])
            row = {
                "unit_id": protein,
                "positive_count": int(matrix.gold[index, :].sum()),
                **binary,
                "smin": smin_for_vector(
                    matrix.prediction[index, :],
                    matrix.gold[index, :],
                    matrix.information_content,
                    matrix.gold.shape[1],
                ),
            }
            rows.append(row)
    else:
        for index, term in enumerate(matrix.terms):
            binary = evaluate_binary(matrix.gold[:, index], matrix.prediction[:, index])
            row = {
                "unit_id": term,
                "positive_count": int(matrix.gold[:, index].sum()),
                **binary,
                "smin": smin_for_vector(
                    matrix.prediction[:, index],
                    matrix.gold[:, index],
                    float(matrix.information_content[index]),
                    matrix.gold.shape[0],
                ),
            }
            rows.append(row)
    return rows


def metric_blocks(matrix: MethodMatrix):
    overall_binary = evaluate_binary(matrix.gold, matrix.prediction)
    overall = {
        **overall_binary,
        "smin": smin_for_vector(
            matrix.prediction,
            matrix.gold,
            matrix.information_content.reshape(1, -1),
            matrix.gold.shape[0],
        ),
    }
    per_gene = unit_metric_rows(matrix, axis=1)
    per_term = unit_metric_rows(matrix, axis=0)
    return overall, per_gene, per_term


def aggregate_unit_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows])) if rows else math.nan
        for metric in METRICS
    }


def method_role(method: str) -> str:
    return "primary" if method in PRIMARY_METHODS else "sensitivity"


def read_detail_index(frontend_data: Path) -> Dict[Tuple[str, str], Path]:
    index_path = frontend_data / "pfp_prediction_details_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    result: Dict[Tuple[str, str], Path] = {}
    for item in payload.get("details", []):
        if not item.get("available"):
            continue
        relative = str(item["path"])
        if relative.startswith("data/"):
            relative = relative[len("data/") :]
        result[(str(item["organism"]), str(item["model"]))] = frontend_data / relative
    return result


def load_method_matrices(frontend_data: Path, organisms: Sequence[str]):
    detail_paths = read_detail_index(frontend_data)
    matrices: Dict[Tuple[str, str], MethodMatrix] = {}
    canonical: Dict[str, Dict[str, object]] = {}
    input_files: Set[Path] = {frontend_data / "pfp_prediction_details_index.json"}

    for organism in organisms:
        for method in METHODS:
            path = detail_paths.get((organism, method))
            if path is None or not path.is_file():
                raise RuntimeError(f"Missing prediction details for {organism} / {method}.")
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            input_files.add(path)
            truth_pairs = {
                (str(row["protein_id"]), str(row["term_id"]))
                for row in rows
                if int(row.get("true_label", 0)) == 1
            }
            if not truth_pairs:
                raise RuntimeError(f"No ground-truth rows found in {path}.")
            proteins = sorted({protein for protein, _ in truth_pairs})
            terms = sorted({term for _, term in truth_pairs})
            term_domain = {}
            term_name = {}
            for row in rows:
                term = str(row["term_id"])
                if row.get("go_domain"):
                    term_domain[term] = str(row["go_domain"])
                if row.get("go_name"):
                    term_name[term] = str(row["go_name"])
            signature = (tuple(proteins), tuple(terms), frozenset(truth_pairs))
            if organism not in canonical:
                canonical[organism] = {
                    "signature": signature,
                    "proteins": proteins,
                    "terms": terms,
                    "domains": term_domain,
                    "names": term_name,
                }
            elif canonical[organism]["signature"] != signature:
                raise RuntimeError(f"Ground truth differs between methods for organism {organism}.")

            protein_to_idx = {protein: index for index, protein in enumerate(proteins)}
            term_to_idx = {term: index for index, term in enumerate(terms)}
            prediction = np.zeros((len(proteins), len(terms)), dtype=np.float64)
            gold = np.zeros_like(prediction)
            for protein, term in truth_pairs:
                gold[protein_to_idx[protein], term_to_idx[term]] = 1.0
            for row in rows:
                protein = str(row["protein_id"])
                term = str(row["term_id"])
                if protein not in protein_to_idx or term not in term_to_idx:
                    continue
                score = float(row.get("score", 0.0))
                index = (protein_to_idx[protein], term_to_idx[term])
                prediction[index] = max(prediction[index], score)
            if np.unique(prediction).size > 10000:
                prediction = np.around(prediction, decimals=4)
            term_counts = gold.sum(axis=0)
            total_annotations = float(gold.sum())
            information_content = np.zeros(len(terms), dtype=float)
            positive = term_counts > 0
            information_content[positive] = -np.log2(term_counts[positive] / total_annotations)
            matrices[(organism, method)] = MethodMatrix(
                organism=organism,
                method=method,
                proteins=proteins,
                terms=terms,
                prediction=prediction,
                gold=gold,
                information_content=information_content,
                term_domains=[term_domain.get(term) or "unknown" for term in terms],
                term_names=[term_name.get(term) or term for term in terms],
                detail_path=path,
            )
    return matrices, input_files


def subset_matrix(matrix: MethodMatrix, term_mask: np.ndarray) -> MethodMatrix:
    indices = np.flatnonzero(term_mask)
    return MethodMatrix(
        organism=matrix.organism,
        method=matrix.method,
        proteins=list(matrix.proteins),
        terms=[matrix.terms[index] for index in indices],
        prediction=matrix.prediction[:, indices],
        gold=matrix.gold[:, indices],
        information_content=matrix.information_content[indices],
        term_domains=[matrix.term_domains[index] for index in indices],
        term_names=[matrix.term_names[index] for index in indices],
        detail_path=matrix.detail_path,
    )


def saved_metric_lookup(path: Path) -> Dict[Tuple[str, str, str, str], float]:
    lookup: Dict[Tuple[str, str, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            organism = str(row["organism"])
            method = str(row["model"])
            for scope in SCOPES:
                for metric in METRICS:
                    if scope == "overall":
                        column = f"overall::{metric}"
                    else:
                        column = f"{metric} {scope}"
                    value = row.get(column)
                    if value not in (None, ""):
                        lookup[(organism, method, scope, metric)] = float(value)
    return lookup


def build_metric_outputs(
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    saved_metrics_path: Path,
    validate: bool,
):
    saved = saved_metric_lookup(saved_metrics_path)
    metric_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    units: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    max_difference = 0.0
    legacy_smin_max_difference = 0.0
    for (organism, method), matrix in matrices.items():
        overall, per_gene, per_term = metric_blocks(matrix)
        units[(organism, method, "per-gene")] = per_gene
        units[(organism, method, "per-term")] = per_term
        blocks = {
            "overall": overall,
            "per-gene": aggregate_unit_metrics(per_gene),
            "per-term": aggregate_unit_metrics(per_term),
        }
        for scope, block in blocks.items():
            for metric in METRICS:
                value = float(block[metric])
                saved_value = saved.get((organism, method, scope, metric))
                difference = abs(value - saved_value) if saved_value is not None else math.nan
                legacy_full_organism_ic = metric == "smin" and method in ADVANCED_METHODS
                if math.isfinite(difference):
                    if legacy_full_organism_ic:
                        legacy_smin_max_difference = max(legacy_smin_max_difference, difference)
                    else:
                        max_difference = max(max_difference, difference)
                metric_rows.append(
                    {
                        "organism": organism,
                        "method": method,
                        "method_role": method_role(method),
                        "scope": scope,
                        "metric": metric,
                        "value": saved_value if saved_value is not None else value,
                        "harmonized_value": value,
                        "saved_value": saved_value,
                        "absolute_difference": difference,
                        "metric_validation_included": not legacy_full_organism_ic,
                        "saved_information_content_scope": (
                            "full_organism_before_shared_filter"
                            if legacy_full_organism_ic
                            else "shared_evaluation_set"
                        ),
                        "harmonized_information_content_scope": "shared_evaluation_set",
                        "proteins": matrix.gold.shape[0],
                        "terms": matrix.gold.shape[1],
                        "annotations": int(matrix.gold.sum()),
                    }
                )
            if scope == "overall":
                source_rows = [overall]
            else:
                source_rows = units[(organism, method, scope)]
            thresholds = [float(row["best_threshold"]) for row in source_rows]
            precisions = [float(row["precision_at_best"]) for row in source_rows]
            recalls = [float(row["recall_at_best"]) for row in source_rows]
            fmaxes = [float(row["F_max"]) for row in source_rows]
            threshold_rows.append(
                {
                    "organism": organism,
                    "method": method,
                    "method_role": method_role(method),
                    "scope": scope,
                    "units": len(source_rows),
                    "oracle_fmax": float(np.mean(fmaxes)),
                    "optimal_threshold_mean": float(np.mean(thresholds)),
                    "optimal_threshold_median": float(np.median(thresholds)),
                    "optimal_threshold_q25": float(np.quantile(thresholds, 0.25)),
                    "optimal_threshold_q75": float(np.quantile(thresholds, 0.75)),
                    "precision_at_oracle_mean": float(np.mean(precisions)),
                    "recall_at_oracle_mean": float(np.mean(recalls)),
                    "degenerate_units": int(sum(bool(row.get("degenerate")) for row in source_rows)),
                }
            )
    if validate and max_difference > 2e-6:
        raise RuntimeError(
            f"Reconstructed metrics differ from saved benchmark values by up to {max_difference:.8g}."
        )
    return metric_rows, threshold_rows, units, max_difference, legacy_smin_max_difference


def downsample_curve(points: Sequence[Dict[str, object]], max_points: int = 401):
    if len(points) <= max_points:
        return list(points)
    f1_values = np.array([float(point["f1"]) for point in points])
    best = int(np.argmax(f1_values))
    indices = set(np.linspace(0, len(points) - 1, max_points - 1, dtype=int).tolist())
    indices.add(best)
    return [points[index] for index in sorted(indices)]


def shared_threshold_curve(matrix: MethodMatrix, scope: str, local_thresholds: Sequence[float]):
    if scope == "overall":
        thresholds, precision, recall, f1, tp, fp, fn = binary_curve(matrix.gold, matrix.prediction)
        points = [
            {
                "threshold": float(thresholds[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "tp": int(tp[index]),
                "fp": int(fp[index]),
                "fn": int(fn[index]),
            }
            for index in range(len(thresholds))
        ]
        return downsample_curve(points)

    thresholds = np.unique(np.r_[np.linspace(0.0, 1.0, 201), np.asarray(local_thresholds, dtype=float)])[::-1]
    axis = 0 if scope == "per-gene" else 1
    units = matrix.gold.shape[axis]
    points = []
    for threshold in thresholds:
        predicted = matrix.prediction >= threshold
        truth = matrix.gold > 0
        if scope == "per-gene":
            tp = np.sum(predicted & truth, axis=1)
            fp = np.sum(predicted & ~truth, axis=1)
            fn = np.sum(~predicted & truth, axis=1)
        else:
            tp = np.sum(predicted & truth, axis=0)
            fp = np.sum(predicted & ~truth, axis=0)
            fn = np.sum(~predicted & truth, axis=0)
        precision = np.divide(tp, tp + fp, out=np.zeros(units, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(units, dtype=float), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(units, dtype=float), where=(precision + recall) > 0)
        points.append(
            {
                "threshold": float(threshold),
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f1": float(np.mean(f1)),
                "tp": int(np.sum(tp)),
                "fp": int(np.sum(fp)),
                "fn": int(np.sum(fn)),
            }
        )
    return points


def build_threshold_curves(
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    units: Mapping[Tuple[str, str, str], Sequence[Mapping[str, object]]],
):
    curves = []
    shared_rows = []
    for (organism, method), matrix in matrices.items():
        for scope in SCOPES:
            local = [] if scope == "overall" else [
                float(row["best_threshold"]) for row in units[(organism, method, scope)]
            ]
            points = shared_threshold_curve(matrix, scope, local)
            best = max(points, key=lambda row: (float(row["f1"]), float(row["threshold"])))
            curves.append(
                {
                    "organism": organism,
                    "method": method,
                    "scope": scope,
                    "points": points,
                }
            )
            shared_rows.append(
                {
                    "organism": organism,
                    "method": method,
                    "method_role": method_role(method),
                    "scope": scope,
                    "shared_threshold": float(best["threshold"]),
                    "shared_threshold_precision": float(best["precision"]),
                    "shared_threshold_recall": float(best["recall"]),
                    "shared_threshold_f1": float(best["f1"]),
                    "shared_threshold_tp": int(best["tp"]),
                    "shared_threshold_fp": int(best["fp"]),
                    "shared_threshold_fn": int(best["fn"]),
                }
            )
    return curves, shared_rows


def build_score_distributions(matrices: Mapping[Tuple[str, str], MethodMatrix]):
    bins = np.linspace(0.0, 1.0, 51)
    summary_rows = []
    histograms = []
    for (organism, method), matrix in matrices.items():
        scores = matrix.prediction.ravel()
        truth = matrix.gold.ravel() > 0
        nonzero = scores[scores > 0]
        rounded = np.round(nonzero, 8)
        repeats = Counter(rounded.tolist())
        summary_rows.append(
            {
                "organism": organism,
                "method": method,
                "method_role": method_role(method),
                "total_pairs": int(scores.size),
                "positive_pairs": int(np.sum(truth)),
                "nonzero_predictions": int(np.sum(scores > 0)),
                "zero_fraction": float(np.mean(scores == 0)),
                "unique_nonzero_scores": int(np.unique(rounded).size),
                "most_repeated_score": float(repeats.most_common(1)[0][0]) if repeats else math.nan,
                "most_repeated_score_count": int(repeats.most_common(1)[0][1]) if repeats else 0,
                "score_q10_nonzero": float(np.quantile(nonzero, 0.1)) if nonzero.size else math.nan,
                "score_median_nonzero": float(np.median(nonzero)) if nonzero.size else math.nan,
                "score_q90_nonzero": float(np.quantile(nonzero, 0.9)) if nonzero.size else math.nan,
                "mean_predictions_per_protein": float(np.sum(matrix.prediction > 0, axis=1).mean()),
            }
        )
        for label, mask in (("true", truth), ("false", ~truth)):
            counts, edges = np.histogram(scores[mask], bins=bins)
            histograms.append(
                {
                    "organism": organism,
                    "method": method,
                    "label": label,
                    "bins": [
                        {
                            "lower": float(edges[index]),
                            "upper": float(edges[index + 1]),
                            "count": int(counts[index]),
                        }
                        for index in range(len(counts))
                    ],
                }
            )
    return summary_rows, histograms


def build_subgroup_metrics(matrices: Mapping[Tuple[str, str], MethodMatrix]):
    ontology_rows = []
    frequency_rows = []
    for (_organism, _method), matrix in matrices.items():
        terms = np.array(matrix.terms, dtype=object)
        domains = np.array(matrix.term_domains, dtype=object)
        counts = matrix.gold.sum(axis=0)
        for ontology in ONTOLOGY_LABELS:
            base_mask = np.ones(len(terms), dtype=bool) if ontology == "all" else domains == ontology
            for roots in ("included", "excluded"):
                mask = base_mask.copy()
                if roots == "excluded":
                    mask &= ~np.isin(terms, list(ONTOLOGY_ROOTS))
                if not np.any(mask):
                    continue
                subset = subset_matrix(matrix, mask)
                overall, per_gene, per_term = metric_blocks(subset)
                blocks = {
                    "overall": overall,
                    "per-gene": aggregate_unit_metrics(per_gene),
                    "per-term": aggregate_unit_metrics(per_term),
                }
                for scope, block in blocks.items():
                    for metric in METRICS:
                        ontology_rows.append(
                            {
                                "organism": matrix.organism,
                                "method": matrix.method,
                                "method_role": method_role(matrix.method),
                                "ontology": ontology,
                                "ontology_label": ONTOLOGY_LABELS[ontology],
                                "ontology_roots": roots,
                                "scope": scope,
                                "metric": metric,
                                "value": float(block[metric]),
                                "proteins": subset.gold.shape[0],
                                "terms": subset.gold.shape[1],
                                "annotations": int(subset.gold.sum()),
                            }
                        )
        for bin_key, lower, upper in FREQUENCY_BINS:
            mask = counts >= lower
            if upper is not None:
                mask &= counts <= upper
            if not np.any(mask):
                continue
            subset = subset_matrix(matrix, mask)
            overall, per_gene, per_term = metric_blocks(subset)
            blocks = {
                "overall": overall,
                "per-gene": aggregate_unit_metrics(per_gene),
                "per-term": aggregate_unit_metrics(per_term),
            }
            for scope, block in blocks.items():
                for metric in METRICS:
                    frequency_rows.append(
                        {
                            "organism": matrix.organism,
                            "method": matrix.method,
                            "method_role": method_role(matrix.method),
                            "frequency_bin": bin_key,
                            "frequency_min": lower,
                            "frequency_max": upper,
                            "scope": scope,
                            "metric": metric,
                            "value": float(block[metric]),
                            "proteins": subset.gold.shape[0],
                            "terms": subset.gold.shape[1],
                            "annotations": int(subset.gold.sum()),
                            "degenerate_terms": int(sum(bool(row["degenerate"]) for row in per_term)),
                        }
                    )
    return ontology_rows, frequency_rows


def paired_bootstrap(
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    units: Mapping[Tuple[str, str, str], Sequence[Mapping[str, object]]],
    replicates: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    rows = []
    delta_rows = []
    for organism in sorted({key[0] for key in matrices}):
        protein_count = len(matrices[(organism, KNN10)].proteins)
        indices = rng.integers(0, protein_count, size=(replicates, protein_count))
        samples = {}
        for method in METHODS:
            method_rows = units[(organism, method, "per-gene")]
            for metric in METRICS:
                values = np.array([float(row[metric]) for row in method_rows], dtype=float)
                boot = values[indices].mean(axis=1)
                samples[(method, metric)] = boot
                rows.append(
                    {
                        "organism": organism,
                        "method": method,
                        "method_role": method_role(method),
                        "scope": "per-gene",
                        "metric": metric,
                        "replicates": replicates,
                        "point_estimate": float(values.mean()),
                        "ci95_lower": float(np.quantile(boot, 0.025)),
                        "ci95_upper": float(np.quantile(boot, 0.975)),
                    }
                )
        for comparator in ADVANCED_METHODS:
            for metric in METRICS:
                delta = samples[(KNN10, metric)] - samples[(comparator, metric)]
                point = float(
                    np.mean([float(row[metric]) for row in units[(organism, KNN10, "per-gene")]])
                    - np.mean([float(row[metric]) for row in units[(organism, comparator, "per-gene")]])
                )
                delta_rows.append(
                    {
                        "organism": organism,
                        "method": KNN10,
                        "comparator": comparator,
                        "scope": "per-gene",
                        "metric": metric,
                        "replicates": replicates,
                        "delta": point,
                        "ci95_lower": float(np.quantile(delta, 0.025)),
                        "ci95_upper": float(np.quantile(delta, 0.975)),
                        "probability_delta_positive": float(np.mean(delta > 0)),
                    }
                )
    return rows, delta_rows


def stratified_folds(gold: np.ndarray, folds: int, rng: np.random.Generator) -> np.ndarray:
    counts = gold.sum(axis=1)
    quantiles = np.unique(np.quantile(counts, [0.2, 0.4, 0.6, 0.8]))
    strata = np.digitize(counts, quantiles, right=True)
    assignments = np.full(gold.shape[0], -1, dtype=int)
    for stratum in np.unique(strata):
        selected = np.flatnonzero(strata == stratum)
        rng.shuffle(selected)
        for offset, index in enumerate(selected):
            assignments[index] = offset % folds
    # Rotate sparse strata so every fold receives a balanced total size.
    if np.any(assignments < 0):
        raise RuntimeError("Could not assign all proteins to cross-validation folds.")
    return assignments


def repeated_threshold_cv(
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    repeats: int,
    seed: int,
):
    fold_rows = []
    summary_rows = []
    for organism in sorted({key[0] for key in matrices}):
        gold = matrices[(organism, KNN10)].gold
        folds = min(5, gold.shape[0])
        prepared = {}
        for method in PRIMARY_METHODS:
            matrix = matrices[(organism, method)]
            local = np.array(
                [float(row["best_threshold"]) for row in unit_metric_rows(matrix, axis=1)],
                dtype=float,
            )
            thresholds = np.unique(np.r_[np.linspace(0.0, 1.0, 201), local])[::-1]
            precision = np.zeros((matrix.gold.shape[0], len(thresholds)), dtype=float)
            recall = np.zeros_like(precision)
            f1 = np.zeros_like(precision)
            tp_total = np.zeros_like(precision, dtype=np.int32)
            fp_total = np.zeros_like(precision, dtype=np.int32)
            fn_total = np.zeros_like(precision, dtype=np.int32)
            truth = matrix.gold > 0
            for threshold_index, threshold in enumerate(thresholds):
                predicted = matrix.prediction >= threshold
                tp = np.sum(predicted & truth, axis=1)
                fp = np.sum(predicted & ~truth, axis=1)
                fn = np.sum(~predicted & truth, axis=1)
                precision[:, threshold_index] = np.divide(
                    tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
                )
                recall[:, threshold_index] = np.divide(
                    tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
                )
                f1[:, threshold_index] = np.divide(
                    2.0 * precision[:, threshold_index] * recall[:, threshold_index],
                    precision[:, threshold_index] + recall[:, threshold_index],
                    out=np.zeros(matrix.gold.shape[0], dtype=float),
                    where=(precision[:, threshold_index] + recall[:, threshold_index]) > 0,
                )
                tp_total[:, threshold_index] = tp
                fp_total[:, threshold_index] = fp
                fn_total[:, threshold_index] = fn
            prepared[method] = {
                "thresholds": thresholds,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
            }
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat + int(organism))
            assignments = stratified_folds(gold, folds, rng)
            for method in PRIMARY_METHODS:
                method_grid = prepared[method]
                for fold in range(folds):
                    test_mask = assignments == fold
                    train_mask = ~test_mask
                    train_f1 = method_grid["f1"][train_mask].mean(axis=0)
                    # Thresholds are descending, so argmax resolves ties to the
                    # most conservative threshold.
                    threshold_index = int(np.argmax(train_f1))
                    threshold = float(method_grid["thresholds"][threshold_index])
                    heldout = {
                        key: float(method_grid[key][test_mask, threshold_index].mean())
                        for key in ("precision", "recall", "f1")
                    }
                    for key in ("tp", "fp", "fn"):
                        heldout[key] = int(method_grid[key][test_mask, threshold_index].sum())
                    fold_rows.append(
                        {
                            "organism": organism,
                            "method": method,
                            "repeat": repeat,
                            "fold": fold,
                            "train_proteins": int(train_mask.sum()),
                            "heldout_proteins": int(test_mask.sum()),
                            "selected_threshold": threshold,
                            "heldout_precision": heldout["precision"],
                            "heldout_recall": heldout["recall"],
                            "heldout_f1": heldout["f1"],
                            "heldout_tp": heldout["tp"],
                            "heldout_fp": heldout["fp"],
                            "heldout_fn": heldout["fn"],
                        }
                    )
        for method in PRIMARY_METHODS:
            selected = [row for row in fold_rows if row["organism"] == organism and row["method"] == method]
            summary_rows.append(
                {
                    "organism": organism,
                    "method": method,
                    "repeats": repeats,
                    "folds": folds,
                    "threshold_mean": float(np.mean([row["selected_threshold"] for row in selected])),
                    "threshold_std": float(np.std([row["selected_threshold"] for row in selected])),
                    "heldout_precision_mean": float(np.mean([row["heldout_precision"] for row in selected])),
                    "heldout_recall_mean": float(np.mean([row["heldout_recall"] for row in selected])),
                    "heldout_f1_mean": float(np.mean([row["heldout_f1"] for row in selected])),
                    "heldout_f1_std": float(np.std([row["heldout_f1"] for row in selected])),
                }
            )
    return fold_rows, summary_rows


def resolve_blacklist_path(
    organism: str,
    explicit_dir: Optional[Path],
    neighbor_payload: Mapping[str, object],
) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_dir is not None:
        candidates.append(explicit_dir / f"{organism}.blacklist")
    env_dir = os.environ.get("PFP_BLACKLIST_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / f"{organism}.blacklist")
    candidates.append(S2F_ROOT / "data" / "blacklists" / f"{organism}.blacklist")
    for item in neighbor_payload.get("metadata", {}).get("blacklist_audit", []):
        if str(item.get("organism")) == organism and item.get("blacklist_path"):
            candidates.append(Path(str(item["blacklist_path"])).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_blacklist(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    result = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            result.add(value)
    return result


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result

    x_rank = ranks(x)
    y_rank = ranks(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def build_neighbor_outputs(
    frontend_data: Path,
    organisms: Sequence[str],
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    units: Mapping[Tuple[str, str, str], Sequence[Mapping[str, object]]],
    blacklist_dir: Optional[Path],
):
    path = frontend_data / "plm_neighbor_embeddings.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = {str(item["protein_id"]): item for item in payload.get("points", [])}
    neighborhoods = payload.get("neighborhoods", {})
    benchmark_ids = {
        str(item["protein_id"])
        for item in payload.get("points", [])
        if item.get("dataset_source") == "benchmark_test"
    }
    audit_rows = []
    donor_rows = []
    profile_rows = []
    taxon_rows = []
    correlation_rows = []
    blacklist_paths: Set[Path] = set()
    neighbor_lookup: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for organism in organisms:
        blacklist_path = resolve_blacklist_path(organism, blacklist_dir, payload)
        if blacklist_path is not None:
            blacklist_paths.add(blacklist_path)
        blacklist = read_blacklist(blacklist_path)
        selected_queries = [
            (protein, item)
            for protein, item in neighborhoods.items()
            if str(item.get("test_organism_taxon")) == organism
        ]
        donor_reuse = Counter()
        organism_reuse = Counter()
        all_method_donors: Set[str] = set()
        violations = []
        benchmark_overlap = []
        for query_id, item in selected_queries:
            for method_neighbors in item.get("methods", {}).values():
                for neighbor in method_neighbors:
                    donor_id = str(neighbor["protein_id"])
                    all_method_donors.add(donor_id)
                    donor = points.get(donor_id, {})
                    taxon = str(donor.get("organism_taxon") or "")
                    if blacklist and taxon in blacklist:
                        violations.append((query_id, donor_id, taxon))
                    if donor_id in benchmark_ids:
                        benchmark_overlap.append((query_id, donor_id))
        for query_id, item in selected_queries:
            neighbors = item.get("methods", {}).get("knn_k_10", [])
            detailed = []
            for rank, neighbor in enumerate(neighbors, start=1):
                donor_id = str(neighbor["protein_id"])
                donor = points.get(donor_id, {})
                go_terms = donor.get("go_terms", [])
                donor_reuse[donor_id] += 1
                organism_name = str(donor.get("organism") or "Unknown")
                organism_reuse[organism_name] += 1
                record = {
                    "organism": organism,
                    "query_protein": query_id,
                    "rank": rank,
                    "donor_protein": donor_id,
                    "cosine_similarity": float(neighbor["cosine_similarity"]),
                    "cosine_distance": float(neighbor["cosine_distance"]),
                    "donor_organism": organism_name,
                    "donor_taxon": str(donor.get("organism_taxon") or ""),
                    "direct_go_term_count": len(go_terms),
                    "direct_go_terms": "; ".join(str(term.get("id")) for term in go_terms),
                    "blacklisted_taxon": bool(blacklist and str(donor.get("organism_taxon") or "") in blacklist),
                }
                donor_rows.append(record)
                detailed.append({**record, "go_terms": go_terms})
            neighbor_lookup[(organism, query_id)] = detailed

        knn_units = {row["unit_id"]: row for row in units[(organism, KNN10, "per-gene")]}
        advanced_units = {
            method: {row["unit_id"]: row for row in units[(organism, method, "per-gene")]}
            for method in ADVANCED_METHODS
        }
        for query_id, item in selected_queries:
            neighbors = neighbor_lookup[(organism, query_id)]
            if not neighbors:
                continue
            direct_counts = Counter()
            for neighbor in neighbors:
                direct_counts.update(term.get("id") for term in neighbor["go_terms"])
            best_advanced_fmax = max(float(advanced_units[method][query_id]["F_max"]) for method in ADVANCED_METHODS)
            best_advanced_aupr = max(float(advanced_units[method][query_id]["AUPR"]) for method in ADVANCED_METHODS)
            profile_rows.append(
                {
                    "organism": organism,
                    "protein_id": query_id,
                    "nearest_similarity": float(neighbors[0]["cosine_similarity"]),
                    "tenth_similarity": float(neighbors[-1]["cosine_similarity"]),
                    "mean_similarity": float(np.mean([row["cosine_similarity"] for row in neighbors])),
                    "unique_donor_organisms": len({row["donor_organism"] for row in neighbors}),
                    "mean_direct_go_terms_per_donor": float(np.mean([row["direct_go_term_count"] for row in neighbors])),
                    "direct_go_union_size": len(direct_counts),
                    "direct_consensus_terms_5_plus": int(sum(count >= 5 for count in direct_counts.values())),
                    "knn_fmax": float(knn_units[query_id]["F_max"]),
                    "knn_aupr": float(knn_units[query_id]["AUPR"]),
                    "knn_auc": float(knn_units[query_id]["AUC"]),
                    "knn_smin": float(knn_units[query_id]["smin"]),
                    "knn_minus_best_advanced_fmax": float(knn_units[query_id]["F_max"]) - best_advanced_fmax,
                    "knn_minus_best_advanced_aupr": float(knn_units[query_id]["AUPR"]) - best_advanced_aupr,
                }
            )
        organism_profiles = [row for row in profile_rows if row["organism"] == organism]
        similarities = np.array([row["nearest_similarity"] for row in organism_profiles])
        quartiles = np.quantile(similarities, [0.25, 0.5, 0.75])
        for row in organism_profiles:
            row["nearest_similarity_quartile"] = int(np.digitize(row["nearest_similarity"], quartiles, right=True) + 1)
        for feature in (
            "nearest_similarity",
            "tenth_similarity",
            "mean_similarity",
            "unique_donor_organisms",
            "mean_direct_go_terms_per_donor",
            "direct_consensus_terms_5_plus",
        ):
            for outcome in ("knn_minus_best_advanced_fmax", "knn_minus_best_advanced_aupr"):
                correlation_rows.append(
                    {
                        "organism": organism,
                        "feature": feature,
                        "outcome": outcome,
                        "spearman_rho": spearman(
                            [float(row[feature]) for row in organism_profiles],
                            [float(row[outcome]) for row in organism_profiles],
                        ),
                        "proteins": len(organism_profiles),
                    }
                )
        for donor_id, count in donor_reuse.items():
            donor = points.get(donor_id, {})
            taxon_rows.append(
                {
                    "organism": organism,
                    "donor_protein": donor_id,
                    "donor_organism": donor.get("organism"),
                    "donor_taxon": donor.get("organism_taxon"),
                    "query_reuse_count": count,
                    "blacklisted_taxon": bool(blacklist and str(donor.get("organism_taxon") or "") in blacklist),
                }
            )
        audit_rows.append(
            {
                "organism": organism,
                "blacklist_filter_enabled": bool(payload.get("metadata", {}).get("blacklist_filter_enabled")),
                "blacklist_file_available": blacklist_path is not None,
                "blacklist_file": blacklist_path.name if blacklist_path else None,
                "blacklist_sha256": sha256_file(blacklist_path) if blacklist_path else None,
                "blacklist_taxa_count": len(blacklist),
                "test_queries": len(selected_queries),
                "unique_knn10_donors": len(donor_reuse),
                "unique_knn10_donor_organisms": len(organism_reuse),
                "all_method_selected_donors": len(all_method_donors),
                "selected_blacklist_violations": len(violations),
                "selected_benchmark_accession_violations": len(benchmark_overlap),
                "median_top1_similarity": float(np.median([row["nearest_similarity"] for row in organism_profiles])),
                "median_tenth_similarity": float(np.median([row["tenth_similarity"] for row in organism_profiles])),
            }
        )
    return {
        "source_path": path,
        "audit_rows": audit_rows,
        "donor_rows": donor_rows,
        "profile_rows": profile_rows,
        "taxon_rows": taxon_rows,
        "correlation_rows": correlation_rows,
        "neighbor_lookup": neighbor_lookup,
        "blacklist_paths": blacklist_paths,
    }


def compact_outcomes(matrix: MethodMatrix, protein_index: int, threshold: float, limit: int = 8):
    scores = matrix.prediction[protein_index]
    truth = matrix.gold[protein_index] > 0
    predicted = scores >= threshold
    outcome_masks = {
        "true_positive": predicted & truth,
        "false_positive": predicted & ~truth,
        "false_negative": ~predicted & truth,
    }
    result = {}
    for outcome, mask in outcome_masks.items():
        indices = np.flatnonzero(mask)
        order = indices[np.argsort(scores[indices])[::-1]] if indices.size else indices
        result[outcome] = [
            {
                "term_id": matrix.terms[index],
                "term_name": matrix.term_names[index],
                "ontology": matrix.term_domains[index],
                "score": float(scores[index]),
            }
            for index in order[:limit]
        ]
        result[f"{outcome}_count"] = int(indices.size)
    return result


def build_examples(
    matrices: Mapping[Tuple[str, str], MethodMatrix],
    units: Mapping[Tuple[str, str, str], Sequence[Mapping[str, object]]],
    neighbor_lookup: Mapping[Tuple[str, str], Sequence[Mapping[str, object]]],
    count: int = 5,
):
    examples = []
    for organism in sorted({key[0] for key in matrices}):
        proteins = matrices[(organism, KNN10)].proteins
        unit_lookup = {
            method: {str(row["unit_id"]): row for row in units[(organism, method, "per-gene")]}
            for method in PRIMARY_METHODS
        }
        comparisons = []
        for protein in proteins:
            knn = unit_lookup[KNN10][protein]
            advanced = sorted(
                ADVANCED_METHODS,
                key=lambda method: (
                    float(unit_lookup[method][protein]["F_max"]),
                    float(unit_lookup[method][protein]["AUPR"]),
                    method,
                ),
                reverse=True,
            )
            winner = advanced[0]
            comparisons.append(
                {
                    "protein_id": protein,
                    "best_advanced_method": winner,
                    "knn_fmax": float(knn["F_max"]),
                    "advanced_fmax": float(unit_lookup[winner][protein]["F_max"]),
                    "fmax_delta": float(knn["F_max"]) - float(unit_lookup[winner][protein]["F_max"]),
                    "knn_aupr": float(knn["AUPR"]),
                    "advanced_aupr": float(unit_lookup[winner][protein]["AUPR"]),
                    "aupr_delta": float(knn["AUPR"]) - float(unit_lookup[winner][protein]["AUPR"]),
                }
            )
        directions = {
            "knn_better": sorted(comparisons, key=lambda row: (row["fmax_delta"], row["aupr_delta"], row["protein_id"]), reverse=True)[:count],
            "advanced_better": sorted(comparisons, key=lambda row: (row["fmax_delta"], row["aupr_delta"], row["protein_id"]))[:count],
        }
        for direction, selected in directions.items():
            for rank, comparison in enumerate(selected, start=1):
                protein = comparison["protein_id"]
                competitor = comparison["best_advanced_method"]
                knn_matrix = matrices[(organism, KNN10)]
                competitor_matrix = matrices[(organism, competitor)]
                protein_index = knn_matrix.proteins.index(protein)
                knn_unit = unit_lookup[KNN10][protein]
                competitor_unit = unit_lookup[competitor][protein]
                examples.append(
                    {
                        "organism": organism,
                        "direction": direction,
                        "rank": rank,
                        **comparison,
                        "knn_threshold": float(knn_unit["best_threshold"]),
                        "advanced_threshold": float(competitor_unit["best_threshold"]),
                        "ground_truth_terms": int(knn_matrix.gold[protein_index].sum()),
                        "knn_outcomes": compact_outcomes(knn_matrix, protein_index, float(knn_unit["best_threshold"])),
                        "advanced_outcomes": compact_outcomes(competitor_matrix, protein_index, float(competitor_unit["best_threshold"])),
                        "knn_neighbors": [
                            {
                                key: value
                                for key, value in neighbor.items()
                                if key not in {"go_terms"}
                            }
                            for neighbor in neighbor_lookup.get((organism, protein), [])
                        ],
                    }
                )
    return examples


def metric_value(metric_rows, organism: str, method: str, scope: str, metric: str) -> float:
    for row in metric_rows:
        if (
            row["organism"] == organism
            and row["method"] == method
            and row["scope"] == scope
            and row["metric"] == metric
        ):
            return float(row["value"])
    raise KeyError((organism, method, scope, metric))


def markdown_metric_table(metric_rows, organism: str, scope: str) -> List[str]:
    lines = [
        f"### {organism} — {scope}",
        "",
        "| Method | Fmax | AUPR | AUROC | Smin |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in PRIMARY_METHODS:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                method.replace("Clean PLM + ", ""),
                metric_value(metric_rows, organism, method, scope, "F_max"),
                metric_value(metric_rows, organism, method, scope, "AUPR"),
                metric_value(metric_rows, organism, method, scope, "AUC"),
                metric_value(metric_rows, organism, method, scope, "smin"),
            )
        )
    lines.append("")
    return lines


def harmonized_metric_value(metric_rows, organism: str, method: str, scope: str, metric: str) -> float:
    for row in metric_rows:
        if (
            row["organism"] == organism
            and row["method"] == method
            and row["scope"] == scope
            and row["metric"] == metric
        ):
            return float(row["harmonized_value"])
    raise KeyError((organism, method, scope, metric))


def write_report(path: Path, payload: Mapping[str, object]) -> None:
    metric_rows = payload["metric_rows"]
    score_rows = payload["score_summary"]
    threshold_rows = payload["threshold_summary"]
    shared_rows = payload["shared_threshold_summary"]
    cv_rows = payload["cross_validation_summary"]
    neighbor_audit = payload["neighbor_audit"]
    bootstrap_delta_rows = payload["bootstrap_delta_rows"]
    ontology_rows = payload["ontology_rows"]
    frequency_rows = payload["frequency_rows"]
    correlation_rows = payload["neighbor_correlations"]
    examples = payload["examples"]
    lines = [
        "# Why can KNN obtain a strong Fmax?",
        "",
        "> This report uses only the active post-blacklist benchmark artifacts. Archived pre-blacklist results are intentionally excluded.",
        "",
        "## Evaluation boundary",
        "",
        "All methods use the same propagated GO ground truth and shared evaluation proteins. The reconstructed matrices contain 160 proteins, 574 GO terms, and 1,922 positive pairs for `83333`; and 33 proteins, 93 GO terms, and 306 positive pairs for `1111708`. Smin is lower-is-better; Fmax, AUPR, and AUROC are higher-is-better.",
        "",
        "The current code defines per-gene Fmax as the mean of a separately maximized F1 score for each protein. Therefore every protein may use a different oracle threshold. This is more permissive than selecting one threshold and deploying it for every protein.",
        "",
        "**Smin comparability warning.** The saved S2F, TALE, ATGO, and PANDA2 rows calculate GO information content on the full organism before applying the shared evaluation-protein filter. The Clean PLM rows calculate it after that filter. Their published Smin values are therefore not strictly comparable. This report preserves them in the main reproduction tables and supplies a harmonized evaluation-set-IC Smin in the generated diagnostics.",
        "",
        "## Metric comparison",
        "",
    ]
    for organism in DEFAULT_ORGANISMS:
        for scope in SCOPES:
            lines.extend(markdown_metric_table(metric_rows, organism, scope))

    lines.extend(
        [
            "## Why the rankings differ",
            "",
            "- **Fmax** keeps only the best precision/recall operating point. A method can score well if one threshold isolates a useful subset, even if scores below and above that point are poorly ordered.",
            "- **AUPR** evaluates the full precision–recall ranking. It is sensitive to false positives appearing ahead of true annotations and is informative for the highly imbalanced protein–GO matrix. Davis and Goadrich give the classic explanation of why PR views are useful for skewed data ([ICML 2006](https://doi.org/10.1145/1143844.1143874)).",
            "- **AUROC** measures positive-versus-negative ranking over all pairs. Because negatives vastly outnumber positives, it can remain high while precision is modest.",
            "- **Smin** weights remaining uncertainty and misinformation by GO-term information content. Incorrect or missed specific terms can hurt Smin more than errors on common terms.",
            "",
            "KNN-10 transfers a sparse, quantized set of scores based on how many of ten almost equally weighted donors support a term. That score geometry can create a good Fmax cutoff while producing ties and false-positive blocks that reduce AUPR. S2F is much denser, so its score ordering can support a stronger full PR curve even when its best single F1 point is lower. CAFA likewise notes that metric choice can affect method rankings and separates protein- from term-centric questions ([Radivojac et al., 2013](https://www.nature.com/articles/nmeth.2340)).",
            "",
            "## Harmonized Smin sensitivity",
            "",
            "The values below recompute Smin for every method using information content from the same shared evaluation set. They should be used for the fair Smin sensitivity analysis; they do not overwrite the historical benchmark table.",
            "",
            "| Organism | Scope | Method | Saved Smin | Harmonized Smin |",
            "|---|---|---|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for scope in SCOPES:
            for method in PRIMARY_METHODS:
                lines.append(
                    "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                        organism,
                        scope,
                        method.replace("Clean PLM + ", ""),
                        metric_value(metric_rows, organism, method, scope, "smin"),
                        harmonized_metric_value(metric_rows, organism, method, scope, "smin"),
                    )
                )
    lines.extend(
        [
            "",
            "## Threshold diagnostics",
            "",
            "The tables below contrast the oracle per-gene result with a shared threshold and with thresholds learned on training folds. The cross-validation values are diagnostics, not replacement benchmark metrics.",
            "",
            "| Organism | Method | Per-gene oracle Fmax | Median local threshold | Best shared-threshold F1 | Held-out threshold F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for method in PRIMARY_METHODS:
            oracle = next(row for row in threshold_rows if row["organism"] == organism and row["method"] == method and row["scope"] == "per-gene")
            shared = next(row for row in shared_rows if row["organism"] == organism and row["method"] == method and row["scope"] == "per-gene")
            cv = next(row for row in cv_rows if row["organism"] == organism and row["method"] == method)
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    organism,
                    method.replace("Clean PLM + ", ""),
                    float(oracle["oracle_fmax"]),
                    float(oracle["optimal_threshold_median"]),
                    float(shared["shared_threshold_f1"]),
                    float(cv["heldout_f1_mean"]),
                )
            )
    lines.extend(
        [
            "",
            "## Paired-bootstrap stability of KNN versus S2F",
            "",
            "The intervals resample the same proteins for both methods. Positive deltas favor KNN for Fmax, AUPR, and AUROC; positive Smin deltas favor S2F because lower is better. Smin uses harmonized shared-set information content.",
            "",
            "| Organism | Metric | KNN − S2F | 95% paired interval | P(delta > 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for metric in METRICS:
            row = next(
                row for row in bootstrap_delta_rows
                if row["organism"] == organism
                and row["comparator"] == "S2F"
                and row["metric"] == metric
            )
            lines.append(
                "| {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.3f} |".format(
                    organism,
                    metric,
                    float(row["delta"]),
                    float(row["ci95_lower"]),
                    float(row["ci95_upper"]),
                    float(row["probability_delta_positive"]),
                )
            )
    lines.extend(
        [
            "",
            "## Score-distribution evidence",
            "",
            "| Organism | Method | Nonzero pairs | Zero fraction | Unique nonzero scores | Median nonzero score |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in score_rows:
        if row["method"] not in PRIMARY_METHODS:
            continue
        lines.append(
            "| {} | {} | {:,} | {:.3f} | {:,} | {:.4f} |".format(
                row["organism"],
                str(row["method"]).replace("Clean PLM + ", ""),
                int(row["nonzero_predictions"]),
                float(row["zero_fraction"]),
                int(row["unique_nonzero_scores"]),
                float(row["score_median_nonzero"]),
            )
        )
    lines.extend(
        [
            "",
            "## Ontology and term-frequency diagnostics",
            "",
            "The ontology table removes GO roots. The frequency table is term-centric and bins terms by positive-protein count.",
            "",
            "| Organism | Ontology | Method | Per-gene Fmax | Per-gene AUPR |",
            "|---|---|---|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for ontology in ("biological_process", "molecular_function", "cellular_component"):
            for method in ("S2F", KNN10):
                values = {
                    metric: next(
                        float(row["value"])
                        for row in ontology_rows
                        if row["organism"] == organism
                        and row["ontology"] == ontology
                        and row["ontology_roots"] == "excluded"
                        and row["scope"] == "per-gene"
                        and row["method"] == method
                        and row["metric"] == metric
                    )
                    for metric in ("F_max", "AUPR")
                }
                lines.append(
                    "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                        organism,
                        ONTOLOGY_LABELS[ontology],
                        method.replace("Clean PLM + ", ""),
                        values["F_max"],
                        values["AUPR"],
                    )
                )
    lines.extend(
        [
            "",
            "| Organism | Term-frequency bin | Method | Per-term Fmax | Per-term AUPR |",
            "|---|---|---|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for frequency_bin, _lower, _upper in FREQUENCY_BINS:
            for method in ("S2F", KNN10):
                values = {
                    metric: next(
                        float(row["value"])
                        for row in frequency_rows
                        if row["organism"] == organism
                        and row["frequency_bin"] == frequency_bin
                        and row["scope"] == "per-term"
                        and row["method"] == method
                        and row["metric"] == metric
                    )
                    for metric in ("F_max", "AUPR")
                }
                lines.append(
                    "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                        organism,
                        frequency_bin,
                        method.replace("Clean PLM + ", ""),
                        values["F_max"],
                        values["AUPR"],
                    )
                )
    lines.extend(
        [
            "",
            "For `83333`, S2F is stronger in every root-excluded ontology and every term-frequency bin. For `1111708`, KNN's advantage is concentrated in Molecular Function and terms observed on at least two proteins; Cellular Component remains weak and the organism has only 33 evaluated proteins.",
            "",
            f"## K and {ACTIVE_KDE_LABEL} sensitivity",
            "",
            "| Organism | Method | Per-gene Fmax | Per-gene AUPR | Per-gene AUROC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for method in (KNN10,) + SENSITIVITY_METHODS:
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    organism,
                    method.replace("Clean PLM + ", ""),
                    metric_value(metric_rows, organism, method, "per-gene", "F_max"),
                    metric_value(metric_rows, organism, method, "per-gene", "AUPR"),
                    metric_value(metric_rows, organism, method, "per-gene", "AUC"),
                )
            )
    lines.extend(
        [
            "",
            "## Post-blacklist neighbor audit",
            "",
            "| Organism | Queries | Unique KNN-10 donors | Donor organisms | Blacklist violations | Benchmark-accession violations | Median top-1 similarity |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in neighbor_audit:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {:.4f} |".format(
                row["organism"], row["test_queries"], row["unique_knn10_donors"],
                row["unique_knn10_donor_organisms"], row["selected_blacklist_violations"],
                row["selected_benchmark_accession_violations"], row["median_top1_similarity"],
            )
        )
    lines.extend(
        [
            "",
            "| Organism | Neighbor feature | Outcome | Spearman rho |",
            "|---|---|---|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for outcome in ("knn_minus_best_advanced_fmax", "knn_minus_best_advanced_aupr"):
            row = next(
                row for row in correlation_rows
                if row["organism"] == organism
                and row["feature"] == "nearest_similarity"
                and row["outcome"] == outcome
            )
            lines.append(
                "| {} | nearest similarity | {} | {:+.3f} |".format(
                    organism,
                    outcome.replace("knn_minus_best_advanced_", "KNN minus best advanced "),
                    float(row["spearman_rho"]),
                )
            )
    lines.extend(
        [
            "",
            "The audit can establish that the configured blacklist and shared benchmark accessions were removed before transfer. It cannot establish that all remaining donors are phylogenetically distant or that PLM pretraining was free of exposure. High cosine similarity outside the configured blacklist is a hypothesis for follow-up, not proof of leakage.",
            "",
            "## Organism-specific interpretation",
            "",
            "### `83333`",
            "",
            "KNN-10 narrowly exceeds S2F in per-gene Fmax, but S2F has much higher per-gene AUPR and AUROC and lower per-gene Smin. This is the clearest case of a threshold-specific KNN advantage: its best local cutoff is effective, while its full within-protein ranking is weaker. Overall and per-term results must be read separately because they pool a different set of decisions.",
            "",
            "### `1111708`",
            "",
            "KNN-10 has a large per-gene Fmax advantage and strong per-gene AUROC, but its per-gene AUPR remains below S2F. The historical saved Smin favors KNN, but the harmonized same-IC Smin slightly favors S2F and the paired interval crosses zero. The organism has only 33 proteins, so paired-bootstrap intervals and protein examples are required before treating the ranking as stable.",
            "",
            "## Example proteins",
            "",
            "The explorer includes five cases in each direction for each organism. The compact table below lists the two largest Fmax differences in each direction; use the explorer buttons to open exact donor neighborhoods and protein–GO outcomes.",
            "",
            "| Organism | Direction | Protein | Best advanced method | KNN Fmax | Advanced Fmax | Delta |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for organism in DEFAULT_ORGANISMS:
        for direction in ("knn_better", "advanced_better"):
            selected = [
                row for row in examples
                if row["organism"] == organism
                and row["direction"] == direction
                and int(row["rank"]) <= 2
            ]
            for row in selected:
                lines.append(
                    "| {} | {} | {} | {} | {:.4f} | {:.4f} | {:+.4f} |".format(
                        organism,
                        direction,
                        row["protein_id"],
                        row["best_advanced_method"],
                        float(row["knn_fmax"]),
                        float(row["advanced_fmax"]),
                        float(row["fmax_delta"]),
                    )
                )
    lines.extend(
        [
            "",
            "## Additional validation experiments",
            "",
            "1. Select one global threshold on a disjoint validation organism or historical time split and apply it unchanged to the benchmark.",
            "2. Repeat KNN after removing donors above sequence-identity and taxonomic-clade cutoffs, without using benchmark GO labels to choose those cutoffs.",
            "3. Remove ontology roots and high-frequency ancestors to measure how much Fmax comes from generic transferred terms.",
            "4. Compare cosine-weighted KNN with equal voting and calibrated probabilities to separate neighborhood quality from score calibration.",
            "5. Use bootstrap intervals and repeated organism-level splits rather than interpreting the 33-protein result as a precise population estimate.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/build_knn_competitor_analysis.py --project-root .",
            "```",
            "",
            "The exporter records SHA-256 hashes for every prediction-detail and neighbor input in `notebooks/exports/knn_competitor_analysis/analysis_manifest.json`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_frontend_index(frontend_data: Path) -> None:
    path = frontend_data / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("clean_plm_blacklist_comparison", None)
    payload["knn_competitor_analysis"] = {
        "schema_version": SCHEMA_VERSION,
        "path": "data/knn_competitor_analysis.json",
        "description": "Post-blacklist KNN-10 versus S2F, TALE, ATGO, and PANDA2 diagnostics.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    frontend_data = project_root / "notebooks" / "esm_go_explorer" / "data"
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    report_path = args.report if args.report.is_absolute() else project_root / args.report
    organisms = [str(value) for value in args.organisms]
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices, input_files = load_method_matrices(frontend_data, organisms)
    saved_metrics_path = frontend_data / "competitor_context_metrics.csv"
    input_files.add(saved_metrics_path)
    metric_rows, threshold_summary, units, max_difference, legacy_smin_max_difference = build_metric_outputs(
        matrices,
        saved_metrics_path,
        validate=not args.skip_metric_validation,
    )
    threshold_curves, shared_threshold_summary = build_threshold_curves(matrices, units)
    score_summary, score_histograms = build_score_distributions(matrices)
    ontology_rows, frequency_rows = build_subgroup_metrics(matrices)
    bootstrap_rows, bootstrap_delta_rows = paired_bootstrap(
        matrices, units, args.bootstrap_replicates, args.seed
    )
    cv_fold_rows, cv_summary = repeated_threshold_cv(matrices, args.cv_repeats, args.seed)
    neighbor = build_neighbor_outputs(
        frontend_data, organisms, matrices, units, args.blacklist_dir
    )
    input_files.add(neighbor["source_path"])
    input_files.update(neighbor["blacklist_paths"])
    examples = build_examples(matrices, units, neighbor["neighbor_lookup"])

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "metadata": {
            "organisms": organisms,
            "primary_methods": list(PRIMARY_METHODS),
            "sensitivity_methods": list(SENSITIVITY_METHODS),
            "primary_knn": KNN10,
            "evaluation_policy": "shared post-blacklist prediction-detail matrices; archived pre-blacklist metrics excluded",
            "pre_blacklist_results_used": False,
            "score_rounding": "round to four decimals when a complete prediction matrix has more than 10,000 unique scores",
            "bootstrap_replicates": args.bootstrap_replicates,
            "cross_validation_repeats": args.cv_repeats,
            "random_seed": args.seed,
            "ontology_roots": sorted(ONTOLOGY_ROOTS),
            "metric_validation_max_absolute_difference": max_difference,
            "legacy_advanced_smin_max_absolute_difference": legacy_smin_max_difference,
            "smin_comparability_warning": (
                "Saved S2F/TALE/ATGO/PANDA2 Smin uses full-organism information content, while Clean PLM "
                "uses shared-evaluation-set information content. Harmonized values use the shared set for all methods."
            ),
            "smin_direction": "lower_is_better",
        },
        "metric_rows": metric_rows,
        "threshold_summary": threshold_summary,
        "shared_threshold_summary": shared_threshold_summary,
        "threshold_curves": threshold_curves,
        "score_summary": score_summary,
        "score_histograms": score_histograms,
        "ontology_rows": ontology_rows,
        "frequency_rows": frequency_rows,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_delta_rows": bootstrap_delta_rows,
        "cross_validation_summary": cv_summary,
        "neighbor_audit": neighbor["audit_rows"],
        "neighbor_profiles": neighbor["profile_rows"],
        "neighbor_correlations": neighbor["correlation_rows"],
        "donor_taxa": neighbor["taxon_rows"],
        "examples": examples,
    }
    payload = json_ready(payload)
    frontend_path = frontend_data / "knn_competitor_analysis.json"
    frontend_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    update_frontend_index(frontend_data)

    write_csv(output_dir / "metric_comparison.csv", metric_rows)
    write_csv(output_dir / "threshold_summary.csv", threshold_summary)
    write_csv(output_dir / "shared_threshold_summary.csv", shared_threshold_summary)
    write_csv(output_dir / "score_summary.csv", score_summary)
    write_csv(output_dir / "ontology_metrics.csv", ontology_rows)
    write_csv(output_dir / "term_frequency_metrics.csv", frequency_rows)
    write_csv(output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    write_csv(output_dir / "bootstrap_deltas.csv", bootstrap_delta_rows)
    write_csv(output_dir / "threshold_cross_validation.csv", cv_fold_rows)
    write_csv(output_dir / "threshold_cross_validation_summary.csv", cv_summary)
    write_csv(output_dir / "neighbor_profiles.csv", neighbor["profile_rows"])
    write_csv(output_dir / "neighbor_audit.csv", neighbor["audit_rows"])
    write_csv(output_dir / "neighbor_donors.csv", neighbor["donor_rows"])
    write_csv(output_dir / "neighbor_correlations.csv", neighbor["correlation_rows"])
    write_csv(output_dir / "donor_reuse.csv", neighbor["taxon_rows"])
    (output_dir / "examples.json").write_text(
        json.dumps(json_ready(examples), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    write_report(report_path, payload)
    input_files.add(frontend_data / "pfp_prediction_details_index.json")
    output_artifacts = [frontend_path, report_path] + sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "command": "python scripts/build_knn_competitor_analysis.py --project-root .",
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "organisms": organisms,
            "bootstrap_replicates": args.bootstrap_replicates,
            "cross_validation_repeats": args.cv_repeats,
            "seed": args.seed,
        },
        "inputs": [
            {
                "path": project_relative_or_name(path, project_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(input_files)
        ],
        "outputs": {
            "frontend_json": str(frontend_path.relative_to(project_root)),
            "report": str(report_path.relative_to(project_root)),
            "export_directory": str(output_dir.relative_to(project_root)),
        },
        "artifacts": [
            {
                "path": project_relative_or_name(path, project_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_artifacts
        ],
        "metric_validation_max_absolute_difference": max_difference,
        "legacy_advanced_smin_max_absolute_difference": legacy_smin_max_difference,
        "pre_blacklist_results_used": False,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "frontend": str(frontend_path),
                "report": str(report_path),
                "output_dir": str(output_dir),
                "metric_validation_max_absolute_difference": max_difference,
                "legacy_advanced_smin_max_absolute_difference": legacy_smin_max_difference,
                "examples": len(examples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
