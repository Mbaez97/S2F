#!/usr/bin/env python3
"""Derive all-method comparison evidence from existing benchmark artifacts.

This script does not run any prediction method.  It combines the current
post-blacklist diagnostics and complete protein--GO score matrices into a
traceable research summary for the explorer and a human-readable report.
Missing method outputs in the complete score matrices are treated as score
zero for metric, calibration, and operating-point calculations, matching the
benchmark reconstruction used by ``build_knn_competitor_analysis.py``.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import itertools
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from build_knn_competitor_analysis import evaluate_binary, smin_for_vector
from clean_plm_benchmark import EVIDENCE_CODES, load_goa_annotations_for_taxon
from GOTool.GeneOntology import GeneOntology


S2F_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
DEFAULT_EXPORT_DIR = S2F_ROOT / "notebooks" / "exports" / "method_comparison_analysis"
DEFAULT_REPORT = S2F_ROOT / "doc" / "method_comparison_research_questions.md"
DIAGNOSTICS_NAME = "knn_competitor_analysis.json"
SCORE_INDEX_NAME = "pfp_score_comparison_index.json"
OUTPUT_NAME = "method_comparison_analysis.json"
SCHEMA_VERSION = 2

S2F_METHOD = "S2F"
KNN10_METHOD = "Clean PLM + KNN k=10 (weighted_support)"
KDE_METHOD = "Clean PLM + KDE (gaussian_kernel_support)"
COMPETITOR_METHODS = ("TALE", "ATGO", "PANDA2")
METRIC_ORDER = ("F_max", "AUPR", "AUC", "smin")
ONTOLOGY_ROOTS = {"GO:0003674", "GO:0005575", "GO:0008150"}
SCORE_ROUNDING_UNIQUE_LIMIT = 10_000
SCORE_ROUNDING_DECIMALS = 4
FULL_ORGANISM_IC_SCOPE = "full_organism_before_shared_evaluation_filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an all-method analysis from existing explorer artifacts."
    )
    parser.add_argument("--frontend-data", type=Path, default=DEFAULT_FRONTEND_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--goa-path",
        type=Path,
        default=None,
        help=(
            "Experimental GOA snapshot used to calculate full-organism information "
            "content. Defaults to the configured filtered_goa path or its mounted-disk "
            "equivalent."
        ),
    )
    parser.add_argument("--obo-path", type=Path, default=S2F_ROOT / "go.obo")
    return parser.parse_args()


def resolve_goa_path(requested: Path | None) -> Path:
    """Resolve the exact filtered GOA snapshot used by the benchmark notebook."""
    candidates: List[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())

    config = configparser.ConfigParser()
    config.read(S2F_ROOT / "s2f.conf")
    configured = config.get("databases", "filtered_goa", fallback="").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.append(configured_path)
        if configured_path.is_absolute() and configured_path.parts[:2] == ("/", "media"):
            candidates.append(Path("/run") / configured_path.relative_to("/"))

    candidates.extend(
        [
            S2F_ROOT / "transfer" / "esm_pfp_work_2026-07-01" / "data" / "uniprot" / "filtered_goa",
            S2F_ROOT.parent / "data" / "uniprot" / "filtered_goa",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "A filtered GOA snapshot is required for full-organism Smin. "
        f"Searched: {searched}"
    )


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return finite_or_none(value)


def method_family(method: str) -> str:
    if method == S2F_METHOD:
        return "proposed"
    if method in COMPETITOR_METHODS:
        return "competitor"
    if method.startswith("Clean PLM +"):
        return "simple_embedding_transfer"
    return "other"


def score_payload_path(frontend_data: Path, descriptor: Mapping[str, object]) -> Path:
    relative = Path(str(descriptor["path"]))
    if relative.parts and relative.parts[0] == "data":
        relative = Path(*relative.parts[1:])
    return frontend_data / relative


def unpack_score_payload(
    payload: Mapping[str, object], methods: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    columns = {key: index for index, key in enumerate(payload["column_keys"])}
    rows = payload["rows"]
    truth = np.asarray([int(row[columns["ground_truth"]]) for row in rows], dtype=np.int8)
    proteins = np.asarray([str(row[columns["protein_id"]]) for row in rows], dtype=object)
    terms = np.asarray([str(row[columns["term_id"]]) for row in rows], dtype=object)
    domains = np.asarray([str(row[columns["go_domain"]]) for row in rows], dtype=object)
    protein_order = list(dict.fromkeys(proteins.tolist()))
    term_order = list(dict.fromkeys(terms.tolist()))
    if len(rows) != len(protein_order) * len(term_order):
        raise RuntimeError("Score payload is not a complete protein-by-GO-term matrix.")
    protein_to_index = {protein: index for index, protein in enumerate(protein_order)}
    term_to_index = {term: index for index, term in enumerate(term_order)}
    row_indices = np.asarray([protein_to_index[protein] for protein in proteins], dtype=int)
    column_indices = np.asarray([term_to_index[term] for term in terms], dtype=int)
    truth_matrix = np.zeros((len(protein_order), len(term_order)), dtype=np.int8)
    truth_matrix[row_indices, column_indices] = truth
    domain_by_term = {}
    for term, domain in zip(terms, domains):
        existing = domain_by_term.setdefault(str(term), str(domain))
        if existing != str(domain):
            raise RuntimeError(f"GO domain differs across rows for {term}.")

    scores = {}
    score_matrices = {}
    available = {}
    score_round_decimals = {}
    for method in methods:
        values = [row[columns[str(method["key"])]] for row in rows]
        model = str(method["source_model"])
        available[model] = np.asarray(
            [value is not None for value in values], dtype=bool
        )
        method_scores = np.asarray(
            [0.0 if value is None else float(value) for value in values], dtype=float
        )
        decimals = (
            SCORE_ROUNDING_DECIMALS
            if np.unique(method_scores).size > SCORE_ROUNDING_UNIQUE_LIMIT
            else None
        )
        if decimals is not None:
            method_scores = np.around(method_scores, decimals=decimals)
        scores[model] = method_scores
        score_round_decimals[model] = decimals
        score_matrix = np.zeros_like(truth_matrix, dtype=float)
        score_matrix[row_indices, column_indices] = method_scores
        score_matrices[model] = score_matrix
    return {
        "truth": truth,
        "proteins": proteins,
        "terms": terms,
        "domains": domains,
        "protein_order": protein_order,
        "term_order": term_order,
        "truth_matrix": truth_matrix,
        "scores": scores,
        "score_matrices": score_matrices,
        "available": available,
        "score_round_decimals": score_round_decimals,
        "domain_by_term": domain_by_term,
    }


def full_organism_information_content(
    organism: str,
    terms: Sequence[str],
    goa_path: Path,
    obo_path: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Calculate GO-term IC before applying the shared evaluation-protein filter."""
    annotations = load_goa_annotations_for_taxon(goa_path, organism)
    if annotations.empty:
        raise RuntimeError(f"No experimental GOA annotations found for taxon {organism}.")

    ontology = GeneOntology(str(obo_path), verbose=False)
    ontology.build_structure()
    valid = annotations["GO ID"].map(
        lambda term: term in ontology.terms or term in ontology.alias_map
    )
    annotations = annotations.loc[valid].copy()
    if annotations.empty:
        raise RuntimeError(f"No GOA annotations for taxon {organism} match {obo_path}.")

    organism_name = f"method_comparison_full_taxon_{organism}"
    ontology.load_annotations(annotations, organism_name)
    ontology.up_propagate_annotations(organism_name)
    propagated = ontology.get_annotations(organism_name)

    values = np.zeros(len(terms), dtype=float)
    missing_terms = []
    for index, term in enumerate(terms):
        try:
            values[index] = float(ontology.find_term(term).information_content(organism_name))
        except KeyError:
            missing_terms.append(term)
    if missing_terms:
        raise RuntimeError(
            f"{len(missing_terms)} evaluated GO terms are absent from {obo_path}: "
            + ", ".join(missing_terms[:5])
        )

    return values, {
        "organism": organism,
        "information_content_scope": FULL_ORGANISM_IC_SCOPE,
        "direct_experimental_annotation_pairs": int(len(annotations)),
        "direct_experimentally_annotated_proteins": int(annotations["Protein"].nunique()),
        "propagated_annotation_pairs": int(len(propagated)),
        "propagated_annotated_proteins": int(propagated["Protein"].nunique()),
        "evaluated_terms": int(len(terms)),
        "evidence_codes": sorted(EVIDENCE_CODES),
    }


def calculated_smin_by_scope(
    prediction: np.ndarray,
    gold: np.ndarray,
    information_content: np.ndarray,
) -> Dict[str, float]:
    if gold.size == 0 or gold.shape[0] == 0 or gold.shape[1] == 0:
        return {"overall": math.nan, "per-gene": math.nan, "per-term": math.nan}
    overall = smin_for_vector(
        prediction,
        gold,
        information_content.reshape(1, -1),
        gold.shape[0],
    )
    per_gene = [
        smin_for_vector(prediction[index, :], gold[index, :], information_content, gold.shape[1])
        for index in range(gold.shape[0])
    ]
    per_term = [
        smin_for_vector(
            prediction[:, index],
            gold[:, index],
            float(information_content[index]),
            gold.shape[0],
        )
        for index in range(gold.shape[1])
    ]
    return {
        "overall": float(overall),
        "per-gene": float(np.mean(per_gene)),
        "per-term": float(np.mean(per_term)),
    }


def calculated_smin_lookup(
    organism: str,
    unpacked: Mapping[str, object],
    methods: Sequence[Mapping[str, object]],
    information_content: np.ndarray,
) -> Dict[Tuple[str, str, str, str], float]:
    """Calculate full-organism-IC Smin for global and ontology metric rows."""
    terms = np.asarray(unpacked["term_order"], dtype=object)
    domains = np.asarray(
        [unpacked["domain_by_term"][str(term)] for term in terms], dtype=object
    )
    truth = unpacked["truth_matrix"]
    result: Dict[Tuple[str, str, str, str], float] = {}
    ontology_names = ("all", "biological_process", "molecular_function", "cellular_component")
    for ontology in ontology_names:
        ontology_mask = np.ones(len(terms), dtype=bool) if ontology == "all" else domains == ontology
        for root_policy in ("included", "excluded"):
            term_mask = ontology_mask.copy()
            if root_policy == "excluded":
                term_mask &= ~np.isin(terms, list(ONTOLOGY_ROOTS))
            selected_gold = truth[:, term_mask]
            selected_ic = information_content[term_mask]
            for method in methods:
                model = str(method["source_model"])
                selected_prediction = unpacked["score_matrices"][model][:, term_mask]
                values = calculated_smin_by_scope(
                    selected_prediction, selected_gold, selected_ic
                )
                for scope, value in values.items():
                    result[(model, ontology, root_policy, scope)] = value
    return result


def replace_smin_values(
    rows: Sequence[Mapping[str, object]],
    organism: str,
    lookup: Mapping[Tuple[str, str, str, str], float],
) -> List[Dict[str, object]]:
    """Keep one calculated Smin value and remove saved/harmonized variants."""
    result = []
    for source in rows:
        row = dict(source)
        if str(row.get("organism")) != organism or row.get("metric") != "smin":
            result.append(row)
            continue
        ontology = str(row.get("ontology", "all"))
        root_policy = str(row.get("ontology_roots", "included"))
        key = (str(row["method"]), ontology, root_policy, str(row["scope"]))
        row["value"] = float(lookup[key])
        for field in (
            "harmonized_value",
            "saved_value",
            "absolute_difference",
            "metric_validation_included",
            "saved_information_content_scope",
            "harmonized_information_content_scope",
        ):
            row.pop(field, None)
        row["value_source"] = "calculated_from_shared_predictions"
        row["information_content_scope"] = FULL_ORGANISM_IC_SCOPE
        result.append(row)
    return result


def calibration_rows(
    organism: str,
    unpacked: Mapping[str, object],
    methods: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    truth = unpacked["truth"].astype(bool)
    rows = []
    for method in methods:
        model = str(method["source_model"])
        scores = unpacked["scores"][model]
        available = unpacked["available"][model]
        positive_scores = scores[truth]
        negative_scores = scores[~truth]
        rows.append(
            {
                "organism": organism,
                "method": model,
                "method_family": method_family(model),
                "pairs": int(scores.size),
                "positive_prevalence": float(truth.mean()),
                "output_coverage": float(available.mean()),
                "nonzero_fraction": float(np.mean(scores > 0)),
                "mean_score": float(scores.mean()),
                "positive_nonzero_fraction": float(np.mean(positive_scores > 0)),
                "negative_nonzero_fraction": float(np.mean(negative_scores > 0)),
                "positive_score_median": float(np.median(positive_scores)),
                "negative_score_median": float(np.median(negative_scores)),
                "negative_score_p90": float(np.quantile(negative_scores, 0.9)),
                "score_round_decimals": unpacked["score_round_decimals"][model],
                "interpretation": (
                    "Descriptive score-output statistics on the benchmark-aligned matrix; "
                    "method score scales are not assumed to be calibrated probabilities."
                ),
            }
        )
    return rows


def safe_correlation(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    pearson = math.nan
    spearman = math.nan
    if left.size and float(np.std(left)) > 0 and float(np.std(right)) > 0:
        pearson = float(np.corrcoef(left, right)[0, 1])
        spearman = float(spearmanr(left, right).statistic)
    return pearson, spearman


def threshold_lookup(diagnostics: Mapping[str, object]) -> Dict[Tuple[str, str], float]:
    return {
        (str(row["organism"]), str(row["method"])): float(row["shared_threshold"])
        for row in diagnostics.get("shared_threshold_summary", [])
        if row.get("scope") == "per-gene"
    }


def pairwise_rows(
    organism: str,
    unpacked: Mapping[str, object],
    methods: Sequence[Mapping[str, object]],
    thresholds: Mapping[Tuple[str, str], float],
) -> List[Dict[str, object]]:
    truth = unpacked["truth"].astype(bool)
    result = []
    for left_method, right_method in itertools.combinations(methods, 2):
        left_name = str(left_method["source_model"])
        right_name = str(right_method["source_model"])
        left = unpacked["scores"][left_name]
        right = unpacked["scores"][right_name]
        left_available = unpacked["available"][left_name]
        right_available = unpacked["available"][right_name]
        left_threshold = thresholds[(organism, left_name)]
        right_threshold = thresholds[(organism, right_name)]
        left_positive = left >= left_threshold
        right_positive = right >= right_threshold
        left_correct = left_positive == truth
        right_correct = right_positive == truth
        pearson, spearman = safe_correlation(left, right)
        result.append(
            {
                "organism": organism,
                "method_a": left_name,
                "method_b": right_name,
                "method_a_family": method_family(left_name),
                "method_b_family": method_family(right_name),
                "pairs": int(truth.size),
                "method_a_threshold": left_threshold,
                "method_b_threshold": right_threshold,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "mean_absolute_score_difference": float(np.mean(np.abs(left - right))),
                "exact_score_agreement_fraction": float(np.mean(np.isclose(left, right, atol=1e-12, rtol=0.0))),
                "prediction_agreement_fraction": float(np.mean(left_positive == right_positive)),
                "both_available": int(np.sum(left_available & right_available)),
                "method_a_only_available": int(np.sum(left_available & ~right_available)),
                "method_b_only_available": int(np.sum(~left_available & right_available)),
                "neither_available": int(np.sum(~left_available & ~right_available)),
                "both_correct": int(np.sum(left_correct & right_correct)),
                "method_a_only_correct": int(np.sum(left_correct & ~right_correct)),
                "method_b_only_correct": int(np.sum(~left_correct & right_correct)),
                "both_wrong": int(np.sum(~left_correct & ~right_correct)),
                "positive_both_recovered": int(np.sum(truth & left_positive & right_positive)),
                "positive_method_a_only": int(np.sum(truth & left_positive & ~right_positive)),
                "positive_method_b_only": int(np.sum(truth & ~left_positive & right_positive)),
                "positive_neither_recovered": int(np.sum(truth & ~left_positive & ~right_positive)),
                "negative_method_a_only_false_positive": int(np.sum(~truth & left_positive & ~right_positive)),
                "negative_method_b_only_false_positive": int(np.sum(~truth & ~left_positive & right_positive)),
            }
        )
    return result


def per_protein_metrics(
    unpacked: Mapping[str, object], methods: Sequence[Mapping[str, object]]
) -> Dict[str, Dict[str, Dict[str, object]]]:
    proteins = unpacked["proteins"]
    truth = unpacked["truth"]
    result: Dict[str, Dict[str, Dict[str, object]]] = {}
    for protein in sorted(set(proteins.tolist())):
        mask = proteins == protein
        result[protein] = {}
        for method in methods:
            model = str(method["source_model"])
            result[protein][model] = evaluate_binary(
                truth[mask], unpacked["scores"][model][mask]
            )
    return result


def pair_example_rows(
    organism: str,
    unpacked: Mapping[str, object],
    methods: Sequence[Mapping[str, object]],
    count: int = 3,
) -> List[Dict[str, object]]:
    metrics = per_protein_metrics(unpacked, methods)
    result = []
    for left_method, right_method in itertools.combinations(methods, 2):
        left_name = str(left_method["source_model"])
        right_name = str(right_method["source_model"])
        candidates = []
        for protein, by_method in metrics.items():
            left = by_method[left_name]
            right = by_method[right_name]
            candidates.append(
                {
                    "organism": organism,
                    "protein_id": protein,
                    "method_a": left_name,
                    "method_b": right_name,
                    "method_a_fmax": float(left["F_max"]),
                    "method_b_fmax": float(right["F_max"]),
                    "fmax_delta_a_minus_b": float(left["F_max"] - right["F_max"]),
                    "method_a_aupr": float(left["AUPR"]),
                    "method_b_aupr": float(right["AUPR"]),
                    "aupr_delta_a_minus_b": float(left["AUPR"] - right["AUPR"]),
                    "ground_truth_terms": int(np.sum(unpacked["truth"][unpacked["proteins"] == protein])),
                }
            )
        left_wins = sorted(
            candidates,
            key=lambda row: (
                row["fmax_delta_a_minus_b"],
                row["aupr_delta_a_minus_b"],
                row["protein_id"],
            ),
            reverse=True,
        )[:count]
        right_wins = sorted(
            candidates,
            key=lambda row: (
                row["fmax_delta_a_minus_b"],
                row["aupr_delta_a_minus_b"],
                row["protein_id"],
            ),
        )[:count]
        for direction, selected in (("method_a_better", left_wins), ("method_b_better", right_wins)):
            for rank, row in enumerate(selected, start=1):
                result.append({**row, "direction": direction, "rank": rank})
    return result


def metric_value(
    diagnostics: Mapping[str, object], organism: str, method: str, scope: str, metric: str,
    field: str = "value",
) -> float:
    row = next(
        item
        for item in diagnostics["metric_rows"]
        if str(item["organism"]) == organism
        and item["method"] == method
        and item["scope"] == scope
        and item["metric"] == metric
    )
    return float(row.get(field, row["value"]))


def short_method(method: str) -> str:
    replacements = {
        KNN10_METHOD: "KNN K=10",
        KDE_METHOD: "Gaussian KDE",
        "Clean PLM + KNN k=3 (weighted_support)": "KNN K=3",
        "Clean PLM + KNN k=5 (weighted_support)": "KNN K=5",
        "Clean PLM + KNN k=7 (weighted_support)": "KNN K=7",
    }
    return replacements.get(method, method)


def format_metric(metric: str, value: float) -> str:
    if metric == "smin":
        return f"{value:.4f}"
    return f"{value:.4f}"


def markdown_metric_table(
    diagnostics: Mapping[str, object], organism: str, methods: Sequence[str], scope: str
) -> List[str]:
    lines = [
        f"### `{organism}` — {scope}",
        "",
        "| Method | Family | Fmax | AUPR | AUROC | Calculated Smin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        values = {
            metric: metric_value(diagnostics, organism, method, scope, metric)
            for metric in METRIC_ORDER
        }
        lines.append(
            f"| {short_method(method)} | {method_family(method)} | "
            f"{values['F_max']:.4f} | {values['AUPR']:.4f} | {values['AUC']:.4f} | "
            f"{values['smin']:.4f} |"
        )
    return lines


def calibration_lookup(rows: Sequence[Mapping[str, object]], organism: str, method: str):
    return next(
        row for row in rows
        if row["organism"] == organism and row["method"] == method
    )


def frequency_value(
    diagnostics: Mapping[str, object], organism: str, method: str, frequency: str, metric: str
) -> float:
    row = next(
        item for item in diagnostics["frequency_rows"]
        if str(item["organism"]) == organism
        and item["method"] == method
        and item["frequency_bin"] == frequency
        and item["scope"] == "per-term"
        and item["metric"] == metric
    )
    return float(row["value"])


def write_report(
    path: Path,
    payload: Mapping[str, object],
    methods: Sequence[str],
) -> None:
    score_rows = payload["calibration_rows"]
    lines = [
        "# Why S2F and simple transfer methods perform well",
        "",
        "> Derived only from the current shared, post-blacklist prediction matrices. No prediction method was rerun.",
        "",
        "## Evaluation boundary",
        "",
        "The analysis compares S2F, TALE, ATGO, PANDA2, Gaussian KDE, and KNN with K = 3, 5, 7, and 10. "
        "All methods are evaluated on the same propagated GO truth within each organism. Taxon `83333` has 160 proteins, 574 terms, and 1,922 positive protein–GO pairs; taxon `1111708` has 33 proteins, 93 terms, and 306 positives. The 33-protein organism is informative but statistically fragile.",
        "",
        "Per-gene Fmax is an oracle average: each protein receives its own best threshold. AUPR evaluates the complete ranking. AUROC evaluates positive-versus-negative ordering. Smin is lower-is-better and weights remaining uncertainty and misinformation by GO-term information content. The Smin shown here is recalculated for every method from the shared prediction matrix using information content computed from the complete experimentally annotated organism before the shared evaluation-protein filter.",
        "",
        "## Research question 1: Why does S2F perform well?",
        "",
        "The experiment directly supports a prediction-level explanation: when S2F has higher AUPR or AUROC, its saved scores order positive protein–GO pairs ahead of negatives more successfully; when it has higher Fmax, its best operating point has a better precision–recall balance. The tables below establish those differences. They do not isolate which S2F component caused them.",
        "",
        "### S2F versus TALE, ATGO, and PANDA2",
        "",
    ]
    advanced_methods = [S2F_METHOD, *COMPETITOR_METHODS]
    for organism in ("83333", "1111708"):
        lines.extend(markdown_metric_table(payload, organism, advanced_methods, "per-gene"))
        lines.append("")

    lines.extend(
        [
            "### Observed prediction-score properties",
            "",
            "These values use the same score rounding as the benchmark metrics. Missing outputs are score zero. Because method scores are not guaranteed to be calibrated probabilities, absolute score magnitudes should not be compared as confidence across methods.",
            "",
            "| Organism | Method | Output coverage | True-pair nonzero | Negative-pair nonzero | Positive median | Negative p90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for organism in ("83333", "1111708"):
        for method in advanced_methods:
            row = calibration_lookup(score_rows, organism, method)
            lines.append(
                f"| {organism} | {short_method(method)} | {row['output_coverage']:.3f} | "
                f"{row['positive_nonzero_fraction']:.3f} | {row['negative_nonzero_fraction']:.3f} | "
                f"{row['positive_score_median']:.4f} | {row['negative_score_p90']:.4f} |"
            )

    lines.extend(
        [
            "",
            "These score properties can explain metric differences only at the output level. For example, greater recovery of true pairs can support recall, while high scores on negatives can reduce precision and AUPR. The current experiment does not contain an S2F component ablation, so it cannot establish why the pipeline produced those score patterns.",
            "",
            "## Research question 2: Why do simpler algorithms sometimes perform better?",
            "",
            "The saved results show whether a simple method's advantage is confined to Fmax or extends to ranking metrics. A higher Fmax demonstrates a better best threshold; it does not by itself demonstrate a better full ranking. AUPR and AUROC must be read separately.",
            "",
            "### All methods",
            "",
        ]
    )
    for organism in ("83333", "1111708"):
        lines.extend(markdown_metric_table(payload, organism, methods, "per-gene"))
        lines.append("")

    lines.extend(
        [
            "## Evidence limits",
            "",
            "The transfer audit reports high cosine similarity to selected donors and zero violations of the configured taxon blacklist and shared-benchmark accession exclusion. It does not measure whether a selected donor carries the correct GO annotation for each query, prove phylogenetic independence, or rule out PLM pretraining exposure. Therefore the available data can identify ranking, coverage, precision, recall, and threshold behavior, but cannot prove an architectural or biological cause for a method's advantage.",
            "",
            "## Conclusions",
            "",
            "1. S2F's advantages over TALE, ATGO, and PANDA2 are supported where its Fmax, AUPR, AUROC, or calculated Smin is better. Higher AUPR/AUROC is evidence of better score ordering; threshold precision and recall show the operating-point behavior. The experiment does not identify which S2F component causes those outputs.",
            "2. A simple method's higher Fmax is evidence of a stronger best operating point, not automatically a stronger ranking. When its AUPR or AUROC is lower, the result is metric-specific. The donor audit is insufficient to prove that neighborhood quality caused the advantage.",
            "3. Method order depends on organism, evaluation scope, ontology, and metric. Results for `1111708` require particular caution because the shared set contains only 33 proteins.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/build_method_comparison_analysis.py",
            "```",
            "",
            "The manifest in `notebooks/exports/method_comparison_analysis/analysis_manifest.json` records the interpreter and SHA-256 hashes of all consumed artifacts.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [key for key in rows[0] if key != "bins"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_none(row.get(key)) for key in fields})


def update_frontend_index(frontend_data: Path) -> None:
    path = frontend_data / "index.json"
    payload = read_json(path)
    payload["method_comparison_analysis"] = {
        "schema_version": SCHEMA_VERSION,
        "path": f"data/{OUTPUT_NAME}",
        "description": (
            "Pairwise diagnostics for S2F, competitor models, KNN, and Gaussian KDE "
            "on shared post-blacklist evaluation matrices."
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def manifest_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(S2F_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    frontend_data = args.frontend_data.resolve()
    export_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    goa_path = resolve_goa_path(args.goa_path)
    obo_path = args.obo_path.expanduser().resolve()
    if not obo_path.is_file():
        raise FileNotFoundError(f"GO ontology file not found: {obo_path}")
    diagnostics_path = frontend_data / DIAGNOSTICS_NAME
    score_index_path = frontend_data / SCORE_INDEX_NAME
    diagnostics = read_json(diagnostics_path)
    score_index = read_json(score_index_path)
    methods = score_index["methods"]
    method_names = [str(method["source_model"]) for method in methods]
    thresholds = threshold_lookup(diagnostics)
    generated_at = datetime.now(timezone.utc).isoformat()

    calibration = []
    pairwise = []
    examples = []
    source_paths = [diagnostics_path, score_index_path, goa_path, obo_path]
    organism_summaries = []
    full_organism_ic_summaries = []
    score_rounding = []
    metric_rows = [dict(row) for row in diagnostics.get("metric_rows", [])]
    ontology_rows = [dict(row) for row in diagnostics.get("ontology_rows", [])]
    advanced_smin_differences = []
    for descriptor in score_index["organisms"]:
        organism = str(descriptor["organism"])
        path = score_payload_path(frontend_data, descriptor)
        source_paths.append(path)
        payload = read_json(path)
        unpacked = unpack_score_payload(payload, methods)
        calibration.extend(calibration_rows(organism, unpacked, methods))
        pairwise.extend(pairwise_rows(organism, unpacked, methods, thresholds))
        examples.extend(pair_example_rows(organism, unpacked, methods))
        organism_summaries.append({"organism": organism, **payload["summary"]})
        information_content, ic_summary = full_organism_information_content(
            organism, unpacked["term_order"], goa_path, obo_path
        )
        full_organism_ic_summaries.append(ic_summary)
        smin_lookup = calculated_smin_lookup(
            organism, unpacked, methods, information_content
        )
        for row in diagnostics.get("metric_rows", []):
            if (
                str(row.get("organism")) == organism
                and row.get("metric") == "smin"
                and row.get("method") in COMPETITOR_METHODS + (S2F_METHOD,)
            ):
                calculated = smin_lookup[
                    (str(row["method"]), "all", "included", str(row["scope"]))
                ]
                advanced_smin_differences.append(abs(calculated - float(row["value"])))
        metric_rows = replace_smin_values(metric_rows, organism, smin_lookup)
        ontology_rows = replace_smin_values(ontology_rows, organism, smin_lookup)
        score_rounding.extend(
            {
                "organism": organism,
                "method": str(method["source_model"]),
                "decimals": unpacked["score_round_decimals"][str(method["source_model"])],
                "unique_score_limit": SCORE_ROUNDING_UNIQUE_LIMIT,
            }
            for method in methods
        )

    metadata = dict(diagnostics.get("metadata", {}))
    metadata.pop("smin_comparability_warning", None)
    metadata.update(
        {
            "methods": [
                {
                    **method,
                    "family": method_family(str(method["source_model"])),
                }
                for method in methods
            ],
            "organism_summaries": organism_summaries,
            "comparison_scope": (
                "shared post-blacklist complete protein-GO matrices; missing method "
                "outputs are score zero for diagnostics"
            ),
            "score_rounding": score_rounding,
            "smin_information_content_scope": FULL_ORGANISM_IC_SCOPE,
            "smin_information_content_source": str(goa_path),
            "smin_ontology_source": str(obo_path),
            "smin_full_organism_summaries": full_organism_ic_summaries,
            "advanced_smin_validation_max_absolute_difference": max(
                advanced_smin_differences, default=math.nan
            ),
        }
    )

    payload = {
        **diagnostics,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "source_diagnostics_schema_version": diagnostics.get("schema_version"),
        "source_diagnostics_generated_at_utc": diagnostics.get("generated_at_utc"),
        "metadata": metadata,
        "metric_rows": metric_rows,
        "ontology_rows": ontology_rows,
        "calibration_rows": calibration,
        "pairwise_rows": pairwise,
        "pair_examples": examples,
        "research_answers": {
            "why_s2f_performs_well": (
                "Where S2F has higher AUPR or AUROC, its scores rank positive pairs "
                "ahead of negatives more successfully; threshold precision, recall, "
                "and score coverage describe the corresponding output behavior."
            ),
            "why_simple_methods_perform_well": (
                "Where a simple method has higher Fmax, the saved results establish a "
                "better best precision-recall operating point. They do not establish "
                "that neighborhood quality caused that advantage."
            ),
            "ranking_caveat": (
                "Method order changes with metric, aggregation scope, organism, ontology, "
                "and whether thresholds are oracle-selected or held out."
            ),
            "causal_limit": (
                "The available experiments support prediction-level explanations but do "
                "not contain the ablations needed to identify a causal model component."
            ),
        },
    }
    payload = json_ready(payload)
    output_path = frontend_data / OUTPUT_NAME
    output_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    update_frontend_index(frontend_data)

    export_dir.mkdir(parents=True, exist_ok=True)
    write_csv(export_dir / "calibration_metrics.csv", calibration)
    write_csv(export_dir / "pairwise_score_diagnostics.csv", pairwise)
    write_csv(export_dir / "pair_examples.csv", examples)
    write_report(report_path, payload, method_names)

    outputs = [
        output_path,
        report_path,
        export_dir / "calibration_metrics.csv",
        export_dir / "pairwise_score_diagnostics.csv",
        export_dir / "pair_examples.csv",
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "command": (
            "/home/marcelo_baez/anaconda3/envs/S2F/bin/python "
            "scripts/build_method_comparison_analysis.py"
        ),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "inputs": [
            {"path": manifest_display_path(path), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "outputs": [
            {"path": manifest_display_path(path), "sha256": sha256_file(path)}
            for path in outputs
        ],
        "smin_information_content_scope": FULL_ORGANISM_IC_SCOPE,
        "advanced_smin_validation_max_absolute_difference": max(
            advanced_smin_differences, default=math.nan
        ),
    }
    manifest_path = export_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
