#!/usr/bin/env python3
"""Build compact all-method protein-GO score matrices for the explorer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple
import argparse
import json

from clean_plm_method_registry import ACTIVE_KDE_LABEL, ACTIVE_KDE_MODEL, ACTIVE_KDE_SCORE_KEY


S2F_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
DETAIL_INDEX_NAME = "pfp_prediction_details_index.json"
OUTPUT_INDEX_NAME = "pfp_score_comparison_index.json"
OUTPUT_DIR_NAME = "pfp_score_comparison"
SCHEMA_VERSION = 2

METHODS = [
    {
        "key": "s2f_score",
        "label": "S2F score",
        "short_label": "S2F",
        "source_model": "S2F",
    },
    {
        "key": "tale_score",
        "label": "TALE score",
        "short_label": "TALE",
        "source_model": "TALE",
    },
    {
        "key": "atgo_score",
        "label": "ATGO score",
        "short_label": "ATGO",
        "source_model": "ATGO",
    },
    {
        "key": "panda2_score",
        "label": "PANDA2 score",
        "short_label": "PANDA2",
        "source_model": "PANDA2",
    },
    {
        "key": ACTIVE_KDE_SCORE_KEY,
        "label": f"{ACTIVE_KDE_LABEL} score",
        "short_label": ACTIVE_KDE_LABEL,
        "source_model": ACTIVE_KDE_MODEL,
    },
    {
        "key": "knn_k10_score",
        "label": "KNN K=10 score",
        "short_label": "KNN K=10",
        "source_model": "Clean PLM + KNN k=10 (weighted_support)",
    },
    {
        "key": "knn_k7_score",
        "label": "KNN K=7 score",
        "short_label": "KNN K=7",
        "source_model": "Clean PLM + KNN k=7 (weighted_support)",
    },
    {
        "key": "knn_k5_score",
        "label": "KNN K=5 score",
        "short_label": "KNN K=5",
        "source_model": "Clean PLM + KNN k=5 (weighted_support)",
    },
    {
        "key": "knn_k3_score",
        "label": "KNN K=3 score",
        "short_label": "KNN K=3",
        "source_model": "Clean PLM + KNN k=3 (weighted_support)",
    },
]

COLUMN_KEYS = [
    "protein_id",
    "term_id",
    "go_name",
    "go_domain",
    *[method["key"] for method in METHODS],
    "ground_truth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact all-method score matrices from PFP prediction details."
    )
    parser.add_argument(
        "--frontend-data",
        type=Path,
        default=DEFAULT_FRONTEND_DATA,
        help="Explorer data directory containing index.json and prediction details.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def excluded_organisms(frontend_index: Mapping[str, object]) -> Set[str]:
    excluded: Set[str] = set()
    for entry in frontend_index.get("frontend_excluded_benchmark_organisms", []):
        if isinstance(entry, Mapping):
            value = entry.get("taxon")
        else:
            value = entry
        if value is not None:
            excluded.add(str(value))
    return excluded


def detail_path(frontend_data: Path, value: str) -> Path:
    relative = Path(value)
    if relative.parts and relative.parts[0] == "data":
        relative = Path(*relative.parts[1:])
    return frontend_data / relative


def detail_descriptors_by_organism(
    details_index: Mapping[str, object],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    result: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for descriptor in details_index.get("details", []):
        if not descriptor.get("available"):
            continue
        organism = str(descriptor["organism"])
        model = str(descriptor["model"])
        result.setdefault(organism, {})[model] = descriptor
    return result


def truth_signature(rows: Sequence[Mapping[str, object]]) -> Set[Tuple[str, str]]:
    return {
        (str(row["protein_id"]), str(row["term_id"]))
        for row in rows
        if int(row.get("true_label", 0)) == 1
    }


def update_term_metadata(
    metadata: MutableMapping[str, Dict[str, str]],
    rows: Iterable[Mapping[str, object]],
) -> None:
    for row in rows:
        term = str(row["term_id"])
        current = metadata.setdefault(term, {"go_name": "", "go_domain": ""})
        for key in ("go_name", "go_domain"):
            value = str(row.get(key) or "")
            if value and current[key] and value != current[key]:
                raise RuntimeError(
                    f"Conflicting {key} metadata for {term}: {current[key]!r} != {value!r}."
                )
            if value:
                current[key] = value


def predicted_scores(
    rows: Iterable[Mapping[str, object]],
    valid_pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if not bool(row.get("is_predicted", False)):
            continue
        pair = (str(row["protein_id"]), str(row["term_id"]))
        if pair not in valid_pairs:
            continue
        score = float(row["score"])
        scores[pair] = max(score, scores.get(pair, score))
    return scores


def build_organism_payload(
    organism: str,
    method_payloads: Mapping[str, Mapping[str, object]],
    generated_at: str,
) -> Dict[str, object]:
    canonical_truth: Set[Tuple[str, str]] | None = None
    term_metadata: Dict[str, Dict[str, str]] = {}
    rows_by_model: Dict[str, Sequence[Mapping[str, object]]] = {}

    for method in METHODS:
        model = method["source_model"]
        payload = method_payloads[model]
        detail_rows = payload.get("rows", [])
        truth = truth_signature(detail_rows)
        if not truth:
            raise RuntimeError(f"No ground-truth rows found for {organism} / {model}.")
        if canonical_truth is None:
            canonical_truth = truth
        elif truth != canonical_truth:
            raise RuntimeError(f"Ground truth differs between methods for organism {organism}.")
        update_term_metadata(term_metadata, detail_rows)
        rows_by_model[model] = detail_rows

    assert canonical_truth is not None
    proteins = sorted({protein for protein, _term in canonical_truth})
    terms = sorted({term for _protein, term in canonical_truth})
    valid_pairs = {(protein, term) for protein in proteins for term in terms}
    score_maps = {
        method["source_model"]: predicted_scores(
            rows_by_model[method["source_model"]], valid_pairs
        )
        for method in METHODS
    }

    rows: List[List[object]] = []
    for protein in proteins:
        for term in terms:
            pair = (protein, term)
            metadata = term_metadata.get(term, {})
            rows.append(
                [
                    protein,
                    term,
                    metadata.get("go_name", ""),
                    metadata.get("go_domain", ""),
                    *[
                        score_maps[method["source_model"]].get(pair)
                        for method in METHODS
                    ],
                    1 if pair in canonical_truth else 0,
                ]
            )

    prediction_counts = {
        method["key"]: len(score_maps[method["source_model"]]) for method in METHODS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "organism": organism,
        "row_scope": (
            "Complete shared evaluation matrix: every ground-truth protein crossed "
            "with every ground-truth GO term. Missing model predictions are null."
        ),
        "column_keys": COLUMN_KEYS,
        "summary": {
            "protein_count": len(proteins),
            "term_count": len(terms),
            "row_count": len(rows),
            "ground_truth_positive_count": len(canonical_truth),
            "prediction_counts": prediction_counts,
        },
        "rows": rows,
    }


def build_comparison(frontend_data: Path) -> Dict[str, object]:
    frontend_data = frontend_data.resolve()
    frontend_index_path = frontend_data / "index.json"
    detail_index_path = frontend_data / DETAIL_INDEX_NAME
    frontend_index = read_json(frontend_index_path)
    details_index = read_json(detail_index_path)
    descriptors = detail_descriptors_by_organism(details_index)
    excluded = excluded_organisms(frontend_index)
    organisms = sorted(organism for organism in descriptors if organism not in excluded)
    generated_at = utc_now()
    output_dir = frontend_data / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    organism_descriptors = []
    for organism in organisms:
        available = descriptors[organism]
        missing = [
            method["source_model"]
            for method in METHODS
            if method["source_model"] not in available
        ]
        if missing:
            raise RuntimeError(
                f"Missing required prediction details for {organism}: {', '.join(missing)}"
            )

        method_payloads = {}
        source_detail_keys = []
        for method in METHODS:
            model = method["source_model"]
            descriptor = available[model]
            path = detail_path(frontend_data, str(descriptor["path"]))
            if not path.is_file():
                raise RuntimeError(f"Prediction detail file does not exist: {path}")
            method_payloads[model] = read_json(path)
            source_detail_keys.append(str(descriptor["detail_key"]))

        payload = build_organism_payload(organism, method_payloads, generated_at)
        output_path = output_dir / f"{organism}.json"
        output_path.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        organism_descriptors.append(
            {
                "organism": organism,
                "path": f"data/{OUTPUT_DIR_NAME}/{organism}.json",
                **payload["summary"],
                "source_detail_keys": source_detail_keys,
            }
        )

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "description": "Complete protein-GO score matrices for active PFP methods.",
        "row_scope": (
            "Every evaluated protein crossed with every evaluated GO term; null means "
            "that a model did not provide a prediction for the pair."
        ),
        "source_details_index": f"data/{DETAIL_INDEX_NAME}",
        "source_details_generated_at_utc": details_index.get("generated_at_utc"),
        "column_keys": COLUMN_KEYS,
        "methods": METHODS,
        "excluded_organisms": sorted(excluded),
        "organisms": organism_descriptors,
    }
    (frontend_data / OUTPUT_INDEX_NAME).write_text(
        json.dumps(index_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    frontend_index["pfp_score_comparison_path"] = f"data/{OUTPUT_INDEX_NAME}"
    frontend_index["pfp_score_comparison_schema_version"] = SCHEMA_VERSION
    frontend_index_path.write_text(
        json.dumps(frontend_index, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return index_payload


def main() -> None:
    args = parse_args()
    payload = build_comparison(args.frontend_data)
    print(
        json.dumps(
            {
                "index": str(args.frontend_data / OUTPUT_INDEX_NAME),
                "organisms": [
                    {
                        "organism": item["organism"],
                        "rows": item["row_count"],
                        "proteins": item["protein_count"],
                        "terms": item["term_count"],
                    }
                    for item in payload["organisms"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
