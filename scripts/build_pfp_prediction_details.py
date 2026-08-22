#!/usr/bin/env python3
"""Build per-method prediction detail tables for the PFP metrics frontend."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

S2F_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = S2F_ROOT.parent
FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
DETAIL_DIR = FRONTEND_DATA / "pfp_prediction_details"
EXPORT_DIR = S2F_ROOT / "notebooks" / "exports" / "clean_plm_benchmark"
DEFAULT_S2F_BACKUP_ROOTS = [
    WORKSPACE_ROOT / "optional_raw_data" / "s2f_backup",
]
PREDICTION_COLUMNS = ["protein_id", "term_id", "score"]
PANDA2_HEADER_TOKENS = {"AUTHOR", "MODEL", "KEYWORDS", "END"}
TALE_PATTERN = re.compile(
    r"^(?P<protein>\S+)\s+\('(?P<term>GO:\d+)',.*\)\s+"
    r"(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

sys.path.insert(0, str(S2F_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_plm_benchmark as bench  # noqa: E402
import build_pfp_score_comparison as score_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write lazy-loaded PFP prediction detail files for the explorer."
    )
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--s2f-backup-root",
        action="append",
        default=[],
        help=(
            "Root containing strict S2F <organism>/prediction.df folders. "
            "Can be repeated. Defaults to S2F_BACKUP_ROOT, then the portable optional_raw_data copy."
        ),
    )
    return parser.parse_args()


def configured_s2f_backup_roots(cli_roots: Sequence[str]) -> List[Path]:
    roots = [Path(value).expanduser() for value in cli_roots if str(value).strip()]
    env_root = os.environ.get("S2F_BACKUP_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(DEFAULT_S2F_BACKUP_ROOTS)

    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def safe_filename(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:180] or "detail"


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    clean = df.replace({np.nan: None})
    records = []
    for row in clean.to_dict(orient="records"):
        out = {}
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            out[key] = value
        records.append(out)
    return records


def empty_prediction_frame(model: str, organism: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=[*PREDICTION_COLUMNS, "model", "organism"])
    df["score"] = df["score"].astype("float32", errors="ignore")
    return df


def normalise_prediction_frame(df: pd.DataFrame, model: str, organism: str) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_prediction_frame(model, organism)
    result = df.copy()
    result = result.rename(columns={"Protein": "protein_id", "GO ID": "term_id", "Score": "score"})
    missing = [column for column in PREDICTION_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"{model} predictions for {organism} are missing columns: {missing}")
    result = result[PREDICTION_COLUMNS].copy()
    result["protein_id"] = result["protein_id"].astype(str).str.strip()
    result["term_id"] = result["term_id"].astype(str).str.strip()
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result = result.dropna(subset=["protein_id", "term_id", "score"])
    result = result[result["protein_id"].ne("") & result["term_id"].str.match(r"^GO:\d+$")]
    if result.empty:
        return empty_prediction_frame(model, organism)
    result = (
        result.groupby(["protein_id", "term_id"], as_index=False, sort=False)["score"]
        .max()
    )
    result["model"] = model
    result["organism"] = str(organism)
    result["score"] = result["score"].astype("float32")
    return result


def s2f_prediction_candidates(organism: str, backup_roots: Sequence[Path]) -> List[Path]:
    return [root / str(organism) / "prediction.df" for root in backup_roots]


def resolve_s2f_prediction_path(organism: str, backup_roots: Sequence[Path]) -> Optional[Path]:
    for path in s2f_prediction_candidates(organism, backup_roots):
        if path.exists():
            return path
    return None


def load_s2f_score_matrix_fallback(
    organism: str,
    protein_set: Set[str],
    valid_terms: Set[str],
) -> Optional[pd.DataFrame]:
    """Recover S2F scores from the previously materialized complete score matrix."""
    path = FRONTEND_DATA / "pfp_score_comparison" / f"{organism}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = [str(value) for value in payload.get("column_keys", [])]
    required = {"protein_id", "term_id", "s2f_score"}
    if not required.issubset(columns):
        return None
    positions = {column: columns.index(column) for column in required}
    rows = []
    for values in payload.get("rows", []):
        score = values[positions["s2f_score"]]
        if score is None:
            continue
        protein = str(values[positions["protein_id"]])
        term = str(values[positions["term_id"]])
        if protein in protein_set and term in valid_terms:
            rows.append((protein, term, score))
    result = normalise_prediction_frame(
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS),
        "S2F",
        organism,
    )
    result.attrs["source_path"] = str(path)
    result.attrs["source_detail"] = (
        "Recovered from the previously materialized complete frontend score matrix "
        "because the raw S2F prediction.df backup is unavailable."
    )
    return result


def load_s2f_predictions(
    organism: str,
    protein_set: Set[str],
    valid_terms: Set[str],
    backup_roots: Sequence[Path],
    chunk_size: int = 1_000_000,
):
    path = resolve_s2f_prediction_path(organism, backup_roots)
    if path is None:
        fallback = load_s2f_score_matrix_fallback(organism, protein_set, valid_terms)
        if fallback is not None and not fallback.empty:
            return fallback, None
        candidates = ", ".join(str(candidate) for candidate in s2f_prediction_candidates(organism, backup_roots))
        return None, f"S2F prediction file is not available. Checked: {candidates}"

    frames = []
    total_rows = 0
    kept_rows = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=PREDICTION_COLUMNS,
        usecols=[0, 1, 2],
        dtype={"protein_id": str, "term_id": str, "score": np.float32},
        na_filter=False,
        chunksize=chunk_size,
        low_memory=False,
        memory_map=True,
    ):
        total_rows += len(chunk)
        mask = chunk["protein_id"].isin(protein_set) & chunk["term_id"].isin(valid_terms)
        if mask.any():
            subset = chunk.loc[mask, PREDICTION_COLUMNS].copy()
            kept_rows += len(subset)
            frames.append(subset)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    result = normalise_prediction_frame(df, "S2F", organism)
    result["source_detail"] = f"{kept_rows:,} kept rows from {total_rows:,} raw rows"
    result.attrs["source_path"] = str(path)
    return result, None


def load_tale_predictions(organism: str, protein_set: Set[str]) -> pd.DataFrame:
    rows = []
    for ontology in ["bp", "mf", "cc"]:
        path = S2F_ROOT / "competitors" / "TALE" / ontology / "output.txt"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = TALE_PATTERN.match(line.rstrip())
                if match is None:
                    continue
                protein = match.group("protein")
                if protein in protein_set:
                    rows.append((protein, match.group("term"), match.group("score")))
    return normalise_prediction_frame(pd.DataFrame(rows, columns=PREDICTION_COLUMNS), "TALE", organism)


def atgo_prediction_file(protein: str, ontology: str) -> Optional[Path]:
    result_dir = S2F_ROOT / "competitors" / "ATGO" / ontology / "result" / protein
    preferred = result_dir / f"final_combine_{ontology}_new"
    fallback = result_dir / f"final_combine_{ontology}"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return None


def load_atgo_predictions(organism: str, protein_set: Set[str]) -> pd.DataFrame:
    rows = []
    for ontology in ["BP", "MF", "CC"]:
        for protein in sorted(protein_set):
            path = atgo_prediction_file(protein, ontology)
            if path is None:
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0].startswith("GO:"):
                        rows.append((protein, parts[0], parts[-1]))
    return normalise_prediction_frame(pd.DataFrame(rows, columns=PREDICTION_COLUMNS), "ATGO", organism)


def load_panda2_predictions(organism: str, protein_set: Set[str]) -> pd.DataFrame:
    path = bench.panda2_prediction_path(organism)
    if path is None:
        return empty_prediction_frame("PANDA2", organism)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or parts[0] in PANDA2_HEADER_TOKENS:
                continue
            if len(parts) != 3 or not parts[1].startswith("GO:"):
                continue
            protein, term, score = parts
            if protein in protein_set:
                rows.append((protein, term, score))
    return normalise_prediction_frame(pd.DataFrame(rows, columns=PREDICTION_COLUMNS), "PANDA2", organism)


def load_clean_predictions(organism: str, model: str) -> pd.DataFrame:
    path = EXPORT_DIR / "clean_plm_predictions.tsv"
    if not path.exists():
        return empty_prediction_frame(model, organism)
    df = pd.read_csv(path, sep="\t")
    df = df[(df["organism"].astype(str) == str(organism)) & (df["model"] == model)].copy()
    return normalise_prediction_frame(df, model, organism)


def term_metadata(ontology, term_id: str) -> Dict[str, str]:
    try:
        term = ontology.find_term(term_id)
        return {"go_name": term.name, "go_domain": term.domain}
    except KeyError:
        return {"go_name": "", "go_domain": ""}


def materialize_detail_rows(
    organism: str,
    model: str,
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    ontology,
    max_rows: int = 0,
) -> pd.DataFrame:
    proteins = set(annotations["Protein"].astype(str))
    gold_terms = set(annotations["GO ID"].astype(str))
    pred = normalise_prediction_frame(predictions, model, organism)
    pred = pred[pred["protein_id"].isin(proteins) & pred["term_id"].isin(gold_terms)].copy()
    pred = pred[PREDICTION_COLUMNS].copy()
    pred["is_predicted"] = True

    gold = annotations[["Protein", "GO ID"]].drop_duplicates().rename(
        columns={"Protein": "protein_id", "GO ID": "term_id"}
    )
    gold["true_label"] = 1
    gold["is_ground_truth"] = True

    detail = pred.merge(gold, on=["protein_id", "term_id"], how="outer")
    detail["score"] = detail["score"].fillna(0.0).astype(float)
    detail["true_label"] = detail["true_label"].fillna(0).astype(int)
    detail["is_predicted"] = detail["is_predicted"].fillna(False).astype(bool)
    detail["is_ground_truth"] = detail["is_ground_truth"].fillna(False).astype(bool)
    detail["organism"] = str(organism)
    detail["model"] = model
    detail["score_rank_within_protein"] = (
        detail.groupby("protein_id")["score"].rank(method="first", ascending=False).astype(int)
    )
    metadata = detail["term_id"].map(lambda term_id: term_metadata(ontology, term_id))
    detail["go_name"] = metadata.map(lambda item: item["go_name"])
    detail["go_domain"] = metadata.map(lambda item: item["go_domain"])
    detail = detail.sort_values(
        ["score", "protein_id", "term_id"],
        ascending=[False, True, True],
    )
    if max_rows > 0 and len(detail) > max_rows:
        detail = detail.head(max_rows).copy()
        detail["row_limit_applied"] = True
    else:
        detail["row_limit_applied"] = False
    columns = [
        "organism",
        "model",
        "protein_id",
        "term_id",
        "go_name",
        "go_domain",
        "score",
        "true_label",
        "is_predicted",
        "is_ground_truth",
        "score_rank_within_protein",
        "row_limit_applied",
    ]
    return detail[columns]


def benchmark_rows() -> List[Dict[str, object]]:
    payload = json.loads((FRONTEND_DATA / "probe_pfp_metrics.json").read_text())
    return payload.get("benchmark_context", {}).get("rows", [])


def active_goa_path() -> Path:
    summary_path = EXPORT_DIR / "clean_plm_benchmark_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorded = summary.get("goa_path")
        if recorded and Path(str(recorded)).is_file():
            return Path(str(recorded)).resolve()
    fallback = bench.DATA_ROOT / "uniprot" / "filtered_goa"
    if fallback.is_file():
        return fallback.resolve()
    raise RuntimeError(
        "No active filtered GOA file is available. Run clean_plm_benchmark.py first "
        "or restore the path recorded in clean_plm_benchmark_summary.json."
    )


def load_predictions_for_model(
    model: str,
    organism: str,
    proteins: Set[str],
    valid_terms: Set[str],
    s2f_backup_roots: Sequence[Path],
):
    if model == "S2F":
        return load_s2f_predictions(organism, proteins, valid_terms, s2f_backup_roots)
    if model == "TALE":
        return load_tale_predictions(organism, proteins), None
    if model == "ATGO":
        return load_atgo_predictions(organism, proteins), None
    if model == "PANDA2":
        return load_panda2_predictions(organism, proteins), None
    if model.startswith("Clean PLM"):
        return load_clean_predictions(organism, model), None
    return None, f"No prediction detail loader is configured for {model}."


def update_index_json(details_path: str) -> None:
    index_path = FRONTEND_DATA / "index.json"
    payload = json.loads(index_path.read_text())
    payload["pfp_prediction_details_path"] = details_path
    payload["pfp_prediction_details_schema_version"] = 1
    index_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))


def main() -> None:
    args = parse_args()
    s2f_backup_roots = configured_s2f_backup_roots(args.s2f_backup_root)
    print("S2F prediction roots:", ", ".join(str(root) for root in s2f_backup_roots))
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    goa_path = active_goa_path()
    go_obo = S2F_ROOT / "go.obo"
    evaluation_sets, evaluation_summary = bench.build_evaluation_sets(goa_path)
    evaluation_summary.to_csv(EXPORT_DIR / "prediction_detail_evaluation_sets.csv", index=False)

    summaries = []
    rows_by_organism = defaultdict(list)
    for row in benchmark_rows():
        rows_by_organism[str(row["organism"])].append(row)

    for organism, rows in sorted(rows_by_organism.items()):
        proteins = evaluation_sets[organism]
        annotations, ontology, _organism_name = bench.prepare_ground_truth(goa_path, go_obo, organism, proteins)
        valid_terms = set(annotations["GO ID"].astype(str))
        for metric_row in rows:
            model = str(metric_row.get("model") or metric_row.get("method"))
            detail_key = f"{organism}::{model}"
            detail_file = DETAIL_DIR / f"{safe_filename(organism, model)}.json"
            predictions, unavailable_reason = load_predictions_for_model(
                model,
                organism,
                proteins,
                valid_terms,
                s2f_backup_roots,
            )
            summary = {
                "detail_key": detail_key,
                "organism": organism,
                "model": model,
                "method_family": metric_row.get("method_family") or metric_row.get("model"),
                "available": unavailable_reason is None,
                "path": f"data/pfp_prediction_details/{detail_file.name}",
                "row_scope": "nonzero predictions plus ground-truth positives absent from the predictions",
            }
            if unavailable_reason is not None:
                summary["available"] = False
                summary["unavailable_reason"] = unavailable_reason
                detail_payload = {
                    "schema_version": 1,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "rows": [],
                }
            else:
                source_path = predictions.attrs.get("source_path") if hasattr(predictions, "attrs") else None
                if source_path:
                    summary["source_path"] = source_path
                detail = materialize_detail_rows(
                    organism,
                    model,
                    predictions,
                    annotations,
                    ontology,
                    max_rows=args.max_rows_per_file,
                )
                summary.update(
                    {
                        "available": True,
                        "row_count": int(len(detail)),
                        "predicted_pair_count": int(detail["is_predicted"].sum()),
                        "ground_truth_pair_count": int(detail["is_ground_truth"].sum()),
                        "row_limit_applied": bool(detail["row_limit_applied"].any()) if not detail.empty else False,
                    }
                )
                detail_payload = {
                    "schema_version": 1,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "rows": dataframe_records(detail),
                }
            detail_file.write_text(json.dumps(detail_payload, separators=(",", ":"), allow_nan=False))
            summaries.append(summary)
            print(f"{organism} {model}: {'unavailable' if not summary['available'] else summary.get('row_count', 0)} detail rows")

    index_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Per-method prediction detail files for the PFP benchmark view.",
        "row_scope": "Each detail file contains all nonzero prediction pairs plus ground-truth positives absent from the predictions.",
        "details": summaries,
    }
    index_path = FRONTEND_DATA / "pfp_prediction_details_index.json"
    index_path.write_text(json.dumps(index_payload, separators=(",", ":"), allow_nan=False))
    update_index_json("data/pfp_prediction_details_index.json")
    comparison_index = score_comparison.build_comparison(FRONTEND_DATA)
    print(
        json.dumps(
            {
                "detail_files": len(summaries),
                "index": str(index_path),
                "comparison_organisms": [
                    item["organism"] for item in comparison_index["organisms"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
