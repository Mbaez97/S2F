#!/usr/bin/env python3
"""Build exact higher-threshold CC compartment result sets.

The source ``protein_compartments.tsv`` already contains the maximum S2F
score for every protein-compartment pair retained at its source threshold.
Consequently, any threshold greater than or equal to that source threshold
can be evaluated exactly without rereading the original prediction matrix or
duplicating the large term-level evidence table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_tsv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def threshold_slug(threshold: float) -> str:
    return f"threshold_{threshold:.2f}".replace(".", "p")


def format_threshold(threshold: float) -> str:
    return f"{threshold:.15g}"


def parse_thresholds(values: Sequence[str]) -> List[float]:
    thresholds = sorted({float(value) for value in values})
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise ValueError("At least one finite threshold is required")
    return thresholds


def source_threshold(rows: Sequence[Mapping[str, str]]) -> float:
    values = {
        float(row["threshold"])
        for row in rows
        if row.get("threshold", "").strip()
    }
    if len(values) != 1:
        raise ValueError(
            "Source protein compartments must have one uniform non-empty threshold"
        )
    return next(iter(values))


def retained_at_threshold(
    rows: Sequence[Mapping[str, str]],
    threshold: float,
) -> List[Dict[str, str]]:
    retained: List[Dict[str, str]] = []
    for source in rows:
        row = dict(source)
        raw_score = row.get("s2f_score", "").strip()
        if not raw_score:
            if row.get("confidence_tier") == "curated_only":
                retained.append(row)
            continue
        if float(raw_score) + 1e-15 < threshold:
            continue
        row["threshold"] = format_threshold(threshold)
        retained.append(row)
    return retained


def rebuild_summary(
    source_rows: Sequence[Mapping[str, str]],
    retained_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    retained: MutableMapping[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in retained_rows:
        retained[row["protein_id"]].append(row)

    output: List[Dict[str, str]] = []
    for source in source_rows:
        protein_id = source["protein_id"]
        rows = sorted(
            retained.get(protein_id, []),
            key=lambda row: row["compartment_id"],
        )
        mapped = int(source.get("n_mapped_cc_rows", "0") or 0)
        status = "assigned" if rows else (
            "below_threshold" if mapped else "unresolved"
        )
        row = dict(source)
        row.update({
            "status": status,
            "compartment_ids": ";".join(
                value["compartment_id"] for value in rows
            ),
            "compartment_labels": ";".join(
                value["compartment_label"] for value in rows
            ),
            "confidence_tiers": ";".join(
                value["confidence_tier"] for value in rows
            ),
            "scores": ";".join(value["s2f_score"] for value in rows),
            "n_high_confidence_compartments": str(len(rows)),
        })
        output.append(row)
    return output


def describe_threshold(
    threshold: float,
    retained: Sequence[Mapping[str, str]],
    summary: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    protein_counts = Counter(row["protein_id"] for row in retained)
    status = Counter(row["status"] for row in summary)
    scores = [
        float(row["s2f_score"])
        for row in retained
        if row.get("s2f_score", "").strip()
    ]
    protein_total = len(summary)
    assigned = status["assigned"]
    return {
        "threshold": format_threshold(threshold),
        "retained_pairs": len(retained),
        "assigned_proteins": assigned,
        "protein_coverage": assigned / protein_total if protein_total else 0.0,
        "below_threshold_proteins": status["below_threshold"],
        "unresolved_proteins": status["unresolved"],
        "single_compartment_proteins": sum(
            count == 1 for count in protein_counts.values()
        ),
        "multi_compartment_proteins": sum(
            count > 1 for count in protein_counts.values()
        ),
        "minimum_retained_score": min(scores) if scores else "",
        "median_retained_score": statistics.median(scores) if scores else "",
        "maximum_retained_score": max(scores) if scores else "",
    }


def build(args: argparse.Namespace) -> Dict[str, object]:
    source_dir = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_path = source_dir / "protein_compartments.tsv"
    summary_path = source_dir / "protein_compartment_summary.tsv"
    metadata_path = source_dir / "run_metadata.json"
    for path in (pair_path, summary_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    pair_rows = read_tsv(pair_path)
    source_summary = read_tsv(summary_path)
    if not pair_rows or not source_summary:
        raise ValueError("Source result tables must not be empty")
    floor = source_threshold(pair_rows)
    thresholds = parse_thresholds(args.thresholds)
    below_floor = [value for value in thresholds if value + 1e-15 < floor]
    if below_floor:
        raise ValueError(
            "Thresholds below the source threshold require the original "
            f"prediction matrix: {', '.join(map(str, below_floor))} < {floor}"
        )

    summary_rows: List[Dict[str, object]] = []
    compartment_rows: List[Dict[str, object]] = []
    stability: Dict[str, Dict[str, object]] = {
        row["protein_id"]: {
            "protein_id": row["protein_id"],
            "source_status": row["status"],
            "maximum_compartment_score": "",
        }
        for row in source_summary
    }
    scores_by_protein: MutableMapping[str, List[float]] = defaultdict(list)
    for row in pair_rows:
        if row.get("s2f_score", "").strip():
            scores_by_protein[row["protein_id"]].append(float(row["s2f_score"]))
    for protein_id, values in scores_by_protein.items():
        stability[protein_id]["maximum_compartment_score"] = max(values)

    pair_fields = list(pair_rows[0])
    source_summary_fields = list(source_summary[0])
    for threshold in thresholds:
        retained = retained_at_threshold(pair_rows, threshold)
        rebuilt = rebuild_summary(source_summary, retained)
        child = output / threshold_slug(threshold)
        write_tsv(child / "protein_compartments.tsv", pair_fields, retained)
        write_tsv(
            child / "protein_compartment_summary.tsv",
            source_summary_fields,
            rebuilt,
        )
        description = describe_threshold(threshold, retained, rebuilt)
        summary_rows.append(description)

        counts = Counter(
            (row["compartment_id"], row["compartment_label"])
            for row in retained
        )
        for (compartment_id, label), count in sorted(counts.items()):
            compartment_rows.append({
                "threshold": format_threshold(threshold),
                "compartment_id": compartment_id,
                "compartment_label": label,
                "retained_pairs": count,
            })

        by_protein: MutableMapping[str, List[str]] = defaultdict(list)
        for row in retained:
            by_protein[row["protein_id"]].append(row["compartment_id"])
        count_field = f"n_compartments_at_{threshold:.2f}"
        labels_field = f"compartments_at_{threshold:.2f}"
        for protein_id in stability:
            labels = sorted(by_protein.get(protein_id, []))
            stability[protein_id][count_field] = len(labels)
            stability[protein_id][labels_field] = ";".join(labels)

        child_metadata = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "method": "exact filtering of saved maximum S2F protein-compartment scores",
            "threshold": threshold,
            "source_threshold": floor,
            "source_dir": str(source_dir),
            "source_protein_compartments_sha256": sha256_file(pair_path),
            "source_protein_summary_sha256": sha256_file(summary_path),
            "term_level_evidence": str(source_dir / "protein_compartment_evidence.tsv"),
            "term_level_evidence_duplicated": False,
            "counts": description,
        }
        (child / "run_metadata.json").write_text(
            json.dumps(child_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary_fields = [
        "threshold", "retained_pairs", "assigned_proteins",
        "protein_coverage", "below_threshold_proteins",
        "unresolved_proteins", "single_compartment_proteins",
        "multi_compartment_proteins", "minimum_retained_score",
        "median_retained_score", "maximum_retained_score",
    ]
    write_tsv(output / "threshold_summary.tsv", summary_fields, summary_rows)
    write_tsv(
        output / "compartment_counts_by_threshold.tsv",
        [
            "threshold", "compartment_id", "compartment_label",
            "retained_pairs",
        ],
        compartment_rows,
    )
    stability_fields = [
        "protein_id", "source_status", "maximum_compartment_score",
    ]
    for threshold in thresholds:
        stability_fields.extend([
            f"n_compartments_at_{threshold:.2f}",
            f"compartments_at_{threshold:.2f}",
        ])
    write_tsv(
        output / "protein_assignment_stability.tsv",
        stability_fields,
        (stability[protein_id] for protein_id in sorted(stability)),
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "source_dir": str(source_dir),
        "source_threshold": floor,
        "thresholds": thresholds,
        "method": (
            "Exact higher-threshold filtering of saved maximum S2F scores; "
            "the 5 GB term-level evidence table was not duplicated"
        ),
        "source_files": {
            "protein_compartments": {
                "path": str(pair_path), "sha256": sha256_file(pair_path),
            },
            "protein_compartment_summary": {
                "path": str(summary_path), "sha256": sha256_file(summary_path),
            },
            "run_metadata": {
                "path": str(metadata_path), "sha256": sha256_file(metadata_path),
            },
        },
        "threshold_results": summary_rows,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    metadata = build(args)
    print(json.dumps(metadata["threshold_results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
