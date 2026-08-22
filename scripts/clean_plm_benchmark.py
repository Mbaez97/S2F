#!/usr/bin/env python3
"""Benchmark clean PLM GO-transfer baselines against existing competitors.

The script intentionally keeps this experiment outside the S2F prediction
pipeline. It reuses the embedding/cache logic from ``plm.py`` and writes
frontend-ready benchmark rows for the ESM GO explorer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import argparse
import csv
import gc
import hashlib
import json
import math
import re
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="`np\\.(float|int|bool)` is a deprecated alias", category=DeprecationWarning)
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


S2F_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = S2F_ROOT.parent


def resolve_bundle_root(repository_root: Path = S2F_ROOT) -> Path:
    """Locate experiment data in either the source tree or portable bundle.

    Source checkout layout::

        S2F/transfer/esm_pfp_work_2026-07-01/data/

    Portable bundle layout::

        esm_pfp_work_2026-07-01/repo/
        esm_pfp_work_2026-07-01/data/

    The portable layout deliberately keeps the large data directory beside the
    code directory, so copying the bundle to another disk does not retain an
    office-machine path dependency.
    """
    candidates = [
        repository_root / "transfer" / "esm_pfp_work_2026-07-01",
        repository_root.parent,
    ]
    for candidate in candidates:
        if (candidate / "data" / "uniprot" / "filtered_goa").is_file():
            return candidate.resolve()
    return candidates[0]


TRANSFER_ROOT = resolve_bundle_root()
DATA_ROOT = TRANSFER_ROOT / "data"
DEFAULT_MODEL_DIR = (
    DATA_ROOT / "models" / "huggingface"
    if (DATA_ROOT / "models" / "huggingface").is_dir()
    else S2F_ROOT / ".cache" / "huggingface"
)
FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
EXPORT_DIR = S2F_ROOT / "notebooks" / "exports" / "clean_plm_benchmark"
DEFAULT_BLACKLIST_DIRS = [
    DATA_ROOT / "blacklists",
]
TARGET_HEADER_TAXON = re.compile(r"(?:^|\s)OX=(\d+)(?:\s|$)")

sys.path.insert(0, str(S2F_ROOT))
import plm  # noqa: E402
from GOTool import GeneOntology  # noqa: E402
from Measures.measures import HX_py  # noqa: E402
from clean_plm_method_registry import (  # noqa: E402
    ACTIVE_KDE_METHOD_KEY,
    ACTIVE_KDE_MODEL,
    LEGACY_ADAPTIVE_KDE_MODEL,
)


ORGANISMS = ["83333", "223283", "1111708"]
FRONTEND_EXCLUDED_ORGANISMS = {"223283"}
K_VALUES = [3, 5, 7, 10]
DEFAULT_RADII = [0.01]
DEFAULT_STRATEGIES = ["knn", "kde"]
TRANSFER_STRATEGIES = ["knn", "radius", "kde", "adaptive_kde"]
KDE_CALIBRATION_SIZE = 512
KDE_MAX_NEIGHBORS = 8192
KDE_WEIGHT_FLOOR = 1e-6
KDE_SELECTION_METRIC = "overall::F_max"
KDE_SELECTION_METRICS = ["overall::F_max", "overall::AUPR", "overall::AUC"]
KDE_RANDOM_SEED = 20260713
KDE_GRID_NEIGHBORS = 50
KDE_GRID_CANDIDATES = 12
KDE_CALIBRATION_SCORE_DECIMALS = 2
EXPECTED_TARGET_PROTEINS = 83_003
EXPECTED_ESM1B_DIMENSION = 1_280
ADAPTIVE_KDE_CALIBRATION_REPEATS = 3
ADAPTIVE_KDE_CALIBRATION_SIZE = 384
ADAPTIVE_KDE_DISTANCE_BINS = 5
ADAPTIVE_KDE_NEIGHBOR_COUNTS = [3, 5]
ADAPTIVE_KDE_BANDWIDTH_SCALES = [1.0, 1.25]
ADAPTIVE_KDE_RELATIVE_WEIGHT_FLOOR = 0.01
ADAPTIVE_KDE_MIN_NEIGHBORS = 3
ADAPTIVE_KDE_CONFIDENCE_DISTANCE_SCALES = [0.005, 0.01, 0.02]
ADAPTIVE_KDE_FMAX_SE_MULTIPLIER = 1.0
ADAPTIVE_KDE_CALIBRATION_POOL_MULTIPLIER = 3
SCORE_MODES = ["binary", "support_fraction", "max_similarity", "weighted_support"]
SCORE_MODE_LABELS = {
    "binary": "binary",
    "support_fraction": "support_fraction",
    "max_similarity": "max_similarity",
    "weighted_support": "weighted_support",
}
EVIDENCE_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
METRIC_COLUMNS = [
    "overall::AUC",
    "overall::AUPR",
    "overall::F_max",
    "overall::smin",
    "AUC per-gene",
    "AUPR per-gene",
    "F_max per-gene",
    "smin per-gene",
    "AUC per-term",
    "AUPR per-term",
    "F_max per-term",
    "smin per-term",
]
PANDA2_HEADER_TOKENS = {"AUTHOR", "MODEL", "KEYWORDS", "END"}
TALE_PATTERN = re.compile(
    r"^(?P<protein>\S+)\s+\('(?P<term>GO:\d+)',.*\)\s+"
    r"(?P<score>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def log(message: str) -> None:
    print(f"[clean-plm] {message}", flush=True)


@dataclass
class ArgsForPlm:
    model_name: str
    model_dir: str
    device: str
    local_files_only: bool
    long_sequence_mode: str
    long_window_size: int
    long_overlap: int
    batch_tokens: int
    query_chunk_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean PLM KNN, radius, and KDE benchmark rows for the PFP metrics frontend."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Backward-compatible single cosine-distance radius. Prefer --radii for multiple values.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=None,
        help="Cosine-distance radii for radius-based transfer. A radius R selects neighbors with cosine distance <= R.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=TRANSFER_STRATEGIES,
        default=None,
        help=(
            "Clean PLM transfer strategies to run. Defaults to KNN plus shared-bandwidth Gaussian KDE. "
            "Passing --radius/--radii also enables the fixed-radius strategy for backward compatibility."
        ),
    )
    parser.add_argument(
        "--kde-calibration-size",
        type=int,
        default=KDE_CALIBRATION_SIZE,
        help="Number of experimentally annotated Swiss-Prot pseudo-queries used for source-only bandwidth calibration.",
    )
    parser.add_argument(
        "--kde-bandwidths",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional Gaussian bandwidth candidates in normalized-Euclidean units. "
            "By default candidates are derived from source-only calibration distances."
        ),
    )
    parser.add_argument(
        "--kde-max-neighbors",
        type=int,
        default=KDE_MAX_NEIGHBORS,
        help=(
            "Safety cap for Gaussian contributors per query. The run fails instead of silently truncating "
            "when the relative kernel contour contains more donors."
        ),
    )
    parser.add_argument(
        "--kde-weight-floor",
        type=float,
        default=KDE_WEIGHT_FLOOR,
        help="Discard donors whose Gaussian weight is below this fraction of the query's peak donor weight.",
    )
    parser.add_argument(
        "--kde-selection-metric",
        choices=KDE_SELECTION_METRICS,
        default=KDE_SELECTION_METRIC,
        help="Source-only PFP metric maximized when selecting the KDE bandwidth.",
    )
    parser.add_argument(
        "--adaptive-kde-calibration-repeats",
        type=int,
        default=ADAPTIVE_KDE_CALIBRATION_REPEATS,
        help="Repeated source-only calibration samples used for adaptive KDE selection.",
    )
    parser.add_argument(
        "--adaptive-kde-calibration-size",
        type=int,
        default=ADAPTIVE_KDE_CALIBRATION_SIZE,
        help="Pseudo-query proteins per repeated adaptive KDE calibration sample.",
    )
    parser.add_argument(
        "--adaptive-kde-distance-bins",
        type=int,
        default=ADAPTIVE_KDE_DISTANCE_BINS,
        help="Nearest-neighbor-distance bins used to match source pseudo-queries to unlabeled benchmark queries.",
    )
    parser.add_argument(
        "--adaptive-kde-neighbor-counts",
        type=int,
        nargs="+",
        default=ADAPTIVE_KDE_NEIGHBOR_COUNTS,
        help="Neighbor ranks used to derive each query-local adaptive KDE bandwidth.",
    )
    parser.add_argument(
        "--adaptive-kde-bandwidth-scales",
        type=float,
        nargs="+",
        default=ADAPTIVE_KDE_BANDWIDTH_SCALES,
        help="Multipliers applied to query-local bandwidths derived from the selected neighbor rank.",
    )
    parser.add_argument(
        "--adaptive-kde-relative-weight-floor",
        type=float,
        default=ADAPTIVE_KDE_RELATIVE_WEIGHT_FLOOR,
        help="Retain a neighbor when its Gaussian weight is at least this fraction of the query's nearest-neighbor weight.",
    )
    parser.add_argument(
        "--adaptive-kde-min-neighbors",
        type=int,
        default=ADAPTIVE_KDE_MIN_NEIGHBORS,
        help="Minimum closest donors retained when the adaptive KDE contour is smaller.",
    )
    parser.add_argument(
        "--adaptive-kde-confidence-distance-scales",
        type=float,
        nargs="+",
        default=ADAPTIVE_KDE_CONFIDENCE_DISTANCE_SCALES,
        help="Nearest-neighbor cosine-distance scales evaluated for adaptive KDE confidence shrinkage.",
    )
    parser.add_argument(
        "--adaptive-kde-fmax-se-multiplier",
        type=float,
        default=ADAPTIVE_KDE_FMAX_SE_MULTIPLIER,
        help="Fmax standard-error tolerance for coverage-aware adaptive KDE selection.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the non-frontend KNN/global-KDE/adaptive-KDE ablation outputs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=KDE_RANDOM_SEED,
        help="Deterministic seed for selecting KDE calibration pseudo-queries.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--force-query-embeddings", action="store_true")
    parser.add_argument("--skip-missing-embeddings", action="store_true")
    parser.add_argument("--output-dir", default=str(EXPORT_DIR))
    parser.add_argument(
        "--goa-path",
        default="",
        help="Optional filtered GOA path. Defaults to <bundle>/data/uniprot/filtered_goa.",
    )
    parser.add_argument(
        "--target-fasta",
        default="",
        help="Swiss-Prot FASTA used to validate or build the target embedding cache.",
    )
    parser.add_argument(
        "--target-cache-dir",
        default="",
        help="Optional target embedding cache directory.",
    )
    parser.add_argument(
        "--build-target-cache",
        action="store_true",
        help="Build a missing target cache from --target-fasta. CUDA is required.",
    )
    parser.add_argument(
        "--blacklist-dir",
        default="",
        help=(
            "Directory containing one <test-taxon>.blacklist file per benchmark organism. "
            "Defaults to data/blacklists in the source or portable bundle."
        ),
    )
    parser.add_argument(
        "--score-modes",
        nargs="+",
        choices=SCORE_MODES,
        default=["weighted_support"],
        help="Clean PLM GO-term scoring modes to evaluate.",
    )
    parser.add_argument(
        "--skip-frontend-refresh",
        action="store_true",
        help="Write benchmark outputs without updating the static explorer data files.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Hugging Face cache directory containing ESM-1b.",
    )
    return parser.parse_args()


def selected_strategies(args: argparse.Namespace) -> List[str]:
    strategies = list(args.strategies or DEFAULT_STRATEGIES)
    if (args.radius is not None or args.radii is not None) and "radius" not in strategies:
        strategies.append("radius")
    return [strategy for strategy in TRANSFER_STRATEGIES if strategy in strategies]


def selected_radii(args: argparse.Namespace) -> List[float]:
    values = args.radii if args.radii is not None else ([args.radius] if args.radius is not None else DEFAULT_RADII)
    radii = sorted({float(value) for value in values})
    invalid = [value for value in radii if value < 0 or value > 2]
    if invalid:
        raise ValueError(f"Cosine-distance radii must be in [0, 2]. Invalid values: {invalid}")
    return radii


def validate_kde_args(args: argparse.Namespace) -> None:
    if args.kde_calibration_size < 2:
        raise ValueError("--kde-calibration-size must be at least 2.")
    if args.kde_max_neighbors < 1:
        raise ValueError("--kde-max-neighbors must be at least 1.")
    if not 0.0 < args.kde_weight_floor < 1.0:
        raise ValueError("--kde-weight-floor must be strictly between 0 and 1.")
    if args.kde_bandwidths is not None and any(value <= 0 for value in args.kde_bandwidths):
        raise ValueError("Every --kde-bandwidths value must be positive.")
    if args.adaptive_kde_calibration_repeats < 2:
        raise ValueError("--adaptive-kde-calibration-repeats must be at least 2 for a standard error.")
    if args.adaptive_kde_calibration_size < 2:
        raise ValueError("--adaptive-kde-calibration-size must be at least 2.")
    if args.adaptive_kde_distance_bins < 1:
        raise ValueError("--adaptive-kde-distance-bins must be at least 1.")
    if not args.adaptive_kde_neighbor_counts or min(args.adaptive_kde_neighbor_counts) < 1:
        raise ValueError("--adaptive-kde-neighbor-counts must contain positive ranks.")
    if not args.adaptive_kde_bandwidth_scales or min(args.adaptive_kde_bandwidth_scales) <= 0:
        raise ValueError("--adaptive-kde-bandwidth-scales must contain positive values.")
    if not 0.0 < args.adaptive_kde_relative_weight_floor < 1.0:
        raise ValueError("--adaptive-kde-relative-weight-floor must be strictly between 0 and 1.")
    if args.adaptive_kde_min_neighbors < 1:
        raise ValueError("--adaptive-kde-min-neighbors must be at least 1.")
    if not args.adaptive_kde_confidence_distance_scales or min(args.adaptive_kde_confidence_distance_scales) <= 0:
        raise ValueError("--adaptive-kde-confidence-distance-scales must contain positive values.")
    if args.adaptive_kde_fmax_se_multiplier < 0:
        raise ValueError("--adaptive-kde-fmax-se-multiplier cannot be negative.")


def read_id_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return {line.strip().split()[0] for line in handle if line.strip()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_blacklist_dir(value: str = "") -> Path:
    """Locate the per-organism taxon blacklists required for clean transfer."""
    candidates = [Path(value).expanduser()] if str(value).strip() else DEFAULT_BLACKLIST_DIRS
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / f"{organism}.blacklist").is_file() for organism in ORGANISMS):
            return candidate.resolve()
    expected = ", ".join(f"{organism}.blacklist" for organism in ORGANISMS)
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "Clean PLM transfer requires per-organism taxon blacklists before neighbour selection. "
        f"Expected {expected}; checked: {checked}. Pass --blacklist-dir to provide them."
    )


def read_taxon_blacklist(path: Path) -> Set[str]:
    values = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            token = raw_line.strip().split("#", 1)[0].strip()
            if token:
                values.add(token.split()[0])
    if not values:
        raise RuntimeError(f"Blacklist is empty: {path}")
    return values


def load_organism_blacklists(blacklist_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Path]]:
    blacklists: Dict[str, Set[str]] = {}
    paths: Dict[str, Path] = {}
    for organism in ORGANISMS:
        path = blacklist_dir / f"{organism}.blacklist"
        if not path.is_file():
            raise RuntimeError(f"Missing blacklist for test organism {organism}: {path}")
        blacklists[organism] = read_taxon_blacklist(path)
        paths[organism] = path
    return blacklists, paths


def target_taxa_from_cache(target_cache: plm.EmbeddingCache) -> List[str]:
    taxa = []
    missing = []
    for protein_id, header in zip(target_cache.ids, target_cache.headers):
        match = TARGET_HEADER_TAXON.search(str(header))
        if match is None:
            missing.append(str(protein_id))
            taxa.append("")
        else:
            taxa.append(match.group(1))
    if missing:
        raise RuntimeError(
            "Cannot apply organism blacklist before neighbor selection because Swiss-Prot target headers "
            f"lack OX taxonomy for {len(missing)} proteins (for example {missing[:5]})."
        )
    return taxa


def build_source_exclusions_by_organism(
    target_cache: plm.EmbeddingCache,
    benchmark_exclusion_ids: Set[str],
    blacklists: Dict[str, Set[str]],
    blacklist_paths: Dict[str, Path],
) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    """Build separate source-ID exclusions for every benchmark organism.

    The blacklist is translated from taxon IDs to source accessions *before*
    any cosine search, so a forbidden protein cannot occupy a KNN, radius, or
    KDE neighbor slot and cannot contribute a GO term.
    """
    target_ids = [str(protein_id) for protein_id in target_cache.ids]
    target_id_set = set(target_ids)
    target_taxa = target_taxa_from_cache(target_cache)
    benchmark_in_target = benchmark_exclusion_ids & target_id_set
    exclusions: Dict[str, Set[str]] = {}
    audit_rows = []
    for organism in ORGANISMS:
        blacklist = blacklists[organism]
        blocked_source_ids = {
            protein_id
            for protein_id, taxon in zip(target_ids, target_taxa)
            if taxon in blacklist
        }
        combined = set(benchmark_in_target) | blocked_source_ids
        if len(combined) >= len(target_ids):
            raise RuntimeError(f"No Swiss-Prot donors remain after blacklist filtering for {organism}.")
        exclusions[organism] = combined
        audit_rows.append(
            {
                "organism": organism,
                "blacklist_filter_enabled": True,
                "blacklist_path": str(blacklist_paths[organism]),
                "blacklist_sha256": file_sha256(blacklist_paths[organism]),
                "blacklist_taxa_count": len(blacklist),
                "blacklisted_source_protein_count": len(blocked_source_ids),
                "benchmark_exclusion_count": len(benchmark_in_target),
                "overlap_with_benchmark_exclusions": len(blocked_source_ids & benchmark_in_target),
                "source_exclusion_count": len(combined),
                "eligible_source_protein_count": len(target_ids) - len(combined),
            }
        )
    return exclusions, pd.DataFrame(audit_rows)


def read_atgo_test_proteins() -> Set[str]:
    proteins: Set[str] = set()
    for ontology in ["BP", "MF", "CC"]:
        proteins.update(read_id_list(S2F_ROOT / "competitors" / "ATGO" / ontology / "test_gene_list"))
    return proteins


def panda2_prediction_path(organism: str) -> Optional[Path]:
    root = S2F_ROOT / "competitors" / "PANDA2" / "by_dataset" / organism
    expected = root / f"{organism}.2" / "panda2_prediction.txt"
    if expected.exists():
        return expected
    candidates = sorted(root.glob("*/panda2_prediction.txt")) if root.exists() else []
    return candidates[0] if candidates else None


def read_panda2_test_proteins(organism: str) -> Set[str]:
    input_dir = S2F_ROOT / "competitors" / "PANDA2" / "by_dataset" / organism / organism
    if input_dir.exists():
        proteins = {path.stem for path in input_dir.glob("*.fasta")}
        if proteins:
            return proteins

    path = panda2_prediction_path(organism)
    if path is None:
        return set()

    proteins: Set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] not in PANDA2_HEADER_TOKENS and parts[1].startswith("GO:"):
                proteins.add(parts[0])
    return proteins


def read_tale_test_proteins() -> Set[str]:
    proteins: Set[str] = set()
    for ontology in ["bp", "mf", "cc"]:
        path = S2F_ROOT / "competitors" / "TALE" / ontology / "output.txt"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = TALE_PATTERN.match(line.rstrip())
                if match is not None:
                    proteins.add(match.group("protein"))
    return proteins


def load_goa_annotations_for_taxon(goa_path: Path, taxon_id: str) -> pd.DataFrame:
    column_names = [
        "DB",
        "DB Object ID",
        "DB Object Symbol",
        "Qualifier",
        "GO ID",
        "DB Reference",
        "Evidence Code",
        "With",
        "Aspect",
        "DB Object Name",
        "Synonym",
        "DB Object Type",
        "Taxon",
        "Date",
        "Assigned By",
        "Annotation Extension",
        "Gene Product Form ID",
    ]
    taxon_pattern = fr"(?:^|\|)taxon:{taxon_id}(?:$|\|)"
    not_pattern = r"(?:^|\|)NOT(?:$|\|)"
    chunks = []
    for chunk in pd.read_csv(
        goa_path,
        sep="\t",
        comment="!",
        header=None,
        names=column_names,
        dtype=str,
        chunksize=200_000,
        low_memory=False,
    ):
        mask = chunk["Taxon"].fillna("").str.contains(taxon_pattern, regex=True)
        if not mask.any():
            continue
        subset = chunk.loc[mask]
        subset = subset[~subset["Qualifier"].fillna("").str.contains(not_pattern, regex=True)]
        subset = subset[subset["Evidence Code"].isin(EVIDENCE_CODES)]
        if subset.empty:
            continue
        trimmed = subset[["DB Object ID", "GO ID"]].drop_duplicates()
        trimmed = trimmed.rename(columns={"DB Object ID": "Protein"})
        trimmed["Score"] = 1.0
        chunks.append(trimmed)
    if not chunks:
        return pd.DataFrame(columns=["Protein", "GO ID", "Score"])
    return pd.concat(chunks, ignore_index=True).drop_duplicates()


def build_evaluation_sets(goa_path: Path) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    atgo_proteins = read_atgo_test_proteins()
    tale_proteins = read_tale_test_proteins()
    evaluation_sets: Dict[str, Set[str]] = {}
    rows = []
    for organism in ORGANISMS:
        goa_annotations = load_goa_annotations_for_taxon(goa_path, organism)
        goa_proteins = set(goa_annotations["Protein"].astype(str))
        panda2_proteins = read_panda2_test_proteins(organism)
        model_sets = []
        for model, proteins in [("ATGO", atgo_proteins), ("PANDA2", panda2_proteins)]:
            overlap = proteins & goa_proteins
            if overlap:
                model_sets.append(overlap)
            rows.append(
                {
                    "organism": organism,
                    "source_model": model,
                    "raw_test_proteins": len(proteins),
                    "goa_overlap_proteins": len(overlap),
                    "goa_proteins": len(goa_proteins),
                    "protein_set_mode": "intersection",
                }
            )
        selected = set.intersection(*model_sets)
        evaluation_sets[organism] = selected
        rows.append(
            {
                "organism": organism,
                "source_model": "SHARED_EVALUATION_SET",
                "raw_test_proteins": math.nan,
                "goa_overlap_proteins": len(selected),
                "goa_proteins": len(goa_proteins),
                "protein_set_mode": "intersection",
            }
        )
    return evaluation_sets, pd.DataFrame(rows)


def fasta_records_by_id(paths: Sequence[Path]) -> Dict[str, plm.FastaRecord]:
    records: Dict[str, plm.FastaRecord] = {}
    for path in paths:
        if path.exists():
            for record in plm.read_fasta(path, "uniprot"):
                records[record.protein_id] = record
    return records


def write_query_fasta(organism: str, proteins: Set[str], output_dir: Path) -> Path:
    candidate_paths = [
        WORKSPACE_ROOT / "PANDA2" / "runs" / "panda2_cafa3" / "filtered_inputs" / "by_dataset" / f"{organism}.fasta",
        WORKSPACE_ROOT / "PANDA2" / "runs" / "panda2_swissprot_2016" / "filtered_inputs" / "by_dataset" / f"{organism}.fasta",
        WORKSPACE_ROOT / "PANDA2" / "data" / "testset" / f"{organism}.fasta",
    ]
    records = fasta_records_by_id(candidate_paths)
    missing = sorted(proteins - set(records))
    if missing:
        per_protein_dir = WORKSPACE_ROOT / "PANDA2" / "data" / "testset" / organism
        records.update(fasta_records_by_id([path for protein in missing for path in [per_protein_dir / f"{protein}.fasta"]]))
        missing = sorted(proteins - set(records))
    if missing:
        raise RuntimeError(f"Missing FASTA sequence(s) for {organism}: {missing[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{organism}.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for protein in sorted(proteins):
            record = records[protein]
            handle.write(f">{record.protein_id}\n")
            sequence = record.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")
    return fasta_path


def load_subset_cache(source_cache: plm.EmbeddingCache, proteins: Set[str], cache_dir: Path) -> Optional[plm.EmbeddingCache]:
    index = {protein_id: i for i, protein_id in enumerate(source_cache.ids)}
    missing = sorted(proteins - set(index))
    if missing:
        return None
    records = []
    vectors = []
    for protein in sorted(proteins):
        i = index[protein]
        records.append(plm.FastaRecord(source_cache.headers[i], protein, "X" * int(source_cache.lengths[i])))
        vectors.append(np.asarray(source_cache.embeddings[i], dtype=np.float32))
    metadata = dict(source_cache.metadata)
    metadata["subset_source"] = str(source_cache.path)
    metadata["subset_proteins"] = len(records)
    return plm.write_embedding_cache(cache_dir, records, np.vstack(vectors), metadata)


def query_cache_for_organism(
    organism: str,
    proteins: Set[str],
    output_dir: Path,
    plm_args: ArgsForPlm,
    force: bool,
    skip_missing: bool,
) -> Optional[plm.EmbeddingCache]:
    cache_dir = output_dir / "embeddings" / f"query_{organism}"
    loaded = plm.load_embedding_cache(cache_dir, expected_metadata=None, validate=False)
    if loaded is not None and not force and set(proteins).issubset(set(loaded.ids)):
        log(f"Using cached query embeddings for {organism}: {len(loaded.ids)} proteins.")
        return loaded

    if organism == "83333":
        source_dir = DATA_ROOT / "embeddings" / "ecoli_83333_query_embeddings"
        source_cache = plm.load_embedding_cache(source_dir, expected_metadata=None, validate=False)
        if source_cache is not None:
            subset = load_subset_cache(source_cache, proteins, cache_dir)
            if subset is not None:
                log(f"Subselected cached 83333 query embeddings: {len(subset.ids)} proteins.")
                return subset

    if skip_missing:
        return None

    fasta_path = write_query_fasta(organism, proteins, output_dir / "query_fastas")
    log(f"Computing query embeddings for {organism}: {len(proteins)} proteins from {fasta_path}.")
    records = plm.read_fasta(fasta_path, "uniprot")
    metadata = plm.build_cache_metadata(
        [fasta_path],
        plm_args.model_name,
        "uniprot",
        plm_args.long_sequence_mode,
        plm_args.long_window_size,
        plm_args.long_overlap,
    )
    return plm.get_or_create_embedding_cache(cache_dir, records, metadata, plm_args, force=force)


def validate_target_records(records: Sequence[plm.FastaRecord], fasta_path: Path) -> None:
    ids = [record.protein_id for record in records]
    if len(ids) != EXPECTED_TARGET_PROTEINS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGET_PROTEINS} Swiss-Prot target proteins in {fasta_path}, found {len(ids)}."
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Target FASTA contains duplicate protein accessions: {fasta_path}")
    missing_taxonomy = [record.protein_id for record in records if TARGET_HEADER_TAXON.search(record.header) is None]
    if missing_taxonomy:
        raise RuntimeError(
            f"Target FASTA lacks OX taxonomy for {len(missing_taxonomy)} proteins "
            f"(for example {missing_taxonomy[:5]}): {fasta_path}"
        )


def target_cache_matches_fasta_content(
    target_cache: plm.EmbeddingCache,
    records: Sequence[plm.FastaRecord],
    expected_metadata: Dict[str, object],
) -> bool:
    """Validate a cache whose FASTA mount prefix changed without trusting the path."""
    for key, expected_value in expected_metadata.items():
        if key == "fasta":
            continue
        if target_cache.metadata.get(key) != expected_value:
            return False
    cached_fasta = target_cache.metadata.get("fasta")
    expected_fasta = expected_metadata.get("fasta")
    if not isinstance(cached_fasta, list) or not isinstance(expected_fasta, list):
        return False
    cached_fingerprints = [
        (item.get("size"), item.get("mtime_ns"))
        for item in cached_fasta
        if isinstance(item, dict)
    ]
    expected_fingerprints = [
        (item.get("size"), item.get("mtime_ns"))
        for item in expected_fasta
        if isinstance(item, dict)
    ]
    if cached_fingerprints != expected_fingerprints:
        return False
    return (
        target_cache.ids == [record.protein_id for record in records]
        and target_cache.lengths == [len(record.sequence) for record in records]
        and target_cache.headers == [record.header for record in records]
    )


def load_or_build_target_cache(
    args: argparse.Namespace,
    output_dir: Path,
    plm_args: ArgsForPlm,
) -> Tuple[plm.EmbeddingCache, Path, Optional[Path]]:
    """Load the configured donor cache or build it from the validated 83k FASTA."""
    target_fasta = Path(args.target_fasta).expanduser().resolve() if args.target_fasta else None
    expected_metadata = None
    records: Optional[List[plm.FastaRecord]] = None
    if target_fasta is not None:
        if not target_fasta.is_file():
            raise RuntimeError(f"Missing target FASTA: {target_fasta}")
        records = plm.read_fasta(target_fasta, "uniprot")
        validate_target_records(records, target_fasta)
        expected_metadata = plm.build_cache_metadata(
            [target_fasta],
            plm_args.model_name,
            "uniprot",
            plm_args.long_sequence_mode,
            plm_args.long_window_size,
            plm_args.long_overlap,
        )

    if args.target_cache_dir:
        target_cache_dir = Path(args.target_cache_dir).expanduser().resolve()
    elif args.build_target_cache:
        if expected_metadata is None:
            raise RuntimeError("--build-target-cache requires --target-fasta.")
        target_cache_dir = (
            output_dir / "embeddings" / f"source_target_{plm.metadata_key(expected_metadata)}"
        ).resolve()
    else:
        target_cache_dir = (DATA_ROOT / "embeddings" / "source_target_57e9174250d35f0d").resolve()

    target_cache = plm.load_embedding_cache(
        target_cache_dir,
        expected_metadata=expected_metadata,
        validate=expected_metadata is not None,
    )
    if (
        target_cache is None
        and args.target_cache_dir
        and expected_metadata is not None
        and records is not None
    ):
        path_agnostic_cache = plm.load_embedding_cache(target_cache_dir, validate=False)
        if path_agnostic_cache is not None and target_cache_matches_fasta_content(
            path_agnostic_cache,
            records,
            expected_metadata,
        ):
            log(
                "Target-cache FASTA path changed mount prefix, but model settings, file fingerprint, "
                "accession order, headers, and sequence lengths match exactly; reusing the cache."
            )
            target_cache = path_agnostic_cache
    if target_cache is None and args.build_target_cache:
        if records is None or expected_metadata is None:
            raise RuntimeError("Target-cache construction requires validated FASTA records and metadata.")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The S2F environment must provide PyTorch to build the target cache.") from exc
        if args.device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(
                "Building the 83,003-protein ESM-1b target cache requires working CUDA; "
                "PyTorch does not currently see an NVIDIA GPU."
            )
        log(f"Building Swiss-Prot target embeddings from {target_fasta} into {target_cache_dir}.")
        target_cache = plm.get_or_create_embedding_cache(
            target_cache_dir,
            records,
            expected_metadata,
            plm_args,
            force=False,
        )
    if target_cache is None:
        raise RuntimeError(
            f"Missing target embedding cache: {target_cache_dir}. "
            "Provide --target-cache-dir or use --build-target-cache with --target-fasta."
        )
    if len(target_cache.ids) != EXPECTED_TARGET_PROTEINS:
        raise RuntimeError(
            f"Target cache contains {len(target_cache.ids)} proteins; expected {EXPECTED_TARGET_PROTEINS}."
        )
    if target_cache.embeddings.ndim != 2 or target_cache.embeddings.shape[1] != EXPECTED_ESM1B_DIMENSION:
        raise RuntimeError(
            f"Target cache shape is {target_cache.embeddings.shape}; expected "
            f"({EXPECTED_TARGET_PROTEINS}, {EXPECTED_ESM1B_DIMENSION})."
        )
    if len(set(target_cache.ids)) != len(target_cache.ids):
        raise RuntimeError(f"Target cache contains duplicate protein accessions: {target_cache_dir}")
    target_taxa_from_cache(target_cache)
    return target_cache, target_cache_dir, target_fasta


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def normalized_target_memmap(embeddings: np.ndarray, cache_path: Path, batch_size: int = 1_024) -> np.ndarray:
    """Normalize the large donor cache without materialising a second RAM array."""
    shape = tuple(np.asarray(embeddings).shape)
    if cache_path.exists():
        try:
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape == shape and cached.dtype == np.float32:
                return cached
        except (OSError, ValueError):
            pass
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, shape[0], batch_size):
        end = min(start + batch_size, shape[0])
        batch = np.asarray(embeddings[start:end], dtype=np.float32)
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized[start:end] = batch / norms
    normalized.flush()
    del normalized
    return np.load(cache_path, mmap_mode="r")


def similarity_blocks(query: np.ndarray, target_norm: np.ndarray, chunk_size: int) -> Iterable[Tuple[int, int, np.ndarray]]:
    query_norm = normalize_embeddings(query)
    for start in range(0, query_norm.shape[0], chunk_size):
        end = min(start + chunk_size, query_norm.shape[0])
        yield start, end, query_norm[start:end].dot(target_norm.T)


def read_target_go_terms(go, goa_path: Path, target_ids: Set[str]) -> Dict[str, Set[str]]:
    return plm.load_go_terms_for_accessions(go, goa_path, target_ids, evidence_codes=EVIDENCE_CODES)


def top_neighbors_for_queries(
    query_embeddings: np.ndarray,
    target_norm: np.ndarray,
    target_ids: Sequence[str],
    excluded_source_ids: Set[str],
    max_neighbors: int,
    chunk_size: int,
    progress_label: str,
) -> Tuple[List[List[int]], List[List[float]], np.ndarray]:
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_ids)}
    excluded_target_indices = np.array(
        [target_index_by_id[protein_id] for protein_id in excluded_source_ids if protein_id in target_index_by_id],
        dtype=int,
    )
    available_count = len(target_ids) - len(excluded_target_indices)
    retained_count = min(max_neighbors, available_count)
    if retained_count < 1:
        raise RuntimeError(f"No source proteins remain after exclusions for {progress_label}.")

    neighbor_indices: List[List[int]] = [[] for _ in range(len(query_embeddings))]
    neighbor_scores: List[List[float]] = [[] for _ in range(len(query_embeddings))]
    for start, end, sims in similarity_blocks(query_embeddings, target_norm, chunk_size):
        if excluded_target_indices.size:
            sims[:, excluded_target_indices] = -np.inf
        top = np.argpartition(-sims, kth=retained_count - 1, axis=1)[:, :retained_count]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for local_row, query_index in enumerate(range(start, end)):
            neighbor_indices[query_index] = top[local_row].astype(int).tolist()
            neighbor_scores[query_index] = top_scores[local_row].astype(float).tolist()
        log(f"Computed {progress_label} neighbors: {end}/{len(query_embeddings)} query proteins.")
    return neighbor_indices, neighbor_scores, excluded_target_indices


def annotations_from_term_map(
    proteins: Sequence[str], accession_to_terms: Dict[str, Set[str]]
) -> pd.DataFrame:
    rows = [
        {"Protein": protein, "GO ID": go_id, "Score": 1.0}
        for protein in proteins
        for go_id in sorted(accession_to_terms.get(protein, set()))
    ]
    return pd.DataFrame(rows, columns=["Protein", "GO ID", "Score"])


def ancestor_ids_for_term(go, go_id: str, cache: Dict[str, Set[str]]) -> Set[str]:
    if go_id in cache:
        return cache[go_id]
    try:
        term = go.find_term(go_id)
    except KeyError:
        cache[go_id] = set()
        return cache[go_id]
    # Store the term itself before descending so malformed ontology cycles cannot recurse forever.
    cache[go_id] = {go_id}
    for parent in term.get_parents():
        cache[go_id].update(ancestor_ids_for_term(go, parent.go_id, cache))
    return cache[go_id]


def propagate_direct_rows_with_ancestors(
    go,
    direct_rows: List[Dict[str, object]],
    ancestor_cache: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """Up-propagate direct scores without retaining one GO annotation namespace per candidate."""
    if not direct_rows:
        return pd.DataFrame(columns=["protein_id", "term_id", "score"])
    direct_df = pd.DataFrame(direct_rows)
    direct_df = direct_df.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    cached_ancestors = ancestor_cache if ancestor_cache is not None else {}
    expanded_rows = []
    for protein, go_id, score in direct_df[["Protein", "GO ID", "Score"]].itertuples(index=False, name=None):
        expanded_rows.extend(
            {"protein_id": protein, "term_id": ancestor_id, "score": float(score)}
            for ancestor_id in ancestor_ids_for_term(go, go_id, cached_ancestors)
        )
    if not expanded_rows:
        return pd.DataFrame(columns=["protein_id", "term_id", "score"])
    propagated = pd.DataFrame(expanded_rows)
    return propagated.groupby(["protein_id", "term_id"], as_index=False)["score"].max()


def derive_kde_bandwidth_candidates(
    neighbor_scores: Sequence[Sequence[float]],
    weight_floor: float,
    explicit_bandwidths: Optional[Sequence[float]] = None,
) -> List[float]:
    if explicit_bandwidths is not None:
        return sorted({float(value) for value in explicit_bandwidths})
    local_distance_gaps = []
    for scores in neighbor_scores:
        distances = np.clip(1.0 - np.asarray(scores[:KDE_GRID_NEIGHBORS], dtype=float), 0.0, 2.0)
        if distances.size == 0:
            continue
        gaps = distances - float(distances[0])
        local_distance_gaps.extend(gap for gap in gaps if gap > 0 and np.isfinite(gap))
    if not local_distance_gaps:
        raise RuntimeError("Could not derive KDE bandwidths because calibration neighbor distance gaps are empty.")
    quantiles = np.linspace(0.05, 0.95, KDE_GRID_CANDIDATES)
    effective_radii = np.unique(np.quantile(np.asarray(local_distance_gaps, dtype=float), quantiles))
    divisor = -math.log(weight_floor)
    return sorted({float(math.sqrt(radius / divisor)) for radius in effective_radii if radius > 0})


def direct_rows_for_neighbors(
    query_ids: Sequence[str],
    target_ids: Sequence[str],
    neighbor_indices: Sequence[Sequence[int]],
    neighbor_scores: Sequence[Sequence[float]],
    accession_to_terms: Dict[str, Set[str]],
    score_mode: str,
    neighbor_weights: Optional[Sequence[Sequence[float]]] = None,
    query_confidences: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    rows = []
    weights_by_query = neighbor_weights if neighbor_weights is not None else neighbor_scores
    confidences = query_confidences if query_confidences is not None else [1.0] * len(query_ids)
    for query_id, indices, scores, weights, confidence in zip(
        query_ids, neighbor_indices, neighbor_scores, weights_by_query, confidences
    ):
        if not indices:
            continue
        term_similarities: Dict[str, List[float]] = defaultdict(list)
        term_weights: Dict[str, List[float]] = defaultdict(list)
        neighbor_weight_sum = sum(max(float(weight), 0.0) for weight in weights)
        for target_index, similarity, weight in zip(indices, scores, weights):
            neighbor_id = target_ids[int(target_index)]
            for go_id in accession_to_terms.get(neighbor_id, set()):
                term_similarities[go_id].append(float(similarity))
                term_weights[go_id].append(max(float(weight), 0.0))
        for go_id, support_scores in sorted(term_similarities.items()):
            support_weights = term_weights[go_id]
            if score_mode == "binary":
                score = 1.0
            elif score_mode == "support_fraction":
                score = len(support_scores) / len(indices)
            elif score_mode == "max_similarity":
                score = max(support_scores)
            elif score_mode == "weighted_support":
                numerator = sum(support_weights)
                score = numerator / neighbor_weight_sum if neighbor_weight_sum > 0 else 0.0
            else:
                raise RuntimeError(f"Unsupported clean PLM score mode: {score_mode}")
            rows.append({"Protein": query_id, "GO ID": go_id, "Score": float(score) * float(confidence)})
    return rows


def kde_effective_cosine_radius(bandwidth: float, weight_floor: float) -> float:
    """Return the distance increment where a relative Gaussian reaches ``weight_floor``.

    Unit-normalized vectors satisfy ``euclidean_distance**2 = 2 * cosine_distance``.
    After subtracting the nearest-donor distance for numerical stability, the
    relative Gaussian weight is ``exp(-(distance - d_min) / bandwidth**2)``.
    The subtraction cancels exactly in normalized GO support and does not move
    any kernel center.
    """
    if bandwidth <= 0:
        raise ValueError("KDE bandwidth must be positive.")
    if not 0.0 < weight_floor < 1.0:
        raise ValueError("KDE weight floor must be strictly between 0 and 1.")
    return float(-bandwidth * bandwidth * math.log(weight_floor))


def gaussian_kde_weights(similarities: Sequence[float], bandwidth: float) -> np.ndarray:
    distances = np.clip(1.0 - np.asarray(similarities, dtype=float), 0.0, 2.0)
    return np.exp(-distances / (bandwidth * bandwidth))


def relative_gaussian_kde_weights(similarities: Sequence[float], bandwidth: float) -> np.ndarray:
    """Return Gaussian weights scaled so the closest donor has weight one.

    Multiplying every absolute Gaussian weight by the same query-specific
    constant leaves normalized kernel support unchanged while avoiding
    underflow for targets far from the donor cloud.
    """
    if bandwidth <= 0:
        raise ValueError("KDE bandwidth must be positive.")
    score_array = np.asarray(similarities, dtype=float)
    if score_array.size == 0:
        return np.asarray([], dtype=float)
    distances = np.clip(1.0 - score_array, 0.0, 2.0)
    nearest_distance = float(np.min(distances))
    return np.exp(-(distances - nearest_distance) / (bandwidth * bandwidth))


def select_kde_neighbors(
    neighbor_indices: Sequence[Sequence[int]],
    neighbor_scores: Sequence[Sequence[float]],
    bandwidth: float,
    weight_floor: float,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]], List[Dict[str, object]]]:
    selected_indices: List[List[int]] = []
    selected_scores: List[List[float]] = []
    selected_weights: List[List[float]] = []
    diagnostics: List[Dict[str, object]] = []
    relative_radius = kde_effective_cosine_radius(bandwidth, weight_floor)
    for indices, scores in zip(neighbor_indices, neighbor_scores):
        score_array = np.asarray(scores, dtype=float)
        index_array = np.asarray(indices, dtype=int)
        weights = relative_gaussian_kde_weights(score_array, bandwidth)
        mask = weights >= weight_floor
        kept_indices = index_array[mask].astype(int).tolist()
        kept_scores = score_array[mask].astype(float).tolist()
        kept_weights = weights[mask].astype(float).tolist()
        total_weight = float(np.sum(weights[mask]))
        squared_weight_sum = float(np.sum(np.square(weights[mask])))
        effective_sample_size = (
            total_weight * total_weight / squared_weight_sum if squared_weight_sum > 0 else 0.0
        )
        selected_indices.append(kept_indices)
        selected_scores.append(kept_scores)
        selected_weights.append(kept_weights)
        diagnostics.append(
            {
                "bandwidth": float(bandwidth),
                "relative_distance_radius": relative_radius,
                "absolute_distance_threshold": (
                    float(1.0 - score_array[0] + relative_radius) if len(score_array) else math.nan
                ),
                "kernel_weight_floor": float(weight_floor),
                "retained_neighbor_count": len(kept_indices),
                "retained_kernel_weight": total_weight,
                "candidate_limit_reached": bool(len(weights) and np.all(mask)),
                "effective_sample_size": effective_sample_size,
                "nearest_cosine_distance": float(1.0 - score_array[0]) if len(score_array) else math.nan,
            }
        )
    return selected_indices, selected_scores, selected_weights, diagnostics


def adaptive_kde_bandwidth(
    similarities: Sequence[float],
    neighbor_rank: int,
    relative_weight_floor: float,
    bandwidth_scale: float,
) -> Tuple[float, float, float, float]:
    """Derive a query-local Gaussian bandwidth from its nearest-neighbor geometry.

    The rank-th source donor is placed on the relative kernel contour when
    ``bandwidth_scale == 1``.  Distances are cosine distances between unit
    vectors, so the same Gaussian convention as ``gaussian_kde_weights`` is
    retained.
    """
    score_array = np.asarray(similarities, dtype=float)
    if score_array.size == 0:
        raise ValueError("Adaptive KDE requires at least one candidate donor.")
    if neighbor_rank < 1:
        raise ValueError("Adaptive KDE neighbor rank must be positive.")
    if not 0.0 < relative_weight_floor < 1.0:
        raise ValueError("Adaptive KDE relative weight floor must be strictly between zero and one.")
    if bandwidth_scale <= 0:
        raise ValueError("Adaptive KDE bandwidth scale must be positive.")

    distances = np.clip(1.0 - score_array, 0.0, 2.0)
    nearest_distance = float(distances[0])
    rank_index = min(neighbor_rank - 1, len(distances) - 1)
    ranked_distance = float(distances[rank_index])
    distance_increment = max(ranked_distance - nearest_distance, 1e-12)
    base_bandwidth = math.sqrt(distance_increment / -math.log(relative_weight_floor))
    bandwidth = max(float(bandwidth_scale) * base_bandwidth, 1e-12)
    relative_radius = -bandwidth * bandwidth * math.log(relative_weight_floor)
    return bandwidth, nearest_distance, ranked_distance, relative_radius


def adaptive_relative_gaussian_weights(
    similarities: Sequence[float],
    neighbor_rank: int,
    relative_weight_floor: float,
    bandwidth_scale: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Return numerically stable Gaussian weights relative to the nearest donor."""
    score_array = np.asarray(similarities, dtype=float)
    bandwidth, nearest_distance, ranked_distance, relative_radius = adaptive_kde_bandwidth(
        score_array,
        neighbor_rank,
        relative_weight_floor,
        bandwidth_scale,
    )
    distances = np.clip(1.0 - score_array, 0.0, 2.0)
    relative_weights = np.exp(-(distances - nearest_distance) / (bandwidth * bandwidth))
    return relative_weights, bandwidth, nearest_distance, ranked_distance, relative_radius


def select_adaptive_kde_neighbors(
    neighbor_indices: Sequence[Sequence[int]],
    neighbor_scores: Sequence[Sequence[float]],
    neighbor_rank: int,
    bandwidth_scale: float,
    relative_weight_floor: float,
    min_neighbors: int,
    confidence_distance_scale: float,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]], List[float], List[Dict[str, object]]]:
    """Select query-local Gaussian KDE donors with relative cutoff and fallback.

    Relative kernel weights are used for GO support because their common
    absolute scale cancels in the support ratio.  A calibrated nearest-donor
    distance factor shrinks low-density predictions without numerical
    underflow from absolute Gaussian masses.
    """
    if min_neighbors < 1:
        raise ValueError("Adaptive KDE minimum neighbors must be positive.")
    if confidence_distance_scale <= 0:
        raise ValueError("Adaptive KDE confidence distance scale must be positive.")

    selected_indices: List[List[int]] = []
    selected_scores: List[List[float]] = []
    selected_relative_weights: List[List[float]] = []
    confidences: List[float] = []
    diagnostics: List[Dict[str, object]] = []
    for indices, scores in zip(neighbor_indices, neighbor_scores):
        score_array = np.asarray(scores, dtype=float)
        index_array = np.asarray(indices, dtype=int)
        relative_weights, bandwidth, nearest_distance, ranked_distance, relative_radius = adaptive_relative_gaussian_weights(
            score_array,
            neighbor_rank,
            relative_weight_floor,
            bandwidth_scale,
        )
        contour_mask = relative_weights >= relative_weight_floor
        contour_count = int(np.count_nonzero(contour_mask))
        retained_count = min(max(contour_count, min_neighbors), len(index_array))
        fallback_applied = contour_count < min_neighbors
        kept_indices = index_array[:retained_count].astype(int).tolist()
        kept_scores = score_array[:retained_count].astype(float).tolist()
        kept_relative_weights = relative_weights[:retained_count].astype(float).tolist()

        exponent = nearest_distance / (bandwidth * bandwidth)
        nearest_absolute_weight = math.exp(-exponent) if exponent < 745 else 0.0
        absolute_mass = nearest_absolute_weight * float(np.sum(relative_weights[:retained_count]))
        confidence = math.exp(-nearest_distance / confidence_distance_scale)

        selected_indices.append(kept_indices)
        selected_scores.append(kept_scores)
        selected_relative_weights.append(kept_relative_weights)
        confidences.append(float(confidence))
        diagnostics.append(
            {
                "adaptive_neighbor_rank": int(neighbor_rank),
                "adaptive_bandwidth_scale": float(bandwidth_scale),
                "adaptive_bandwidth": float(bandwidth),
                "nearest_cosine_distance": nearest_distance,
                "ranked_cosine_distance": ranked_distance,
                "relative_weight_floor": float(relative_weight_floor),
                "relative_distance_radius": float(relative_radius),
                "absolute_distance_threshold": float(nearest_distance + relative_radius),
                "contour_neighbor_count": contour_count,
                "retained_neighbor_count": retained_count,
                "minimum_neighbor_fallback_applied": fallback_applied,
                "nearest_absolute_kernel_weight": nearest_absolute_weight,
                "absolute_kernel_mass": absolute_mass,
                "confidence_distance_scale": float(confidence_distance_scale),
                "density_confidence": float(confidence),
                "effective_sample_size": (
                    float(np.sum(relative_weights[:retained_count])) ** 2
                    / float(np.sum(np.square(relative_weights[:retained_count])))
                    if retained_count else 0.0
                ),
            }
        )
    return selected_indices, selected_scores, selected_relative_weights, confidences, diagnostics


def propagate_predictions(go, direct_rows: List[Dict[str, object]], namespace: str) -> pd.DataFrame:
    if not direct_rows:
        return pd.DataFrame(columns=["protein_id", "term_id", "score"])
    direct_df = pd.DataFrame(direct_rows)
    direct_df = direct_df.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    log(f"Up-propagating {len(direct_df)} direct transfer rows for {namespace}.")
    go.load_annotations(direct_df, namespace)
    go.up_propagate_annotations(namespace)
    propagated = go.get_annotations(namespace)
    propagated = propagated[["Protein", "GO ID", "Score"]]
    propagated = propagated.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    propagated = propagated.rename(columns={"Protein": "protein_id", "GO ID": "term_id", "Score": "score"})
    return propagated


def build_prediction_and_gold_matrices(predictions: pd.DataFrame, annotations: pd.DataFrame):
    pred_proteins = set(predictions["protein_id"]) if not predictions.empty else set()
    pred_terms = set(predictions["term_id"]) if not predictions.empty else set()
    proteins = sorted(pred_proteins | set(annotations["Protein"]))
    terms = sorted(pred_terms | set(annotations["GO ID"]))
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}
    term_to_idx = {term: idx for idx, term in enumerate(terms)}

    prediction_matrix = np.zeros((len(proteins), len(terms)), dtype=np.float32)
    if not predictions.empty:
        rows = predictions["protein_id"].map(protein_to_idx).to_numpy()
        cols = predictions["term_id"].map(term_to_idx).to_numpy()
        prediction_matrix[rows, cols] = predictions["score"].to_numpy(dtype=float)

    gold_matrix = np.zeros((len(proteins), len(terms)), dtype=np.float32)
    rows = annotations["Protein"].map(protein_to_idx).to_numpy()
    cols = annotations["GO ID"].map(term_to_idx).to_numpy()
    gold_matrix[rows, cols] = 1.0
    return prediction_matrix, gold_matrix, term_to_idx


def compute_information_content(term_to_idx: Dict[str, int], ontology, organism_name: str) -> np.ndarray:
    ic = np.zeros(len(term_to_idx), dtype=np.float32)
    for term, idx in term_to_idx.items():
        try:
            ic[idx] = ontology.find_term(term).information_content(organism_name)
        except KeyError:
            ic[idx] = 0.0
    return ic


def flatten_metrics(block: str, metrics: Dict[str, object]) -> Dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (list, tuple, dict)):
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            label = key if block != "overall" else f"{block}::{key}"
            flat[label] = float(arr)
    return flat


def prepare_ground_truth_from_annotations(
    annotations: pd.DataFrame, go_obo: Path, organism_name: str
):
    ontology = GeneOntology.GeneOntology(str(go_obo), verbose=False)
    ontology.build_structure()
    ontology.load_annotations(annotations, organism_name)
    ontology.up_propagate_annotations(organism_name)
    propagated = ontology.get_annotations(organism_name)
    return propagated, ontology, organism_name


def prepare_ground_truth(goa_path: Path, go_obo: Path, organism: str, proteins: Set[str]):
    annotations = load_goa_annotations_for_taxon(goa_path, organism)
    annotations = annotations[annotations["Protein"].astype(str).isin(proteins)].copy()
    organism_name = f"clean_plm_gt_{organism}"
    return prepare_ground_truth_from_annotations(annotations, go_obo, organism_name)


def evaluate_prediction_table(
    model: str,
    organism: str,
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    ontology,
    organism_name: str,
    evaluation_size: int,
    source_note: str,
    extra_metadata: Optional[Dict[str, object]] = None,
    metric_blocks: Sequence[str] = ("overall", "per-gene", "per-term"),
    score_round_decimals: Optional[int] = None,
) -> Dict[str, object]:
    prediction_matrix, gold_matrix, term_to_idx = build_prediction_and_gold_matrices(predictions, annotations)
    ic = compute_information_content(term_to_idx, ontology, organism_name)
    sumrow = gold_matrix.sum(axis=1)
    sumcol = gold_matrix.sum(axis=0)
    row_mask = sumrow >= 1
    col_mask = sumcol >= 1
    pred = prediction_matrix[row_mask][:, col_mask]
    gold = gold_matrix[row_mask][:, col_mask]
    filtered_ic = ic[col_mask]
    if score_round_decimals is not None:
        pred = np.around(pred, decimals=score_round_decimals)
    elif len(np.unique(pred)) > 10000:
        pred = np.around(pred, decimals=4)

    measure = HX_py(pred, filtered_ic, organism_id=f"{model}_{organism}", verbose=False)
    metrics = {}
    if "overall" in metric_blocks:
        metrics.update(flatten_metrics("overall", measure.compute_overall(gold)))
    if "per-gene" in metric_blocks:
        metrics.update(flatten_metrics("per-gene", measure.compute_per_gene(gold)))
    if "per-term" in metric_blocks:
        metrics.update(flatten_metrics("per-term", measure.compute_per_term(gold)))

    row = {
        "organism": organism,
        "model": model,
        "label": model,
        "plot_label": model,
        "is_s2f": False,
        "evaluation_protein_set_size": evaluation_size,
        "predicted_proteins_in_evaluation_set": int(predictions["protein_id"].nunique()) if not predictions.empty else 0,
        "proteins_considered": int(gold.shape[0]),
        "terms_considered": int(gold.shape[1]),
        "total_annotations": int(gold.sum()),
        "matrix_shape": str(tuple(gold.shape)),
        "source_csv": "notebooks/esm_go_explorer/data/clean_plm_benchmark_metrics.csv",
        "source_note": source_note,
    }
    if extra_metadata:
        row.update(extra_metadata)
    row.update(metrics)
    for column in METRIC_COLUMNS:
        row.setdefault(column, math.nan)
    return row


def evaluate_kde_calibration_prediction(
    model: str,
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    score_round_decimals: int,
) -> Dict[str, object]:
    """Compute calibration ranking metrics without the expensive semantic-distance sweep."""
    prediction_matrix, gold_matrix, _term_to_idx = build_prediction_and_gold_matrices(
        predictions, annotations
    )
    row_mask = gold_matrix.sum(axis=1) >= 1
    col_mask = gold_matrix.sum(axis=0) >= 1
    pred = np.around(
        prediction_matrix[row_mask][:, col_mask], decimals=score_round_decimals
    )
    gold = gold_matrix[row_mask][:, col_mask]
    metrics = HX_py.HX_iteration(pred.flatten(), gold.flatten())
    return {
        "organism": "source_calibration",
        "model": model,
        "overall::AUC": float(metrics["AUC"]),
        "overall::AUPR": float(metrics["AUPR"]),
        "overall::F_max": float(metrics["F_max"]),
        "proteins_considered": int(gold.shape[0]),
        "terms_considered": int(gold.shape[1]),
        "total_annotations": int(gold.sum()),
        "matrix_shape": str(tuple(gold.shape)),
        "predicted_proteins_in_evaluation_set": (
            int(predictions["protein_id"].nunique()) if not predictions.empty else 0
        ),
    }


def select_kde_calibration_ids(
    accession_to_terms: Dict[str, Set[str]],
    target_ids: Sequence[str],
    excluded_ids: Set[str],
    calibration_size: int,
    random_seed: int,
) -> List[str]:
    candidates = sorted(
        protein_id
        for protein_id in target_ids
        if protein_id not in excluded_ids and accession_to_terms.get(protein_id)
    )
    if len(candidates) < calibration_size:
        raise RuntimeError(
            f"KDE calibration requested {calibration_size} proteins but only {len(candidates)} annotated targets are available."
        )
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(np.asarray(candidates, dtype=object), size=calibration_size, replace=False)
    return sorted(str(protein_id) for protein_id in selected.tolist())


def calibrate_kde_bandwidth(
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    accession_to_terms: Dict[str, Set[str]],
    benchmark_exclusion_ids: Set[str],
    go,
    go_obo: Path,
    chunk_size: int,
    calibration_size: int,
    max_neighbors: int,
    weight_floor: float,
    selection_metric: str,
    random_seed: int,
    explicit_bandwidths: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, object], pd.DataFrame, List[str]]:
    calibration_ids = select_kde_calibration_ids(
        accession_to_terms,
        target_cache.ids,
        benchmark_exclusion_ids,
        calibration_size,
        random_seed,
    )
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_cache.ids)}
    calibration_indices = [target_index_by_id[protein_id] for protein_id in calibration_ids]
    calibration_embeddings = np.asarray(target_cache.embeddings[calibration_indices], dtype=np.float32)
    calibration_exclusion_ids = set(benchmark_exclusion_ids) | set(calibration_ids)
    neighbor_indices, neighbor_scores, excluded_target_indices = top_neighbors_for_queries(
        calibration_embeddings,
        target_norm,
        target_cache.ids,
        calibration_exclusion_ids,
        max_neighbors,
        chunk_size,
        "KDE calibration",
    )
    bandwidths = derive_kde_bandwidth_candidates(
        neighbor_scores,
        weight_floor,
        explicit_bandwidths=explicit_bandwidths,
    )
    log(
        "Evaluating source-only KDE bandwidth candidates: "
        + ", ".join(f"{bandwidth:.6g}" for bandwidth in bandwidths)
    )

    calibration_annotations = annotations_from_term_map(calibration_ids, accession_to_terms)
    calibration_gold, _, _ = prepare_ground_truth_from_annotations(
        calibration_annotations,
        go_obo,
        f"clean_plm_kde_calibration_{random_seed}",
    )
    candidate_rows = []
    ancestor_cache: Dict[str, Set[str]] = {}
    for bandwidth in bandwidths:
        log(f"Materializing KDE calibration predictions for h={bandwidth:.6g}.")
        selected_indices, selected_scores, selected_weights, diagnostics = select_kde_neighbors(
            neighbor_indices,
            neighbor_scores,
            bandwidth,
            weight_floor,
        )
        direct = direct_rows_for_neighbors(
            calibration_ids,
            target_cache.ids,
            selected_indices,
            selected_scores,
            accession_to_terms,
            "weighted_support",
            neighbor_weights=selected_weights,
        )
        log(f"KDE calibration h={bandwidth:.6g}: {len(direct)} direct protein/GO support rows.")
        predictions = propagate_direct_rows_with_ancestors(go, direct, ancestor_cache=ancestor_cache)
        log(f"KDE calibration h={bandwidth:.6g}: {len(predictions)} propagated prediction rows.")
        effective_radius = kde_effective_cosine_radius(bandwidth, weight_floor)
        row = evaluate_kde_calibration_prediction(
            f"KDE calibration h={bandwidth:.6g}",
            predictions,
            calibration_gold,
            score_round_decimals=KDE_CALIBRATION_SCORE_DECIMALS,
        )
        row.update(
            {
                "transfer_strategy": "kde",
                "kernel": "gaussian",
                "bandwidth": float(bandwidth),
                "effective_radius": effective_radius,
                "kernel_weight_floor": float(weight_floor),
                "score_mode": "kernel_support",
                "calibration_size": len(calibration_ids),
                "calibration_seed": random_seed,
                "calibration_source_exclusion_count": int(len(excluded_target_indices)),
                "max_neighbors": max_neighbors,
                "queries_with_neighbors": int(sum(item["retained_neighbor_count"] > 0 for item in diagnostics)),
                "zero_neighbor_queries": int(sum(item["retained_neighbor_count"] == 0 for item in diagnostics)),
                "median_neighbor_count": float(np.median([item["retained_neighbor_count"] for item in diagnostics])),
                "mean_neighbor_count": float(np.mean([item["retained_neighbor_count"] for item in diagnostics])),
                "max_neighbor_count": int(max(item["retained_neighbor_count"] for item in diagnostics)),
                "median_effective_sample_size": float(
                    np.median([item["effective_sample_size"] for item in diagnostics])
                ),
                "score_round_decimals": KDE_CALIBRATION_SCORE_DECIMALS,
            }
        )
        candidate_rows.append(row)
        log(
            f"KDE calibration h={bandwidth:.6g}, R_eff={effective_radius:.6g}: "
            f"{selection_metric}={row.get(selection_metric, math.nan):.6g}."
        )

    candidates_df = pd.DataFrame(candidate_rows)
    values = pd.to_numeric(candidates_df[selection_metric], errors="coerce")
    finite = candidates_df[np.isfinite(values)].copy()
    if finite.empty:
        raise RuntimeError(f"No finite KDE calibration value was produced for {selection_metric}.")
    ascending = "smin" in selection_metric.lower()
    finite = finite.sort_values([selection_metric, "bandwidth"], ascending=[ascending, True])
    selected_row = finite.iloc[0].to_dict()
    selected_bandwidth = float(selected_row["bandwidth"])
    candidates_df["selected"] = np.isclose(
        pd.to_numeric(candidates_df["bandwidth"], errors="coerce"), selected_bandwidth
    )
    selection = {
        "kernel": "gaussian",
        "bandwidth": selected_bandwidth,
        "effective_radius": kde_effective_cosine_radius(selected_bandwidth, weight_floor),
        "kernel_weight_floor": float(weight_floor),
        "score_mode": "kernel_support",
        "selection_metric": selection_metric,
        "selection_metric_value": float(selected_row[selection_metric]),
        "calibration_size": len(calibration_ids),
        "calibration_seed": random_seed,
        "max_neighbors": max_neighbors,
        "candidate_count": len(bandwidths),
        "benchmark_exclusion_count": len(benchmark_exclusion_ids),
        "calibration_exclusion_count": len(calibration_ids),
        "calibration_score_round_decimals": KDE_CALIBRATION_SCORE_DECIMALS,
    }
    log(
        f"Selected KDE bandwidth h={selected_bandwidth:.6g}, "
        f"effective cosine radius={selection['effective_radius']:.6g}, "
        f"{selection_metric}={selection['selection_metric_value']:.6g}."
    )
    return selection, candidates_df, calibration_ids


def calibrate_shared_kde_bandwidth(
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    accession_to_terms: Dict[str, Set[str]],
    source_exclusions_by_organism: Dict[str, Set[str]],
    go,
    go_obo: Path,
    chunk_size: int,
    calibration_size: int,
    max_neighbors: int,
    weight_floor: float,
    selection_metric: str,
    random_seed: int,
    explicit_bandwidths: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Select one Gaussian bandwidth by macro-averaged source-only PFP quality.

    Each organism keeps its own blacklist-filtered donor pool, but every
    candidate bandwidth is evaluated for every organism.  The selected scalar
    bandwidth is therefore shared by the final benchmark methods and is never
    derived from a fixed neighbor rank.
    """
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_cache.ids)}
    calibration_data: Dict[str, Dict[str, object]] = {}
    calibration_records: List[Dict[str, object]] = []
    pooled_neighbor_scores: List[List[float]] = []

    for offset, organism in enumerate(ORGANISMS):
        excluded_ids = set(source_exclusions_by_organism[organism])
        calibration_ids = select_kde_calibration_ids(
            accession_to_terms,
            target_cache.ids,
            excluded_ids,
            calibration_size,
            random_seed + offset,
        )
        calibration_indices = [target_index_by_id[protein_id] for protein_id in calibration_ids]
        calibration_embeddings = np.asarray(target_cache.embeddings[calibration_indices], dtype=np.float32)
        calibration_exclusions = excluded_ids | set(calibration_ids)
        neighbor_indices, neighbor_scores, excluded_target_indices = top_neighbors_for_queries(
            calibration_embeddings,
            target_norm,
            target_cache.ids,
            calibration_exclusions,
            max_neighbors,
            chunk_size,
            f"shared KDE calibration {organism}",
        )
        annotations = annotations_from_term_map(calibration_ids, accession_to_terms)
        calibration_gold, _, _ = prepare_ground_truth_from_annotations(
            annotations,
            go_obo,
            f"clean_plm_shared_kde_calibration_{organism}_{random_seed + offset}",
        )
        calibration_data[organism] = {
            "ids": calibration_ids,
            "neighbor_indices": neighbor_indices,
            "neighbor_scores": neighbor_scores,
            "gold": calibration_gold,
            "excluded_target_count": int(len(excluded_target_indices)),
        }
        pooled_neighbor_scores.extend(neighbor_scores)
        calibration_records.extend(
            {
                "organism": organism,
                "protein_id": protein_id,
                "calibration_seed": random_seed + offset,
            }
            for protein_id in calibration_ids
        )

    bandwidths = derive_kde_bandwidth_candidates(
        pooled_neighbor_scores,
        weight_floor,
        explicit_bandwidths=explicit_bandwidths,
    )
    log(
        "Evaluating shared Gaussian KDE bandwidth candidates: "
        + ", ".join(f"{bandwidth:.6g}" for bandwidth in bandwidths)
    )

    candidate_rows: List[Dict[str, object]] = []
    ancestor_cache: Dict[str, Set[str]] = {}
    for bandwidth in bandwidths:
        organism_values = []
        bandwidth_cap_hit_count = 0
        for organism in ORGANISMS:
            data = calibration_data[organism]
            neighbor_indices = data["neighbor_indices"]
            neighbor_scores = data["neighbor_scores"]
            selected_indices, selected_scores, selected_weights, diagnostics = select_kde_neighbors(
                neighbor_indices,
                neighbor_scores,
                bandwidth,
                weight_floor,
            )
            cap_hit_query_count = sum(
                item["candidate_limit_reached"] and len(scores) >= max_neighbors
                for item, scores in zip(diagnostics, neighbor_scores)
            )
            if cap_hit_query_count:
                bandwidth_cap_hit_count += int(cap_hit_query_count)
                candidate_rows.append(
                    {
                        "organism": organism,
                        "model": f"Shared KDE h={bandwidth:.6g}",
                        "transfer_strategy": "kde",
                        "kernel": "gaussian",
                        "bandwidth": float(bandwidth),
                        "relative_distance_radius": kde_effective_cosine_radius(bandwidth, weight_floor),
                        "relative_weight_floor": float(weight_floor),
                        "score_mode": "gaussian_kernel_support",
                        "calibration_size": len(data["ids"]),
                        "calibration_seed": random_seed + ORGANISMS.index(organism),
                        "calibration_source_exclusion_count": data["excluded_target_count"],
                        "max_neighbors": max_neighbors,
                        "safety_cap_exceeded": True,
                        "cap_hit_query_count": int(cap_hit_query_count),
                        "eligible_for_selection": False,
                        selection_metric: math.nan,
                    }
                )
                continue
            direct = direct_rows_for_neighbors(
                data["ids"],
                target_cache.ids,
                selected_indices,
                selected_scores,
                accession_to_terms,
                "weighted_support",
                neighbor_weights=selected_weights,
            )
            predictions = propagate_direct_rows_with_ancestors(go, direct, ancestor_cache=ancestor_cache)
            row = evaluate_kde_calibration_prediction(
                f"Shared KDE h={bandwidth:.6g}",
                predictions,
                data["gold"],
                score_round_decimals=KDE_CALIBRATION_SCORE_DECIMALS,
            )
            row.update(
                {
                    "organism": organism,
                    "transfer_strategy": "kde",
                    "kernel": "gaussian",
                    "bandwidth": float(bandwidth),
                    "relative_distance_radius": kde_effective_cosine_radius(bandwidth, weight_floor),
                    "relative_weight_floor": float(weight_floor),
                    "score_mode": "gaussian_kernel_support",
                    "calibration_size": len(data["ids"]),
                    "calibration_seed": random_seed + ORGANISMS.index(organism),
                    "calibration_source_exclusion_count": data["excluded_target_count"],
                    "max_neighbors": max_neighbors,
                    "zero_neighbor_queries": int(sum(item["retained_neighbor_count"] == 0 for item in diagnostics)),
                    "median_neighbor_count": float(np.median([item["retained_neighbor_count"] for item in diagnostics])),
                    "mean_neighbor_count": float(np.mean([item["retained_neighbor_count"] for item in diagnostics])),
                    "max_neighbor_count": int(max(item["retained_neighbor_count"] for item in diagnostics)),
                    "safety_cap_exceeded": False,
                    "cap_hit_query_count": 0,
                    "eligible_for_selection": True,
                    "median_effective_sample_size": float(
                        np.median([item["effective_sample_size"] for item in diagnostics])
                    ),
                }
            )
            candidate_rows.append(row)
            organism_values.append(float(row[selection_metric]))
        candidate_eligible = len(organism_values) == len(ORGANISMS)
        macro_row = {
            "organism": "macro",
            "model": f"Shared KDE h={bandwidth:.6g}",
            "transfer_strategy": "kde",
            "kernel": "gaussian",
            "bandwidth": float(bandwidth),
            "relative_distance_radius": kde_effective_cosine_radius(bandwidth, weight_floor),
            "relative_weight_floor": float(weight_floor),
            "score_mode": "gaussian_kernel_support",
            selection_metric: float(np.mean(organism_values)) if candidate_eligible else math.nan,
            "organism_count": len(organism_values),
            "calibration_size": calibration_size * len(ORGANISMS),
            "max_neighbors": max_neighbors,
            "safety_cap_exceeded": not candidate_eligible,
            "cap_hit_query_count": bandwidth_cap_hit_count,
            "eligible_for_selection": candidate_eligible,
        }
        candidate_rows.append(macro_row)
        if candidate_eligible:
            log(
                f"Shared KDE h={bandwidth:.6g}: macro {selection_metric}="
                f"{macro_row[selection_metric]:.6g}."
            )
        else:
            log(
                f"Shared KDE h={bandwidth:.6g}: ineligible because {bandwidth_cap_hit_count} "
                f"calibration query/queries reached the {max_neighbors}-donor safety boundary."
            )

    candidates_df = pd.DataFrame(candidate_rows)
    macro = candidates_df[candidates_df["organism"] == "macro"].copy()
    metric_values = pd.to_numeric(macro[selection_metric], errors="coerce")
    macro = macro[np.isfinite(metric_values)]
    if macro.empty:
        raise RuntimeError(f"No finite shared KDE calibration value was produced for {selection_metric}.")
    ascending = "smin" in selection_metric.lower()
    selected_row = macro.sort_values(
        [selection_metric, "bandwidth"],
        ascending=[ascending, False],
    ).iloc[0]
    selected_bandwidth = float(selected_row["bandwidth"])
    candidates_df["selected"] = np.isclose(
        pd.to_numeric(candidates_df["bandwidth"], errors="coerce"), selected_bandwidth
    )
    selection = {
        "kernel": "gaussian",
        "bandwidth": selected_bandwidth,
        "relative_distance_radius": kde_effective_cosine_radius(selected_bandwidth, weight_floor),
        "relative_weight_floor": float(weight_floor),
        "kernel_weight_floor": float(weight_floor),
        "score_mode": "gaussian_kernel_support",
        "shared_across_organisms": True,
        "selection_metric": f"macro_mean::{selection_metric}",
        "selection_metric_value": float(selected_row[selection_metric]),
        "calibration_size_per_organism": calibration_size,
        "calibration_size": calibration_size * len(ORGANISMS),
        "calibration_seed": random_seed,
        "max_neighbors": max_neighbors,
        "candidate_count": len(bandwidths),
        "organisms": list(ORGANISMS),
        "calibration_score_round_decimals": KDE_CALIBRATION_SCORE_DECIMALS,
    }
    return selection, candidates_df, pd.DataFrame(calibration_records)


def constrain_shared_kde_to_unlabeled_query_support(
    selection: Dict[str, object],
    candidates: pd.DataFrame,
    query_caches: Dict[str, plm.EmbeddingCache],
    target_norm: np.ndarray,
    target_ids: Sequence[str],
    source_exclusions_by_organism: Dict[str, Set[str]],
    chunk_size: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Reject bandwidths that would truncate meaningful held-out query support.

    This check uses only target embedding geometry, never target GO labels.  It
    keeps the safety boundary from becoming a hidden fixed-K selection rule.
    """
    max_neighbors = int(selection["max_neighbors"])
    weight_floor = float(selection["relative_weight_floor"])
    query_neighbor_scores: Dict[str, List[List[float]]] = {}
    for organism, query_cache in query_caches.items():
        _indices, scores, _excluded = top_neighbors_for_queries(
            query_cache.embeddings,
            target_norm,
            target_ids,
            source_exclusions_by_organism[organism],
            max_neighbors=max_neighbors,
            chunk_size=chunk_size,
            progress_label=f"unlabeled KDE support check {organism}",
        )
        query_neighbor_scores[organism] = scores

    macro_mask = candidates["organism"].astype(str) == "macro"
    metric_name = str(selection["selection_metric"]).removeprefix("macro_mean::")
    macro = candidates[macro_mask].copy()
    macro_metric = pd.to_numeric(macro[metric_name], errors="coerce")
    macro = macro[np.isfinite(macro_metric)].sort_values(
        [metric_name, "bandwidth"],
        ascending=[False, False],
    )
    support_cap_counts: Dict[float, int] = {}
    for bandwidth in pd.to_numeric(candidates["bandwidth"], errors="coerce").dropna().unique():
        cap_count = 0
        for scores in query_neighbor_scores.values():
            dummy_indices = [list(range(len(row))) for row in scores]
            _indices, _scores, _weights, diagnostics = select_kde_neighbors(
                dummy_indices,
                scores,
                float(bandwidth),
                weight_floor,
            )
            cap_count += sum(
                item["candidate_limit_reached"] and len(row) >= max_neighbors
                for item, row in zip(diagnostics, scores)
            )
        support_cap_counts[float(bandwidth)] = int(cap_count)

    candidates = candidates.copy()
    candidates["unlabeled_query_cap_hit_count"] = candidates["bandwidth"].map(
        lambda value: support_cap_counts.get(float(value), 0)
    )
    candidates["unlabeled_query_support_eligible"] = (
        candidates["unlabeled_query_cap_hit_count"] == 0
    )
    eligible_macro = macro[
        macro["bandwidth"].map(lambda value: support_cap_counts.get(float(value), 0) == 0)
    ]
    if eligible_macro.empty:
        raise RuntimeError(
            "No calibrated Gaussian KDE bandwidth fits the donor safety boundary for every unlabeled "
            "benchmark target. Increase --kde-max-neighbors and rerun."
        )
    selected_row = eligible_macro.iloc[0]
    selected_bandwidth = float(selected_row["bandwidth"])
    original_bandwidth = float(selection["bandwidth"])
    selection = {
        **selection,
        "bandwidth": selected_bandwidth,
        "relative_distance_radius": kde_effective_cosine_radius(selected_bandwidth, weight_floor),
        "selection_metric_value": float(selected_row[metric_name]),
        "source_only_optimal_bandwidth_before_support_check": original_bandwidth,
        "unlabeled_benchmark_support_check": True,
        "unlabeled_benchmark_support_cap_hit_count": 0,
        "selection_rule": (
            "maximize source-only macro calibration metric among bandwidths that do not reach the "
            "donor safety boundary on calibration or unlabeled benchmark query geometry"
        ),
    }
    candidates["selected"] = np.isclose(
        pd.to_numeric(candidates["bandwidth"], errors="coerce"), selected_bandwidth
    )
    if not math.isclose(original_bandwidth, selected_bandwidth):
        log(
            f"Source-only optimum h={original_bandwidth:.6g} exceeded the donor safety boundary on "
            f"unlabeled benchmark geometry; selected eligible h={selected_bandwidth:.6g}."
        )
    else:
        log(f"Shared KDE h={selected_bandwidth:.6g} passed the unlabeled benchmark support check.")
    return selection, candidates


def query_nearest_distance_profile(
    query_caches: Dict[str, plm.EmbeddingCache],
    target_norm: np.ndarray,
    target_ids: Sequence[str],
    excluded_source_ids: Set[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Build an unlabeled nearest-donor distance profile for benchmark queries."""
    frames = []
    for organism, cache in query_caches.items():
        _indices, scores, _excluded = top_neighbors_for_queries(
            cache.embeddings,
            target_norm,
            target_ids,
            excluded_source_ids,
            max_neighbors=1,
            chunk_size=chunk_size,
            progress_label=f"adaptive KDE profile {organism}",
        )
        frames.extend(
            {
                "organism": organism,
                "protein_id": protein_id,
                "nearest_cosine_distance": float(1.0 - score_row[0]),
            }
            for protein_id, score_row in zip(cache.ids, scores)
        )
    return pd.DataFrame(frames)


def select_stratified_adaptive_calibration_ids(
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    accession_to_terms: Dict[str, Set[str]],
    excluded_ids: Set[str],
    query_profile: pd.DataFrame,
    calibration_size: int,
    calibration_repeats: int,
    distance_bins: int,
    random_seed: int,
    chunk_size: int,
) -> Tuple[List[List[str]], pd.DataFrame, pd.DataFrame]:
    """Select repeated source pseudo-query sets matching unlabeled query density.

    The benchmark proteins contribute only their embedding nearest-neighbor
    distances.  Their GO labels are never loaded or inspected here.
    """
    candidates = sorted(
        protein_id
        for protein_id in target_cache.ids
        if protein_id not in excluded_ids and accession_to_terms.get(protein_id)
    )
    total_needed = calibration_size * calibration_repeats
    if len(candidates) < total_needed:
        raise RuntimeError(
            f"Adaptive KDE calibration needs {total_needed} distinct proteins but only {len(candidates)} are available."
        )
    rng = np.random.default_rng(random_seed)
    pool_size = min(
        len(candidates),
        max(total_needed * ADAPTIVE_KDE_CALIBRATION_POOL_MULTIPLIER, total_needed + distance_bins),
    )
    pool_ids = sorted(str(item) for item in rng.choice(np.asarray(candidates, dtype=object), size=pool_size, replace=False))
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_cache.ids)}
    pool_embeddings = np.asarray(
        target_cache.embeddings[[target_index_by_id[protein_id] for protein_id in pool_ids]], dtype=np.float32
    )
    _pool_indices, pool_scores, _excluded = top_neighbors_for_queries(
        pool_embeddings,
        target_norm,
        target_cache.ids,
        set(excluded_ids) | set(pool_ids),
        max_neighbors=1,
        chunk_size=chunk_size,
        progress_label="adaptive KDE calibration-pool profile",
    )
    profile_values = query_profile["nearest_cosine_distance"].to_numpy(dtype=float)
    if profile_values.size == 0:
        raise RuntimeError("Adaptive KDE requires a nonempty unlabeled benchmark distance profile.")
    edges = np.unique(np.quantile(profile_values, np.linspace(0.0, 1.0, distance_bins + 1)))
    effective_bins = max(len(edges) - 1, 1)

    def bin_index(values: np.ndarray) -> np.ndarray:
        if len(edges) <= 1:
            return np.zeros(values.shape[0], dtype=int)
        return np.clip(np.digitize(values, edges[1:-1], right=True), 0, effective_bins - 1)

    query_bins = bin_index(profile_values)
    desired_counts = np.bincount(query_bins, minlength=effective_bins).astype(float)
    desired_counts /= desired_counts.sum()
    per_repeat_counts = np.floor(desired_counts * calibration_size).astype(int)
    remainder = calibration_size - int(per_repeat_counts.sum())
    if remainder:
        for bin_id in np.argsort(-desired_counts)[:remainder]:
            per_repeat_counts[bin_id] += 1

    pool_distances = np.asarray([1.0 - score_row[0] for score_row in pool_scores], dtype=float)
    pool_df = pd.DataFrame(
        {
            "protein_id": pool_ids,
            "nearest_cosine_distance": pool_distances,
            "distance_bin": bin_index(pool_distances),
        }
    )
    remaining_by_bin = {
        bin_id: pool_df[pool_df["distance_bin"] == bin_id]["protein_id"].tolist()
        for bin_id in range(effective_bins)
    }
    for values in remaining_by_bin.values():
        rng.shuffle(values)
    all_remaining = set(pool_ids)
    repeats: List[List[str]] = []
    records = []
    for repeat in range(calibration_repeats):
        selected: List[str] = []
        for bin_id, requested in enumerate(per_repeat_counts.tolist()):
            available = remaining_by_bin[bin_id]
            take = min(requested, len(available))
            chosen = available[:take]
            del available[:take]
            selected.extend(chosen)
            all_remaining.difference_update(chosen)
        if len(selected) < calibration_size:
            fill_count = calibration_size - len(selected)
            fallback = sorted(all_remaining)
            if len(fallback) < fill_count:
                raise RuntimeError("Adaptive KDE calibration pool was exhausted while filling stratified repeats.")
            chosen = rng.choice(np.asarray(fallback, dtype=object), size=fill_count, replace=False).tolist()
            selected.extend(str(item) for item in chosen)
            all_remaining.difference_update(selected)
            for values in remaining_by_bin.values():
                selected_set = set(selected)
                values[:] = [protein_id for protein_id in values if protein_id not in selected_set]
        selected = sorted(selected)
        repeats.append(selected)
        selected_frame = pool_df[pool_df["protein_id"].isin(selected)].copy()
        selected_frame["repeat"] = repeat
        records.extend(selected_frame.to_dict(orient="records"))

    calibration_records = pd.DataFrame(records).sort_values(["repeat", "protein_id"])
    profile_summary = pd.DataFrame(
        {
            "distance_bin": list(range(effective_bins)),
            "benchmark_query_count": np.bincount(query_bins, minlength=effective_bins),
            "benchmark_query_fraction": desired_counts,
            "calibration_per_repeat_target": per_repeat_counts,
            "bin_left": [float(edges[index]) if len(edges) > 1 else float(profile_values.min()) for index in range(effective_bins)],
            "bin_right": [float(edges[index + 1]) if len(edges) > 1 else float(profile_values.max()) for index in range(effective_bins)],
        }
    )
    return repeats, calibration_records, profile_summary


def adaptive_kde_candidate_configurations(
    neighbor_counts: Sequence[int],
    bandwidth_scales: Sequence[float],
    confidence_distance_scales: Sequence[float],
    relative_weight_floor: float,
    min_neighbors: int,
) -> List[Dict[str, object]]:
    configs = []
    for neighbor_rank in sorted({int(value) for value in neighbor_counts}):
        for bandwidth_scale in sorted({float(value) for value in bandwidth_scales}):
            for confidence_distance_scale in sorted({float(value) for value in confidence_distance_scales}):
                configs.append(
                    {
                        "adaptive_neighbor_rank": neighbor_rank,
                        "adaptive_bandwidth_scale": bandwidth_scale,
                        "relative_weight_floor": float(relative_weight_floor),
                        "min_neighbors": int(min_neighbors),
                        "confidence_distance_scale": confidence_distance_scale,
                    }
                )
    return configs


def calibrate_adaptive_kde(
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    accession_to_terms: Dict[str, Set[str]],
    benchmark_exclusion_ids: Set[str],
    query_profile: pd.DataFrame,
    go,
    go_obo: Path,
    chunk_size: int,
    calibration_size: int,
    calibration_repeats: int,
    distance_bins: int,
    neighbor_counts: Sequence[int],
    bandwidth_scales: Sequence[float],
    relative_weight_floor: float,
    min_neighbors: int,
    confidence_distance_scales: Sequence[float],
    fmax_se_multiplier: float,
    max_neighbors: int,
    random_seed: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select an adaptive KDE configuration using repeated density-matched calibration."""
    repeats, calibration_records, profile_summary = select_stratified_adaptive_calibration_ids(
        target_cache,
        target_norm,
        accession_to_terms,
        benchmark_exclusion_ids,
        query_profile,
        calibration_size,
        calibration_repeats,
        distance_bins,
        random_seed,
        chunk_size,
    )
    calibration_ids = [protein_id for repeat_ids in repeats for protein_id in repeat_ids]
    calibration_exclusion_ids = set(benchmark_exclusion_ids) | set(calibration_ids)
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_cache.ids)}
    calibration_embeddings = np.asarray(
        target_cache.embeddings[[target_index_by_id[protein_id] for protein_id in calibration_ids]], dtype=np.float32
    )
    neighbor_indices, neighbor_scores, excluded_target_indices = top_neighbors_for_queries(
        calibration_embeddings,
        target_norm,
        target_cache.ids,
        calibration_exclusion_ids,
        max_neighbors,
        chunk_size,
        "adaptive KDE calibration",
    )
    configs = adaptive_kde_candidate_configurations(
        neighbor_counts,
        bandwidth_scales,
        confidence_distance_scales,
        relative_weight_floor,
        min_neighbors,
    )
    repeat_offsets = []
    offset = 0
    for repeat_ids in repeats:
        repeat_offsets.append((offset, offset + len(repeat_ids)))
        offset += len(repeat_ids)
    repeat_gold = []
    for repeat, repeat_ids in enumerate(repeats):
        annotations = annotations_from_term_map(repeat_ids, accession_to_terms)
        gold, _ontology, _name = prepare_ground_truth_from_annotations(
            annotations,
            go_obo,
            f"adaptive_kde_calibration_{random_seed}_{repeat}",
        )
        repeat_gold.append(gold)

    repeat_rows = []
    ancestor_cache: Dict[str, Set[str]] = {}
    for config_index, config in enumerate(configs):
        log(
            "Evaluating adaptive KDE calibration configuration "
            f"{config_index + 1}/{len(configs)}: k={config['adaptive_neighbor_rank']}, "
            f"scale={config['adaptive_bandwidth_scale']:g}, confidence_distance={config['confidence_distance_scale']:g}."
        )
        selected_indices, selected_scores, selected_weights, confidences, diagnostics = select_adaptive_kde_neighbors(
            neighbor_indices,
            neighbor_scores,
            neighbor_rank=int(config["adaptive_neighbor_rank"]),
            bandwidth_scale=float(config["adaptive_bandwidth_scale"]),
            relative_weight_floor=float(config["relative_weight_floor"]),
            min_neighbors=int(config["min_neighbors"]),
            confidence_distance_scale=float(config["confidence_distance_scale"]),
        )
        for repeat, (start, end) in enumerate(repeat_offsets):
            repeat_ids = repeats[repeat]
            direct = direct_rows_for_neighbors(
                repeat_ids,
                target_cache.ids,
                selected_indices[start:end],
                selected_scores[start:end],
                accession_to_terms,
                "weighted_support",
                neighbor_weights=selected_weights[start:end],
                query_confidences=confidences[start:end],
            )
            predictions = propagate_direct_rows_with_ancestors(go, direct, ancestor_cache=ancestor_cache)
            metric = evaluate_kde_calibration_prediction(
                f"adaptive KDE k={config['adaptive_neighbor_rank']}",
                predictions,
                repeat_gold[repeat],
                score_round_decimals=KDE_CALIBRATION_SCORE_DECIMALS,
            )
            subset_diagnostics = diagnostics[start:end]
            metric.update(config)
            metric.update(
                {
                    "repeat": repeat,
                    "predicted_coverage": (
                        float(predictions["protein_id"].nunique()) / len(repeat_ids) if not predictions.empty else 0.0
                    ),
                    "median_adaptive_bandwidth": float(np.median([item["adaptive_bandwidth"] for item in subset_diagnostics])),
                    "median_neighbor_count": float(np.median([item["retained_neighbor_count"] for item in subset_diagnostics])),
                    "mean_neighbor_count": float(np.mean([item["retained_neighbor_count"] for item in subset_diagnostics])),
                    "fallback_query_count": int(sum(item["minimum_neighbor_fallback_applied"] for item in subset_diagnostics)),
                    "mean_density_confidence": float(np.mean([item["density_confidence"] for item in subset_diagnostics])),
                    "median_density_confidence": float(np.median([item["density_confidence"] for item in subset_diagnostics])),
                    "calibration_source_exclusion_count": int(len(excluded_target_indices)),
                }
            )
            repeat_rows.append(metric)

    repeat_df = pd.DataFrame(repeat_rows)
    config_columns = [
        "adaptive_neighbor_rank",
        "adaptive_bandwidth_scale",
        "relative_weight_floor",
        "min_neighbors",
        "confidence_distance_scale",
    ]
    aggregate = repeat_df.groupby(config_columns, as_index=False).agg(
        mean_fmax=("overall::F_max", "mean"),
        std_fmax=("overall::F_max", "std"),
        mean_aupr=("overall::AUPR", "mean"),
        mean_auc=("overall::AUC", "mean"),
        mean_coverage=("predicted_coverage", "mean"),
        mean_median_bandwidth=("median_adaptive_bandwidth", "mean"),
        mean_neighbor_count=("mean_neighbor_count", "mean"),
        mean_fallback_query_count=("fallback_query_count", "mean"),
        mean_density_confidence=("mean_density_confidence", "mean"),
        repeats=("repeat", "count"),
    )
    aggregate["fmax_standard_error"] = aggregate["std_fmax"].fillna(0.0) / np.sqrt(aggregate["repeats"])
    best = aggregate.sort_values(
        ["mean_fmax", "mean_coverage", "mean_median_bandwidth"], ascending=[False, False, True]
    ).iloc[0]
    fmax_tolerance = float(fmax_se_multiplier) * float(best["fmax_standard_error"])
    aggregate["within_fmax_one_se"] = aggregate["mean_fmax"] >= float(best["mean_fmax"]) - fmax_tolerance
    eligible = aggregate[aggregate["within_fmax_one_se"]].copy()
    selected = eligible.sort_values(
        ["mean_coverage", "mean_median_bandwidth", "adaptive_neighbor_rank", "adaptive_bandwidth_scale"],
        ascending=[False, True, True, True],
    ).iloc[0]
    aggregate["selected"] = (
        (aggregate["adaptive_neighbor_rank"] == selected["adaptive_neighbor_rank"])
        & np.isclose(aggregate["adaptive_bandwidth_scale"], selected["adaptive_bandwidth_scale"])
        & np.isclose(aggregate["confidence_distance_scale"], selected["confidence_distance_scale"])
    )
    selection = {
        "kernel": "adaptive_gaussian",
        "adaptive_neighbor_rank": int(selected["adaptive_neighbor_rank"]),
        "adaptive_bandwidth_scale": float(selected["adaptive_bandwidth_scale"]),
        "relative_weight_floor": float(selected["relative_weight_floor"]),
        "min_neighbors": int(selected["min_neighbors"]),
        "confidence_distance_scale": float(selected["confidence_distance_scale"]),
        "selection_metric": "mean overall::F_max",
        "selection_metric_value": float(selected["mean_fmax"]),
        "selection_metric_standard_error": float(selected["fmax_standard_error"]),
        "best_mean_fmax": float(best["mean_fmax"]),
        "best_mean_fmax_standard_error": float(best["fmax_standard_error"]),
        "fmax_standard_error_multiplier": float(fmax_se_multiplier),
        "eligible_configuration_count": int(len(eligible)),
        "selection_rule": "within one standard error of best mean Fmax, maximize coverage, then minimize median adaptive bandwidth",
        "mean_calibration_coverage": float(selected["mean_coverage"]),
        "mean_calibration_neighbor_count": float(selected["mean_neighbor_count"]),
        "mean_calibration_density_confidence": float(selected["mean_density_confidence"]),
        "calibration_size": calibration_size,
        "calibration_repeats": calibration_repeats,
        "distance_bins": distance_bins,
        "calibration_seed": random_seed,
        "max_neighbors": max_neighbors,
        "candidate_count": int(len(configs)),
        "benchmark_exclusion_count": len(benchmark_exclusion_ids),
        "calibration_exclusion_count": len(calibration_ids),
        "calibration_score_round_decimals": KDE_CALIBRATION_SCORE_DECIMALS,
    }
    return selection, aggregate, repeat_df, calibration_records, profile_summary


def clean_plm_model_name(strategy: str, score_mode: str, k_value: Optional[int] = None, radius: Optional[float] = None) -> str:
    if strategy == "knn":
        base = f"Clean PLM + KNN k={k_value}"
    elif strategy == "radius":
        base = f"Clean PLM + Radius R={radius:g}"
    elif strategy == "kde":
        return ACTIVE_KDE_MODEL
    elif strategy == "adaptive_kde":
        return LEGACY_ADAPTIVE_KDE_MODEL
    else:
        raise RuntimeError(f"Unsupported clean PLM strategy: {strategy}")
    if score_mode == "binary":
        return base
    return f"{base} ({SCORE_MODE_LABELS[score_mode]})"


def build_clean_plm_predictions(
    organism: str,
    query_cache: plm.EmbeddingCache,
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    go,
    goa_path: Path,
    radii: Sequence[float],
    chunk_size: int,
    score_modes: Sequence[str],
    strategies: Sequence[str],
    kde_selection: Optional[Dict[str, object]] = None,
    adaptive_kde_selection: Optional[Dict[str, object]] = None,
    preloaded_accession_to_terms: Optional[Dict[str, Set[str]]] = None,
    source_exclusion_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, Dict[str, object]]], List[Dict[str, object]]]:
    max_k = max(K_VALUES)
    knn_indices = {k: [[] for _ in query_cache.ids] for k in K_VALUES}
    knn_scores = {k: [[] for _ in query_cache.ids] for k in K_VALUES}
    radius_indices = {radius: [[] for _ in query_cache.ids] for radius in radii}
    radius_scores = {radius: [[] for _ in query_cache.ids] for radius in radii}
    neighbor_counts = []
    target_ids = target_cache.ids
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_ids)}
    excluded_source_ids = set(source_exclusion_ids or set())
    excluded_target_indices = np.array(
        [target_index_by_id[protein_id] for protein_id in excluded_source_ids if protein_id in target_index_by_id],
        dtype=int,
    )
    wanted_targets: Set[str] = set()
    radius_similarity_thresholds = {radius: 1.0 - radius for radius in radii}
    kde_indices: List[List[int]] = [[] for _ in query_cache.ids]
    kde_scores: List[List[float]] = [[] for _ in query_cache.ids]
    kde_weights: List[List[float]] = [[] for _ in query_cache.ids]
    adaptive_indices: List[List[int]] = [[] for _ in query_cache.ids]
    adaptive_scores: List[List[float]] = [[] for _ in query_cache.ids]
    adaptive_weights: List[List[float]] = [[] for _ in query_cache.ids]
    adaptive_confidences: List[float] = [0.0 for _ in query_cache.ids]
    if "kde" in strategies:
        if kde_selection is None:
            raise RuntimeError("The KDE strategy requires a calibrated bandwidth selection.")
        kde_bandwidth = float(kde_selection["bandwidth"])
        kde_weight_floor_value = kde_selection.get("relative_weight_floor")
        if kde_weight_floor_value is None:
            kde_weight_floor_value = kde_selection["kernel_weight_floor"]
        kde_weight_floor = float(kde_weight_floor_value)
        kde_relative_radius = float(
            kde_selection.get(
                "relative_distance_radius",
                kde_effective_cosine_radius(kde_bandwidth, kde_weight_floor),
            )
        )
        kde_max_neighbors = int(kde_selection["max_neighbors"])
    else:
        kde_bandwidth = math.nan
        kde_weight_floor = math.nan
        kde_relative_radius = math.nan
        kde_max_neighbors = 0
    if "adaptive_kde" in strategies:
        if adaptive_kde_selection is None:
            raise RuntimeError("The adaptive KDE strategy requires a calibrated selection.")
        adaptive_max_neighbors = int(adaptive_kde_selection["max_neighbors"])
    else:
        adaptive_max_neighbors = 0

    search_neighbor_count = max(
        max_k if "knn" in strategies else 0,
        kde_max_neighbors if "kde" in strategies else 0,
        adaptive_max_neighbors if "adaptive_kde" in strategies else 0,
        1,
    )
    available_count = len(target_ids) - len(excluded_target_indices)
    search_neighbor_count = min(search_neighbor_count, available_count)

    for start, end, sims in similarity_blocks(query_cache.embeddings, target_norm, chunk_size):
        if excluded_target_indices.size:
            sims[:, excluded_target_indices] = -np.inf
        top = np.argpartition(-sims, kth=search_neighbor_count - 1, axis=1)[:, :search_neighbor_count]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for local_row, query_index in enumerate(range(start, end)):
            if "knn" in strategies:
                for k_value in K_VALUES:
                    selected = top[local_row, : min(k_value, top.shape[1])].astype(int).tolist()
                    selected_scores = top_scores[local_row, : min(k_value, top_scores.shape[1])].astype(float).tolist()
                    knn_indices[k_value][query_index] = selected
                    knn_scores[k_value][query_index] = selected_scores
                    wanted_targets.update(target_ids[i] for i in selected)
            if "kde" in strategies:
                kde_top_indices = top[local_row, : min(kde_max_neighbors, top.shape[1])].astype(int)
                kde_top_scores = top_scores[local_row, : min(kde_max_neighbors, top_scores.shape[1])].astype(float)
                nearest_distance = float(1.0 - kde_top_scores[0])
                absolute_distance_threshold = nearest_distance + kde_relative_radius
                candidate_count = int(
                    np.count_nonzero(sims[local_row] >= 1.0 - absolute_distance_threshold)
                )
                if candidate_count > kde_max_neighbors:
                    raise RuntimeError(
                        f"Gaussian KDE for {organism}/{query_cache.ids[query_index]} has {candidate_count} "
                        f"contributors above the relative floor, exceeding the {kde_max_neighbors}-donor "
                        "safety cap. Increase --kde-max-neighbors and rerun."
                    )
                weights = relative_gaussian_kde_weights(kde_top_scores, kde_bandwidth)
                mask = weights >= kde_weight_floor
                retained_indices = kde_top_indices[mask].astype(int).tolist()
                retained_scores = kde_top_scores[mask].astype(float).tolist()
                retained_weights = weights[mask].astype(float).tolist()
                kde_indices[query_index] = retained_indices
                kde_scores[query_index] = retained_scores
                kde_weights[query_index] = retained_weights
                wanted_targets.update(target_ids[i] for i in retained_indices)
                total_weight = float(np.sum(weights[mask]))
                squared_weight_sum = float(np.sum(np.square(weights[mask])))
                finite_scores = sims[local_row, np.isfinite(sims[local_row])]
                full_weights = relative_gaussian_kde_weights(finite_scores, kde_bandwidth)
                full_weight_sum = float(np.sum(full_weights))
                neighbor_counts.append(
                    {
                        "organism": organism,
                        "protein_id": query_cache.ids[query_index],
                        "transfer_strategy": "kde",
                        "distance_metric": "cosine_distance",
                        "kernel": "gaussian",
                        "bandwidth": kde_bandwidth,
                        "relative_distance_radius": kde_relative_radius,
                        "absolute_distance_threshold": absolute_distance_threshold,
                        "relative_weight_floor": kde_weight_floor,
                        "candidate_neighbor_count": candidate_count,
                        "retained_neighbor_count": len(retained_indices),
                        "max_neighbors": kde_max_neighbors,
                        "neighbor_cap_applied": False,
                        "retained_kernel_weight": total_weight,
                        "retained_kernel_mass_fraction": (
                            total_weight / full_weight_sum if full_weight_sum > 0 else 0.0
                        ),
                        "effective_sample_size": (
                            total_weight * total_weight / squared_weight_sum if squared_weight_sum > 0 else 0.0
                        ),
                        "top10_min_similarity": float(np.min(top_scores[local_row, : min(10, top_scores.shape[1])])),
                        "top1_similarity": float(top_scores[local_row, 0]),
                        "nearest_cosine_distance": nearest_distance,
                        "source_exclusion_count": int(excluded_target_indices.size),
                        "query_accession_excluded": query_cache.ids[query_index] in excluded_source_ids,
                    }
                )
            if "adaptive_kde" in strategies:
                selected = select_adaptive_kde_neighbors(
                    [top[local_row, : min(adaptive_max_neighbors, top.shape[1])].astype(int).tolist()],
                    [top_scores[local_row, : min(adaptive_max_neighbors, top_scores.shape[1])].astype(float).tolist()],
                    neighbor_rank=int(adaptive_kde_selection["adaptive_neighbor_rank"]),
                    bandwidth_scale=float(adaptive_kde_selection["adaptive_bandwidth_scale"]),
                    relative_weight_floor=float(adaptive_kde_selection["relative_weight_floor"]),
                    min_neighbors=int(adaptive_kde_selection["min_neighbors"]),
                    confidence_distance_scale=float(adaptive_kde_selection["confidence_distance_scale"]),
                )
                retained_indices, retained_scores, retained_weights, confidences, diagnostics = selected
                adaptive_indices[query_index] = retained_indices[0]
                adaptive_scores[query_index] = retained_scores[0]
                adaptive_weights[query_index] = retained_weights[0]
                adaptive_confidences[query_index] = confidences[0]
                diagnostic = diagnostics[0]
                full_contour_count = int(
                    np.count_nonzero(
                        sims[local_row] >= 1.0 - float(diagnostic["absolute_distance_threshold"])
                    )
                )
                neighbor_counts.append(
                    {
                        "organism": organism,
                        "protein_id": query_cache.ids[query_index],
                        "transfer_strategy": "adaptive_kde",
                        "distance_metric": "cosine_distance",
                        "kernel": "adaptive_gaussian",
                        "candidate_neighbor_count": full_contour_count,
                        "max_neighbors": adaptive_max_neighbors,
                        "neighbor_cap_applied": full_contour_count > adaptive_max_neighbors,
                        "top10_min_similarity": float(np.min(top_scores[local_row, : min(10, top_scores.shape[1])])),
                        "top1_similarity": float(top_scores[local_row, 0]),
                        "source_exclusion_count": int(excluded_target_indices.size),
                        "query_accession_excluded": query_cache.ids[query_index] in excluded_source_ids,
                        **diagnostic,
                    }
                )
            if "radius" in strategies:
                for radius, radius_similarity in radius_similarity_thresholds.items():
                    radius_selected = np.flatnonzero(sims[local_row] >= radius_similarity).astype(int).tolist()
                    radius_indices[radius][query_index] = radius_selected
                    radius_scores[radius][query_index] = sims[local_row, radius_selected].astype(float).tolist()
                    wanted_targets.update(target_ids[i] for i in radius_selected)
                    neighbor_counts.append(
                        {
                            "organism": organism,
                            "protein_id": query_cache.ids[query_index],
                            "transfer_strategy": "radius",
                            "distance_metric": "cosine_distance",
                            "radius": radius,
                            "radius_distance_threshold": radius,
                            "radius_similarity_threshold": radius_similarity,
                            "radius_neighbor_count": len(radius_selected),
                            "top10_min_similarity": float(np.min(top_scores[local_row, : min(10, top_scores.shape[1])])),
                            "top1_similarity": float(top_scores[local_row, 0]),
                            "source_exclusion_count": int(excluded_target_indices.size),
                            "query_accession_excluded": query_cache.ids[query_index] in excluded_source_ids,
                        }
                    )
        log(f"Computed clean PLM similarities for {organism}: {end}/{len(query_cache.ids)} query proteins.")

    if preloaded_accession_to_terms is None:
        log(f"Loading GO terms for {len(wanted_targets)} Swiss-Prot neighbor proteins for {organism}.")
        accession_to_terms = read_target_go_terms(go, goa_path, wanted_targets)
    else:
        accession_to_terms = preloaded_accession_to_terms
    predictions = {}
    for score_mode in score_modes:
        if "knn" in strategies:
            for k_value in K_VALUES:
                direct = direct_rows_for_neighbors(
                    query_cache.ids,
                    target_ids,
                    knn_indices[k_value],
                    knn_scores[k_value],
                    accession_to_terms,
                    score_mode,
                )
                model = clean_plm_model_name("knn", score_mode, k_value=k_value)
                predictions[model] = (
                    propagate_predictions(
                        go, direct, f"clean_plm_knn_{organism}_{k_value}_{score_mode}"
                    ),
                    {
                        "transfer_strategy": "knn",
                        "k": k_value,
                        "radius": math.nan,
                        "score_mode": score_mode,
                        "score_mode_label": SCORE_MODE_LABELS[score_mode],
                    },
                )
        if "radius" in strategies:
            for radius in radii:
                radius_direct = direct_rows_for_neighbors(
                    query_cache.ids,
                    target_ids,
                    radius_indices[radius],
                    radius_scores[radius],
                    accession_to_terms,
                    score_mode,
                )
                model = clean_plm_model_name("radius", score_mode, radius=radius)
                predictions[model] = (
                    propagate_predictions(
                        go, radius_direct, f"clean_plm_radius_{organism}_{str(radius).replace('.', '_')}_{score_mode}"
                    ),
                    {
                        "transfer_strategy": "radius",
                        "k": math.nan,
                        "radius": radius,
                        "score_mode": score_mode,
                        "score_mode_label": SCORE_MODE_LABELS[score_mode],
                        "distance_metric": "cosine_distance",
                    },
                )
    if "kde" in strategies:
        kde_direct = direct_rows_for_neighbors(
            query_cache.ids,
            target_ids,
            kde_indices,
            kde_scores,
            accession_to_terms,
            "weighted_support",
            neighbor_weights=kde_weights,
        )
        model = clean_plm_model_name("kde", "kernel_support")
        predictions[model] = (
            propagate_predictions(go, kde_direct, f"clean_plm_kde_{organism}_kernel_support"),
            {
                "transfer_strategy": "kde",
                "k": math.nan,
                "radius": math.nan,
                "score_mode": "gaussian_kernel_support",
                "score_mode_label": "gaussian_kernel_support",
                "distance_metric": "cosine_distance",
                **dict(kde_selection or {}),
            },
        )
    if "adaptive_kde" in strategies:
        adaptive_direct = direct_rows_for_neighbors(
            query_cache.ids,
            target_ids,
            adaptive_indices,
            adaptive_scores,
            accession_to_terms,
            "weighted_support",
            neighbor_weights=adaptive_weights,
            query_confidences=adaptive_confidences,
        )
        model = clean_plm_model_name("adaptive_kde", "gaussian_confidence")
        predictions[model] = (
            propagate_predictions(go, adaptive_direct, f"clean_plm_adaptive_kde_{organism}_gaussian_confidence"),
            {
                "transfer_strategy": "adaptive_kde",
                "k": math.nan,
                "radius": math.nan,
                "score_mode": "adaptive_kernel_support",
                "score_mode_label": "adaptive_kernel_support",
                "distance_metric": "cosine_distance",
                **dict(adaptive_kde_selection or {}),
            },
        )
    return predictions, neighbor_counts


def build_adaptive_kde_ablation_predictions(
    organism: str,
    query_cache: plm.EmbeddingCache,
    target_cache: plm.EmbeddingCache,
    target_norm: np.ndarray,
    go,
    accession_to_terms: Dict[str, Set[str]],
    source_exclusion_ids: Set[str],
    global_kde_selection: Dict[str, object],
    adaptive_kde_selection: Dict[str, object],
    chunk_size: int,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, object]]]:
    """Create non-frontend controls separating topology, weighting, and confidence."""
    target_ids = target_cache.ids
    target_index_by_id = {protein_id: index for index, protein_id in enumerate(target_ids)}
    excluded_target_indices = np.array(
        [target_index_by_id[protein_id] for protein_id in source_exclusion_ids if protein_id in target_index_by_id],
        dtype=int,
    )
    fixed_k = int(adaptive_kde_selection["adaptive_neighbor_rank"])
    max_neighbors = max(int(global_kde_selection["max_neighbors"]), int(adaptive_kde_selection["max_neighbors"]), fixed_k)
    available_count = len(target_ids) - len(excluded_target_indices)
    max_neighbors = min(max_neighbors, available_count)

    variant_neighbors = {
        "Ablation | Fixed KNN cosine": {"indices": [], "scores": [], "weights": [], "confidences": []},
        "Ablation | Fixed KNN gaussian": {"indices": [], "scores": [], "weights": [], "confidences": []},
        "Ablation | Global KDE contour cosine": {"indices": [], "scores": [], "weights": [], "confidences": []},
        "Ablation | Adaptive KDE gaussian": {"indices": [], "scores": [], "weights": [], "confidences": []},
        "Ablation | Adaptive KDE gaussian + confidence": {"indices": [], "scores": [], "weights": [], "confidences": []},
    }
    diagnostics = []
    global_bandwidth = float(global_kde_selection["bandwidth"])
    global_floor = float(global_kde_selection["kernel_weight_floor"])
    for start, end, sims in similarity_blocks(query_cache.embeddings, target_norm, chunk_size):
        if excluded_target_indices.size:
            sims[:, excluded_target_indices] = -np.inf
        top = np.argpartition(-sims, kth=max_neighbors - 1, axis=1)[:, :max_neighbors]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for local_row, query_index in enumerate(range(start, end)):
            protein_id = query_cache.ids[query_index]
            top_indices = top[local_row].astype(int)
            top_score_row = top_scores[local_row].astype(float)
            fixed_indices = top_indices[:fixed_k].tolist()
            fixed_scores = top_score_row[:fixed_k].tolist()
            fixed_relative_weights, fixed_bandwidth, _d1, _dk, _relative_radius = adaptive_relative_gaussian_weights(
                fixed_scores,
                fixed_k,
                float(adaptive_kde_selection["relative_weight_floor"]),
                float(adaptive_kde_selection["adaptive_bandwidth_scale"]),
            )
            variant_neighbors["Ablation | Fixed KNN cosine"]["indices"].append(fixed_indices)
            variant_neighbors["Ablation | Fixed KNN cosine"]["scores"].append(fixed_scores)
            variant_neighbors["Ablation | Fixed KNN cosine"]["weights"].append(fixed_scores)
            variant_neighbors["Ablation | Fixed KNN cosine"]["confidences"].append(1.0)
            variant_neighbors["Ablation | Fixed KNN gaussian"]["indices"].append(fixed_indices)
            variant_neighbors["Ablation | Fixed KNN gaussian"]["scores"].append(fixed_scores)
            variant_neighbors["Ablation | Fixed KNN gaussian"]["weights"].append(fixed_relative_weights.tolist())
            variant_neighbors["Ablation | Fixed KNN gaussian"]["confidences"].append(1.0)

            global_weights = gaussian_kde_weights(top_score_row, global_bandwidth)
            global_mask = global_weights >= global_floor
            global_indices = top_indices[global_mask].astype(int).tolist()
            global_scores = top_score_row[global_mask].astype(float).tolist()
            variant_neighbors["Ablation | Global KDE contour cosine"]["indices"].append(global_indices)
            variant_neighbors["Ablation | Global KDE contour cosine"]["scores"].append(global_scores)
            variant_neighbors["Ablation | Global KDE contour cosine"]["weights"].append(global_scores)
            variant_neighbors["Ablation | Global KDE contour cosine"]["confidences"].append(1.0)

            adaptive = select_adaptive_kde_neighbors(
                [top_indices.tolist()],
                [top_score_row.tolist()],
                neighbor_rank=int(adaptive_kde_selection["adaptive_neighbor_rank"]),
                bandwidth_scale=float(adaptive_kde_selection["adaptive_bandwidth_scale"]),
                relative_weight_floor=float(adaptive_kde_selection["relative_weight_floor"]),
                min_neighbors=int(adaptive_kde_selection["min_neighbors"]),
                confidence_distance_scale=float(adaptive_kde_selection["confidence_distance_scale"]),
            )
            adaptive_indices, adaptive_scores, adaptive_weights, adaptive_confidences, adaptive_diagnostics = adaptive
            for label, confidence in [
                ("Ablation | Adaptive KDE gaussian", 1.0),
                ("Ablation | Adaptive KDE gaussian + confidence", adaptive_confidences[0]),
            ]:
                variant_neighbors[label]["indices"].append(adaptive_indices[0])
                variant_neighbors[label]["scores"].append(adaptive_scores[0])
                variant_neighbors[label]["weights"].append(adaptive_weights[0])
                variant_neighbors[label]["confidences"].append(confidence)
            adaptive_diagnostic = adaptive_diagnostics[0]
            diagnostics.extend(
                [
                    {
                        "organism": organism,
                        "protein_id": protein_id,
                        "ablation_model": "Fixed KNN cosine",
                        "retained_neighbor_count": fixed_k,
                        "median_weight": float(np.median(fixed_scores)),
                        "density_confidence": 1.0,
                    },
                    {
                        "organism": organism,
                        "protein_id": protein_id,
                        "ablation_model": "Fixed KNN gaussian",
                        "retained_neighbor_count": fixed_k,
                        "adaptive_bandwidth": fixed_bandwidth,
                        "median_weight": float(np.median(fixed_relative_weights)),
                        "density_confidence": 1.0,
                    },
                    {
                        "organism": organism,
                        "protein_id": protein_id,
                        "ablation_model": "Global KDE contour cosine",
                        "retained_neighbor_count": len(global_indices),
                        "density_confidence": 1.0,
                    },
                    {
                        "organism": organism,
                        "protein_id": protein_id,
                        "ablation_model": "Adaptive KDE gaussian",
                        **adaptive_diagnostic,
                        "density_confidence": 1.0,
                    },
                    {
                        "organism": organism,
                        "protein_id": protein_id,
                        "ablation_model": "Adaptive KDE gaussian + confidence",
                        **adaptive_diagnostic,
                    },
                ]
            )
        log(f"Computed adaptive KDE ablation similarities for {organism}: {end}/{len(query_cache.ids)} query proteins.")

    predictions = {}
    for label, neighbors in variant_neighbors.items():
        direct = direct_rows_for_neighbors(
            query_cache.ids,
            target_ids,
            neighbors["indices"],
            neighbors["scores"],
            accession_to_terms,
            "weighted_support",
            neighbor_weights=neighbors["weights"],
            query_confidences=neighbors["confidences"],
        )
        namespace = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        predictions[label] = propagate_predictions(go, direct, f"ablation_{organism}_{namespace}")
    return predictions, diagnostics


def saved_global_kde_selections(output_dir: Path) -> Dict[str, Dict[str, object]]:
    """Load blacklist-aware per-organism global KDE controls for the ablation."""
    path = output_dir / "kde_bandwidth_selection.json"
    if not path.exists():
        return {}
    selection = json.loads(path.read_text())
    if selection.get("status") != "blacklist_filter_enabled":
        return {}
    per_organism = selection.get("per_organism", {})
    required = {"bandwidth", "kernel_weight_floor", "max_neighbors"}
    if not all(required.issubset(per_organism.get(organism, {})) for organism in ORGANISMS):
        return {}
    return {organism: dict(per_organism[organism]) for organism in ORGANISMS}


def write_adaptive_kde_effect_summary(
    output_dir: Path,
    active_metrics: pd.DataFrame,
    neighbor_rows: List[Dict[str, object]],
    ablation_rows: List[Dict[str, object]],
    ablation_diagnostics: List[Dict[str, object]],
) -> None:
    """Write one compact table joining performance, coverage, neighborhoods, and confidence."""
    frames = []
    neighbor_df = pd.DataFrame(neighbor_rows)
    if not active_metrics.empty and not neighbor_df.empty:
        adaptive_metrics = active_metrics[active_metrics["transfer_strategy"] == "adaptive_kde"].copy()
        adaptive_neighbors = neighbor_df[neighbor_df["transfer_strategy"] == "adaptive_kde"].copy()
        if not adaptive_metrics.empty and not adaptive_neighbors.empty:
            summary = adaptive_neighbors.groupby("organism", as_index=False).agg(
                median_neighbor_count=("retained_neighbor_count", "median"),
                mean_neighbor_count=("retained_neighbor_count", "mean"),
                fallback_query_count=("minimum_neighbor_fallback_applied", "sum"),
                mean_density_confidence=("density_confidence", "mean"),
                median_density_confidence=("density_confidence", "median"),
                mean_adaptive_bandwidth=("adaptive_bandwidth", "mean"),
                max_neighbor_count=("retained_neighbor_count", "max"),
            )
            summary["configuration"] = "active adaptive KDE gaussian + confidence"
            frames.append(adaptive_metrics.merge(summary, on="organism", how="left"))
    if ablation_rows and ablation_diagnostics:
        ablation_metrics = pd.DataFrame(ablation_rows)
        ablation_neighbors = pd.DataFrame(ablation_diagnostics)
        ablation_metrics = ablation_metrics.rename(columns={"model": "ablation_model"})
        ablation_metrics["ablation_join_key"] = ablation_metrics["ablation_model"].str.replace(
            "Ablation | ", "", regex=False
        )
        summary = ablation_neighbors.groupby(["organism", "ablation_model"], as_index=False).agg(
            median_neighbor_count=("retained_neighbor_count", "median"),
            mean_neighbor_count=("retained_neighbor_count", "mean"),
            mean_density_confidence=("density_confidence", "mean"),
            median_density_confidence=("density_confidence", "median"),
        )
        summary = summary.rename(columns={"ablation_model": "ablation_join_key"})
        ablation_metrics["configuration"] = ablation_metrics["ablation_model"]
        frames.append(ablation_metrics.merge(summary, on=["organism", "ablation_join_key"], how="left"))
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            output_dir / "adaptive_kde_effect_summary.csv", index=False
        )


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    records = []
    clean = df.replace({np.nan: None})
    for row in clean.to_dict(orient="records"):
        records.append({key: value for key, value in row.items() if value is not None})
    return records


def write_kde_knn3_comparison(
    output_dir: Path,
    predictions: pd.DataFrame,
    neighbor_diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Dict[str, object]:
    """Audit whether active Gaussian KDE is observably distinct from weighted KNN-3."""
    knn_model = clean_plm_model_name("knn", "weighted_support", k_value=3)
    required_models = {knn_model, ACTIVE_KDE_MODEL}
    available_models = set(predictions.get("model", pd.Series(dtype=str)).astype(str))
    comparison_rows: List[Dict[str, object]] = []

    if predictions.empty or not required_models.issubset(available_models):
        payload = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "required_models_unavailable",
            "knn_model": knn_model,
            "kde_model": ACTIVE_KDE_MODEL,
            "available_models": sorted(available_models),
            "organisms": [],
        }
        (output_dir / "kde_vs_knn_k3_comparison.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        )
        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "kde_vs_knn_k3_comparison.csv", index=False
        )
        return payload

    organisms = sorted(
        set(predictions.loc[predictions["model"] == knn_model, "organism"].astype(str))
        & set(predictions.loc[predictions["model"] == ACTIVE_KDE_MODEL, "organism"].astype(str))
    )
    for organism in organisms:
        organism_predictions = predictions[predictions["organism"].astype(str) == organism]
        knn = organism_predictions[organism_predictions["model"] == knn_model][
            ["protein_id", "term_id", "score"]
        ].rename(columns={"score": "knn_k3_score"})
        kde = organism_predictions[organism_predictions["model"] == ACTIVE_KDE_MODEL][
            ["protein_id", "term_id", "score"]
        ].rename(columns={"score": "kde_score"})
        joined = knn.merge(kde, on=["protein_id", "term_id"], how="outer", indicator=True)
        same_pair_set = bool((joined["_merge"] == "both").all())
        joined[["knn_k3_score", "kde_score"]] = joined[
            ["knn_k3_score", "kde_score"]
        ].fillna(0.0)
        differences = np.abs(
            joined["knn_k3_score"].to_numpy(dtype=float)
            - joined["kde_score"].to_numpy(dtype=float)
        )
        close = np.isclose(
            joined["knn_k3_score"].to_numpy(dtype=float),
            joined["kde_score"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        kde_neighbors = neighbor_diagnostics[
            (neighbor_diagnostics.get("transfer_strategy") == "kde")
            & (neighbor_diagnostics.get("organism").astype(str) == organism)
        ] if not neighbor_diagnostics.empty else pd.DataFrame()
        retained = pd.to_numeric(
            kde_neighbors.get("retained_neighbor_count", pd.Series(dtype=float)),
            errors="coerce",
        )
        metric_subset = metrics[metrics["organism"].astype(str) == organism]
        knn_metric = metric_subset[metric_subset["model"] == knn_model]
        kde_metric = metric_subset[metric_subset["model"] == ACTIVE_KDE_MODEL]

        def metric_value(frame: pd.DataFrame, column: str) -> Optional[float]:
            if frame.empty or column not in frame:
                return None
            value = pd.to_numeric(frame.iloc[0][column], errors="coerce")
            return float(value) if pd.notna(value) else None

        comparison_rows.append(
            {
                "organism": organism,
                "knn_model": knn_model,
                "kde_model": ACTIVE_KDE_MODEL,
                "knn_prediction_pair_count": int(len(knn)),
                "kde_prediction_pair_count": int(len(kde)),
                "same_prediction_pair_set": same_pair_set,
                "score_comparison_pair_count": int(len(joined)),
                "different_score_pair_count": int(np.count_nonzero(~close)),
                "exact_score_match_count": int(np.count_nonzero(differences == 0.0)),
                "mean_absolute_score_difference": float(differences.mean()) if len(differences) else 0.0,
                "max_absolute_score_difference": float(differences.max()) if len(differences) else 0.0,
                "predictions_identical_within_1e_12": bool(same_pair_set and close.all()),
                "kde_target_count": int(len(retained)),
                "kde_targets_with_more_than_three_donors": int((retained > 3).sum()),
                "kde_min_retained_donors": int(retained.min()) if len(retained) else None,
                "kde_median_retained_donors": float(retained.median()) if len(retained) else None,
                "kde_max_retained_donors": int(retained.max()) if len(retained) else None,
                "knn_overall_fmax": metric_value(knn_metric, "overall::F_max"),
                "kde_overall_fmax": metric_value(kde_metric, "overall::F_max"),
                "knn_overall_aupr": metric_value(knn_metric, "overall::AUPR"),
                "kde_overall_aupr": metric_value(kde_metric, "overall::AUPR"),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    all_distinct = bool(
        not comparison_df.empty
        and (~comparison_df["predictions_identical_within_1e_12"]).all()
    )
    more_than_three_observed = bool(
        not comparison_df.empty
        and (comparison_df["kde_targets_with_more_than_three_donors"] > 0).any()
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all_distinct and more_than_three_observed else "fail",
        "knn_model": knn_model,
        "kde_model": ACTIVE_KDE_MODEL,
        "all_organisms_have_distinct_predictions": all_distinct,
        "more_than_three_donors_observed": more_than_three_observed,
        "organisms": dataframe_records(comparison_df),
    }
    comparison_df.to_csv(output_dir / "kde_vs_knn_k3_comparison.csv", index=False)
    (output_dir / "kde_vs_knn_k3_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    )
    if not all_distinct:
        raise RuntimeError(
            "Gaussian KDE and weighted KNN-3 predictions are still identical for at least one organism; "
            "see kde_vs_knn_k3_comparison.json."
        )
    if not more_than_three_observed:
        raise RuntimeError(
            "Gaussian KDE did not retain more than three donors for any target; "
            "see kde_vs_knn_k3_comparison.json."
        )
    return payload


def snapshot_pre_blacklist_metrics(output_dir: Path) -> Path:
    """Preserve the current unfiltered controls before replacing them."""
    comparison_dir = output_dir / "blacklist_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    destination = comparison_dir / "pre_blacklist_metrics.csv"
    if destination.exists():
        return destination

    sources = [
        output_dir / "clean_plm_benchmark_metrics.csv",
        output_dir / "fixed_radius_baseline_metrics.csv",
        output_dir / "global_kde_baseline_metrics.csv",
    ]
    frames = []
    source_manifest = []
    for source in sources:
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["pre_blacklist_source"] = source.name
        frames.append(frame)
        source_manifest.append({"path": str(source), "sha256": file_sha256(source), "rows": len(frame)})
    if not frames:
        raise RuntimeError("Cannot snapshot pre-blacklist results: no Clean PLM metric files are available.")
    baseline = pd.concat(frames, ignore_index=True, sort=False)
    baseline = baseline.drop_duplicates(subset=["organism", "model"], keep="first")
    baseline.to_csv(destination, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_status": "unfiltered baseline retained before per-organism blacklist filtering",
        "sources": source_manifest,
        "metric_rows": len(baseline),
    }
    (comparison_dir / "pre_blacklist_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return destination


def write_blacklist_comparison(
    output_dir: Path,
    blacklist_audit: pd.DataFrame,
    filtered_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join the preserved unfiltered controls to the new filtered results."""
    baseline_path = output_dir / "blacklist_comparison" / "pre_blacklist_metrics.csv"
    if not baseline_path.exists():
        raise RuntimeError("Pre-blacklist metric snapshot is missing.")
    baseline = pd.read_csv(baseline_path)
    baseline["organism"] = baseline["organism"].astype(str)
    baseline["model"] = baseline["model"].astype(str)
    baseline = baseline.drop_duplicates(subset=["organism", "model"], keep="first")
    filtered = filtered_metrics.copy()
    filtered["organism"] = filtered["organism"].astype(str)
    filtered["model"] = filtered["model"].astype(str)
    filtered = filtered.drop_duplicates(subset=["organism", "model"], keep="last")
    blacklist_audit = blacklist_audit.copy()
    blacklist_audit["organism"] = blacklist_audit["organism"].astype(str)
    join_columns = ["organism", "model"]
    metadata_columns = ["label", "transfer_strategy", "k", "radius", "score_mode", "score_mode_label"]
    rows = filtered.merge(
        baseline[join_columns + [column for column in METRIC_COLUMNS if column in baseline.columns]],
        on=join_columns,
        how="left",
        suffixes=("", "__pre_blacklist"),
    )
    for column in METRIC_COLUMNS:
        previous_column = f"{column}__pre_blacklist"
        if column not in rows or previous_column not in rows:
            continue
        rows[f"previous::{column}"] = pd.to_numeric(rows[previous_column], errors="coerce")
        rows[f"blacklist_filtered::{column}"] = pd.to_numeric(rows[column], errors="coerce")
        rows[f"delta::{column}"] = rows[f"blacklist_filtered::{column}"] - rows[f"previous::{column}"]
    keep_columns = join_columns + [column for column in metadata_columns if column in rows]
    keep_columns += [column for column in rows.columns if column.startswith(("previous::", "blacklist_filtered::", "delta::"))]
    comparison = rows[keep_columns].merge(blacklist_audit, on="organism", how="left")
    comparison_path = output_dir / "blacklist_comparison" / "clean_plm_blacklist_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blacklist_filter_enabled",
        "description": (
            "Clean PLM comparison between the preserved unfiltered baseline and results generated after "
            "per-test-organism taxon blacklists were applied before neighbour selection and GO transfer."
        ),
        "organism_filters": dataframe_records(blacklist_audit),
        "rows": dataframe_records(comparison),
    }
    # Preserve the historical comparison beside the run for audit only.  Its
    # pre-blacklist baseline used an older evaluation boundary and therefore
    # must not be surfaced as a current explorer result.
    audit_path = output_dir / "blacklist_comparison" / "clean_plm_blacklist_comparison.json"
    audit_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return comparison


def load_competitor_context() -> pd.DataFrame:
    frames = []
    for path in sorted((S2F_ROOT / "figures").glob("*/old/competitors_comparison.csv")):
        df = pd.read_csv(path)
        df["source_csv"] = str(path.relative_to(S2F_ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def benchmark_spreadsheet_rows(rows: pd.DataFrame) -> List[Dict[str, object]]:
    out = []
    for row in rows.replace({np.nan: None}).to_dict(orient="records"):
        model = str(row.get("model", ""))
        method_key = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        out.append(
            {
                "row_type": "benchmark",
                "organism": row.get("organism"),
                "taxon": row.get("organism"),
                "method": model,
                "method_key": method_key,
                "method_family": "Clean PLM" if model.startswith("Clean PLM") else model,
                "display_in_frontend": True,
                "drilldown_available": False,
                "proteins_considered": row.get("proteins_considered"),
                "terms_considered": row.get("terms_considered"),
                "total_annotations": row.get("total_annotations"),
                "matrix_shape": row.get("matrix_shape"),
                "AUC": row.get("overall::AUC"),
                "AUPR": row.get("overall::AUPR"),
                "F_max": row.get("overall::F_max"),
                "smin": row.get("overall::smin"),
                "source_type": "full_benchmark_context",
                "source_file": row.get("source_csv"),
                "source_note": row.get(
                    "source_note",
                    "Full benchmark aggregate; per-annotation drill-down is not available in this artifact.",
                ),
            }
        )
    return out


def archive_fixed_radius_outputs(output_dir: Path) -> Dict[str, object]:
    """Preserve existing fixed-radius controls before KDE becomes the active radius replacement."""
    metric_frames = []
    for path in [
        output_dir / "clean_plm_benchmark_metrics.csv",
        FRONTEND_DATA / "clean_plm_benchmark_metrics.csv",
        output_dir / "fixed_radius_baseline_metrics.csv",
    ]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "model" in frame:
            frame = frame[frame["model"].astype(str).str.startswith("Clean PLM + Radius")]
        if not frame.empty:
            metric_frames.append(frame)
    archived_metric_rows = 0
    metric_archive = output_dir / "fixed_radius_baseline_metrics.csv"
    if metric_frames:
        archived_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        archived_metrics = archived_metrics.drop_duplicates(subset=["organism", "model"], keep="last")
        archived_metrics = archived_metrics.sort_values(["organism", "model"])
        archived_metrics.to_csv(metric_archive, index=False)
        archived_metric_rows = len(archived_metrics)

    prediction_frames = []
    prediction_archive = output_dir / "fixed_radius_predictions.tsv"
    for path in [output_dir / "clean_plm_predictions.tsv", prediction_archive]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t")
        if "model" in frame:
            frame = frame[frame["model"].astype(str).str.startswith("Clean PLM + Radius")]
        if not frame.empty:
            prediction_frames.append(frame)
    archived_prediction_rows = 0
    if prediction_frames:
        archived_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
        archived_predictions = archived_predictions.drop_duplicates(
            subset=["organism", "model", "protein_id", "term_id"], keep="last"
        )
        archived_predictions.to_csv(prediction_archive, sep="\t", index=False)
        archived_prediction_rows = len(archived_predictions)

    detail_files = sorted(
        path.relative_to(S2F_ROOT).as_posix()
        for path in (FRONTEND_DATA / "pfp_prediction_details").glob("*Radius*.json")
    )
    archive = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "fixed-radius controls archived; KNN and shared-bandwidth Gaussian KDE are active",
        "metric_archive": str(metric_archive),
        "metric_rows": archived_metric_rows,
        "prediction_archive": str(prediction_archive),
        "prediction_rows": archived_prediction_rows,
        "retained_detail_files": detail_files,
    }
    (output_dir / "fixed_radius_archive_manifest.json").write_text(
        json.dumps(archive, indent=2, sort_keys=True)
    )
    return archive


def archive_global_kde_outputs(output_dir: Path) -> Dict[str, object]:
    """Preserve the earlier global-bandwidth KDE as a reproducible control."""
    metric_archive = output_dir / "global_kde_baseline_metrics.csv"
    prediction_archive = output_dir / "global_kde_baseline_predictions.tsv"
    metric_frames = []
    for path in [output_dir / "clean_plm_benchmark_metrics.csv", FRONTEND_DATA / "clean_plm_benchmark_metrics.csv", metric_archive]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["model"].astype(str) == "Clean PLM + KDE (kernel_support)"]
        if not frame.empty:
            metric_frames.append(frame)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        metrics.drop_duplicates(subset=["organism", "model"], keep="last").to_csv(metric_archive, index=False)

    prediction_frames = []
    for path in [output_dir / "clean_plm_predictions.tsv", prediction_archive]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t")
        frame = frame[frame["model"].astype(str) == "Clean PLM + KDE (kernel_support)"]
        if not frame.empty:
            prediction_frames.append(frame)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
        predictions.drop_duplicates(subset=["organism", "model", "protein_id", "term_id"], keep="last").to_csv(
            prediction_archive, sep="\t", index=False
        )
    archive = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "legacy global KDE archived; adaptive KDE is the active kernel method",
        "metric_archive": str(metric_archive),
        "metric_rows": int(len(pd.read_csv(metric_archive))) if metric_archive.exists() else 0,
        "prediction_archive": str(prediction_archive),
        "prediction_rows": int(len(pd.read_csv(prediction_archive, sep="\t"))) if prediction_archive.exists() else 0,
    }
    (output_dir / "global_kde_archive_manifest.json").write_text(json.dumps(archive, indent=2, sort_keys=True))
    return archive


def archive_adaptive_kde_outputs(output_dir: Path) -> Dict[str, object]:
    """Preserve the former rank-adaptive active method before KDE replacement."""
    metric_archive = output_dir / "adaptive_kde_legacy_metrics.csv"
    prediction_archive = output_dir / "adaptive_kde_legacy_predictions.tsv"
    metric_frames = []
    for path in [output_dir / "clean_plm_benchmark_metrics.csv", FRONTEND_DATA / "clean_plm_benchmark_metrics.csv", metric_archive]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["model"].astype(str) == LEGACY_ADAPTIVE_KDE_MODEL]
        if not frame.empty:
            metric_frames.append(frame)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        metrics.drop_duplicates(subset=["organism", "model"], keep="last").to_csv(metric_archive, index=False)

    prediction_frames = []
    for path in [output_dir / "clean_plm_predictions.tsv", prediction_archive]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t")
        frame = frame[frame["model"].astype(str) == LEGACY_ADAPTIVE_KDE_MODEL]
        if not frame.empty:
            prediction_frames.append(frame)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
        predictions.drop_duplicates(
            subset=["organism", "model", "protein_id", "term_id"], keep="last"
        ).to_csv(prediction_archive, sep="\t", index=False)

    selection_source = output_dir / "adaptive_kde_selection.json"
    selection_archive = output_dir / "adaptive_kde_legacy_selection.json"
    if selection_source.exists() and not selection_archive.exists():
        shutil.copy2(selection_source, selection_archive)
    archive = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "rank-adaptive KDE retained as a legacy control; shared-bandwidth Gaussian KDE is active",
        "metric_archive": str(metric_archive),
        "metric_rows": int(len(pd.read_csv(metric_archive))) if metric_archive.exists() else 0,
        "prediction_archive": str(prediction_archive),
        "prediction_rows": int(len(pd.read_csv(prediction_archive, sep="\t"))) if prediction_archive.exists() else 0,
        "selection_archive": str(selection_archive) if selection_archive.exists() else None,
        "retained_detail_files": sorted(
            path.relative_to(S2F_ROOT).as_posix()
            for path in (FRONTEND_DATA / "pfp_prediction_details").glob("*Adaptive_KDE*.json")
        ),
    }
    (output_dir / "adaptive_kde_legacy_archive_manifest.json").write_text(
        json.dumps(archive, indent=2, sort_keys=True)
    )
    return archive


def refresh_frontend_data(clean_metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    competitor = load_competitor_context()
    combined = pd.concat([competitor, clean_metrics], ignore_index=True, sort=False)
    order = {
        "S2F": 0,
        "TALE": 1,
        "ATGO": 2,
        "PANDA2": 3,
        "Clean PLM + KNN k=3": 4,
        "Clean PLM + KNN k=5": 5,
        "Clean PLM + KNN k=7": 6,
        "Clean PLM + KNN k=10": 7,
        "Clean PLM + KNN k=3 (weighted_support)": 4,
        "Clean PLM + KNN k=5 (weighted_support)": 5,
        "Clean PLM + KNN k=7 (weighted_support)": 6,
        "Clean PLM + KNN k=10 (weighted_support)": 7,
        ACTIVE_KDE_MODEL: 8,
        LEGACY_ADAPTIVE_KDE_MODEL: 9,
        "Clean PLM + KDE (kernel_support)": 10,
        "Clean PLM + Radius R=0.01 (weighted_support)": 10,
        "Clean PLM + Radius R=0.02 (weighted_support)": 11,
        "Clean PLM + Radius R=0.03 (weighted_support)": 12,
    }
    combined["_rank"] = combined["model"].map(order).fillna(20).astype(int)
    combined = combined.sort_values(["organism", "_rank", "model"]).drop(columns=["_rank"])
    frontend_combined = combined[
        ~combined["organism"].astype(str).isin(FRONTEND_EXCLUDED_ORGANISMS)
    ].copy()

    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
    frontend_combined.to_csv(FRONTEND_DATA / "competitor_context_metrics.csv", index=False)
    clean_metrics.to_csv(FRONTEND_DATA / "clean_plm_benchmark_metrics.csv", index=False)

    pfp_path = FRONTEND_DATA / "probe_pfp_metrics.json"
    payload = json.loads(pfp_path.read_text())
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("source", {})["full_benchmark_source"] = (
        "figures/*/old/competitors_comparison.csv plus clean_plm_benchmark_metrics.csv"
    )
    payload["source"]["frontend_excluded_organisms"] = sorted(FRONTEND_EXCLUDED_ORGANISMS)
    payload["benchmark_context"] = {
        "description": (
            "Full benchmark metrics on the shared competitor/S2F evaluation protein set. "
            "Taxon 223283 is excluded from the frontend because only three proteins remain."
        ),
        "rows": dataframe_records(frontend_combined),
    }
    pfp_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))

    spreadsheet_path = FRONTEND_DATA / "pfp_spreadsheet.json"
    spreadsheet = json.loads(spreadsheet_path.read_text())
    spreadsheet["generated_at_utc"] = payload["generated_at_utc"]
    spreadsheet.setdefault("source", {})["benchmark_source"] = (
        "figures/*/old/competitors_comparison.csv plus clean_plm_benchmark_metrics.csv"
    )
    spreadsheet["benchmark_rows"] = benchmark_spreadsheet_rows(frontend_combined)
    spreadsheet_path.write_text(json.dumps(spreadsheet, separators=(",", ":"), allow_nan=False))
    pd.DataFrame(spreadsheet["benchmark_rows"]).to_csv(FRONTEND_DATA / "pfp_spreadsheet.csv", index=False)

    index_path = FRONTEND_DATA / "index.json"
    index_payload = json.loads(index_path.read_text())
    index_payload.pop("clean_plm_blacklist_comparison", None)
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True, allow_nan=False))
    return frontend_combined


def main() -> None:
    args = parse_args()
    strategies = selected_strategies(args)
    if "kde" in strategies or "adaptive_kde" in strategies:
        validate_kde_args(args)
    radii = selected_radii(args) if "radius" in strategies else []
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_blacklist_metrics = snapshot_pre_blacklist_metrics(output_dir)
    saved_kde_selections = saved_global_kde_selections(output_dir)
    fixed_radius_archive = (
        archive_fixed_radius_outputs(output_dir) if "radius" not in strategies else None
    )
    global_kde_archive = (
        archive_global_kde_outputs(output_dir)
        if "adaptive_kde" in strategies and "kde" not in strategies
        else None
    )
    adaptive_kde_legacy_archive = (
        archive_adaptive_kde_outputs(output_dir)
        if "kde" in strategies and "adaptive_kde" not in strategies
        else None
    )

    plm_args = ArgsForPlm(
        model_name="facebook/esm1b_t33_650M_UR50S",
        model_dir=args.model_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        long_sequence_mode="sliding_mean",
        long_window_size=1022,
        long_overlap=128,
        batch_tokens=args.batch_tokens,
        query_chunk_size=args.query_chunk_size,
    )
    goa_path = (
        Path(args.goa_path).expanduser().resolve()
        if args.goa_path
        else (DATA_ROOT / "uniprot" / "filtered_goa").resolve()
    )
    if not goa_path.is_file():
        raise RuntimeError(f"Missing filtered GOA file: {goa_path}")
    go_obo = S2F_ROOT / "go.obo"
    target_cache, target_cache_dir, target_fasta = load_or_build_target_cache(
        args,
        output_dir,
        plm_args,
    )
    log(f"Loaded Swiss-Prot target embeddings: {len(target_cache.ids)} proteins from {target_cache_dir}.")
    target_norm_path = Path("/tmp") / f"clean_plm_target_norm_{target_cache_dir.name}.npy"
    log(f"Normalizing Swiss-Prot target embeddings once for all clean PLM searches at {target_norm_path}.")
    target_norm = normalized_target_memmap(target_cache.embeddings, target_norm_path)
    target_cache.embeddings = target_norm
    gc.collect()

    evaluation_sets, evaluation_summary = build_evaluation_sets(goa_path)
    evaluation_summary.to_csv(output_dir / "evaluation_protein_sets.csv", index=False)
    for organism in ORGANISMS:
        log(f"Shared evaluation set for {organism}: {len(evaluation_sets[organism])} proteins.")
    benchmark_exclusion_ids = set().union(*evaluation_sets.values())
    blacklist_dir = resolve_blacklist_dir(args.blacklist_dir)
    organism_blacklists, blacklist_paths = load_organism_blacklists(blacklist_dir)
    source_exclusions_by_organism, blacklist_audit = build_source_exclusions_by_organism(
        target_cache,
        benchmark_exclusion_ids,
        organism_blacklists,
        blacklist_paths,
    )
    blacklist_audit["blacklist_dir"] = str(blacklist_dir)
    blacklist_audit["pre_blacklist_metrics"] = str(pre_blacklist_metrics)
    blacklist_audit.to_csv(output_dir / "clean_plm_blacklist_audit.csv", index=False)
    blacklist_filter_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blacklist_filter_enabled",
        "policy": (
            "Each test organism excludes the complete shared benchmark evaluation set and its own related-organism "
            "taxon blacklist before every KNN, fixed-radius, KDE, adaptive-KDE, calibration, and ablation search."
        ),
        "organisms": dataframe_records(blacklist_audit),
    }
    (output_dir / "clean_plm_blacklist_filter.json").write_text(
        json.dumps(blacklist_filter_payload, indent=2, sort_keys=True, allow_nan=False)
    )
    for audit in dataframe_records(blacklist_audit):
        log(
            f"Blacklist filtering for {audit['organism']}: {audit['blacklisted_source_protein_count']} related-organism "
            f"donors + {audit['benchmark_exclusion_count']} shared-evaluation accessions -> "
            f"{audit['eligible_source_protein_count']} eligible Swiss-Prot donors."
        )
    log("Active Clean PLM strategies: " + ", ".join(strategies))
    if radii:
        log("Radius transfer will use fixed cosine-distance thresholds: " + ", ".join(f"{radius:g}" for radius in radii))

    go = GeneOntology.GeneOntology(str(go_obo), verbose=False)
    go.build_structure()
    kde_selections: Dict[str, Dict[str, object]] = {}
    shared_kde_selection: Dict[str, object] = {}
    adaptive_kde_selections: Dict[str, Dict[str, object]] = {}
    preloaded_accession_to_terms: Optional[Dict[str, Set[str]]] = None
    if "kde" in strategies or "adaptive_kde" in strategies:
        log("Loading experimental GO terms for the source-only KDE calibration pool.")
        preloaded_accession_to_terms = read_target_go_terms(go, goa_path, set(target_cache.ids))
        log(
            f"Experimental GO terms are available for {len(preloaded_accession_to_terms)}/"
            f"{len(target_cache.ids)} target-cache proteins."
        )
    query_caches: Dict[str, plm.EmbeddingCache] = {}
    skipped = []
    for organism in ORGANISMS:
        query_cache = query_cache_for_organism(
            organism,
            evaluation_sets[organism],
            output_dir,
            plm_args,
            force=args.force_query_embeddings,
            skip_missing=args.skip_missing_embeddings,
        )
        if query_cache is None:
            skipped.append({"organism": organism, "reason": "query embeddings unavailable"})
        else:
            query_caches[organism] = query_cache

    audit_by_organism = {
        str(row["organism"]): row
        for row in blacklist_audit.to_dict(orient="records")
    }
    if "kde" in strategies:
        shared_selection, candidates, calibration_records = calibrate_shared_kde_bandwidth(
            target_cache,
            target_norm,
            preloaded_accession_to_terms or {},
            source_exclusions_by_organism,
            go,
            go_obo,
            chunk_size=args.query_chunk_size,
            calibration_size=args.kde_calibration_size,
            max_neighbors=args.kde_max_neighbors,
            weight_floor=args.kde_weight_floor,
            selection_metric=args.kde_selection_metric,
            random_seed=args.random_seed,
            explicit_bandwidths=args.kde_bandwidths,
        )
        shared_selection, candidates = constrain_shared_kde_to_unlabeled_query_support(
            shared_selection,
            candidates,
            query_caches,
            target_norm,
            target_cache.ids,
            source_exclusions_by_organism,
            args.query_chunk_size,
        )
        shared_kde_selection = shared_selection
        for organism in ORGANISMS:
            kde_selections[organism] = {
                **shared_selection,
                **audit_by_organism[organism],
                "organism": organism,
            }
        candidates.to_csv(output_dir / "kde_bandwidth_calibration.csv", index=False)
        calibration_records.to_csv(output_dir / "kde_calibration_proteins.csv", index=False)
        (output_dir / "kde_bandwidth_selection.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "status": "blacklist_filter_enabled",
                    "shared": shared_selection,
                    "per_organism": kde_selections,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    if "adaptive_kde" in strategies:
        if not query_caches:
            raise RuntimeError("Adaptive KDE requires at least one query embedding cache for unlabeled distance matching.")
        candidate_frames = []
        repeat_frames = []
        calibration_frames = []
        profile_frames = []
        strata_frames = []
        for index, organism in enumerate(ORGANISMS):
            query_cache = query_caches.get(organism)
            if query_cache is None:
                continue
            query_profile = query_nearest_distance_profile(
                {organism: query_cache},
                target_norm,
                target_cache.ids,
                source_exclusions_by_organism[organism],
                args.query_chunk_size,
            )
            (
                selection,
                candidates,
                repeat_results,
                calibration_records,
                profile_summary,
            ) = calibrate_adaptive_kde(
                target_cache,
                target_norm,
                preloaded_accession_to_terms or {},
                source_exclusions_by_organism[organism],
                query_profile,
                go,
                go_obo,
                chunk_size=args.query_chunk_size,
                calibration_size=args.adaptive_kde_calibration_size,
                calibration_repeats=args.adaptive_kde_calibration_repeats,
                distance_bins=args.adaptive_kde_distance_bins,
                neighbor_counts=args.adaptive_kde_neighbor_counts,
                bandwidth_scales=args.adaptive_kde_bandwidth_scales,
                relative_weight_floor=args.adaptive_kde_relative_weight_floor,
                min_neighbors=args.adaptive_kde_min_neighbors,
                confidence_distance_scales=args.adaptive_kde_confidence_distance_scales,
                fmax_se_multiplier=args.adaptive_kde_fmax_se_multiplier,
                max_neighbors=args.kde_max_neighbors,
                random_seed=args.random_seed + index,
            )
            selection.update(audit_by_organism[organism])
            adaptive_kde_selections[organism] = selection
            candidate_frames.append(candidates.assign(organism=organism))
            repeat_frames.append(repeat_results.assign(organism=organism))
            calibration_frames.append(calibration_records.assign(organism=organism))
            profile_frames.append(query_profile.assign(organism=organism))
            strata_frames.append(profile_summary.assign(organism=organism))
        if len(adaptive_kde_selections) != len(query_caches):
            raise RuntimeError("Adaptive KDE calibration did not produce a selection for every available test organism.")
        pd.concat(profile_frames, ignore_index=True).to_csv(
            output_dir / "adaptive_kde_query_distance_profile.csv", index=False
        )
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            output_dir / "adaptive_kde_calibration_summary.csv", index=False
        )
        pd.concat(repeat_frames, ignore_index=True).to_csv(
            output_dir / "adaptive_kde_calibration_repeats.csv", index=False
        )
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            output_dir / "adaptive_kde_calibration_proteins.csv", index=False
        )
        pd.concat(strata_frames, ignore_index=True).to_csv(
            output_dir / "adaptive_kde_distance_strata.csv", index=False
        )
        (output_dir / "adaptive_kde_selection.json").write_text(
            json.dumps(
                {"schema_version": 2, "status": "blacklist_filter_enabled", "per_organism": adaptive_kde_selections},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    metric_rows = []
    neighbor_rows = []
    prediction_frames = []

    for organism in ORGANISMS:
        proteins = evaluation_sets[organism]
        log(f"Starting clean PLM benchmark for {organism}.")
        query_cache = query_caches.get(organism)
        if query_cache is None:
            continue

        annotations, ontology, organism_name = prepare_ground_truth(goa_path, go_obo, organism, proteins)
        predictions_by_model, counts = build_clean_plm_predictions(
            organism,
            query_cache,
            target_cache,
            target_norm,
            go,
            goa_path,
            radii=radii,
            chunk_size=args.query_chunk_size,
            score_modes=args.score_modes,
            strategies=strategies,
            kde_selection=kde_selections.get(organism),
            adaptive_kde_selection=adaptive_kde_selections.get(organism),
            preloaded_accession_to_terms=preloaded_accession_to_terms,
            source_exclusion_ids=source_exclusions_by_organism[organism],
        )
        for count in counts:
            count.update(
                {
                    "blacklist_filter_enabled": True,
                    "blacklist_taxa_count": int(audit_by_organism[organism]["blacklist_taxa_count"]),
                    "blacklisted_source_protein_count": int(
                        audit_by_organism[organism]["blacklisted_source_protein_count"]
                    ),
                    "eligible_source_protein_count": int(
                        audit_by_organism[organism]["eligible_source_protein_count"]
                    ),
                }
            )
        neighbor_rows.extend(counts)
        for model, (predictions, model_metadata) in predictions_by_model.items():
            model_metadata = {**model_metadata, **audit_by_organism[organism]}
            if model_metadata["transfer_strategy"] == "radius":
                radius_value = float(model_metadata["radius"])
                strategy_note = (
                    f"radius={radius_value:g} cosine-distance threshold "
                    f"(cosine similarity >= {1.0 - radius_value:g})"
                )
            elif model_metadata["transfer_strategy"] == "knn":
                strategy_note = f"KNN k={int(model_metadata['k'])}"
            elif model_metadata["transfer_strategy"] == "adaptive_kde":
                strategy_note = (
                    f"adaptive Gaussian KDE rank={int(model_metadata['adaptive_neighbor_rank'])}; "
                    f"bandwidth scale={float(model_metadata['adaptive_bandwidth_scale']):g}; "
                    f"relative weight floor={float(model_metadata['relative_weight_floor']):g}; "
                    f"minimum donors={int(model_metadata['min_neighbors'])}; "
                    f"density confidence distance scale={float(model_metadata['confidence_distance_scale']):g}; "
                    f"configuration selected by repeated source-only, query-distance-matched calibration"
                )
            else:
                strategy_note = (
                    f"Gaussian KDE bandwidth h={float(model_metadata['bandwidth']):.6g}; "
                    f"relative cosine-distance radius="
                    f"{float(model_metadata['relative_distance_radius']):.6g} at "
                    f"relative kernel weight floor={float(model_metadata['relative_weight_floor']):g}; "
                    f"bandwidth selected on {int(model_metadata['calibration_size'])} source-only pseudo-queries "
                    f"by {model_metadata['selection_metric']}"
                )
            source_note = (
                f"Clean PLM transfer using ESM-1b embeddings from plm.py; "
                f"target cache={target_cache_dir.name}; {strategy_note}; "
                f"per-test-organism related-taxon blacklist applied before neighbour selection and GO transfer "
                f"({int(audit_by_organism[organism]['blacklist_taxa_count'])} taxa, "
                f"{int(audit_by_organism[organism]['blacklisted_source_protein_count'])} source proteins); "
                f"shared benchmark evaluation accessions excluded from Swiss-Prot source pool "
                f"({int(audit_by_organism[organism]['benchmark_exclusion_count'])} accessions); "
                f"score_mode={model_metadata['score_mode']}."
            )
            row = evaluate_prediction_table(
                model,
                organism,
                predictions,
                annotations,
                ontology,
                organism_name,
                evaluation_size=len(proteins),
                source_note=source_note,
                extra_metadata=model_metadata,
            )
            metric_rows.append(row)
            predictions_with_meta = predictions.copy()
            predictions_with_meta["organism"] = organism
            predictions_with_meta["model"] = model
            prediction_frames.append(predictions_with_meta)
            print(f"Evaluated {model} on {organism}: {row['matrix_shape']}")

    ablation_rows = []
    ablation_diagnostics = []
    ablation_prediction_frames = []
    global_kde_controls = kde_selections or saved_kde_selections
    if not args.skip_ablation and adaptive_kde_selections and global_kde_controls:
        log("Running non-frontend ablation: fixed KNN, global KDE contour, and adaptive KDE controls.")
        for organism in ORGANISMS:
            query_cache = query_caches.get(organism)
            if query_cache is None or organism not in adaptive_kde_selections or organism not in global_kde_controls:
                continue
            ablation_go = GeneOntology.GeneOntology(str(go_obo), verbose=False)
            ablation_go.build_structure()
            predictions, diagnostics = build_adaptive_kde_ablation_predictions(
                organism,
                query_cache,
                target_cache,
                target_norm,
                ablation_go,
                preloaded_accession_to_terms or {},
                source_exclusions_by_organism[organism],
                global_kde_controls[organism],
                adaptive_kde_selections[organism],
                args.query_chunk_size,
            )
            annotations, ontology, organism_name = prepare_ground_truth(goa_path, go_obo, organism, evaluation_sets[organism])
            for model, predictions_df in predictions.items():
                row = evaluate_prediction_table(
                    model,
                    organism,
                    predictions_df,
                    annotations,
                    ontology,
                    organism_name,
                    evaluation_size=len(evaluation_sets[organism]),
                    source_note=(
                        "Non-frontend adaptive KDE ablation. It uses the same per-test-organism blacklist-filtered "
                        "donor pool, query proteins, GO annotations, ontology propagation, and evaluator as the active benchmark."
                    ),
                    extra_metadata={
                        "ablation": True,
                        **audit_by_organism[organism],
                        "adaptive_kde_selection": json.dumps(adaptive_kde_selections[organism], sort_keys=True),
                        "global_kde_control": json.dumps(global_kde_controls[organism], sort_keys=True),
                    },
                )
                ablation_rows.append(row)
                frame = predictions_df.copy()
                frame["organism"] = organism
                frame["model"] = model
                ablation_prediction_frames.append(frame)
            ablation_diagnostics.extend(diagnostics)
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_csv(output_dir / "adaptive_kde_ablation_metrics.csv", index=False)
            pd.DataFrame(ablation_diagnostics).to_csv(
                output_dir / "adaptive_kde_ablation_neighbor_diagnostics.csv", index=False
            )
            pd.concat(ablation_prediction_frames, ignore_index=True).to_csv(
                output_dir / "adaptive_kde_ablation_predictions.tsv", sep="\t", index=False
            )
    elif not args.skip_ablation:
        log("Skipping ablation because no saved or freshly calibrated global KDE control is available.")

    metrics_df = pd.DataFrame(metric_rows)
    kde_knn3_comparison: Dict[str, object] = {
        "status": "not_run",
        "reason": "KNN and Gaussian KDE predictions were not both available.",
    }
    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_clean_plm.csv", index=False)
    if not metrics_df.empty:
        for column in METRIC_COLUMNS:
            metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")
        metrics_df.to_csv(output_dir / "clean_plm_benchmark_metrics.csv", index=False)
        all_predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["protein_id", "term_id", "score", "organism", "model"])
        )
        all_predictions.to_csv(output_dir / "clean_plm_predictions.tsv", sep="\t", index=False)
        neighbor_df = pd.DataFrame(neighbor_rows)
        neighbor_df.to_csv(output_dir / "clean_plm_neighbor_counts.csv", index=False)
        if not neighbor_df.empty and "transfer_strategy" in neighbor_df:
            neighbor_df[neighbor_df["transfer_strategy"] == "kde"].to_csv(
                output_dir / "clean_plm_kde_neighbor_diagnostics.csv", index=False
            )
            neighbor_df[neighbor_df["transfer_strategy"] == "adaptive_kde"].to_csv(
                output_dir / "clean_plm_adaptive_kde_neighbor_diagnostics.csv", index=False
            )
        if "knn" in strategies and "kde" in strategies and "weighted_support" in args.score_modes:
            kde_knn3_comparison = write_kde_knn3_comparison(
                output_dir,
                all_predictions,
                neighbor_df,
                metrics_df,
            )
        write_blacklist_comparison(output_dir, blacklist_audit, metrics_df)
        active_metrics = metrics_df[metrics_df["transfer_strategy"].isin(DEFAULT_STRATEGIES)].copy()
        if args.skip_frontend_refresh:
            combined = active_metrics
        else:
            combined = refresh_frontend_data(active_metrics, output_dir)
    else:
        combined = load_competitor_context()

    write_adaptive_kde_effect_summary(
        output_dir,
        metrics_df,
        neighbor_rows,
        ablation_rows,
        ablation_diagnostics,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "organisms": ORGANISMS,
        "strategies": strategies,
        "k_values": K_VALUES,
        "radii": radii,
        "radius": radii[0] if len(radii) == 1 else None,
        "score_modes": args.score_modes,
        "kde": kde_selections,
        "shared_kde": shared_kde_selection,
        "kde_calibration_proteins": int(shared_kde_selection.get("calibration_size", 0)),
        "adaptive_kde": adaptive_kde_selections,
        "adaptive_kde_calibration_proteins": int(
            sum(
                int(selection.get("calibration_size", 0)) * int(selection.get("calibration_repeats", 0))
                for selection in adaptive_kde_selections.values()
            )
        ),
        "adaptive_kde_query_distance_profile_rows": int(
            sum(len(cache.ids) for cache in query_caches.values())
        ),
        "ablation_rows": int(len(ablation_rows)),
        "ablation_global_kde_control": global_kde_controls,
        "fixed_radius_archive": fixed_radius_archive,
        "global_kde_archive": global_kde_archive,
        "adaptive_kde_legacy_archive": adaptive_kde_legacy_archive,
        "kde_vs_knn_k3_comparison": kde_knn3_comparison,
        "blacklist_filter": blacklist_filter_payload,
        "pre_blacklist_metrics": str(pre_blacklist_metrics),
        "target_cache": str(target_cache_dir),
        "target_fasta": str(target_fasta) if target_fasta is not None else None,
        "goa_path": str(goa_path),
        "output_dir": str(output_dir),
        "clean_metric_rows": int(len(metrics_df)),
        "frontend_refreshed": not args.skip_frontend_refresh,
        "frontend_benchmark_rows": int(len(combined)) if not args.skip_frontend_refresh else None,
        "skipped": skipped,
    }
    (output_dir / "clean_plm_benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
