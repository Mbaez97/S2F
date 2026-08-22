#!/usr/bin/env python3
"""Create joint 3D embedding projections for Clean PLM neighbour inspection.

The Clean PLM benchmark has a comparatively small shared test set and a large
Swiss-Prot donor cache.  This exporter therefore retains every test protein,
every Swiss-Prot donor selected by an actual benchmark neighbourhood, the 32
closest donors for every test protein, and a deterministic background sample
of the remaining donor cache.  PCA, UMAP, and t-SNE are each fitted *once* to
this same combined cohort.  Test and donor coordinates are consequently
directly comparable inside a projection.

The data payload also records the exact donor IDs selected by the active KNN,
fixed-radius, and shared-bandwidth Gaussian-KDE configurations.  The
frontend can use those records to draw a test protein's real transfer
neighbourhood rather than an approximate 3D nearest-neighbour relation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-s2f-neighbour-embedding")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2f-neighbour-embedding")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT))

import clean_plm_benchmark as benchmark  # noqa: E402
import plm  # noqa: E402
from GOTool import GeneOntology  # noqa: E402


RANDOM_SEED = 20260715
BACKGROUND_DONORS = 0
LOCAL_NEIGHBORS = 10
FIXED_RADIUS = 0.01
PROJECTION_KEYS = ("pca", "umap", "tsne")
SOURCE_HEADER_ORGANISM = re.compile(r"(?:^|\s)OS=(.+?)(?:\sOX=|$)")
SOURCE_HEADER_TAXON = re.compile(r"(?:^|\s)OX=(\d+)(?:\s|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build jointly fitted PCA, UMAP, and t-SNE Clean PLM neighbourhood projections.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=S2F_ROOT,
        help="S2F repository root.",
    )
    parser.add_argument(
        "--background-donors",
        type=int,
        default=BACKGROUND_DONORS,
        help="Deterministic Swiss-Prot background sample size (default: %(default)s).",
    )
    parser.add_argument(
        "--local-neighbors",
        type=int,
        default=LOCAL_NEIGHBORS,
        help="Closest donors retained for every test protein (default: %(default)s).",
    )
    parser.add_argument(
        "--blacklist-dir",
        default="",
        help=(
            "Directory containing the Clean PLM per-test-organism taxon blacklists. "
            "Defaults to data/blacklists in the source or portable bundle."
        ),
    )
    parser.add_argument(
        "--sync-transfer",
        action="store_true",
        help="Mirror the generated data and explorer assets into the local transfer bundle.",
    )
    return parser.parse_args()


def normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def organism_label(taxon: str) -> str:
    return f"NCBI taxon {taxon}"


def parse_swissprot_header(header: str, fallback_taxon: str | None = None) -> Dict[str, str | None]:
    organism_match = SOURCE_HEADER_ORGANISM.search(header or "")
    taxon_match = SOURCE_HEADER_TAXON.search(header or "")
    return {
        "organism": organism_match.group(1).strip() if organism_match else None,
        "taxon": taxon_match.group(1) if taxon_match else fallback_taxon,
    }


def read_go_names(go_obo: Path) -> Dict[str, str]:
    names: Dict[str, str] = {}
    current_id: str | None = None
    with go_obo.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ", 1)[1]
            elif current_id and line.startswith("name: "):
                names[current_id] = line.split("name: ", 1)[1]
    return names


def top_donors(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    excluded_indices: Set[int],
    count: int,
    block_size: int = 4_096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact cosine top-k donors without materialising the full 83k matrix."""
    query_norm = normalized_rows(query_embeddings)
    query_count = query_norm.shape[0]
    available = target_embeddings.shape[0] - len(excluded_indices)
    retained = min(count, available)
    if retained < 1:
        raise RuntimeError("No Swiss-Prot donor remains after benchmark exclusions.")

    best_scores = np.full((query_count, retained), -np.inf, dtype=np.float32)
    best_indices = np.full((query_count, retained), -1, dtype=np.int32)
    excluded = np.asarray(sorted(excluded_indices), dtype=np.int64)

    for start in range(0, target_embeddings.shape[0], block_size):
        end = min(start + block_size, target_embeddings.shape[0])
        target_block = normalized_rows(target_embeddings[start:end])
        scores = query_norm.dot(target_block.T)
        if excluded.size:
            local = excluded[(excluded >= start) & (excluded < end)] - start
            if local.size:
                scores[:, local] = -np.inf
        candidate_indices = np.broadcast_to(
            np.arange(start, end, dtype=np.int32),
            scores.shape,
        )
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_indices = np.concatenate([best_indices, candidate_indices], axis=1)
        positions = np.argpartition(-merged_scores, kth=retained - 1, axis=1)[:, :retained]
        best_scores = np.take_along_axis(merged_scores, positions, axis=1)
        best_indices = np.take_along_axis(merged_indices, positions, axis=1)
        ordering = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, ordering, axis=1)
        best_indices = np.take_along_axis(best_indices, ordering, axis=1)
        print(f"[neighbour-embedding] searched Swiss-Prot donors: {end:,}/{target_embeddings.shape[0]:,}", flush=True)
    return best_indices, best_scores


def method_neighbourhoods(
    indices: Sequence[int],
    similarities: Sequence[float],
    target_ids: Sequence[str],
    kde_selection: Dict[str, Any],
) -> Dict[str, List[Dict[str, float | str]]]:
    """Reproduce the active KNN, fixed-radius, and Gaussian-KDE transfer rules."""
    index_values = [int(value) for value in indices]
    score_values = [float(value) for value in similarities]

    def records(
        selected_indices: Sequence[int],
        selected_scores: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> List[Dict[str, float | str]]:
        return [
            {
                "protein_id": str(target_ids[int(source_index)]),
                "cosine_similarity": round(float(score), 8),
                "cosine_distance": round(float(1.0 - score), 8),
                **({"weight": round(float(weight), 8)} if weights is not None else {}),
            }
            for source_index, score, weight in zip(
                selected_indices,
                selected_scores,
                weights if weights is not None else [None] * len(selected_indices),
            )
        ]

    result: Dict[str, List[Dict[str, float | str]]] = {}
    for k in benchmark.K_VALUES:
        result[f"knn_k_{k}"] = records(index_values[:k], score_values[:k])

    radius_mask = [1.0 - score <= FIXED_RADIUS for score in score_values]
    result["radius_r_0_01"] = records(
        [index for index, keep in zip(index_values, radius_mask) if keep],
        [score for score, keep in zip(score_values, radius_mask) if keep],
    )

    selected_indices, selected_scores, selected_weights, diagnostics = benchmark.select_kde_neighbors(
        [index_values],
        [score_values],
        bandwidth=float(kde_selection["bandwidth"]),
        weight_floor=float(kde_selection["relative_weight_floor"]),
    )
    if diagnostics[0]["candidate_limit_reached"] and len(index_values) >= int(kde_selection["max_neighbors"]):
        raise RuntimeError("Gaussian KDE neighbour export reached its configured donor safety cap.")
    result[benchmark.ACTIVE_KDE_METHOD_KEY] = records(
        selected_indices[0],
        selected_scores[0],
        selected_weights[0],
    )
    return result


def rounded_coordinates(values: np.ndarray) -> List[List[float]]:
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def make_projections(embeddings: np.ndarray) -> Dict[str, Dict[str, Any]]:
    sample_size = embeddings.shape[0]
    if sample_size < 6:
        raise ValueError("At least six joint points are needed for the 3D projections.")
    pre_reduction_components = min(50, sample_size - 1, embeddings.shape[1])
    pca = PCA(n_components=pre_reduction_components, random_state=RANDOM_SEED)
    pca_reduced = pca.fit_transform(embeddings)
    pca_coordinates = pca_reduced[:, :3]
    print("[neighbour-embedding] fitted joint PCA.", flush=True)

    umap_coordinates = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_jobs=1,
    ).fit_transform(pca_reduced)
    print("[neighbour-embedding] fitted joint UMAP.", flush=True)

    perplexity = min(35, max(5, (sample_size - 1) // 3))
    tsne_coordinates = TSNE(
        n_components=3,
        perplexity=perplexity,
        metric="euclidean",
        init="random",
        learning_rate="auto",
        max_iter=1_250,
        random_state=RANDOM_SEED,
    ).fit_transform(pca_reduced)
    print("[neighbour-embedding] fitted joint t-SNE.", flush=True)
    return {
        "pca": {
            "label": "PCA",
            "coordinates": rounded_coordinates(pca_coordinates),
            "parameters": {
                "n_components": 3,
                "input": "L2-normalized ESM-1b embeddings",
                "explained_variance_ratio": np.round(pca.explained_variance_ratio_[:3], 8).tolist(),
            },
        },
        "umap": {
            "label": "UMAP",
            "coordinates": rounded_coordinates(umap_coordinates),
            "parameters": {
                "n_components": 3,
                "n_neighbors": 30,
                "min_dist": 0.1,
                "metric": "euclidean",
                "input": f"joint PCA-{pre_reduction_components} representation of L2-normalized ESM-1b embeddings",
                "random_state": RANDOM_SEED,
            },
        },
        "tsne": {
            "label": "t-SNE",
            "coordinates": rounded_coordinates(tsne_coordinates),
            "parameters": {
                "n_components": 3,
                "perplexity": perplexity,
                "metric": "euclidean",
                "input": f"joint PCA-{pre_reduction_components} representation of L2-normalized ESM-1b embeddings",
                "init": "random",
                "n_iter": 1_250,
                "random_state": RANDOM_SEED,
            },
        },
    }


def validate_payload(payload: Dict[str, Any]) -> None:
    points = payload["points"]
    point_keys = [point["key"] for point in points]
    if len(point_keys) != len(set(point_keys)):
        raise AssertionError("Projection payload contains duplicate point keys.")
    test_ids = {
        point["protein_id"]
        for point in points
        if point["dataset_source"] == "benchmark_test"
    }
    donor_ids = {
        point["protein_id"]
        for point in points
        if point["dataset_source"] == "swissprot_donor"
    }
    if test_ids & donor_ids:
        raise AssertionError("Benchmark test proteins leaked into the donor display cohort.")
    for key in PROJECTION_KEYS:
        coordinates = np.asarray(payload["projections"][key]["coordinates"], dtype=float)
        if coordinates.shape != (len(points), 3) or not np.all(np.isfinite(coordinates)):
            raise AssertionError(f"Invalid joint {key} projection: {coordinates.shape}.")
    for neighborhood in payload["neighborhoods"].values():
        for selected in neighborhood["methods"].values():
            if not {row["protein_id"] for row in selected}.issubset(donor_ids):
                raise AssertionError("A recorded transfer neighbour is absent from the displayed Swiss-Prot donors.")


def sync_transfer(project_root: Path, relative_paths: Sequence[Path]) -> None:
    source_root = project_root
    transfer_root = project_root / "transfer" / "esm_pfp_work_2026-07-01" / "repo"
    if not transfer_root.exists():
        return
    for relative_path in relative_paths:
        source = source_root / relative_path
        destination = transfer_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def refresh_transfer_checksums(project_root: Path) -> None:
    """Keep the transfer bundle's integrity manifest valid after a sync."""
    bundle_root = project_root / "transfer" / "esm_pfp_work_2026-07-01"
    if not bundle_root.exists():
        return
    checksum_path = bundle_root / "CHECKSUMS.sha256"
    rows = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(f"{digest.hexdigest()}  ./{path.relative_to(bundle_root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if args.background_donors < 0 or args.local_neighbors < 1:
        raise ValueError("--background-donors must be non-negative and --local-neighbors must be positive.")

    frontend_dir = project_root / "notebooks" / "esm_go_explorer"
    data_dir = frontend_dir / "data"
    export_dir = project_root / "notebooks" / "exports" / "clean_plm_benchmark"
    benchmark_summary_path = export_dir / "clean_plm_benchmark_summary.json"
    if not benchmark_summary_path.is_file():
        raise RuntimeError("Missing Clean PLM benchmark summary; rerun the benchmark first.")
    benchmark_summary = json.loads(benchmark_summary_path.read_text())
    target_cache_path = Path(benchmark_summary["target_cache"])
    goa_path = Path(benchmark_summary["goa_path"])
    go_obo = project_root / "go.obo"
    output_path = data_dir / "plm_neighbor_embeddings.json"

    target_cache = plm.load_embedding_cache(target_cache_path, expected_metadata=None, validate=False)
    if target_cache is None:
        raise RuntimeError(f"Missing Swiss-Prot target cache: {target_cache_path}")
    evaluation_counts: Dict[str, int] = {}
    with (export_dir / "evaluation_protein_sets.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_model"] == "SHARED_EVALUATION_SET":
                evaluation_counts[str(row["organism"])] = int(float(row["goa_overlap_proteins"]))
    query_caches = {}
    query_rows: List[Tuple[str, str, int]] = []
    query_embeddings: List[np.ndarray] = []
    for taxon in benchmark.ORGANISMS:
        query_path = export_dir / "embeddings" / f"query_{taxon}"
        cache = plm.load_embedding_cache(query_path, expected_metadata=None, validate=False)
        if cache is None:
            raise RuntimeError(f"Missing benchmark query cache: {query_path}")
        expected_count = evaluation_counts.get(taxon)
        if expected_count is None:
            raise AssertionError(f"No shared evaluation-set count was recorded for taxon {taxon}.")
        if len(cache.ids) != expected_count:
            raise AssertionError(
                f"Query cache {taxon} does not match its recorded shared benchmark test-set count: "
                f"{len(cache.ids)} cache IDs versus {expected_count} expected IDs."
            )
        query_caches[taxon] = cache
        query_rows.extend((taxon, str(protein_id), int(length)) for protein_id, length in zip(cache.ids, cache.lengths))
        query_embeddings.append(np.asarray(cache.embeddings, dtype=np.float32))
    query_matrix = np.vstack(query_embeddings)
    test_ids = {protein_id for _taxon, protein_id, _length in query_rows}
    if len(test_ids) != len(query_rows):
        raise AssertionError("A test protein occurs in more than one shared benchmark organism set.")

    source_index = {str(protein_id): index for index, protein_id in enumerate(target_cache.ids)}
    if len(test_ids & set(source_index)) != len(test_ids):
        missing = sorted(test_ids - set(source_index))
        raise AssertionError(f"Test protein(s) absent from the Swiss-Prot cache: {missing[:5]}")
    filter_path = export_dir / "clean_plm_blacklist_filter.json"
    if not filter_path.exists():
        raise RuntimeError(
            "Missing Clean PLM blacklist audit. Re-run clean_plm_benchmark.py before building the neighbour view."
        )
    filter_payload = json.loads(filter_path.read_text())
    if filter_payload.get("status") != "blacklist_filter_enabled":
        raise RuntimeError("The Clean PLM neighbour view requires blacklist-filtered benchmark results.")
    blacklist_dir = benchmark.resolve_blacklist_dir(args.blacklist_dir)
    blacklists, blacklist_paths = benchmark.load_organism_blacklists(blacklist_dir)
    source_exclusions, blacklist_audit = benchmark.build_source_exclusions_by_organism(
        target_cache,
        test_ids,
        blacklists,
        blacklist_paths,
    )
    recorded_audit = {str(row["organism"]): row for row in filter_payload.get("organisms", [])}
    for row in blacklist_audit.to_dict(orient="records"):
        expected = recorded_audit.get(str(row["organism"]), {})
        if expected.get("blacklist_sha256") != row["blacklist_sha256"]:
            raise RuntimeError(
                f"Blacklist content for {row['organism']} differs from the completed Clean PLM benchmark; "
                "re-run the benchmark before rebuilding this view."
            )
    excluded_indices_by_organism = {
        organism: {source_index[protein_id] for protein_id in exclusions}
        for organism, exclusions in source_exclusions.items()
    }
    kde_selection_payload = json.loads((export_dir / "kde_bandwidth_selection.json").read_text())
    kde_selection = kde_selection_payload.get("shared", {})
    kde_by_organism = kde_selection_payload.get("per_organism", {})
    if (
        kde_selection_payload.get("status") != "blacklist_filter_enabled"
        or not kde_selection.get("shared_across_organisms")
        or not all(taxon in kde_by_organism for taxon in benchmark.ORGANISMS)
    ):
        raise RuntimeError("The Gaussian KDE selection is not a complete shared-bandwidth blacklist-filtered export.")
    max_method_neighbors = int(kde_selection["max_neighbors"])
    top_indices_by_organism: Dict[str, np.ndarray] = {}
    top_scores_by_organism: Dict[str, np.ndarray] = {}
    for taxon in benchmark.ORGANISMS:
        indices, scores = top_donors(
            np.asarray(query_caches[taxon].embeddings, dtype=np.float32),
            target_cache.embeddings,
            excluded_indices_by_organism[taxon],
            count=max_method_neighbors,
        )
        top_indices_by_organism[taxon] = indices
        top_scores_by_organism[taxon] = scores
    top_indices = np.vstack([top_indices_by_organism[taxon] for taxon in benchmark.ORGANISMS])
    top_scores = np.vstack([top_scores_by_organism[taxon] for taxon in benchmark.ORGANISMS])

    neighborhoods: Dict[str, Dict[str, Any]] = {}
    local_donor_indices: Set[int] = set()
    donor_organisms: Dict[int, Set[str]] = defaultdict(set)
    for row_index, (taxon, protein_id, _length) in enumerate(query_rows):
        methods = method_neighbourhoods(
            top_indices[row_index].tolist(),
            top_scores[row_index].tolist(),
            target_cache.ids,
            kde_by_organism[taxon],
        )
        local_indices = set(int(value) for value in top_indices[row_index, : args.local_neighbors])
        for selected in methods.values():
            local_indices.update(source_index[row["protein_id"]] for row in selected)
        local_donor_indices.update(local_indices)
        for source_position in local_indices:
            donor_organisms[source_position].add(taxon)
        neighborhoods[protein_id] = {
            "test_organism_taxon": taxon,
            "nearest_cosine_distance": round(float(1.0 - top_scores[row_index, 0]), 8),
            "methods": methods,
        }

    excluded_background_indices = set().union(*excluded_indices_by_organism.values())
    available_background = sorted(
        set(range(len(target_cache.ids))) - excluded_background_indices - local_donor_indices,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    background_count = min(args.background_donors, len(available_background))
    background_indices = set(
        int(value)
        for value in rng.choice(
            np.asarray(available_background, dtype=np.int32),
            size=background_count,
            replace=False,
        )
    ) if background_count else set()
    donor_indices = sorted(local_donor_indices | background_indices)
    display_ids = test_ids | {str(target_cache.ids[index]) for index in donor_indices}
    print(
        "[neighbour-embedding] display cohort: "
        f"{len(query_rows)} test proteins, {len(local_donor_indices):,} retained local donors, "
        f"{len(background_indices):,} background donors.",
        flush=True,
    )

    go = GeneOntology.GeneOntology(str(go_obo), verbose=False)
    go.build_structure()
    go_names = read_go_names(go_obo)
    accession_to_terms = plm.load_go_terms_for_accessions(
        go,
        goa_path,
        display_ids,
        evidence_codes=benchmark.EVIDENCE_CODES,
    )
    target_headers = {str(protein_id): str(header) for protein_id, header in zip(target_cache.ids, target_cache.headers)}
    target_lengths = {str(protein_id): int(length) for protein_id, length in zip(target_cache.ids, target_cache.lengths)}

    points: List[Dict[str, Any]] = []
    matrices: List[np.ndarray] = []
    for row_index, (taxon, protein_id, length) in enumerate(query_rows):
        source_info = parse_swissprot_header(target_headers.get(protein_id, ""), taxon)
        points.append(
            {
                "key": f"test:{taxon}:{protein_id}",
                "protein_id": protein_id,
                "dataset_source": "benchmark_test",
                "dataset_source_label": "Shared benchmark test set",
                "test_organism_taxon": taxon,
                "organism": source_info["organism"] or organism_label(taxon),
                "organism_taxon": source_info["taxon"] or taxon,
                "sequence_length": length,
                "go_terms": [
                    {"id": term_id, "name": go_names.get(term_id, "GO term name unavailable")}
                    for term_id in sorted(accession_to_terms.get(protein_id, set()))
                ],
                "display_role": "test",
            },
        )
        matrices.append(query_matrix[row_index])

    for source_position in donor_indices:
        protein_id = str(target_cache.ids[source_position])
        source_info = parse_swissprot_header(target_headers[protein_id])
        is_local = source_position in local_donor_indices
        points.append(
            {
                "key": f"swissprot:{protein_id}",
                "protein_id": protein_id,
                "dataset_source": "swissprot_donor",
                "dataset_source_label": "Swiss-Prot donor candidate",
                "test_organism_taxon": None,
                "organism": source_info["organism"] or "Swiss-Prot organism unavailable",
                "organism_taxon": source_info["taxon"],
                "sequence_length": target_lengths[protein_id],
                "go_terms": [
                    {"id": term_id, "name": go_names.get(term_id, "GO term name unavailable")}
                    for term_id in sorted(accession_to_terms.get(protein_id, set()))
                ],
                "display_role": "local_donor" if is_local else "background_donor",
                "neighbor_for_test_organisms": sorted(donor_organisms.get(source_position, set())),
            },
        )
        matrices.append(np.asarray(target_cache.embeddings[source_position], dtype=np.float32))

    joint_embeddings = normalized_rows(np.vstack(matrices))
    projections = make_projections(joint_embeddings)
    payload = {
        "schema_version": 2,
        "metadata": {
            "model_name": target_cache.metadata.get("model_name"),
            "embedding_dimension": int(joint_embeddings.shape[1]),
            "normalization": "row-wise L2 normalization before projection and cosine-neighbour search",
            "joint_projection_rule": (
                "For every method, test proteins and Swiss-Prot donors were combined into one matrix and "
                "fitted by the same dimensionality-reduction model. No separate test/donor projection was fitted."
            ),
            "test_set_definition": "Shared ATGO/PANDA2/GOA benchmark evaluation accessions used by Clean PLM.",
            "donor_pool_definition": (
                "Swiss-Prot source cache used by Clean PLM after excluding every shared benchmark accession and "
                "the corresponding related-organism taxon blacklist before neighbour selection."
            ),
            "blacklist_filter_enabled": True,
            "full_swissprot_donor_count_by_test_organism": {
                organism: len(target_cache.ids) - len(excluded_indices_by_organism[organism])
                for organism in benchmark.ORGANISMS
            },
            "blacklist_audit": benchmark.dataframe_records(blacklist_audit),
            "test_protein_count": len(query_rows),
            "local_donor_count": len(local_donor_indices),
            "background_donor_count": len(background_indices),
            "display_donor_count": len(donor_indices),
            "display_point_count": len(points),
            "background_sampling": {
                "method": "deterministic uniform sample without replacement",
                "seed": RANDOM_SEED,
                "requested_count": args.background_donors,
                "actual_count": len(background_indices),
            },
            "local_donor_retention": {
                "nearest_donors_per_test": args.local_neighbors,
                "all_method_selected_donors": True,
                "method_candidate_limit": max_method_neighbors,
            },
        },
        "organisms": [
            {
                "taxon": taxon,
                "label": organism_label(taxon),
                "test_protein_count": sum(1 for row_taxon, _protein, _length in query_rows if row_taxon == taxon),
            }
            for taxon in benchmark.ORGANISMS
        ],
        "methods": {
            "knn_k_3": {"label": "KNN k=3", "type": "knn"},
            "knn_k_5": {"label": "KNN k=5", "type": "knn"},
            "knn_k_7": {"label": "KNN k=7", "type": "knn"},
            "knn_k_10": {"label": "KNN k=10", "type": "knn"},
            "radius_r_0_01": {"label": "Fixed radius R=0.01", "type": "radius", "radius": FIXED_RADIUS},
            benchmark.ACTIVE_KDE_METHOD_KEY: {
                "label": "Gaussian KDE (active; shared bandwidth)",
                "type": "kde",
                "bandwidth": float(kde_selection["bandwidth"]),
                "relative_weight_floor": float(kde_selection["relative_weight_floor"]),
                "shared_across_organisms": True,
            },
        },
        "points": points,
        "projections": projections,
        "neighborhoods": neighborhoods,
    }
    validate_payload(payload)
    output_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")

    index_path = data_dir / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["plm_neighbor_embedding_analysis"] = {
        "schema_version": 2,
        "path": "data/plm_neighbor_embeddings.json",
        "description": "Joint PCA, UMAP, and t-SNE projection of the Clean PLM shared test proteins and Swiss-Prot donors.",
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    print(
        "[neighbour-embedding] wrote "
        f"{len(points):,} joint points ({len(query_rows)} test, {len(donor_indices):,} Swiss-Prot donors) to {output_path}",
        flush=True,
    )
    if args.sync_transfer:
        sync_transfer(
            project_root,
            [
                Path("notebooks/esm_go_explorer/data/plm_neighbor_embeddings.json"),
                Path("notebooks/esm_go_explorer/data/index.json"),
                Path("notebooks/esm_go_explorer/index.html"),
                Path("notebooks/esm_go_explorer/app.js"),
                Path("notebooks/esm_go_explorer/styles.css"),
                Path("notebooks/esm_go_explorer/README.md"),
                Path("scripts/build_plm_neighbor_embedding_view.py"),
                Path("tests/test_plm_neighbor_embedding_view.py"),
                Path("doc/plm_neighbor_embedding_view.md"),
            ],
        )
        refresh_transfer_checksums(project_root)
        print("[neighbour-embedding] synchronized explorer assets to the transfer bundle.", flush=True)


if __name__ == "__main__":
    main()
