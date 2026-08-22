#!/usr/bin/env python3
"""Build annotated-versus-unannotated ESM latent-space explorer artifacts.

The existing GO-group datasets supply 60 directly annotated proteins for each
organism/domain comparison. This script pairs them with 60 proteins that have
no accepted direct experimental GO rows anywhere in MF, BP, or CC, matching
sequence length without replacement. PCA, UMAP, and t-SNE are fitted jointly
to the balanced 120-protein sample.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-s2f-annotation-latent")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2f-annotation-latent")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
import umap


RANDOM_SEED = 42
PERMUTATIONS = 10_000
HISTOGRAM_BINS = 36
UNANNOTATED_COLOR = "#707780"
GOA_COLUMNS = [
    "DB",
    "Protein",
    "Symbol",
    "Qualifier",
    "GO",
    "Reference",
    "Evidence",
    "With",
    "Aspect",
    "ObjectName",
    "Synonym",
    "ObjectType",
    "Taxon",
    "Date",
    "AssignedBy",
    "AnnotationExtension",
    "GeneProductForm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate annotation-status cosine and 3D projection data.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="S2F repository root.",
    )
    parser.add_argument(
        "--sync-transfer",
        action="store_true",
        help="Mirror generated explorer files into the local transfer bundle.",
    )
    return parser.parse_args()


def first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate {label}. Checked:\n{rendered}")


def resolve_inputs(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "s2f.conf"
    config = configparser.ConfigParser()
    config.read(config_path)
    transfer_data_candidates = [
        project_root / "transfer" / "esm_pfp_work_2026-07-01" / "data",
        project_root.parent / "data",
    ]
    transfer_data = next(
        (candidate for candidate in transfer_data_candidates if candidate.exists()),
        transfer_data_candidates[0],
    )

    configured_goa = Path(config.get("databases", "filtered_goa")).expanduser()
    if not configured_goa.is_absolute():
        configured_goa = config_path.parent / configured_goa
    filtered_goa = first_existing(
        [configured_goa, transfer_data / "uniprot" / "filtered_goa"],
        "filtered GOA",
    )
    cache_paths = {
        "ecoli": first_existing(
            [
                transfer_data / "embeddings" / "ecoli_83333_query_embeddings",
            ],
            "E. coli embedding cache",
        ),
        "human": first_existing(
            [
                transfer_data / "embeddings" / "human_9606_all_swissprot",
            ],
            "human embedding cache",
        ),
    }
    go_obo = first_existing(
        [project_root / "go.obo", project_root / "transfer" / "esm_pfp_work_2026-07-01" / "repo" / "go.obo"],
        "GO ontology",
    )
    evidence_codes = {
        value.strip()
        for value in config.get("options", "evidence_codes").split(",")
        if value.strip()
    }
    return {
        "config_path": config_path.resolve(),
        "filtered_goa": filtered_goa,
        "cache_paths": cache_paths,
        "go_obo": go_obo,
        "evidence_codes": evidence_codes,
    }


def load_embedding_cache(cache_dir: Path) -> dict[str, Any]:
    ids_frame = pd.read_csv(cache_dir / "ids.tsv", sep="\t", dtype={"protein_id": str})
    embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids_frame):
        raise ValueError(
            f"Embedding cache mismatch at {cache_dir}: {embeddings.shape} vs {len(ids_frame)} IDs",
        )
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    ids = ids_frame["protein_id"].astype(str).tolist()
    lengths = ids_frame["length"].astype(int).tolist()
    return {
        "path": cache_dir,
        "ids": ids,
        "indices": {protein_id: index for index, protein_id in enumerate(ids)},
        "lengths": dict(zip(ids, lengths)),
        "embeddings": embeddings,
        "metadata": metadata,
    }


def parse_go_names(go_obo: Path) -> dict[str, str]:
    names: dict[str, str] = {}
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


def load_annotations(
    goa_path: Path,
    cache_ids: dict[str, set[str]],
    taxa: dict[str, str],
    evidence_codes: set[str],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, str]]]:
    annotations = {key: defaultdict(set) for key in cache_ids}
    object_names = {key: {} for key in cache_ids}
    for chunk in pd.read_csv(
        goa_path,
        sep="\t",
        header=None,
        names=GOA_COLUMNS,
        dtype=str,
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["Evidence"].isin(evidence_codes)]
        chunk = chunk[~chunk["Qualifier"].fillna("").str.contains(r"(?:^|\|)NOT(?:$|\|)", regex=True)]
        for organism_key, taxon in taxa.items():
            taxon_pattern = rf"(?:^|\|)taxon:{taxon}(?:$|\|)"
            subset = chunk[
                chunk["Taxon"].fillna("").str.contains(taxon_pattern, regex=True)
                & chunk["Protein"].isin(cache_ids[organism_key])
            ]
            for protein_id, go_id, object_name in subset[
                ["Protein", "GO", "ObjectName"]
            ].itertuples(index=False, name=None):
                annotations[organism_key][protein_id].add(go_id)
                if object_name and protein_id not in object_names[organism_key]:
                    object_names[organism_key][protein_id] = object_name
    return annotations, object_names


def stable_tie_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def match_unannotated_by_length(
    annotated_ids: list[str],
    unannotated_ids: set[str],
    lengths: dict[str, int],
    dataset_id: str,
) -> tuple[list[str], dict[str, float]]:
    remaining = sorted(unannotated_ids, key=lambda protein_id: (lengths[protein_id], protein_id))
    remaining_lengths = np.asarray([lengths[protein_id] for protein_id in remaining], dtype=int)
    selected: list[str] = []
    absolute_differences: list[int] = []

    for annotated_id in sorted(annotated_ids, key=lambda protein_id: (lengths[protein_id], protein_id)):
        target = lengths[annotated_id]
        differences = np.abs(remaining_lengths - target)
        closest = np.flatnonzero(differences == differences.min())
        chosen_offset = min(
            closest.tolist(),
            key=lambda index: stable_tie_key(dataset_id, annotated_id, remaining[index]),
        )
        selected.append(remaining.pop(chosen_offset))
        absolute_differences.append(int(differences[chosen_offset]))
        remaining_lengths = np.delete(remaining_lengths, chosen_offset)

    return selected, {
        "mean_absolute_length_difference": float(np.mean(absolute_differences)),
        "median_absolute_length_difference": float(np.median(absolute_differences)),
        "max_absolute_length_difference": int(np.max(absolute_differences)),
    }


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered a zero-norm embedding.")
    return values / norms


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "pairs": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def histogram(values: np.ndarray, edges: np.ndarray) -> list[dict[str, float | int]]:
    counts, _ = np.histogram(values, bins=edges)
    return [
        {
            "start": float(edges[index]),
            "end": float(edges[index + 1]),
            "count": int(count),
            "fraction": float(count / values.size),
        }
        for index, count in enumerate(counts)
    ]


def status_separation_statistics(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | str]:
    row_indices, col_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
    pair_values = similarity_matrix[row_indices, col_indices]
    within_mask = labels[row_indices] == labels[col_indices]
    observed = float(pair_values[within_mask].mean() - pair_values[~within_mask].mean())

    rng = np.random.default_rng(RANDOM_SEED)
    exceedances = 0
    batch_size = 250
    for start in range(0, PERMUTATIONS, batch_size):
        count = min(batch_size, PERMUTATIONS - start)
        shuffled = np.stack([rng.permutation(labels) for _ in range(count)])
        shuffled_within = shuffled[:, row_indices] == shuffled[:, col_indices]
        within_counts = shuffled_within.sum(axis=1)
        between_counts = pair_values.size - within_counts
        within_means = (shuffled_within * pair_values).sum(axis=1) / within_counts
        between_means = ((~shuffled_within) * pair_values).sum(axis=1) / between_counts
        exceedances += int(np.count_nonzero((within_means - between_means) >= observed))

    cosine_distances = np.clip(1.0 - similarity_matrix, 0.0, 2.0)
    np.fill_diagonal(cosine_distances, 0.0)
    return {
        "separation_definition": "within_status_minus_between_status",
        "within_status_mean": float(pair_values[within_mask].mean()),
        "between_status_mean": float(pair_values[~within_mask].mean()),
        "separation_mean_difference": observed,
        "permutation_p_value": float((1 + exceedances) / (PERMUTATIONS + 1)),
        "permutations": PERMUTATIONS,
        "silhouette_cosine": float(silhouette_score(cosine_distances, labels, metric="precomputed")),
    }


def rounded_coordinates(values: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(values, dtype=np.float32), 8).tolist()


def build_dataset(
    descriptor: dict[str, Any],
    annotated_dataset: dict[str, Any],
    cache: dict[str, Any],
    protein_annotations: dict[str, set[str]],
    object_names: dict[str, str],
    go_names: dict[str, str],
    evidence_codes: set[str],
    goa_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_id = descriptor["dataset_id"]
    annotated_proteins = annotated_dataset["proteins"]
    annotated_ids = [protein["protein_id"] for protein in annotated_proteins]
    all_cache_ids = set(cache["ids"])
    unannotated_candidates = all_cache_ids - set(protein_annotations)
    if len(unannotated_candidates) < len(annotated_ids):
        raise ValueError(
            f"{dataset_id} has only {len(unannotated_candidates)} unannotated candidates for "
            f"{len(annotated_ids)} annotated proteins.",
        )
    unannotated_ids, length_matching = match_unannotated_by_length(
        annotated_ids,
        unannotated_candidates,
        cache["lengths"],
        dataset_id,
    )
    ordered_ids = annotated_ids + unannotated_ids
    positions = [cache["indices"][protein_id] for protein_id in ordered_ids]
    embeddings = normalize_rows(cache["embeddings"][positions])
    annotated_count = len(annotated_ids)
    status_labels = np.asarray([1] * annotated_count + [0] * annotated_count, dtype=int)

    cosine_distances = pairwise_distances(embeddings, metric="cosine").astype(np.float32)
    cosine_distances = np.clip(cosine_distances, 0.0, 2.0)
    np.fill_diagonal(cosine_distances, 0.0)
    similarities = np.clip(1.0 - cosine_distances, -1.0, 1.0)
    np.fill_diagonal(similarities, 1.0)

    annotated_triangle = np.triu_indices(annotated_count, k=1)
    unannotated_triangle = np.triu_indices(annotated_count, k=1)
    annotated_within = similarities[:annotated_count, :annotated_count][annotated_triangle]
    unannotated_within = similarities[annotated_count:, annotated_count:][unannotated_triangle]
    between_status = similarities[:annotated_count, annotated_count:].ravel()
    distributions = [
        ("annotated_within", "Annotated / annotated", annotated_within, "#002fa7"),
        ("unannotated_within", "Unannotated / unannotated", unannotated_within, UNANNOTATED_COLOR),
        ("annotated_unannotated", "Annotated / unannotated", between_status, "#203040"),
    ]
    all_pair_values = np.concatenate([values for _, _, values, _ in distributions])
    lower = float(np.min(all_pair_values))
    upper = float(np.max(all_pair_values))
    if np.isclose(lower, upper):
        lower -= 0.01
        upper += 0.01
    edges = np.linspace(lower, upper, HISTOGRAM_BINS + 1)

    pca_model = PCA(n_components=3, random_state=RANDOM_SEED)
    pca_coordinates = pca_model.fit_transform(embeddings)
    umap_coordinates = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=3,
        metric="cosine",
        random_state=RANDOM_SEED,
        n_jobs=1,
    ).fit_transform(embeddings)
    tsne_coordinates = TSNE(
        n_components=3,
        perplexity=15,
        metric="cosine",
        init="random",
        learning_rate="auto",
        max_iter=1500,
        random_state=RANDOM_SEED,
    ).fit_transform(embeddings)

    groups_by_index = {
        int(group["group_index"]): group for group in annotated_dataset["groups"]
    }
    proteins: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, protein_id in enumerate(ordered_ids):
        is_annotated = index < annotated_count
        if is_annotated:
            source = annotated_proteins[index]
            group = groups_by_index[int(source["group_index"])]
            group_index = int(source["group_index"])
            group_id = group["go_id"]
            group_name = group["name"]
            color = group["color"]
        else:
            group_index = -1
            group_id = None
            group_name = "No accepted direct experimental GO terms"
            color = UNANNOTATED_COLOR
        go_term_rows = [
            {"id": go_id, "name": go_names.get(go_id, "GO term name unavailable")}
            for go_id in sorted(protein_annotations.get(protein_id, set()))
        ]
        protein = {
            "index": index,
            "protein_id": protein_id,
            "object_name": object_names.get(protein_id),
            "sequence_length": int(cache["lengths"][protein_id]),
            "annotation_status": "annotated" if is_annotated else "unannotated",
            "go_terms": go_term_rows,
            "group_index": group_index,
            "embedding_group": group_id or "unannotated",
            "group_name": group_name,
            "primary_go_id": group_id,
            "primary_go_name": group_name if is_annotated else None,
            "color": color,
        }
        proteins.append(protein)
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "organism": descriptor["organism"],
                "taxon": descriptor["taxon"],
                "domain": descriptor["domain"],
                "protein_id": protein_id,
                "annotation_status": protein["annotation_status"],
                "selected_go_group": group_id,
                "selected_go_name": group_name if is_annotated else None,
                "accepted_go_term_count": len(go_term_rows),
                "sequence_length": protein["sequence_length"],
            },
        )

    comparison_rows: list[dict[str, Any]] = []
    comparison_payloads = []
    separation = status_separation_statistics(similarities, status_labels)
    annotated_centroid = embeddings[:annotated_count].mean(axis=0)
    unannotated_centroid = embeddings[annotated_count:].mean(axis=0)
    centroid_similarity = float(
        np.dot(annotated_centroid, unannotated_centroid)
        / (np.linalg.norm(annotated_centroid) * np.linalg.norm(unannotated_centroid))
    )
    separation["centroid_cosine_similarity"] = centroid_similarity
    separation["centroid_cosine_distance"] = float(1.0 - centroid_similarity)

    for key, label, values, color in distributions:
        summary = distribution_summary(values)
        comparison_payloads.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "summary": summary,
                "histogram": histogram(values, edges),
            },
        )
        comparison_rows.append(
            {
                "dataset_id": dataset_id,
                "organism": descriptor["organism"],
                "taxon": descriptor["taxon"],
                "domain": descriptor["domain"],
                "comparison": key,
                **summary,
                **separation,
            },
        )

    result = {
        "schema_version": 1,
        "metadata": {
            "dataset_id": dataset_id,
            "organism_key": descriptor["organism_key"],
            "organism": descriptor["organism"],
            "taxon": descriptor["taxon"],
            "domain": descriptor["domain"],
            "domain_name": descriptor["domain_name"],
            "model_name": cache["metadata"].get("model_name"),
            "embedding_dimension": int(embeddings.shape[1]),
            "sample_size_per_status": annotated_count,
            "random_seed": RANDOM_SEED,
            "normalization": "row-wise L2 normalization before all comparisons and projections",
            "annotation_definition": (
                "At least one direct, non-NOT GO annotation in the filtered GOA snapshot with an "
                "accepted experimental evidence code, across MF, BP, or CC."
            ),
            "unannotated_definition": (
                "No accepted direct experimental GO row in the filtered GOA snapshot across MF, BP, or CC; "
                "this does not assert absence from all-evidence or future GOA releases."
            ),
            "sampling": (
                "The existing 60 balanced selected-GO proteins are compared with 60 unannotated proteins "
                "chosen without replacement by nearest sequence length."
            ),
            "length_matching": length_matching,
            "filtered_goa_path": str(goa_path),
            "embedding_cache_path": str(cache["path"]),
            "evidence_codes": sorted(evidence_codes),
        },
        "groups": [
            {
                "group_index": int(group["group_index"]),
                "go_id": group["go_id"],
                "name": group["name"],
                "color": group["color"],
                "selected_count": int(group["selected_count"]),
            }
            for group in annotated_dataset["groups"]
        ],
        "proteins": proteins,
        "cosine_similarity": {
            "comparisons": comparison_payloads,
            "shared_histogram_range": [float(edges[0]), float(edges[-1])],
            "status_separation": separation,
        },
        "projections": {
            "pca": {
                "label": "PCA",
                "coordinates": rounded_coordinates(pca_coordinates),
                "explained_variance_ratio": np.round(
                    pca_model.explained_variance_ratio_,
                    8,
                ).tolist(),
                "parameters": {
                    "n_components": 3,
                    "input": "L2-normalized ESM embeddings",
                },
            },
            "umap": {
                "label": "UMAP",
                "coordinates": rounded_coordinates(umap_coordinates),
                "parameters": {
                    "n_components": 3,
                    "n_neighbors": 10,
                    "min_dist": 0.1,
                    "metric": "cosine",
                    "random_state": RANDOM_SEED,
                },
            },
            "tsne": {
                "label": "t-SNE",
                "coordinates": rounded_coordinates(tsne_coordinates),
                "parameters": {
                    "n_components": 3,
                    "perplexity": 15,
                    "metric": "cosine",
                    "init": "random",
                    "max_iter": 1500,
                    "random_state": RANDOM_SEED,
                },
            },
        },
    }
    return result, manifest_rows, comparison_rows


def validate_result(result: dict[str, Any]) -> None:
    protein_count = len(result["proteins"])
    if protein_count != 120:
        raise AssertionError(f"Expected 120 proteins, found {protein_count}.")
    status_counts = pd.Series(
        [protein["annotation_status"] for protein in result["proteins"]],
    ).value_counts().to_dict()
    if status_counts != {"annotated": 60, "unannotated": 60}:
        raise AssertionError(f"Unexpected status counts: {status_counts}")
    for key in ["pca", "umap", "tsne"]:
        coordinates = np.asarray(result["projections"][key]["coordinates"], dtype=float)
        if coordinates.shape != (120, 3) or not np.all(np.isfinite(coordinates)):
            raise AssertionError(f"Invalid {key} coordinates: {coordinates.shape}")
    comparisons = result["cosine_similarity"]["comparisons"]
    expected_pairs = {
        "annotated_within": 1770,
        "unannotated_within": 1770,
        "annotated_unannotated": 3600,
    }
    observed_pairs = {item["key"]: item["summary"]["pairs"] for item in comparisons}
    if observed_pairs != expected_pairs:
        raise AssertionError(f"Unexpected pair counts: {observed_pairs}")


def sync_transfer_files(project_root: Path, filenames: list[str]) -> None:
    source_dir = project_root / "notebooks" / "esm_go_explorer"
    transfer_dir = (
        project_root
        / "transfer"
        / "esm_pfp_work_2026-07-01"
        / "repo"
        / "notebooks"
        / "esm_go_explorer"
    )
    if not transfer_dir.exists():
        return
    import shutil

    for filename in filenames:
        source = source_dir / filename
        destination = transfer_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def sync_transfer_sources(project_root: Path) -> None:
    transfer_repo = project_root / "transfer" / "esm_pfp_work_2026-07-01" / "repo"
    if not transfer_repo.exists():
        return
    import shutil

    source_pairs = [
        (
            project_root / "scripts" / "build_annotation_latent_analysis.py",
            transfer_repo / "scripts" / "build_annotation_latent_analysis.py",
        ),
        (
            project_root / "notebooks" / "esm_go_embedding_analysis.ipynb",
            transfer_repo / "notebooks" / "esm_go_embedding_analysis.ipynb",
        ),
    ]
    for source, destination in source_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    inputs = resolve_inputs(project_root)
    frontend_dir = project_root / "notebooks" / "esm_go_explorer"
    data_dir = frontend_dir / "data"
    index_path = data_dir / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)

    caches = {
        organism_key: load_embedding_cache(cache_path)
        for organism_key, cache_path in inputs["cache_paths"].items()
    }
    taxa = {
        descriptor["organism_key"]: str(descriptor["taxon"])
        for descriptor in index_payload["datasets"]
    }
    annotations, object_names = load_annotations(
        inputs["filtered_goa"],
        {key: set(cache["ids"]) for key, cache in caches.items()},
        taxa,
        inputs["evidence_codes"],
    )
    go_names = parse_go_names(inputs["go_obo"])

    all_manifest_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    generated_filenames: list[str] = []
    for descriptor in index_payload["datasets"]:
        dataset_path = frontend_dir / descriptor["path"]
        with dataset_path.open("r", encoding="utf-8") as handle:
            annotated_dataset = json.load(handle)
        organism_key = descriptor["organism_key"]
        result, manifest_rows, comparison_rows = build_dataset(
            descriptor,
            annotated_dataset,
            caches[organism_key],
            annotations[organism_key],
            object_names[organism_key],
            go_names,
            inputs["evidence_codes"],
            inputs["filtered_goa"],
        )
        validate_result(result)
        output_name = f"annotation_status_{descriptor['dataset_id']}.json"
        output_path = data_dir / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        descriptor["annotation_comparison_path"] = f"data/{output_name}"
        all_manifest_rows.extend(manifest_rows)
        all_comparison_rows.extend(comparison_rows)
        generated_filenames.append(f"data/{output_name}")
        separation = result["cosine_similarity"]["status_separation"]
        print(
            f"{descriptor['dataset_id']}: n=120, "
            f"status silhouette={separation['silhouette_cosine']:.4f}, "
            f"permutation p={separation['permutation_p_value']:.5g}",
        )

    manifest_path = data_dir / "annotation_status_selected_proteins.csv"
    summary_path = data_dir / "annotation_status_summary.csv"
    pd.DataFrame(all_manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(all_comparison_rows).to_csv(summary_path, index=False)
    generated_filenames.extend(
        [
            "data/annotation_status_selected_proteins.csv",
            "data/annotation_status_summary.csv",
        ],
    )

    index_payload["schema_version"] = max(3, int(index_payload.get("schema_version", 1)))
    index_payload["annotation_status_analysis"] = {
        "schema_version": 1,
        "summary_path": "data/annotation_status_summary.csv",
        "selected_proteins_path": "data/annotation_status_selected_proteins.csv",
        "annotation_scope": "direct experimental GO annotations across MF, BP, and CC",
    }
    index_payload["scientific_scope"] = (
        "These unsupervised similarities and projections test GO-group and annotation-status structure in "
        "ESM-1b embeddings. They do not establish that GO labels were memorized during pretraining, prove "
        "functional causality, or replace experimental validation."
    )
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2, sort_keys=True, allow_nan=False)
    generated_filenames.append("data/index.json")

    if args.sync_transfer:
        sync_transfer_files(
            project_root,
            generated_filenames + ["index.html", "app.js", "styles.css", "README.md"],
        )
        sync_transfer_sources(project_root)
    print(f"Wrote {len(index_payload['datasets'])} annotation-status datasets to {data_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
