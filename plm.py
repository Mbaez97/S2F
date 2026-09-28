import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


UNIPROT_ACC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])$"
)
NON_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
EMBEDDING_CACHE_VERSION = 1
TRANSFER_METADATA_VERSION = 1
TARGET_HEADER_TAXON_RE = re.compile(r"(?:^|\s)OX=(\d+)(?:\s|$)")


def log_info(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {timestamp}] {message}")


def log_warn(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WARN {timestamp}] {message}", file=sys.stderr)


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def extract_uniprot_from_id(identifier: str) -> Optional[str]:
    match = re.search(r"(?:sp|tr)\|([A-Z0-9]+)\|", identifier)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[>\s_])(?:sp|tr)_([A-Z0-9]+)", identifier)
    if match:
        return match.group(1)
    for token in re.split(r"\W+", identifier):
        if UNIPROT_ACC_RE.match(token):
            return token
    return None


def protein_id_from_header(header: str, protein_id_mode: str) -> str:
    token = header.split()[0]
    if protein_id_mode == "uniprot":
        accession = extract_uniprot_from_id(token) or extract_uniprot_from_id(header)
        return accession or token
    return token


@dataclass
class FastaRecord:
    header: str
    protein_id: str
    sequence: str


def read_fasta(path: Path, protein_id_mode: str) -> List[FastaRecord]:
    records: List[FastaRecord] = []
    header = None
    chunks: List[str] = []
    with open_maybe_gzip(path) as handler:
        for raw_line in handler:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = NON_AA_RE.sub("", "".join(chunks).upper())
                    records.append(
                        FastaRecord(
                            header=header,
                            protein_id=protein_id_from_header(
                                header, protein_id_mode
                            ),
                            sequence=sequence,
                        )
                    )
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        sequence = NON_AA_RE.sub("", "".join(chunks).upper())
        records.append(
            FastaRecord(
                header=header,
                protein_id=protein_id_from_header(header, protein_id_mode),
                sequence=sequence,
            )
        )
    return records


def read_many_fastas(paths: Sequence[Path], protein_id_mode: str) -> List[FastaRecord]:
    records: List[FastaRecord] = []
    for path in paths:
        loaded = read_fasta(path, protein_id_mode)
        log_info(f"Loaded {len(loaded):,} protein sequence(s) from {path}.")
        records.extend(loaded)
    return records


def fasta_fingerprint(paths: Sequence[Path]) -> List[Dict[str, object]]:
    fingerprints = []
    for path in paths:
        stat = path.stat()
        fingerprints.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprints


def cache_fingerprint(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    cache_dir = Path(path).expanduser()
    return {
        name: fasta_fingerprint([cache_dir / name])[0]
        for name in ("meta.json", "ids.tsv", "embeddings.npy")
        if (cache_dir / name).is_file()
    }


def build_cache_metadata(
    fasta_paths: Sequence[Path],
    model_name: str,
    protein_id_mode: str,
    long_sequence_mode: str,
    long_window_size: int,
    long_overlap: int,
) -> Dict[str, object]:
    return {
        "version": EMBEDDING_CACHE_VERSION,
        "fasta": fasta_fingerprint(fasta_paths),
        "model_name": model_name,
        "protein_id_mode": protein_id_mode,
        "long_sequence_mode": long_sequence_mode,
        "long_window_size": long_window_size,
        "long_overlap": long_overlap,
    }


def metadata_key(metadata: Dict[str, object]) -> str:
    encoded = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


@dataclass
class EmbeddingCache:
    ids: List[str]
    headers: List[str]
    lengths: List[int]
    embeddings: np.ndarray
    path: Path
    metadata: Dict[str, object]


def cache_matches(cache_metadata: Dict[str, object], expected: Dict[str, object]) -> bool:
    for key, value in expected.items():
        cached_value = cache_metadata.get(key)
        if key == "fasta" and isinstance(cached_value, list) and isinstance(value, list):
            if len(cached_value) != len(value):
                return False
            for cached_item, expected_item in zip(cached_value, value):
                cached_copy = dict(cached_item)
                expected_copy = dict(expected_item)
                cached_path = str(cached_copy.pop("path", ""))
                expected_path = str(expected_copy.pop("path", ""))
                cached_candidates = {cached_path}
                expected_candidates = {expected_path}
                if cached_path.startswith("/media/"):
                    cached_candidates.add("/run" + cached_path)
                if expected_path.startswith("/media/"):
                    expected_candidates.add("/run" + expected_path)
                if cached_candidates.isdisjoint(expected_candidates):
                    return False
                if cached_copy != expected_copy:
                    return False
            continue
        if cached_value != value:
            return False
    return True


def load_embedding_cache(
    cache_dir: Path,
    expected_metadata: Optional[Dict[str, object]] = None,
    validate: bool = True,
) -> Optional[EmbeddingCache]:
    ids_path = cache_dir / "ids.tsv"
    embeddings_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "meta.json"
    if not ids_path.exists() or not embeddings_path.exists() or not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as handler:
        metadata = json.load(handler)
    if validate and expected_metadata is not None and not cache_matches(
        metadata, expected_metadata
    ):
        log_warn(f"Embedding cache metadata mismatch for {cache_dir}.")
        return None

    ids: List[str] = []
    headers: List[str] = []
    lengths: List[int] = []
    with open(ids_path, "r", encoding="utf-8") as handler:
        reader = csv.DictReader(handler, delimiter="\t")
        for row in reader:
            ids.append(row["protein_id"])
            headers.append(row["header"])
            lengths.append(int(row["length"]))

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(ids):
        raise RuntimeError(
            f"Embedding cache {cache_dir} has {embeddings.shape[0]} vectors "
            f"but {len(ids)} sequence IDs."
        )
    return EmbeddingCache(ids, headers, lengths, embeddings, cache_dir, metadata)


def write_embedding_cache(
    cache_dir: Path,
    records: Sequence[FastaRecord],
    embeddings: np.ndarray,
    metadata: Dict[str, object],
) -> EmbeddingCache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "ids.tsv"
    embeddings_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "meta.json"

    with open(ids_path, "w", newline="", encoding="utf-8") as handler:
        writer = csv.writer(handler, delimiter="\t")
        writer.writerow(["index", "protein_id", "header", "length"])
        for index, record in enumerate(records):
            writer.writerow(
                [index, record.protein_id, record.header, len(record.sequence)]
            )

    np.save(embeddings_path, embeddings.astype(np.float32, copy=False))
    metadata = dict(metadata)
    metadata["embedding_dim"] = int(embeddings.shape[1])
    metadata["sequence_count"] = len(records)
    with open(meta_path, "w", encoding="utf-8") as handler:
        json.dump(metadata, handler, indent=2, sort_keys=True)

    return EmbeddingCache(
        ids=[record.protein_id for record in records],
        headers=[record.header for record in records],
        lengths=[len(record.sequence) for record in records],
        embeddings=np.load(embeddings_path, mmap_mode="r"),
        path=cache_dir,
        metadata=metadata,
    )


def sequence_windows(
    sequence: str,
    long_sequence_mode: str,
    long_window_size: int,
    long_overlap: int,
) -> List[str]:
    if len(sequence) <= long_window_size:
        return [sequence]
    if long_sequence_mode == "skip":
        return []
    if long_sequence_mode == "truncate":
        return [sequence[:long_window_size]]
    if long_sequence_mode != "sliding_mean":
        raise RuntimeError(f"Unsupported long sequence mode: {long_sequence_mode}")
    if long_overlap >= long_window_size:
        raise RuntimeError("PLM long_overlap must be smaller than long_window_size.")

    step = long_window_size - long_overlap
    windows = []
    start = 0
    while start < len(sequence):
        end = min(start + long_window_size, len(sequence))
        windows.append(sequence[start:end])
        if end == len(sequence):
            break
        start += step
    return windows


def load_esm_model(
    model_name: str,
    model_dir: Optional[Path],
    device_arg: str,
    local_files_only: bool,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "PLM seed requires optional dependencies that are not available. "
            "Install the packages in requirements-plm.txt in the active S2F "
            "environment, then rerun."
        ) from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    cache_dir = str(model_dir) if model_dir else None
    log_info(
        f"Loading ESM model {model_name} on device {device} "
        f"(local_files_only={local_files_only})."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def embed_batch(
    torch_module,
    tokenizer,
    model,
    device: str,
    sequences: Sequence[str],
) -> np.ndarray:
    encoded = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch_module.inference_mode():
        output = model(**encoded)
        hidden = output.last_hidden_state

    attention_mask = encoded["attention_mask"].bool()
    token_mask = attention_mask.clone()
    special_ids = {
        token_id
        for token_id in (
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    for token_id in special_ids:
        token_mask &= encoded["input_ids"] != token_id

    token_mask_float = token_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * token_mask_float).sum(dim=1)
    counts = token_mask_float.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.detach().cpu().float().numpy()


def compute_embeddings(
    records: Sequence[FastaRecord],
    model_name: str,
    model_dir: Optional[Path],
    device_arg: str,
    local_files_only: bool,
    long_sequence_mode: str,
    long_window_size: int,
    long_overlap: int,
    batch_tokens: int,
) -> np.ndarray:
    torch_module, tokenizer, model, device = load_esm_model(
        model_name,
        model_dir,
        device_arg,
        local_files_only,
    )

    embedding_sums: Dict[int, np.ndarray] = {}
    embedding_counts: Dict[int, int] = defaultdict(int)
    skipped: List[str] = []
    batch_sequences: List[str] = []
    batch_record_indices: List[int] = []
    batch_token_count = 0
    embedding_dim: Optional[int] = None
    processed_windows = 0

    def flush_batch() -> None:
        nonlocal batch_sequences
        nonlocal batch_record_indices
        nonlocal batch_token_count
        nonlocal embedding_dim
        nonlocal processed_windows
        if not batch_sequences:
            return
        batch_embeddings = embed_batch(
            torch_module, tokenizer, model, device, batch_sequences
        )
        embedding_dim = int(batch_embeddings.shape[1])
        for record_index, vector in zip(batch_record_indices, batch_embeddings):
            if record_index not in embedding_sums:
                embedding_sums[record_index] = vector.astype(np.float64)
            else:
                embedding_sums[record_index] += vector
            embedding_counts[record_index] += 1
        processed_windows += len(batch_sequences)
        if processed_windows % 1000 == 0:
            log_info(f"Embedded {processed_windows:,} PLM sequence window(s).")
        batch_sequences = []
        batch_record_indices = []
        batch_token_count = 0

    for record_index, record in enumerate(records, start=0):
        windows = sequence_windows(
            record.sequence,
            long_sequence_mode,
            long_window_size,
            long_overlap,
        )
        if not windows:
            skipped.append(record.protein_id)
            continue
        for window in windows:
            token_count = len(window) + 2
            if batch_sequences and batch_token_count + token_count > batch_tokens:
                flush_batch()
            batch_sequences.append(window)
            batch_record_indices.append(record_index)
            batch_token_count += token_count
            if token_count > batch_tokens:
                flush_batch()
    flush_batch()

    if skipped:
        log_warn(
            f"Skipped {len(skipped):,} sequence(s) longer than the ESM limit "
            "because long_sequence_mode=skip."
        )
    if embedding_dim is None:
        raise RuntimeError("No embeddings were produced.")

    embeddings = np.zeros((len(records), embedding_dim), dtype=np.float32)
    for record_index, record in enumerate(records):
        count = embedding_counts.get(record_index, 0)
        if count == 0:
            continue
        embeddings[record_index] = (
            embedding_sums[record_index] / count
        ).astype(np.float32)
        if (record_index + 1) % 1000 == 0:
            log_info(
                f"Prepared final averaged embedding for "
                f"{record_index + 1:,}/{len(records):,} protein(s)."
            )

    return embeddings


def get_or_create_embedding_cache(
    cache_dir: Path,
    records: Sequence[FastaRecord],
    expected_metadata: Dict[str, object],
    args,
    force: bool = False,
) -> EmbeddingCache:
    embedding_records = list(records)
    if args.long_sequence_mode == "skip":
        before = len(embedding_records)
        embedding_records = [
            record
            for record in embedding_records
            if len(record.sequence) <= args.long_window_size
        ]
        skipped = before - len(embedding_records)
        if skipped:
            log_warn(
                f"Excluded {skipped:,} protein sequence(s) from {cache_dir.name} "
                "because long_sequence_mode=skip."
            )

    if not force:
        cache = load_embedding_cache(cache_dir, expected_metadata=expected_metadata)
        if cache is not None:
            log_info(f"Using cached PLM embeddings from {cache_dir}.")
            return cache

    log_info(f"Computing PLM embeddings for {len(embedding_records):,} protein(s).")
    embeddings = compute_embeddings(
        embedding_records,
        model_name=args.model_name,
        model_dir=Path(args.model_dir).expanduser() if args.model_dir else None,
        device_arg=args.device,
        local_files_only=args.local_files_only,
        long_sequence_mode=args.long_sequence_mode,
        long_window_size=args.long_window_size,
        long_overlap=args.long_overlap,
        batch_tokens=args.batch_tokens,
    )
    cache = write_embedding_cache(cache_dir, embedding_records, embeddings, expected_metadata)
    log_info(f"Wrote PLM embeddings to {cache.path}.")
    return cache


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def compute_knn(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int,
    query_chunk_size: int,
    excluded_target_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise RuntimeError("PLM knn_k must be greater than zero.")
    if target_embeddings.shape[0] == 0:
        raise RuntimeError("Target embedding cache is empty.")

    excluded = np.unique(
        np.asarray([] if excluded_target_indices is None else excluded_target_indices, dtype=np.int64)
    )
    if excluded.size and (excluded.min() < 0 or excluded.max() >= target_embeddings.shape[0]):
        raise RuntimeError("PLM target exclusion index is outside the embedding cache.")
    available_targets = target_embeddings.shape[0] - excluded.size
    if available_targets < 1:
        raise RuntimeError("No target proteins remain after blacklist exclusion.")
    k = min(k, available_targets)
    target_norm = normalize_embeddings(target_embeddings)
    query_norm = normalize_embeddings(query_embeddings)
    all_indices = np.zeros((query_norm.shape[0], k), dtype=np.int64)
    all_scores = np.zeros((query_norm.shape[0], k), dtype=np.float32)

    for start in range(0, query_norm.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_norm.shape[0])
        similarities = query_norm[start:end].dot(target_norm.T)
        if excluded.size:
            similarities[:, excluded] = -np.inf
        partition = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        partition_scores = np.take_along_axis(similarities, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        top_indices = np.take_along_axis(partition, order, axis=1)
        top_scores = np.take_along_axis(partition_scores, order, axis=1)
        all_indices[start:end] = top_indices
        all_scores[start:end] = top_scores
        log_info(f"Computed PLM neighbors for {end:,}/{query_norm.shape[0]:,} query protein(s).")

    return all_indices, all_scores


def read_blacklist(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    values = set()
    with open(path, "r", encoding="utf-8") as handler:
        for line in handler:
            token = line.strip()
            if token:
                values.add(token)
    return values or None


def read_accession_exclusions(path: Optional[str]) -> Optional[Set[str]]:
    """Read one-accession-per-line files or benchmark CSV/TSV exports."""
    if not path:
        return None
    exclusion_path = Path(path).expanduser()
    if not exclusion_path.is_file():
        raise RuntimeError(
            f"PLM accession exclusion file does not exist: {exclusion_path}"
        )
    with open(exclusion_path, "r", encoding="utf-8", newline="") as handler:
        first_line = handler.readline()
        handler.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        header = [value.strip() for value in first_line.rstrip("\n").split(delimiter)]
        accession_columns = (
            "protein_id", "Protein", "accession", "uniprot_accession"
        )
        selected_column = next(
            (column for column in accession_columns if column in header), None
        )
        if selected_column is not None:
            reader = csv.DictReader(handler, delimiter=delimiter)
            values = {
                str(row.get(selected_column, "")).strip()
                for row in reader
                if str(row.get(selected_column, "")).strip()
            }
        else:
            values = {
                line.strip().split()[0]
                for line in handler
                if line.strip() and not line.lstrip().startswith("#")
            }
    return values or None


def blacklisted_target_indices(target_cache: EmbeddingCache, blacklist: Optional[Set[str]]) -> np.ndarray:
    """Return target-cache rows forbidden before PLM KNN retrieval.

    Filtering only the GOA rows after KNN allows a forbidden close relative to
    consume a neighbour rank.  Taxon filtering here removes that donor from the
    cosine search itself, which is the required leakage boundary.
    """
    if not blacklist:
        return np.empty(0, dtype=np.int64)
    indices = []
    missing_taxonomy = []
    for index, (protein_id, header) in enumerate(zip(target_cache.ids, target_cache.headers)):
        match = TARGET_HEADER_TAXON_RE.search(str(header))
        if match is None:
            missing_taxonomy.append(str(protein_id))
            continue
        if match.group(1) in blacklist:
            indices.append(index)
    if missing_taxonomy:
        raise RuntimeError(
            "Cannot apply the PLM taxon blacklist before KNN because target headers lack OX taxonomy "
            f"for {len(missing_taxonomy)} proteins (for example {missing_taxonomy[:5]})."
        )
    return np.asarray(indices, dtype=np.int64)


def excluded_accession_indices(
    target_cache: EmbeddingCache, accessions: Optional[Set[str]]
) -> np.ndarray:
    if not accessions:
        return np.empty(0, dtype=np.int64)
    return np.asarray(
        [
            index
            for index, protein_id in enumerate(target_cache.ids)
            if protein_id in accessions
        ],
        dtype=np.int64,
    )


def relative_gaussian_kde_weights(
    similarities: Sequence[float], bandwidth: float
) -> np.ndarray:
    if bandwidth <= 0:
        raise RuntimeError("PLM kde_bandwidth must be greater than zero.")
    scores = np.asarray(similarities, dtype=np.float64)
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)
    distances = np.clip(1.0 - scores, 0.0, 2.0)
    return np.exp(
        -(distances - float(np.min(distances))) / (bandwidth * bandwidth)
    )


def compute_kde_neighbors(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    bandwidth: float,
    weight_floor: float,
    max_neighbors: int,
    query_chunk_size: int,
    excluded_target_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]], List[Dict[str, object]]]:
    """Select all Gaussian contributors above the checked relative floor."""
    if bandwidth <= 0:
        raise RuntimeError("PLM kde_bandwidth must be greater than zero.")
    if not 0.0 < weight_floor < 1.0:
        raise RuntimeError("PLM kde_weight_floor must be strictly between zero and one.")
    if max_neighbors <= 0:
        raise RuntimeError("PLM kde_max_neighbors must be greater than zero.")
    excluded = np.unique(
        np.asarray(
            [] if excluded_target_indices is None else excluded_target_indices,
            dtype=np.int64,
        )
    )
    available_targets = target_embeddings.shape[0] - excluded.size
    if available_targets < 1:
        raise RuntimeError("No target proteins remain after PLM exclusions.")
    target_norm = normalize_embeddings(target_embeddings)
    query_norm = normalize_embeddings(query_embeddings)
    relative_radius = -bandwidth * bandwidth * math.log(weight_floor)
    all_indices: List[List[int]] = []
    all_scores: List[List[float]] = []
    all_weights: List[List[float]] = []
    diagnostics: List[Dict[str, object]] = []

    for start in range(0, query_norm.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_norm.shape[0])
        similarities = query_norm[start:end].dot(target_norm.T)
        if excluded.size:
            similarities[:, excluded] = -np.inf
        for local_index, row in enumerate(similarities):
            nearest_similarity = float(np.max(row))
            minimum_similarity = nearest_similarity - relative_radius
            selected = np.flatnonzero(row >= minimum_similarity)
            if selected.size > max_neighbors:
                query_index = start + local_index
                raise RuntimeError(
                    f"Gaussian KDE for query row {query_index} has {selected.size} "
                    f"contributors above the relative floor, exceeding the "
                    f"{max_neighbors}-donor safety cap. Increase kde_max_neighbors "
                    "and rerun."
                )
            selected_scores = row[selected]
            order = np.argsort(-selected_scores)
            selected = selected[order]
            selected_scores = selected_scores[order]
            weights = relative_gaussian_kde_weights(selected_scores, bandwidth)
            all_indices.append(selected.astype(int).tolist())
            all_scores.append(selected_scores.astype(float).tolist())
            all_weights.append(weights.astype(float).tolist())
            weight_sum = float(np.sum(weights))
            squared_sum = float(np.sum(np.square(weights)))
            diagnostics.append(
                {
                    "retained_neighbor_count": int(selected.size),
                    "nearest_cosine_distance": 1.0 - nearest_similarity,
                    "relative_distance_radius": relative_radius,
                    "absolute_distance_threshold": 1.0 - minimum_similarity,
                    "retained_kernel_weight": weight_sum,
                    "effective_sample_size": (
                        weight_sum * weight_sum / squared_sum
                        if squared_sum > 0
                        else 0.0
                    ),
                    "neighbor_cap_applied": False,
                }
            )
        log_info(
            f"Computed PLM KDE contributors for {end:,}/{query_norm.shape[0]:,} "
            "query protein(s)."
        )
    return all_indices, all_scores, all_weights, diagnostics


def load_go_terms_for_accessions(
    go,
    goa_path: Path,
    accessions: Set[str],
    blacklist: Optional[Set[str]] = None,
    evidence_codes: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    accession_to_terms: Dict[str, Set[str]] = defaultdict(set)
    skipped = Counter()
    with open_maybe_gzip(goa_path) as handler:
        for raw_line in handler:
            if not raw_line or raw_line.startswith("!"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 13:
                skipped["short"] += 1
                continue
            accession = fields[1]
            if accession not in accessions:
                continue
            if evidence_codes is not None and fields[6] not in evidence_codes:
                skipped["evidence"] += 1
                continue
            if blacklist is not None:
                taxons = [taxon.split(":")[-1] for taxon in fields[12].split("|")]
                if any(taxon in blacklist for taxon in taxons):
                    skipped["blacklist"] += 1
                    continue
            qualifiers = fields[3].split("|") if fields[3] else []
            if "NOT" in qualifiers:
                skipped["not"] += 1
                continue
            go_id = fields[4]
            try:
                term = go.find_term(go_id)
            except KeyError:
                skipped["unknown_go"] += 1
                continue
            if term.is_obsolete:
                skipped["obsolete"] += 1
                continue
            accession_to_terms[accession].add(term.go_id)

    if skipped:
        log_warn(
            "Skipped GOA rows during PLM transfer: "
            + ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
        )
    log_info(
        f"Loaded GO terms for {len(accession_to_terms):,}/"
        f"{len(accessions):,} neighbor accession(s)."
    )
    return dict(accession_to_terms)


def write_neighbors(
    output_path: Path,
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    neighbor_indices: Sequence[Sequence[int]],
    neighbor_scores: Sequence[Sequence[float]],
    neighbor_weights: Optional[Sequence[Sequence[float]]] = None,
) -> Set[str]:
    wanted_accessions: Set[str] = set()
    with open(output_path, "w", newline="", encoding="utf-8") as handler:
        writer = csv.writer(handler, delimiter="\t")
        writer.writerow(["Protein", "Neighbor", "Rank", "Cosine", "Weight"])
        for query_index, query_id in enumerate(query_cache.ids):
            weights = (
                neighbor_scores[query_index]
                if neighbor_weights is None
                else neighbor_weights[query_index]
            )
            for rank, target_index in enumerate(neighbor_indices[query_index], start=1):
                neighbor_id = target_cache.ids[int(target_index)]
                wanted_accessions.add(neighbor_id)
                writer.writerow(
                    [
                        query_id,
                        neighbor_id,
                        rank,
                        f"{float(neighbor_scores[query_index][rank - 1]):.8f}",
                        f"{float(weights[rank - 1]):.8f}",
                    ]
                )
    return wanted_accessions


def build_direct_assignments(
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    neighbor_indices: Sequence[Sequence[int]],
    neighbor_scores: Sequence[Sequence[float]],
    accession_to_terms: Dict[str, Set[str]],
    outdir: Path,
    score_mode: str = "all_ones",
    neighbor_weights: Optional[Sequence[Sequence[float]]] = None,
    diagnostics: Optional[Sequence[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    direct_rows: List[Dict[str, object]] = []
    for query_index, query_id in enumerate(query_cache.ids):
        term_weights: Dict[str, float] = defaultdict(float)
        summary_neighbors = []
        weights = (
            neighbor_scores[query_index]
            if neighbor_weights is None
            else neighbor_weights[query_index]
        )
        denominator = sum(max(float(weight), 0.0) for weight in weights)
        for rank, target_index in enumerate(neighbor_indices[query_index], start=1):
            neighbor_id = target_cache.ids[int(target_index)]
            terms = sorted(accession_to_terms.get(neighbor_id, set()))
            weight = max(float(weights[rank - 1]), 0.0)
            for go_id in terms:
                term_weights[go_id] += weight
            summary_neighbors.append(
                {
                    "neighbor": neighbor_id,
                    "rank": rank,
                    "cosine": float(neighbor_scores[query_index][rank - 1]),
                    "weight": weight,
                    "go_count": len(terms),
                }
            )
        for go_id in sorted(term_weights):
            score = 1.0
            if score_mode == "weighted_support":
                score = term_weights[go_id] / denominator if denominator > 0 else 0.0
            elif score_mode != "all_ones":
                raise RuntimeError(f"Unsupported PLM score mode: {score_mode}")
            if score > 0:
                direct_rows.append(
                    {"Protein": query_id, "GO ID": go_id, "Score": float(score)}
                )

        summary = {
            "protein_id": query_id,
            "neighbors": summary_neighbors,
            "direct_go_count": len(term_weights),
            "score_mode": score_mode,
        }
        if diagnostics is not None:
            summary["transfer_diagnostics"] = diagnostics[query_index]
        summary_path = outdir / f"{safe_filename(query_id)}.summary.json"
        with open(summary_path, "w", encoding="utf-8") as handler:
            json.dump(summary, handler, indent=2, sort_keys=True)

        if (query_index + 1) % 500 == 0:
            log_info(
                f"Prepared PLM direct assignments for "
                f"{query_index + 1:,}/{len(query_cache.ids):,} query protein(s)."
            )
    return direct_rows


def safe_filename(value: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", value)[:100] or "query"


def write_propagated_assignments(go, direct_rows: List[Dict[str, object]], out_path: Path) -> None:
    import pandas as pd

    if not direct_rows:
        with open(out_path, "w", newline="", encoding="utf-8") as handler:
            writer = csv.writer(handler, delimiter="\t")
            writer.writerow(["Protein", "GO ID", "Score"])
        log_warn("PLM transfer produced no direct GO assignments.")
        return

    started_at = time.monotonic()
    log_info(
        f"Materializing {len(direct_rows):,} direct PLM assignment row(s) "
        "for deduplication."
    )
    direct_df = pd.DataFrame(direct_rows)
    log_info(
        f"Deduplicating {len(direct_df):,} direct PLM assignment row(s) by "
        "protein and GO term."
    )
    direct_df = direct_df.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    log_info(
        f"Deduplicated to {len(direct_df):,} direct PLM assignment row(s) in "
        f"{time.monotonic() - started_at:.1f} seconds."
    )
    log_info("Loading direct PLM assignments into the Gene Ontology.")
    go.load_annotations(direct_df, "PLM seed")
    log_info("Up-propagating PLM assignments through the Gene Ontology.")
    go.up_propagate_annotations("PLM seed")
    log_info("Collecting propagated PLM assignments.")
    propagated = go.get_annotations("PLM seed")
    propagated = propagated[["Protein", "GO ID", "Score"]]
    log_info(
        f"Deduplicating {len(propagated):,} propagated PLM assignment row(s)."
    )
    propagated = propagated.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    log_info(f"Writing propagated PLM assignments to {out_path}.")
    propagated.to_csv(out_path, sep="\t", index=False)
    log_info(
        f"Wrote {len(propagated):,} propagated PLM assignment row(s) to "
        f"{out_path} in {time.monotonic() - started_at:.1f} seconds."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PLM seed by transferring GO terms from ESM1b nearest SwissProt embeddings."
    )
    parser.add_argument("--fastas", nargs="+", help="input FASTA file(s)")
    parser.add_argument("--target-fasta", help="SwissProt target FASTA")
    parser.add_argument("--goa", help="filtered GOA annotation file")
    parser.add_argument("--go-obo", help="GO OBO file")
    parser.add_argument(
        "--protein-id-mode",
        choices=["uniprot", "entire_id"],
        default="uniprot",
        help="how FASTA identifiers are parsed",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/esm1b_t33_650M_UR50S",
        help="Hugging Face model name or local model path",
    )
    parser.add_argument("--model-dir", default="", help="model cache directory")
    parser.add_argument(
        "--embeddings-dir",
        default="",
        help="directory for reusable target embedding caches",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--knn-k", type=int, default=10, help="number of nearest SwissProt proteins")
    parser.add_argument(
        "--transfer-strategy", choices=["knn", "kde"], default="knn"
    )
    parser.add_argument(
        "--long-sequence-mode",
        choices=["sliding_mean", "truncate", "skip"],
        default="sliding_mean",
        help="how to handle sequences longer than the ESM1b residue limit",
    )
    parser.add_argument("--long-window-size", type=int, default=1022)
    parser.add_argument("--long-overlap", type=int, default=128)
    parser.add_argument(
        "--score-mode",
        choices=["all_ones", "weighted_support"],
        default="all_ones",
        help="GO transfer scoring mode",
    )
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--precomputed-target-embeddings", default="")
    parser.add_argument("--precomputed-query-embeddings", default="")
    parser.add_argument("--force-target-embeddings", action="store_true")
    parser.add_argument("--force-query-embeddings", action="store_true")
    parser.add_argument("--download-model-only", action="store_true")
    parser.add_argument("--blacklist", default="", help="optional taxon blacklist")
    parser.add_argument(
        "--exclude-accessions",
        default="",
        help="optional one-per-line or CSV/TSV accession exclusion set",
    )
    parser.add_argument("--kde-bandwidth", type=float, default=0.025409690504535225)
    parser.add_argument("--kde-weight-floor", type=float, default=1e-6)
    parser.add_argument("--kde-max-neighbors", type=int, default=8192)
    parser.add_argument(
        "--evidence-codes",
        default="EXP,IDA,IPI,IMP,IGI,IEP,TAS,IC",
        help="comma-separated GO evidence codes accepted for PLM transfer",
    )
    parser.add_argument("--outdir", required=False, default="plm_results")
    return parser.parse_args()


def require_path(value: Optional[str], label: str) -> Path:
    if not value:
        raise RuntimeError(f"{label} is required.")
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.download_model_only:
        load_esm_model(
            args.model_name,
            Path(args.model_dir).expanduser() if args.model_dir else None,
            args.device,
            args.local_files_only,
        )
        log_info("Model download/load check completed.")
        return

    if args.knn_k <= 0:
        raise RuntimeError("--knn-k must be greater than zero.")
    if args.transfer_strategy == "kde" and args.score_mode != "weighted_support":
        raise RuntimeError("KDE transfer requires --score-mode weighted_support.")
    if args.batch_tokens <= 0:
        raise RuntimeError("--batch-tokens must be greater than zero.")
    if args.query_chunk_size <= 0:
        raise RuntimeError("--query-chunk-size must be greater than zero.")

    fasta_paths = [require_path(path, "input FASTA") for path in (args.fastas or [])]
    if not fasta_paths:
        raise RuntimeError("--fastas is required.")
    target_fasta = require_path(args.target_fasta, "--target-fasta")
    goa_path = require_path(args.goa, "--goa")
    go_obo = require_path(args.go_obo, "--go-obo")

    from GOTool import GeneOntology

    go = GeneOntology.GeneOntology(str(go_obo), verbose=False)
    go.build_structure()

    target_records = read_fasta(target_fasta, "uniprot")
    query_records = read_many_fastas(fasta_paths, args.protein_id_mode)
    if not target_records:
        raise RuntimeError("Target FASTA contains no protein sequences.")
    if not query_records:
        raise RuntimeError("Input FASTA contains no protein sequences.")
    log_info(
        f"PLM seed will process {len(query_records):,} query protein(s) "
        f"against {len(target_records):,} SwissProt target protein(s)."
    )

    model_dir = Path(args.model_dir).expanduser() if args.model_dir else None
    default_embeddings_dir = Path(args.embeddings_dir).expanduser() if args.embeddings_dir else outdir / "embeddings"
    default_embeddings_dir.mkdir(parents=True, exist_ok=True)

    target_metadata = build_cache_metadata(
        [target_fasta],
        args.model_name,
        "uniprot",
        args.long_sequence_mode,
        args.long_window_size,
        args.long_overlap,
    )
    query_metadata = build_cache_metadata(
        fasta_paths,
        args.model_name,
        args.protein_id_mode,
        args.long_sequence_mode,
        args.long_window_size,
        args.long_overlap,
    )

    if args.precomputed_target_embeddings:
        target_cache_dir = Path(args.precomputed_target_embeddings).expanduser()
        target_cache = load_embedding_cache(target_cache_dir, target_metadata)
        if target_cache is None:
            raise RuntimeError(
                f"Unable to load compatible precomputed target embeddings from {target_cache_dir}."
            )
    else:
        target_cache_dir = default_embeddings_dir / f"target_{metadata_key(target_metadata)}"
        target_cache = get_or_create_embedding_cache(
            target_cache_dir,
            target_records,
            target_metadata,
            args,
            force=args.force_target_embeddings,
        )

    if args.precomputed_query_embeddings:
        query_cache_dir = Path(args.precomputed_query_embeddings).expanduser()
        query_cache = load_embedding_cache(query_cache_dir, query_metadata)
        if query_cache is None:
            raise RuntimeError(
                f"Unable to load compatible precomputed query embeddings from {query_cache_dir}."
            )
    else:
        query_cache_dir = outdir / "query_embeddings"
        query_cache = get_or_create_embedding_cache(
            query_cache_dir,
            query_records,
            query_metadata,
            args,
            force=args.force_query_embeddings,
        )

    blacklist = read_blacklist(args.blacklist) if args.blacklist else None
    excluded_accessions = read_accession_exclusions(args.exclude_accessions)
    blacklist_indices = blacklisted_target_indices(target_cache, blacklist)
    accession_indices = excluded_accession_indices(
        target_cache, excluded_accessions
    )
    excluded_target_indices = np.unique(
        np.concatenate((blacklist_indices, accession_indices))
    )
    if blacklist:
        log_info(
            f"Excluding {len(blacklist_indices):,} SwissProt target protein(s) from PLM transfer "
            f"before neighbour selection using {len(blacklist):,} blacklist taxon ID(s)."
        )
    if excluded_accessions:
        log_info(
            f"Excluding {len(accession_indices):,} SwissProt target protein(s) "
            f"matching {len(excluded_accessions):,} configured accession(s)."
        )

    diagnostics = None
    neighbor_weights = None
    if args.transfer_strategy == "knn":
        neighbor_indices, neighbor_scores = compute_knn(
            query_cache.embeddings,
            target_cache.embeddings,
            args.knn_k,
            args.query_chunk_size,
            excluded_target_indices=excluded_target_indices,
        )
    else:
        (
            neighbor_indices,
            neighbor_scores,
            neighbor_weights,
            diagnostics,
        ) = compute_kde_neighbors(
            query_cache.embeddings,
            target_cache.embeddings,
            args.kde_bandwidth,
            args.kde_weight_floor,
            args.kde_max_neighbors,
            args.query_chunk_size,
            excluded_target_indices=excluded_target_indices,
        )
    neighbors_path = outdir / "neighbors.tsv"
    wanted_accessions = write_neighbors(
        neighbors_path,
        query_cache,
        target_cache,
        neighbor_indices,
        neighbor_scores,
        neighbor_weights=neighbor_weights,
    )
    log_info(
        f"Wrote PLM nearest neighbors to {neighbors_path}; "
        f"{len(wanted_accessions):,} unique SwissProt neighbor(s) will be checked in GOA."
    )

    accession_to_terms = load_go_terms_for_accessions(
        go,
        goa_path,
        wanted_accessions,
        blacklist=blacklist,
        evidence_codes={
            code.strip() for code in args.evidence_codes.split(",") if code.strip()
        } if args.evidence_codes else None,
    )
    direct_rows = build_direct_assignments(
        query_cache,
        target_cache,
        neighbor_indices,
        neighbor_scores,
        accession_to_terms,
        outdir,
        score_mode=args.score_mode,
        neighbor_weights=neighbor_weights,
        diagnostics=diagnostics,
    )
    write_propagated_assignments(go, direct_rows, outdir / "assignments.tsv")
    request = {
        "transfer_strategy": args.transfer_strategy,
        "knn_k": args.knn_k,
        "score_mode": args.score_mode,
        "kde_bandwidth": args.kde_bandwidth,
        "kde_weight_floor": args.kde_weight_floor,
        "kde_max_neighbors": args.kde_max_neighbors,
        "blacklist": str(Path(args.blacklist).expanduser().resolve()) if args.blacklist else "",
        "exclude_accessions": str(Path(args.exclude_accessions).expanduser().resolve()) if args.exclude_accessions else "",
    }
    parameters = {
        **request,
        "target_cache": str(target_cache.path.resolve()),
        "query_cache": str(query_cache.path.resolve()),
        "excluded_target_count": int(len(excluded_target_indices)),
    }
    transfer_metadata = {
        "version": TRANSFER_METADATA_VERSION,
        "request": request,
        "input_fingerprints": {
            "query_fasta": fasta_fingerprint(fasta_paths),
            "target_fasta": fasta_fingerprint([target_fasta]),
            "goa": fasta_fingerprint([goa_path]),
            "go_obo": fasta_fingerprint([go_obo]),
            "blacklist": fasta_fingerprint([Path(args.blacklist).expanduser()]) if args.blacklist else [],
            "exclude_accessions": fasta_fingerprint([Path(args.exclude_accessions).expanduser()]) if args.exclude_accessions else [],
            "precomputed_target_embeddings": cache_fingerprint(args.precomputed_target_embeddings),
            "precomputed_query_embeddings": cache_fingerprint(args.precomputed_query_embeddings),
        },
        "parameters": parameters,
        "signature": hashlib.sha256(
            json.dumps(parameters, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "query_count": len(query_cache.ids),
        "target_count": len(target_cache.ids),
        "direct_assignment_count": len(direct_rows),
    }
    with open(outdir / "transfer_metadata.json", "w", encoding="utf-8") as handler:
        json.dump(transfer_metadata, handler, indent=2, sort_keys=True)
        handler.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_warn(str(exc))
        raise
