import argparse
import csv
import gzip
import json
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
    import hashlib

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
        if cache_metadata.get(key) != value:
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
) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise RuntimeError("PLM knn_k must be greater than zero.")
    if target_embeddings.shape[0] == 0:
        raise RuntimeError("Target embedding cache is empty.")

    k = min(k, target_embeddings.shape[0])
    target_norm = normalize_embeddings(target_embeddings)
    query_norm = normalize_embeddings(query_embeddings)
    all_indices = np.zeros((query_norm.shape[0], k), dtype=np.int64)
    all_scores = np.zeros((query_norm.shape[0], k), dtype=np.float32)

    for start in range(0, query_norm.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_norm.shape[0])
        similarities = query_norm[start:end].dot(target_norm.T)
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
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
) -> Set[str]:
    wanted_accessions: Set[str] = set()
    with open(output_path, "w", newline="", encoding="utf-8") as handler:
        writer = csv.writer(handler, delimiter="\t")
        writer.writerow(["Protein", "Neighbor", "Rank", "Cosine"])
        for query_index, query_id in enumerate(query_cache.ids):
            for rank, target_index in enumerate(neighbor_indices[query_index], start=1):
                neighbor_id = target_cache.ids[int(target_index)]
                wanted_accessions.add(neighbor_id)
                writer.writerow(
                    [
                        query_id,
                        neighbor_id,
                        rank,
                        f"{float(neighbor_scores[query_index, rank - 1]):.8f}",
                    ]
                )
    return wanted_accessions


def build_direct_assignments(
    query_cache: EmbeddingCache,
    target_cache: EmbeddingCache,
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    accession_to_terms: Dict[str, Set[str]],
    outdir: Path,
) -> List[Dict[str, object]]:
    direct_rows: List[Dict[str, object]] = []
    for query_index, query_id in enumerate(query_cache.ids):
        go_terms: Set[str] = set()
        summary_neighbors = []
        for rank, target_index in enumerate(neighbor_indices[query_index], start=1):
            neighbor_id = target_cache.ids[int(target_index)]
            terms = sorted(accession_to_terms.get(neighbor_id, set()))
            go_terms.update(terms)
            summary_neighbors.append(
                {
                    "neighbor": neighbor_id,
                    "rank": rank,
                    "cosine": float(neighbor_scores[query_index, rank - 1]),
                    "go_count": len(terms),
                }
            )
        for go_id in sorted(go_terms):
            direct_rows.append({"Protein": query_id, "GO ID": go_id, "Score": 1.0})

        summary = {
            "protein_id": query_id,
            "neighbors": summary_neighbors,
            "direct_go_count": len(go_terms),
        }
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

    direct_df = pd.DataFrame(direct_rows)
    direct_df = direct_df.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    log_info(
        f"Up-propagating {len(direct_df):,} direct PLM assignment row(s)."
    )
    go.load_annotations(direct_df, "PLM seed")
    go.up_propagate_annotations("PLM seed")
    propagated = go.get_annotations("PLM seed")
    propagated = propagated[["Protein", "GO ID", "Score"]]
    propagated = propagated.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    propagated.to_csv(out_path, sep="\t", index=False)
    log_info(f"Wrote {len(propagated):,} propagated PLM assignment row(s) to {out_path}.")


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
        "--long-sequence-mode",
        choices=["sliding_mean", "truncate", "skip"],
        default="sliding_mean",
        help="how to handle sequences longer than the ESM1b residue limit",
    )
    parser.add_argument("--long-window-size", type=int, default=1022)
    parser.add_argument("--long-overlap", type=int, default=128)
    parser.add_argument(
        "--score-mode",
        choices=["all_ones"],
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

    neighbor_indices, neighbor_scores = compute_knn(
        query_cache.embeddings,
        target_cache.embeddings,
        args.knn_k,
        args.query_chunk_size,
    )
    neighbors_path = outdir / "neighbors.tsv"
    wanted_accessions = write_neighbors(
        neighbors_path,
        query_cache,
        target_cache,
        neighbor_indices,
        neighbor_scores,
    )
    log_info(
        f"Wrote PLM nearest neighbors to {neighbors_path}; "
        f"{len(wanted_accessions):,} unique SwissProt neighbor(s) will be checked in GOA."
    )

    accession_to_terms = load_go_terms_for_accessions(
        go,
        goa_path,
        wanted_accessions,
        blacklist=read_blacklist(args.blacklist) if args.blacklist else None,
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
    )
    write_propagated_assignments(go, direct_rows, outdir / "assignments.tsv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_warn(str(exc))
        raise
