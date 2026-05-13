import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from Utils.progress_bar import progressBar

UNIPROT_ACC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])$"
)
NON_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
FOLDSEEK_ALL_MAX_SEQS = 2147483647
FOLDSEEK_STREAM_PROGRESS_EVERY = 5_000_000


def log_info(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {timestamp}] {message}")


def log_warn(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WARN {timestamp}] {message}", file=sys.stderr)


def format_name_preview(names: List[str], limit: int = 5) -> str:
    if not names:
        return "none"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview += ", ..."
    return preview


def run_streaming_command(
    command: List[str],
    prefix: str,
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str]:
    captured_lines: List[str] = []
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(f"\n# Command: {' '.join(command)}\n")
        log_handle.flush()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            captured_lines.append(line)
            print(f"[{prefix}] {line}")
            if log_handle is not None:
                log_handle.write(line + "\n")
        process.stdout.close()
        return_code = process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()

    return return_code, "\n".join(captured_lines)


def resolve_prostt5_model_path(prostt5_model: Optional[str]) -> Optional[str]:
    if not prostt5_model:
        return None

    model_path = Path(prostt5_model).expanduser()
    if model_path.is_dir():
        candidates = [
            model_path / "prostt5-f16.gguf",
            model_path / "prostt5.gguf",
        ]
        candidates.extend(sorted(model_path.glob("*.gguf")))
        for candidate in candidates:
            if candidate.exists():
                log_info(
                    f"Resolved ProstT5 model directory {model_path} to {candidate}."
                )
                return str(candidate)
        log_warn(
            f"ProstT5 model path is a directory but no .gguf model was found: {model_path}"
        )
        return str(model_path)

    if not model_path.exists():
        log_warn(f"ProstT5 model path does not exist: {model_path}")
    return str(model_path)


def resolve_foldseek_target_db(target_db: str) -> str:
    db_path = Path(target_db).expanduser()

    if db_path.is_dir():
        candidate = db_path / db_path.name
        if candidate.exists():
            log_warn(
                f"Foldseek target DB points to a directory. Using base DB prefix {candidate}."
            )
            return str(candidate)
        log_warn(
            f"Foldseek target DB points to a directory, but {candidate} was not found. "
            "Pass the database prefix, not the containing directory."
        )
        return str(db_path)

    for suffix in ("_ca", "_h", "_ss"):
        if db_path.name.endswith(suffix):
            base_path = db_path.with_name(db_path.name[: -len(suffix)])
            if (Path(str(base_path) + ".lookup")).exists():
                log_warn(
                    f"Foldseek target DB points to internal component {db_path}. "
                    f"Using base DB prefix {base_path} instead."
                )
                return str(base_path)

    lookup_path = Path(str(db_path) + ".lookup")
    if not lookup_path.exists():
        log_warn(
            f"Foldseek lookup file was not found for target DB prefix {db_path}: "
            f"{lookup_path}"
        )
    return str(db_path)


def build_foldseek_gpu_settings(
    gpu: Optional[str],
    cuda_visible_devices: Optional[str],
) -> Tuple[List[str], Dict[str, str]]:
    env_overrides: Dict[str, str] = {}
    gpu_args: List[str] = []

    if cuda_visible_devices:
        env_overrides["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        log_info(f"Foldseek CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        if not gpu:
            gpu = "1"
            log_info(
                "--cuda-visible-devices was provided, so Foldseek GPU mode is enabled with --gpu 1."
            )

    if gpu:
        gpu = str(gpu).strip()
        gpu_args = ["--gpu", gpu]
        if gpu == "0":
            log_warn(
                "Foldseek --gpu 0 disables GPU. Use --gpu 1 to enable CUDA; "
                "use --cuda-visible-devices 0 to select GPU device 0."
            )
        else:
            log_info(f"Foldseek GPU switch: --gpu {gpu}")

    return gpu_args, env_overrides


def resolve_foldseek_gpu_target_db(target_db: str, gpu: Optional[str]) -> str:
    if not gpu or str(gpu).strip() == "0":
        return target_db

    db_path = Path(target_db)
    if db_path.name.endswith("_pad"):
        return target_db

    padded_db = Path(str(db_path) + "_pad")
    if padded_db.exists():
        log_warn(
            f"Foldseek GPU mode is enabled. Using padded target DB {padded_db}."
        )
        return str(padded_db)

    ss_db = Path(str(db_path) + "_ss")
    if ss_db.exists():
        raise RuntimeError(
            "Foldseek GPU search requires a padded target sequence database. "
            f"Found {ss_db}, but padded target DB {padded_db} is missing.\n"
            "Create it once with:\n"
            f"foldseek makepaddedseqdb {db_path} {padded_db}\n"
            f"Then rerun with target DB {padded_db} and --gpu 1."
        )

    return target_db


def open_maybe_gzip(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def read_blacklist(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    blacklist = set()
    with open(path, "r", encoding="utf-8") as handler:
        for line in handler:
            token = line.strip()
            if token:
                blacklist.add(token)
    return blacklist or None


def wrap_seq(sequence: str, width: int = 60) -> str:
    return "\n".join(
        sequence[i : i + width] for i in range(0, len(sequence), width)
    )


def sanitize_name(header: str) -> str:
    base = header.split()[0]
    base = re.sub(r"[^\w\-\.]+", "_", base)
    return base[:80]


def extract_uniprot_from_id(identifier: str) -> Optional[str]:
    m = re.search(r"AF-([A-Z0-9]+)-F\d", identifier)
    if m:
        return m.group(1)
    m = re.search(r"(?:sp|tr)\|([A-Z0-9]+)\|", identifier)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[>\s_])(?:sp|tr)_([A-Z0-9]+)", identifier)
    if m:
        return m.group(1)
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
class FastaSeq:
    header: str
    seq: str
    name: str
    protein_id: str


def read_fasta(path: Path, protein_id_mode: str) -> List[FastaSeq]:
    records: List[FastaSeq] = []
    header = None
    seq_chunks: List[str] = []
    with open(path, "r", encoding="utf-8") as handler:
        for raw_line in handler:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = re.sub(r"\s+", "", "".join(seq_chunks)).upper()
                    sequence = NON_AA_RE.sub("", sequence)
                    records.append(
                        FastaSeq(
                            header=header,
                            seq=sequence,
                            name=sanitize_name(header),
                            protein_id=protein_id_from_header(
                                header, protein_id_mode
                            ),
                        )
                    )
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            sequence = re.sub(r"\s+", "", "".join(seq_chunks)).upper()
            sequence = NON_AA_RE.sub("", sequence)
            records.append(
                FastaSeq(
                    header=header,
                    seq=sequence,
                    name=sanitize_name(header),
                    protein_id=protein_id_from_header(header, protein_id_mode),
                )
            )
    return records


@dataclass
class Hit:
    query: str
    target: str
    evalue: float
    bits: float
    fident: float
    alnlen: int
    qstart: int
    qend: int
    tstart: int
    tend: int
    qlen: int
    tlen: int
    qtmscore: Optional[float] = None
    ttmscore: Optional[float] = None
    alntmscore: Optional[float] = None
    lddt: Optional[float] = None

    @property
    def qcov(self) -> float:
        length = max(0, self.qend - self.qstart + 1)
        return length / self.qlen if self.qlen else 0.0

    @property
    def tcov(self) -> float:
        length = max(0, self.tend - self.tstart + 1)
        return length / self.tlen if self.tlen else 0.0

    @property
    def avg_tm(self) -> Optional[float]:
        if self.qtmscore is None or self.ttmscore is None:
            return None
        return 0.5 * (self.qtmscore + self.ttmscore)


def to_optional_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def parse_hit_from_parts(parts: List[str]) -> Optional[Hit]:
    if len(parts) < 12:
        return None
    try:
        return Hit(
            query=parts[0],
            target=parts[1],
            evalue=float(parts[2]),
            bits=float(parts[3]),
            fident=float(parts[4]),
            alnlen=int(float(parts[5])),
            qstart=int(float(parts[6])),
            qend=int(float(parts[7])),
            tstart=int(float(parts[8])),
            tend=int(float(parts[9])),
            qlen=int(float(parts[10])),
            tlen=int(float(parts[11])),
            qtmscore=to_optional_float(parts[12]) if len(parts) > 12 else None,
            ttmscore=to_optional_float(parts[13]) if len(parts) > 13 else None,
            alntmscore=to_optional_float(parts[14]) if len(parts) > 14 else None,
            lddt=to_optional_float(parts[15]) if len(parts) > 15 else None,
        )
    except Exception:
        return None


def hit_passes_thresholds(
    hit: Hit,
    evalue_max: float,
    min_qcov: float,
    min_tcov: float,
    min_avg_tm: Optional[float],
    tm_available: bool,
) -> bool:
    if hit.evalue > evalue_max:
        return False
    if hit.qcov < min_qcov or hit.tcov < min_tcov:
        return False
    if tm_available and min_avg_tm is not None:
        if hit.avg_tm is None or hit.avg_tm < min_avg_tm:
            return False
    return True


def run_foldseek_easy_search(
    query_path: Path,
    target_db: str,
    tmpdir: Path,
    alignment_type: int,
    max_seqs: int,
    prostt5_model: Optional[str] = None,
    gpu: Optional[str] = None,
    cuda_visible_devices: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> Path:
    foldseek = shutil.which("foldseek")
    if not foldseek:
        raise RuntimeError("Foldseek not found in PATH.")

    prostt5_model = resolve_prostt5_model_path(prostt5_model)
    effective_gpu = gpu or ("1" if cuda_visible_devices else None)
    target_db = resolve_foldseek_gpu_target_db(target_db, effective_gpu)
    out_tsv = tmpdir / f"{query_path.stem}.foldseek.tsv"
    format_fasta = ",".join(
        [
            "query",
            "target",
            "evalue",
            "bits",
            "fident",
            "alnlen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "qlen",
            "tlen",
        ]
    )
    format_struct = ",".join(
        [
            "query",
            "target",
            "evalue",
            "bits",
            "fident",
            "alnlen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "qlen",
            "tlen",
            "qtmscore",
            "ttmscore",
            "alntmscore",
            "lddt",
        ]
    )
    fmt = format_fasta if prostt5_model else format_struct

    command = [
        foldseek,
        "easy-search",
        str(query_path),
        target_db,
        str(out_tsv),
        str(tmpdir),
        "--format-output",
        fmt,
        "--max-seqs",
        str(max_seqs),
    ]

    if prostt5_model:
        command += ["--prostt5-model", prostt5_model]
        alignment_type = 0
    command += ["--alignment-type", str(alignment_type)]

    gpu_args, env_overrides = build_foldseek_gpu_settings(
        gpu,
        cuda_visible_devices,
    )
    command += gpu_args

    log_info(f"Running Foldseek command: {' '.join(command)}")
    env = os.environ.copy()
    env.update(env_overrides)
    return_code, command_output = run_streaming_command(
        command,
        prefix="FOLDSEEK",
        log_path=log_path,
        env=env,
    )
    if return_code != 0:
        raise RuntimeError(
            f"Foldseek failed for {query_path.name}:\n{command_output}"
        )
    return out_tsv


def select_hits(
    hits: List[Hit],
    evalue_max: float,
    min_qcov: float,
    min_tcov: float,
    min_avg_tm: Optional[float],
    tm_available: bool,
) -> List[Hit]:
    selected: List[Hit] = []
    seen_targets: Set[str] = set()
    for hit in hits:
        if hit.evalue > evalue_max:
            continue
        if hit.qcov < min_qcov or hit.tcov < min_tcov:
            continue
        if tm_available and min_avg_tm is not None:
            if hit.avg_tm is None or hit.avg_tm < min_avg_tm:
                continue
        if hit.target in seen_targets:
            continue
        selected.append(hit)
        seen_targets.add(hit.target)
    return selected


def resolve_max_seqs(max_seqs: int) -> int:
    if max_seqs < 0:
        raise ValueError("--max-seqs cannot be negative.")
    if max_seqs == 0:
        log_warn(
            f"Foldseek --max-seqs set to {FOLDSEEK_ALL_MAX_SEQS} "
            "to retrieve all available prefilter hits."
        )
        return FOLDSEEK_ALL_MAX_SEQS
    return max_seqs


def group_hits_by_query(hits: List[Hit]) -> Dict[str, List[Hit]]:
    by_query: Dict[str, List[Hit]] = defaultdict(list)
    for hit in hits:
        by_query[hit.query].append(hit)
    return by_query


def stream_filter_hits_from_tsv(
    tsv_path: Path,
    accepted_hits_path: Path,
    evalue_max: float,
    min_qcov: float,
    min_tcov: float,
    min_avg_tm: Optional[float],
    progress_every: int = FOLDSEEK_STREAM_PROGRESS_EVERY,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, bool], Set[str]]:
    raw_hit_counts: Dict[str, int] = defaultdict(int)
    tm_available: Dict[str, bool] = defaultdict(bool)

    log_info(
        f"Streaming raw Foldseek hits from {tsv_path} to count per-query hits and TM availability."
    )
    with open(tsv_path, "r", encoding="utf-8") as handler:
        for line_number, line in enumerate(handler, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            query_id = parts[0]
            raw_hit_counts[query_id] += 1
            if len(parts) > 13:
                qtmscore = to_optional_float(parts[12])
                ttmscore = to_optional_float(parts[13])
                if qtmscore is not None and ttmscore is not None:
                    tm_available[query_id] = True
            if line_number % progress_every == 0:
                log_info(
                    f"Scanned {line_number:,} raw Foldseek hit row(s) from {tsv_path.name}."
                )

    log_info(
        f"First streaming pass finished for {tsv_path.name}: "
        f"{sum(raw_hit_counts.values()):,} raw hit row(s) across "
        f"{len(raw_hit_counts):,} query identifier(s)."
    )

    accepted_hit_counts: Dict[str, int] = defaultdict(int)
    wanted_accessions: Set[str] = set()
    seen_targets_by_query: Dict[str, Set[str]] = defaultdict(set)

    log_info(
        f"Streaming {tsv_path.name} again and writing only accepted hits to {accepted_hits_path.name}."
    )
    with open(tsv_path, "r", encoding="utf-8") as handler, open(
        accepted_hits_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as accepted_handler:
        writer = csv.writer(accepted_handler, delimiter="\t")
        for line_number, line in enumerate(handler, start=1):
            parts = line.rstrip("\n").split("\t")
            hit = parse_hit_from_parts(parts)
            if hit is None:
                continue
            if not hit_passes_thresholds(
                hit,
                evalue_max=evalue_max,
                min_qcov=min_qcov,
                min_tcov=min_tcov,
                min_avg_tm=min_avg_tm,
                tm_available=tm_available.get(hit.query, False),
            ):
                continue
            seen_targets = seen_targets_by_query[hit.query]
            if hit.target in seen_targets:
                continue
            seen_targets.add(hit.target)

            accession = extract_uniprot_from_id(hit.target) or ""
            if accession:
                wanted_accessions.add(accession)
            writer.writerow(
                [
                    hit.query,
                    hit.target,
                    accession,
                    f"{hit.evalue:.12g}",
                    f"{hit.bits:.12g}",
                    f"{hit.qcov:.6f}",
                    f"{hit.tcov:.6f}",
                    "" if hit.avg_tm is None else f"{hit.avg_tm:.6f}",
                ]
            )
            accepted_hit_counts[hit.query] += 1

            if line_number % progress_every == 0:
                log_info(
                    f"Second streaming pass retained {sum(accepted_hit_counts.values()):,} "
                    f"accepted Foldseek hit row(s) after scanning {line_number:,} rows from {tsv_path.name}."
                )

    log_info(
        f"Second streaming pass finished for {tsv_path.name}: "
        f"{sum(accepted_hit_counts.values()):,} accepted hit row(s) across "
        f"{sum(1 for count in accepted_hit_counts.values() if count > 0):,} query identifier(s)."
    )

    return (
        dict(raw_hit_counts),
        dict(accepted_hit_counts),
        dict(tm_available),
        wanted_accessions,
    )


def process_accepted_hits_stream(
    accepted_hits_path: Path,
    accession_to_terms: Dict[str, Set[str]],
    query_id_map: Dict[str, str],
    protein_id_mode: str,
    score_mode: str,
    outdir: Path,
    parameters: Dict[str, object],
    progress_every: int = 1_000_000,
) -> List[List[object]]:
    aggregate_rows: List[List[object]] = []
    if not accepted_hits_path.exists() or accepted_hits_path.stat().st_size == 0:
        return aggregate_rows

    current_query_id: Optional[str] = None
    current_hits_handle = None
    current_hits_writer = None
    current_go_support: Counter = Counter()
    current_summary_hits: List[Dict[str, object]] = []
    current_accepted_count = 0
    finalized_queries: Set[str] = set()

    def finalize_current_query() -> None:
        nonlocal current_query_id
        nonlocal current_hits_handle
        nonlocal current_hits_writer
        nonlocal current_go_support
        nonlocal current_summary_hits
        nonlocal current_accepted_count

        if current_query_id is None:
            return

        protein_id = resolve_query_protein_id(
            current_query_id, query_id_map, protein_id_mode
        )
        direct_scores = {}
        for go_id, count in sorted(current_go_support.items()):
            if score_mode == "support_fraction" and current_accepted_count > 0:
                score = count / current_accepted_count
            else:
                score = 1.0
            direct_scores[go_id] = score
            aggregate_rows.append([protein_id, go_id, score])

        summary = {
            "query": current_query_id,
            "protein_id": protein_id,
            "parameters": parameters,
            "accepted_hits": current_summary_hits,
            "go_support": dict(sorted(current_go_support.items())),
            "direct_go_scores": direct_scores,
        }
        summary_path = outdir / f"{sanitize_name(current_query_id)}.summary.json"
        with open(summary_path, "w", encoding="utf-8") as handler:
            json.dump(summary, handler, indent=2)

        if current_hits_handle is not None:
            current_hits_handle.close()

        log_info(
            f"Protein {protein_id}: transferred {len(direct_scores)} direct GO term(s) "
            f"from {current_accepted_count} accepted hit(s)."
        )

        finalized_queries.add(current_query_id)
        current_query_id = None
        current_hits_handle = None
        current_hits_writer = None
        current_go_support = Counter()
        current_summary_hits = []
        current_accepted_count = 0

    with open(accepted_hits_path, "r", encoding="utf-8") as handler:
        reader = csv.reader(handler, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) < 8:
                continue
            query_id, target, accession, evalue, bits, qcov, tcov, avg_tm = row[:8]

            if current_query_id is None or query_id != current_query_id:
                if current_query_id is not None:
                    finalize_current_query()
                if query_id in finalized_queries:
                    raise RuntimeError(
                        "Accepted Foldseek hits are not grouped by query identifier. "
                        "This should not happen with Foldseek easy-search output."
                    )
                current_query_id = query_id
                hits_path = outdir / f"{sanitize_name(query_id)}.hits.tsv"
                current_hits_handle = open(
                    hits_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                )
                current_hits_writer = csv.writer(current_hits_handle, delimiter="\t")
                current_hits_writer.writerow(
                    [
                        "target",
                        "uniprot",
                        "evalue",
                        "bits",
                        "qcov",
                        "tcov",
                        "avg_tm",
                        "go_count",
                    ]
                )

            go_terms = accession_to_terms.get(accession, set()) if accession else set()
            assert current_hits_writer is not None
            current_hits_writer.writerow(
                [
                    target,
                    accession,
                    evalue,
                    bits,
                    qcov,
                    tcov,
                    avg_tm,
                    str(len(go_terms)),
                ]
            )
            for go_id in go_terms:
                current_go_support[go_id] += 1
            current_summary_hits.append(
                {
                    "target": target,
                    "uniprot": accession,
                    "evalue": float(evalue),
                    "bits": float(bits),
                    "qcov": round(float(qcov), 3),
                    "tcov": round(float(tcov), 3),
                    "avg_tm": round(float(avg_tm), 3) if avg_tm else None,
                }
            )
            current_accepted_count += 1

            if line_number % progress_every == 0:
                log_info(
                    f"Processed {line_number:,} accepted Foldseek hit row(s) into final outputs."
                )

    finalize_current_query()
    return aggregate_rows


def resolve_python_near_executable(executable_path: str) -> Optional[str]:
    exec_path = Path(executable_path).resolve()
    for name in ("python", "python3"):
        candidate = exec_path.parent / name
        if candidate.exists():
            return str(candidate)
    try:
        with open(exec_path, "r", encoding="utf-8", errors="ignore") as handler:
            shebang = handler.readline().strip()
    except OSError:
        return None
    if shebang.startswith("#!"):
        return shebang[2:].split()[0]
    return None


def inspect_colabfold_runtime(colabfold_executable: str) -> Tuple[str, List[str], Optional[str]]:
    python_executable = resolve_python_near_executable(colabfold_executable)
    if python_executable is None:
        return "unknown", [], "Could not resolve the Python executable used by colabfold_batch."

    command = [
        python_executable,
        "-c",
        (
            "import jax; "
            "print(jax.default_backend()); "
            "print('\\n'.join(str(device) for device in jax.devices()))"
        ),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return "unknown", [], (
            "Failed to inspect the ColabFold JAX runtime: "
            f"{message or 'unknown error'}"
        )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return "unknown", [], "ColabFold JAX runtime probe returned no output."
    backend = lines[0]
    devices = lines[1:]
    return backend, devices, None


def find_colabfold_structure(out_dir: Path, record_name: str) -> Optional[Path]:
    patterns = [
        f"*{record_name}*rank*_model*.pdb",
        f"*{record_name}*rank*_model*.cif",
        f"{record_name}.pdb",
        f"{record_name}.cif",
    ]
    for pattern in patterns:
        candidates = sorted(out_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def ensure_structure_for_fasta(
    fasta_path: Path,
    mode: str,
    structures_dir: Path,
    protein_id_mode: str,
    search_recursive: bool = False,
    colabfold_msa_mode: Optional[str] = None,
    colabfold_num_models: Optional[int] = None,
    colabfold_num_recycle: Optional[int] = None,
    colabfold_use_gpu_relax: bool = False,
    colabfold_debug_logging: bool = False,
) -> Tuple[List[Tuple[Path, FastaSeq]], List[FastaSeq]]:
    sequences = read_fasta(fasta_path, protein_id_mode)
    structure_pairs: List[Tuple[Path, FastaSeq]] = []
    missing: List[FastaSeq] = []
    globber = structures_dir.rglob if search_recursive else structures_dir.glob

    log_info(
        f"Loaded {len(sequences)} protein(s) from {fasta_path.name} "
        f"using structure mode '{mode}'."
    )

    if mode == "existing":
        log_info(
            f"Searching for existing structures in {structures_dir} "
            f"(recursive={search_recursive})."
        )
        for record in sequences:
            log_info(f"Resolving existing structure for protein {record.name}.")
            candidates = [
                structures_dir / f"{record.name}.pdb",
                structures_dir / f"{record.name}.cif",
                structures_dir / f"{record.name}.pdb.gz",
                structures_dir / f"{record.name}.cif.gz",
            ]
            hit = next((candidate for candidate in candidates if candidate.exists()), None)
            if hit is None and protein_id_mode == "uniprot":
                accession = extract_uniprot_from_id(record.protein_id)
                if accession:
                    patterns = [
                        f"AF-{accession}-F*-model*.pdb",
                        f"AF-{accession}-F*-model*.cif",
                        f"AF-{accession}-F*-model*.pdb.gz",
                        f"AF-{accession}-F*-model*.cif.gz",
                    ]
                    for pattern in patterns:
                        found = sorted(globber(pattern))
                        if found:
                            hit = found[0]
                            break
            if hit is None:
                log_info(f"No existing structure found for protein {record.name}.")
                missing.append(record)
                continue
            log_info(
                f"Using existing structure for protein {record.name}: {hit}"
            )
            structure_pairs.append((hit, record))
        return structure_pairs, missing

    if mode == "colabfold":
        out_dir = structures_dir / f"colabfold_{fasta_path.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_info(
            f"Using ColabFold output directory for {len(sequences)} protein(s) "
            f"from {fasta_path.name}. Output directory: {out_dir}"
        )
        resolved_paths: Dict[str, Path] = {}
        missing_for_prediction: List[FastaSeq] = []

        log_info("Processing existing ColabFold structures before prediction.")
        for record in progressBar(sequences):
            existing_path = find_colabfold_structure(out_dir, record.name)
            if existing_path is not None:
                resolved_paths[record.name] = existing_path
                # log_info(
                #     f"Reusing existing ColabFold structure for protein {record.name}: "
                #     f"{existing_path}"
                # )
            else:
                missing_for_prediction.append(record)
                # log_info(
                #     f"No existing ColabFold structure found for protein {record.name}; "
                #     "queued for prediction."
                # )

        log_info(
            f"ColabFold resume status for {fasta_path.name}: "
            f"{len(resolved_paths)} ready, {len(missing_for_prediction)} missing."
        )

        if missing_for_prediction:
            colabfold = shutil.which("colabfold_batch")
            if not colabfold:
                raise RuntimeError(
                    "colabfold_batch not found in PATH; install ColabFold or use --structure-mode existing."
                )

            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                log_info(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

            backend, devices, runtime_warning = inspect_colabfold_runtime(colabfold)
            if runtime_warning:
                log_warn(runtime_warning)
            else:
                device_preview = ", ".join(devices) if devices else "no devices reported"
                log_info(
                    f"ColabFold runtime backend: {backend}; visible devices: {device_preview}"
                )
                if not devices or all("CpuDevice" in device for device in devices):
                    log_warn(
                        "ColabFold appears to see CPU only. "
                        "Prediction will be slow until the NVIDIA driver/CUDA/JAX setup is fixed."
                    )

            prepared_fasta = out_dir / f"{fasta_path.stem}.sanitized.fasta"
            with open(prepared_fasta, "w", encoding="utf-8") as handler:
                for index, record in enumerate(missing_for_prediction, start=1):
                    handler.write(f">{record.name}\n{wrap_seq(record.seq)}\n")
                    log_info(
                        f"ColabFold batch item {index}/{len(missing_for_prediction)}: "
                        f"{record.name}"
                    )
            command = [colabfold, str(prepared_fasta), str(out_dir)]
            if colabfold_msa_mode:
                command += ["--msa-mode", colabfold_msa_mode]
            if colabfold_num_models is not None:
                command += ["--num-models", str(colabfold_num_models)]
            if colabfold_num_recycle is not None:
                command += ["--num-recycle", str(colabfold_num_recycle)]
            if colabfold_use_gpu_relax:
                command.append("--use-gpu-relax")
            if colabfold_debug_logging:
                command.append("--debug-logging")
            colabfold_log_path = out_dir / "colabfold_batch.log"
            log_info(
                f"Running colabfold_batch for {len(missing_for_prediction)} missing protein(s) "
                f"on {prepared_fasta.name}."
            )
            log_info(f"Running ColabFold command: {' '.join(command)}")
            log_info(
                f"Streaming ColabFold logs to the console and to {colabfold_log_path}."
            )
            return_code, command_output = run_streaming_command(
                command,
                prefix="COLABFOLD",
                log_path=colabfold_log_path,
            )
            if return_code != 0:
                raise RuntimeError(
                    f"colabfold_batch failed for {fasta_path.name}:\n{command_output}"
                )

            for record in missing_for_prediction:
                predicted_path = find_colabfold_structure(out_dir, record.name)
                if predicted_path is None:
                    raise FileNotFoundError(
                        f"No PDB/CIF produced by ColabFold for sequence {record.name}."
                    )
                resolved_paths[record.name] = predicted_path
                log_info(
                    f"Using newly predicted ColabFold structure for protein {record.name}: "
                    f"{predicted_path}"
                )
        else:
            log_info(
                f"All proteins from {fasta_path.name} already have ColabFold structures. "
                "Skipping prediction."
            )

        for record in sequences:
            structure_path = resolved_paths.get(record.name)
            if structure_path is None:
                raise FileNotFoundError(
                    f"Unable to resolve a ColabFold structure for sequence {record.name}."
                )
            structure_pairs.append((structure_path, record))
        return structure_pairs, []

    raise ValueError(f"Unsupported structure mode: {mode}")


def resolve_query_protein_id(
    query_id: str,
    query_id_map: Dict[str, str],
    protein_id_mode: str,
) -> str:
    if query_id in query_id_map:
        return query_id_map[query_id]

    first_token = query_id.split()[0]
    if first_token in query_id_map:
        return query_id_map[first_token]

    matched = [
        key
        for key in query_id_map.keys()
        if query_id.startswith(key) or key in query_id or key.startswith(query_id)
    ]
    if matched:
        best = sorted(set(matched), key=len, reverse=True)[0]
        return query_id_map[best]

    if protein_id_mode == "uniprot":
        return extract_uniprot_from_id(query_id) or first_token
    return first_token


def stream_goa_terms_for_wanted(
    goa_path: str,
    wanted: Set[str],
    go,
    blacklist: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    accession_to_terms: Dict[str, Set[str]] = defaultdict(set)
    if not wanted:
        return accession_to_terms

    with open_maybe_gzip(goa_path) as handler:
        for line in handler:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            accession = fields[1].strip()
            if accession not in wanted:
                continue
            qualifiers = [q for q in fields[3].split("|") if q]
            if "NOT" in qualifiers:
                continue
            if blacklist:
                taxons = []
                for taxon in fields[12].split("|"):
                    taxons.append(taxon.split(":", 1)[1] if ":" in taxon else taxon)
                if any(taxon in blacklist for taxon in taxons):
                    continue
            go_id = fields[4].strip()
            try:
                term = go.find_term(go_id)
            except KeyError:
                continue
            if term.is_obsolete:
                continue
            accession_to_terms[accession].add(term.go_id)
    return accession_to_terms


def up_propagate_assignments(go, assignments, organism_name: str = "foldseek_transfer"):
    import pandas as pd

    if assignments.empty:
        return pd.DataFrame(columns=["Protein", "GO ID", "Score"])

    assignments = (
        assignments.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
    )
    go.load_annotations(assignments, organism_name)
    go.up_propagate_annotations(organism_name)
    propagated = go.get_annotations(organism_name)
    return propagated[["Protein", "GO ID", "Score"]]


def main():
    parser = argparse.ArgumentParser(
        description="Batch S2F Foldseek pipeline: FASTA/structure to experimental GO transfer."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--fastas",
        nargs="+",
        help="Input FASTA files (glob accepted by the shell).",
    )
    input_group.add_argument(
        "--query-structures",
        nargs="+",
        help="Input query structure files (PDB/CIF).",
    )

    parser.add_argument(
        "--structure-mode",
        choices=["existing", "colabfold"],
        default="existing",
        help="How to obtain structures for FASTA inputs (default: existing).",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("structures"),
        help="Directory to look for or write structures for FASTA inputs.",
    )
    parser.add_argument(
        "--search-recursive",
        action="store_true",
        help="Search structures_dir recursively for AlphaFold files.",
    )
    parser.add_argument(
        "--target-db",
        default=None,
        help="Foldseek target DB base path (from `foldseek createdb`).",
    )
    parser.add_argument(
        "--precomputed-tsv",
        default=None,
        help="Optional existing Foldseek TSV to parse instead of running easy-search.",
    )
    parser.add_argument(
        "--goa",
        required=True,
        help="SwissProt GOA file used for GO transfer (optionally gzipped).",
    )
    parser.add_argument(
        "--go-obo",
        default="go.obo",
        help="Gene Ontology OBO file (default: go.obo).",
    )
    parser.add_argument(
        "--protein-id-mode",
        choices=["uniprot", "entire_id"],
        default="uniprot",
        help="How output Protein identifiers should be written (default: uniprot).",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="Optional taxon blacklist file, one taxon ID per line.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["binary", "support_fraction"],
        default="binary",
        help="How transferred GO terms are scored before GO up-propagation.",
    )
    parser.add_argument(
        "--alignment-type",
        type=int,
        default=1,
        choices=[0, 1],
        help="Foldseek alignment type: 0=local 3Di+AA, 1=TM-align re-score/rank.",
    )
    parser.add_argument(
        "--evalue-max",
        type=float,
        default=1e-5,
        help="Max E-value for an accepted Foldseek hit (default: 1e-5).",
    )
    parser.add_argument(
        "--min-qcov",
        type=float,
        default=0.70,
        help="Minimum query coverage for accepted hits (default: 0.70).",
    )
    parser.add_argument(
        "--min-tcov",
        type=float,
        default=0.70,
        help="Minimum target coverage for accepted hits (default: 0.70).",
    )
    parser.add_argument(
        "--min-avg-tm",
        type=float,
        default=0.50,
        help="Minimum average TM-score if TM-scores are available.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=0,
        help=(
            "Foldseek --max-seqs retrieval limit. Use 0 to request every "
            "available prefilter hit."
        ),
    )
    parser.add_argument(
        "--prostt5-model",
        default=None,
        help="Enable FASTA-to-3Di mode. Pass the ProstT5 .gguf model or weights directory.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Foldseek GPU switch: 1 enables CUDA, 0 disables it.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="CUDA device selector for Foldseek/ProstT5, for example '0' or '0,1'.",
    )
    parser.add_argument(
        "--colabfold-msa-mode",
        choices=[
            "mmseqs2_uniref_env",
            "mmseqs2_uniref_env_envpair",
            "mmseqs2_uniref",
            "single_sequence",
        ],
        default=None,
        help="Optional ColabFold MSA mode override.",
    )
    parser.add_argument(
        "--colabfold-num-models",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Optional ColabFold --num-models override.",
    )
    parser.add_argument(
        "--colabfold-num-recycle",
        type=int,
        default=None,
        help="Optional ColabFold --num-recycle override.",
    )
    parser.add_argument(
        "--colabfold-use-gpu-relax",
        action="store_true",
        help="Pass --use-gpu-relax to ColabFold.",
    )
    parser.add_argument(
        "--colabfold-debug-logging",
        action="store_true",
        help="Pass --debug-logging to ColabFold for more verbose output.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Output directory.",
    )
    args = parser.parse_args()
    
    # Validate and resolve arguments
    args.prostt5_model = resolve_prostt5_model_path(args.prostt5_model)
    args.precomputed_tsv = (
        Path(args.precomputed_tsv).expanduser() if args.precomputed_tsv else None
    )
    if args.precomputed_tsv is not None and not args.precomputed_tsv.exists():
        sys.exit(f"Precomputed Foldseek TSV not found: {args.precomputed_tsv}")
    if args.target_db not in (None, ""):
        args.target_db = resolve_foldseek_target_db(args.target_db)
    elif args.precomputed_tsv is None:
        sys.exit("--target-db is required unless --precomputed-tsv is provided.")
    try:
        args.max_seqs = resolve_max_seqs(args.max_seqs)
    except ValueError as exc:
        sys.exit(str(exc))
    if args.target_db not in (None, ""):
        effective_foldseek_gpu = args.gpu or (
            "1" if args.cuda_visible_devices else None
        )
        args.target_db = resolve_foldseek_gpu_target_db(
            args.target_db,
            effective_foldseek_gpu,
        )

    import pandas as pd
    from GOTool import GeneOntology

    args.outdir.mkdir(parents=True, exist_ok=True)

    log_info("Building Gene Ontology structure...")
    go = GeneOntology.GeneOntology(args.go_obo, verbose=True)
    go.build_structure()

    blacklist = read_blacklist(args.blacklist)
    if blacklist:
        log_info(f"Loaded {len(blacklist)} blacklist taxon identifiers.")
    if args.structure_mode == "colabfold":
        log_info(
            "ColabFold options: "
            f"msa_mode={args.colabfold_msa_mode or 'default'}, "
            f"num_models={args.colabfold_num_models or 'default'}, "
            f"num_recycle={args.colabfold_num_recycle or 'default'}, "
            f"use_gpu_relax={args.colabfold_use_gpu_relax}, "
            f"debug_logging={args.colabfold_debug_logging}"
        )
    log_info(
        "Foldseek hit limits: "
        f"max_seqs={args.max_seqs}"
    )
    if args.precomputed_tsv is not None:
        log_info(f"Using precomputed Foldseek TSV: {args.precomputed_tsv}")

    query_structures: List[Tuple[Path, Optional[FastaSeq]]] = []
    fallback_sequences: List[FastaSeq] = []
    query_id_map: Dict[str, str] = {}
    input_query_count = 0

    log_info("Preparing query inputs...")
    if args.query_structures:
        log_info(
            f"Received {len(args.query_structures)} query structure file(s) as input."
        )
        for query in args.query_structures:
            query_path = Path(query)
            if not query_path.exists():
                sys.exit(f"Query structure not found: {query_path}")
            if args.precomputed_tsv is None:
                query_structures.append((query_path, None))
            query_id_map[query_path.stem] = protein_id_from_header(
                query_path.stem, args.protein_id_mode
            )
            input_query_count += 1
            log_info(f"Using provided structure: {query_path}")
    else:
        log_info(
            f"Received {len(args.fastas)} FASTA file(s); "
            f"structure mode is '{args.structure_mode}'."
        )
        for fasta in args.fastas:
            fasta_path = Path(fasta)
            if not fasta_path.exists():
                sys.exit(f"FASTA not found: {fasta_path}")
            log_info(f"Inspecting FASTA input: {fasta_path}")
            if args.precomputed_tsv is not None:
                records = read_fasta(fasta_path, args.protein_id_mode)
                input_query_count += len(records)
                log_info(
                    f"Using precomputed Foldseek TSV, so only FASTA identifiers are loaded "
                    f"from {fasta_path.name}: {len(records)} protein(s)."
                )
                for record in records:
                    query_id_map[record.name] = record.protein_id
                    first_token = record.header.split()[0]
                    query_id_map.setdefault(first_token, record.protein_id)
                continue
            try:
                structure_pairs, missing = ensure_structure_for_fasta(
                    fasta_path,
                    args.structure_mode,
                    args.structures_dir,
                    args.protein_id_mode,
                    search_recursive=args.search_recursive,
                    colabfold_msa_mode=args.colabfold_msa_mode,
                    colabfold_num_models=args.colabfold_num_models,
                    colabfold_num_recycle=args.colabfold_num_recycle,
                    colabfold_use_gpu_relax=args.colabfold_use_gpu_relax,
                    colabfold_debug_logging=args.colabfold_debug_logging,
                )
            except Exception as exc:
                sys.exit(str(exc))

            for structure_path, record in structure_pairs:
                query_structures.append((structure_path, record))
                query_id_map[record.name] = record.protein_id
                query_id_map[structure_path.stem] = record.protein_id
                input_query_count += 1
                log_info(
                    f"Protein {record.name} is ready for Foldseek using structure {structure_path.name}."
                )

            for record in missing:
                fallback_sequences.append(record)
                query_id_map[record.name] = record.protein_id
                input_query_count += 1
                log_info(
                    f"Protein {record.name} requires ProstT5/Foldseek fallback because no structure was found."
                )

    if args.precomputed_tsv is None and fallback_sequences and not args.prostt5_model:
        preview = ", ".join(record.name for record in fallback_sequences[:5])
        if len(fallback_sequences) > 5:
            preview += ", ..."
        sys.exit(
            "Missing structures detected but --prostt5-model not provided. "
            f"Examples: {preview}"
        )

    if args.precomputed_tsv is None and not query_structures and not fallback_sequences:
        sys.exit("No queries available after inspecting inputs.")
    if not query_id_map:
        sys.exit("No query identifiers could be resolved from the provided inputs.")

    raw_hit_counts: Dict[str, int] = defaultdict(int)
    accepted_hit_counts: Dict[str, int] = defaultdict(int)
    wanted_accessions: Set[str] = set()
    processed_query_ids: Set[str] = set()
    total_query_count = input_query_count

    if args.precomputed_tsv is not None:
        log_info(
            f"Starting processing for {total_query_count} protein(s) "
            f"from precomputed Foldseek TSV {args.precomputed_tsv}."
        )
    else:
        log_info(
            f"Starting processing for {total_query_count} protein(s): "
            f"{len(query_structures)} with structures and "
            f"{len(fallback_sequences)} requiring ProstT5 fallback."
        )
    if fallback_sequences:
        log_info(
            "Proteins queued for ProstT5 fallback: "
            f"{format_name_preview([record.name for record in fallback_sequences])}"
        )

    log_info("Running Foldseek...")
    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="s2f_foldseek_") as temp_dir:
        tmpdir = Path(temp_dir)
        accepted_hits_path = tmpdir / "accepted_hits.tsv"
        jobs: List[Tuple[Path, Optional[str], str, List[str]]] = []

        if args.precomputed_tsv is None:
            for structure_path, record in query_structures:
                label = record.name if record is not None else structure_path.stem
                jobs.append((structure_path, None, "structure", [label]))
                if record is not None:
                    query_id_map.setdefault(structure_path.stem, record.protein_id)

            if fallback_sequences:
                fallback_fasta = tmpdir / "queries_prostt5.fasta"
                with open(fallback_fasta, "w", encoding="utf-8") as handler:
                    for record in fallback_sequences:
                        handler.write(f">{record.name}\n{wrap_seq(record.seq)}\n")
                jobs.append(
                    (
                        fallback_fasta,
                        args.prostt5_model,
                        "prostt5",
                        [record.name for record in fallback_sequences],
                    )
                )
                log_info(
                    f"Prepared ProstT5 fallback FASTA with {len(fallback_sequences)} sequence(s)."
                )

        if args.precomputed_tsv is not None:
            try:
                (
                    job_raw_counts,
                    job_accepted_counts,
                    _job_tm_availability,
                    job_wanted_accessions,
                ) = stream_filter_hits_from_tsv(
                    args.precomputed_tsv,
                    accepted_hits_path,
                    evalue_max=args.evalue_max,
                    min_qcov=args.min_qcov,
                    min_tcov=args.min_tcov,
                    min_avg_tm=args.min_avg_tm,
                )
            except Exception as exc:
                sys.exit(str(exc))
            for query_id, raw_count in job_raw_counts.items():
                protein_id = resolve_query_protein_id(
                    query_id, query_id_map, args.protein_id_mode
                )
                processed_query_ids.add(query_id)
                raw_hit_counts[query_id] += raw_count
                accepted_hit_counts[query_id] += job_accepted_counts.get(query_id, 0)
                log_info(
                    f"Foldseek returned {raw_count} raw hit(s) for protein "
                    f"{protein_id} (query identifier: {query_id})."
                )
                log_info(
                    f"Protein {protein_id}: accepted "
                    f"{job_accepted_counts.get(query_id, 0)} of {raw_count} raw hit(s) "
                    "after thresholding."
                )
            wanted_accessions.update(job_wanted_accessions)

        for query_path, prostt5_model, job_mode, job_labels in jobs:
            if job_mode == "structure":
                log_info(
                    f"Foldseek search ({job_mode}) for protein {job_labels[0]} "
                    f"using input {query_path.name}."
                )
            else:
                log_info(
                    f"Foldseek search ({job_mode}) for {len(job_labels)} protein(s): "
                    f"{format_name_preview(job_labels)}."
                )
            try:
                out_tsv = run_foldseek_easy_search(
                    query_path,
                    args.target_db,
                    tmpdir,
                    alignment_type=args.alignment_type,
                    max_seqs=args.max_seqs,
                    prostt5_model=prostt5_model,
                    gpu=args.gpu,
                    cuda_visible_devices=args.cuda_visible_devices,
                    log_path=args.outdir / f"{sanitize_name(query_path.stem)}.foldseek.log",
                )
            except Exception as exc:
                sys.stderr.write(f"[WARN] Foldseek failed for {query_path.name}: {exc}\n")
                continue

            try:
                (
                    job_raw_counts,
                    job_accepted_counts,
                    _job_tm_availability,
                    job_wanted_accessions,
                ) = stream_filter_hits_from_tsv(
                    out_tsv,
                    accepted_hits_path,
                    evalue_max=args.evalue_max,
                    min_qcov=args.min_qcov,
                    min_tcov=args.min_tcov,
                    min_avg_tm=args.min_avg_tm,
                )
            except Exception as exc:
                sys.stderr.write(
                    f"[WARN] Foldseek TSV parsing failed for {query_path.name}: {exc}\n"
                )
                continue

            if not job_raw_counts:
                log_info(f"No Foldseek hits returned for input {query_path.name}.")
            for query_id, raw_count in job_raw_counts.items():
                protein_id = resolve_query_protein_id(
                    query_id, query_id_map, args.protein_id_mode
                )
                processed_query_ids.add(query_id)
                raw_hit_counts[query_id] += raw_count
                accepted_hit_counts[query_id] += job_accepted_counts.get(query_id, 0)
                log_info(
                    f"Foldseek returned {raw_count} raw hit(s) for protein "
                    f"{protein_id} (query identifier: {query_id})."
                )
                log_info(
                    f"Protein {protein_id}: accepted "
                    f"{job_accepted_counts.get(query_id, 0)} of {raw_count} raw hit(s) "
                    "after thresholding."
                )
            wanted_accessions.update(job_wanted_accessions)

        end_time = time.time()
        log_info(f"Foldseek finished in {end_time - start_time:.1f} seconds.")
        log_info(
            f"{len(processed_query_ids)} unique query identifier(s) returned hits."
        )

        log_info(
            f"Loading GO annotations for {len(wanted_accessions)} target accession(s)..."
        )
        accession_to_terms = stream_goa_terms_for_wanted(
            args.goa, wanted_accessions, go, blacklist=blacklist
        )
        log_info(
            f"Loaded GO terms for {len(accession_to_terms)} accession(s) from the GOA file."
        )

        aggregate_rows = process_accepted_hits_stream(
            accepted_hits_path,
            accession_to_terms=accession_to_terms,
            query_id_map=query_id_map,
            protein_id_mode=args.protein_id_mode,
            score_mode=args.score_mode,
            outdir=args.outdir,
            parameters={
                "evalue_max": args.evalue_max,
                "min_qcov": args.min_qcov,
                "min_tcov": args.min_tcov,
                "min_avg_tm": args.min_avg_tm,
                "max_seqs": args.max_seqs,
                "alignment_type": args.alignment_type,
                "score_mode": args.score_mode,
                "protein_id_mode": args.protein_id_mode,
                "prostt5_model": bool(args.prostt5_model),
                "precomputed_tsv": str(args.precomputed_tsv)
                if args.precomputed_tsv is not None
                else None,
            },
        )

        if aggregate_rows:
            assignments = pd.DataFrame(
                aggregate_rows, columns=["Protein", "GO ID", "Score"]
            )
            assignments = up_propagate_assignments(go, assignments)
            assignments = (
                assignments.groupby(["Protein", "GO ID"], as_index=False)["Score"].max()
            )
        else:
            assignments = pd.DataFrame(columns=["Protein", "GO ID", "Score"])

        assignments_path = args.outdir / "assignments.tsv"
        assignments.to_csv(assignments_path, sep="\t", index=False)

        total_structure_queries = len(query_structures)
        total_prostt5_queries = len(fallback_sequences)
        total_attempted = total_query_count
        retained_with_hits = sum(
            1 for count in accepted_hit_counts.values() if count > 0
        )

        if args.precomputed_tsv is not None:
            log_info(
                f"Attempted {total_attempted} query sequence(s) from precomputed Foldseek TSV."
            )
        else:
            log_info(
                f"Attempted {total_attempted} query sequence(s) "
                f"({total_structure_queries} structure, {total_prostt5_queries} ProstT5 fallback)."
            )
        log_info(
            f"Accepted hits for {retained_with_hits} query identifier(s)."
        )
        log_info(
            "Per-query files: <query>.hits.tsv and <query>.summary.json; aggregate: assignments.tsv"
        )


if __name__ == "__main__":
    main()
