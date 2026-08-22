#!/usr/bin/env python3
"""Strict direct-EC reaction-compartment analysis for T. cruzi Dm28c.

Only complete EC numbers explicitly present in an archived UniProt protein
record are accepted.  The archived sequence must match exactly one protein in
the S2F target FASTA.  InterPro, GO-to-EC mappings, names, and ortholog EC
transfer are deliberately outside this module.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import kegg_compartments as common


UNISAVE_URL = "https://rest.uniprot.org/unisave/{accession}"
STRICT_SCHEMA_VERSION = 1
FASTA_ACCESSION = re.compile(r"^(?:sp|tr)\|([^|]+)\|")
ENTRY_VERSION = re.compile(r"entry version (\d+)", re.IGNORECASE)
COMPLETE_EC = re.compile(r"EC=(\d+\.\d+\.\d+\.\d+)")
ANY_EC = re.compile(r"EC=([^;\s]+)")
AMINO_ACIDS = set("ABCDEFGHIKLMNPQRSTVWXYZUO*")


@dataclass(frozen=True)
class ArchivedRecord:
    accession: str
    reviewed: bool
    entry_name: str
    entry_version: str
    last_updated: str
    protein_name: str
    gene_ids: Tuple[str, ...]
    complete_ecs: Tuple[str, ...]
    rejected_ecs: Tuple[str, ...]
    location: str
    sequence: str
    source_path: Path


@dataclass(frozen=True)
class DirectEcAssignment:
    protein_id: str
    gene_id: str
    accession: str
    reviewed: bool
    entry_name: str
    entry_version: str
    last_updated: str
    protein_name: str
    ec_number: str
    raw_location: str
    source_path: str
    provenance_url: str


class RequestPacer:
    """Small process-local delay to avoid bursting the public UniSave API."""

    def __init__(self, minimum_interval: float = 0.12) -> None:
        self.minimum_interval = minimum_interval
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.minimum_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_sequence(sequence: str) -> str:
    """Normalise FASTA formatting without changing residue identities."""
    return "".join(sequence.split()).upper().rstrip("*")


def parse_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current = ""
    chunks: List[str] = []

    def flush() -> None:
        if not current:
            return
        sequence = normalise_sequence("".join(chunks))
        if current in records:
            raise ValueError(f"Duplicate FASTA identifier {current} in {path}")
        records[current] = sequence

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current = line[1:].split()[0]
                chunks = []
            elif not current:
                raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
            else:
                chunks.append(line)
    flush()
    return records


def accession_inventory(found_accessions: Path, missing_fasta: Path) -> List[str]:
    accessions = {
        line.strip() for line in found_accessions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    with missing_fasta.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith(">"):
                continue
            token = raw[1:].split()[0]
            match = FASTA_ACCESSION.match(token)
            accessions.add(match.group(1) if match else token)
    invalid = sorted(value for value in accessions if not re.fullmatch(r"[A-Z0-9]+", value))
    if invalid:
        raise ValueError(f"Invalid UniProt accessions: {', '.join(invalid[:5])}")
    return sorted(accessions)


def archive_record_path(archive_dir: Path, accession: str) -> Path:
    return archive_dir / "records" / accession[:3] / f"{accession}.txt"


def read_first_unisave_entry(response) -> bytes:
    chunks: List[bytes] = []
    for raw in response:
        chunks.append(raw)
        if raw.strip() == b"//":
            break
    content = b"".join(chunks)
    if not content.rstrip().endswith(b"//"):
        raise ValueError("UniSave response did not contain a complete first entry")
    return content


def latest_unisave_version(content: bytes) -> str:
    lines = content.decode("utf-8").splitlines()
    if len(lines) < 2 or not lines[0].startswith("Entry version\t"):
        raise ValueError("UniSave TSV response has no version table")
    version = lines[1].split("\t", 1)[0].strip()
    if not version.isdigit():
        raise ValueError("UniSave TSV response has an invalid latest version")
    return version


def fetch_one_unisave(
    accession: str,
    archive_dir: Path,
    force: bool,
    pacer: RequestPacer,
    retries: int = 4,
) -> Tuple[str, str]:
    destination = archive_record_path(archive_dir, accession)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return accession, "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_url = UNISAVE_URL.format(accession=accession)
    last_error: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            pacer.wait()
            version_request = urllib.request.Request(
                f"{base_url}?format=tsv",
                headers={"User-Agent": "S2F-strict-direct-EC/1.0"},
            )
            with urllib.request.urlopen(version_request, timeout=90) as response:
                entry_version = latest_unisave_version(response.read())
            pacer.wait()
            record_request = urllib.request.Request(
                f"{base_url}?format=txt&versions={entry_version}",
                headers={"User-Agent": "S2F-strict-direct-EC/1.0"},
            )
            with urllib.request.urlopen(record_request, timeout=90) as response:
                content = read_first_unisave_entry(response)
            with tempfile.NamedTemporaryFile(
                "wb", delete=False, dir=str(destination.parent), suffix=".partial"
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_path.replace(destination)
            return accession, "downloaded"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(min(2 ** attempt, 8))
    return accession, f"failed:{type(last_error).__name__}:{last_error}"


def fetch_unisave_archive(
    found_accessions: Path,
    missing_fasta: Path,
    archive_dir: Path,
    workers: int = 4,
    force: bool = False,
) -> Dict[str, object]:
    accessions = accession_inventory(found_accessions, missing_fasta)
    archive_dir.mkdir(parents=True, exist_ok=True)
    pacer = RequestPacer()
    statuses: Counter[str] = Counter()
    failures: Dict[str, str] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_one_unisave, accession, archive_dir, force, pacer): accession
            for accession in accessions
        }
        for index, future in enumerate(as_completed(futures), 1):
            accession, status = future.result()
            category = status.split(":", 1)[0]
            statuses[category] += 1
            if category == "failed":
                failures[accession] = status
            if index % 50 == 0 or index == len(accessions):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                remaining = (len(accessions) - index) / rate if rate else 0.0
                print(
                    f"UniSave {index}/{len(accessions)} "
                    f"downloaded={statuses['downloaded']} reused={statuses['reused']} "
                    f"failed={statuses['failed']} rate={rate:.2f}/s "
                    f"eta_minutes={remaining / 60:.1f}",
                    flush=True,
                )
    metadata = {
        "schema_version": STRICT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "found_accessions": str(found_accessions.resolve()),
        "found_accessions_sha256": common.sha256_file(found_accessions),
        "missing_fasta": str(missing_fasta.resolve()),
        "missing_fasta_sha256": common.sha256_file(missing_fasta),
        "expected_accessions": len(accessions),
        "status_counts": dict(sorted(statuses.items())),
        "complete": not failures,
        "failures": failures,
        "endpoint_template": UNISAVE_URL,
        "retrieval": "latest entry version from TSV metadata, then exact version in TXT format",
    }
    (archive_dir / "unisave_fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_unisave_record(path: Path) -> ArchivedRecord:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("ID   "):
        raise ValueError(f"{path}: not a UniProt flat-file record")
    id_parts = lines[0].split()
    entry_name = id_parts[1]
    reviewed = "Reviewed;" in lines[0]
    accession = ""
    gene_ids: Set[str] = set()
    entry_version = ""
    last_updated = ""
    protein_name = ""
    location_parts: List[str] = []
    sequence_parts: List[str] = []
    in_location = False
    in_sequence = False
    for line in lines:
        if line.startswith("AC   ") and not accession:
            accession = line[5:].split(";", 1)[0].strip()
        if line.startswith("DT   "):
            match = ENTRY_VERSION.search(line)
            if match:
                entry_version = match.group(1)
                last_updated = line[5:].split(",", 1)[0].strip()
        if line.startswith("GN   "):
            gene_ids.update(value.upper() for value in common.TCDM_PATTERN.findall(line))
        if line.startswith("DE   ") and not protein_name and "Full=" in line:
            protein_name = line.split("Full=", 1)[1].split(";", 1)[0].strip()
        if line.startswith("CC   -!- SUBCELLULAR LOCATION:"):
            in_location = True
            location_parts.append(line.split("SUBCELLULAR LOCATION:", 1)[1].strip())
            continue
        if in_location:
            if line.startswith("CC       "):
                location_parts.append(line[9:].strip())
                continue
            in_location = False
        if line.startswith("SQ   SEQUENCE"):
            in_sequence = True
            continue
        if in_sequence:
            if line.strip() == "//":
                in_sequence = False
            else:
                sequence_parts.append("".join(char for char in line.upper() if char in AMINO_ACIDS))
    text = "\n".join(lines)
    complete_ecs = sorted(set(COMPLETE_EC.findall(text)))
    all_ecs = set(ANY_EC.findall(text))
    rejected_ecs = sorted(all_ecs - set(complete_ecs))
    if not accession:
        raise ValueError(f"{path}: missing accession")
    return ArchivedRecord(
        accession=accession,
        reviewed=reviewed,
        entry_name=entry_name,
        entry_version=entry_version,
        last_updated=last_updated,
        protein_name=protein_name,
        gene_ids=tuple(sorted(gene_ids)),
        complete_ecs=tuple(complete_ecs),
        rejected_ecs=tuple(rejected_ecs),
        location=" ".join(location_parts),
        sequence=normalise_sequence("".join(sequence_parts)),
        source_path=path,
    )


def exact_target_match(
    record: ArchivedRecord,
    target_by_sequence: Mapping[str, List[str]],
) -> Tuple[Optional[str], str]:
    candidates = target_by_sequence.get(record.sequence, [])
    if not candidates:
        return None, "no_exact_sequence_match"
    if len(candidates) == 1:
        return candidates[0], "exact_unique_sequence"
    gene_candidates = [
        protein_id for protein_id in candidates
        if common.canonical_gene(protein_id) in set(record.gene_ids)
    ]
    if len(gene_candidates) == 1:
        return gene_candidates[0], "exact_sequence_gene_disambiguated"
    return None, "ambiguous_exact_sequence_match"


def load_direct_ec_assignments(
    archive_dir: Path,
    target_fasta: Path,
) -> Tuple[List[DirectEcAssignment], List[Dict[str, object]], Dict[str, ArchivedRecord]]:
    target_records = parse_fasta(target_fasta)
    target_by_sequence: MutableMapping[str, List[str]] = defaultdict(list)
    for protein_id, sequence in target_records.items():
        target_by_sequence[sequence].append(protein_id)
    assignments: List[DirectEcAssignment] = []
    excluded: List[Dict[str, object]] = []
    accepted_record_by_protein: Dict[str, ArchivedRecord] = {}
    for path in sorted((archive_dir / "records").rglob("*.txt")):
        try:
            record = parse_unisave_record(path)
        except ValueError as error:
            excluded.append({
                "accession": path.stem, "stage": "parse_archive",
                "reason": str(error), "source_path": str(path),
            })
            continue
        protein_id, match_status = exact_target_match(record, target_by_sequence)
        if protein_id is None:
            excluded.append({
                "accession": record.accession, "stage": "sequence_match",
                "reason": match_status, "source_path": str(path),
            })
            continue
        if record.rejected_ecs:
            excluded.append({
                "accession": record.accession, "stage": "ec_validation",
                "reason": "incomplete_ec:" + ";".join(record.rejected_ecs),
                "source_path": str(path),
            })
        if not record.complete_ecs:
            excluded.append({
                "accession": record.accession, "stage": "protein_to_ec",
                "reason": "no_explicit_complete_ec", "source_path": str(path),
            })
            continue
        accepted_record_by_protein[protein_id] = record
        for ec in record.complete_ecs:
            assignments.append(DirectEcAssignment(
                protein_id=protein_id,
                gene_id=common.canonical_gene(protein_id),
                accession=record.accession,
                reviewed=record.reviewed,
                entry_name=record.entry_name,
                entry_version=record.entry_version,
                last_updated=record.last_updated,
                protein_name=record.protein_name,
                ec_number=ec,
                raw_location=record.location,
                source_path=str(path.resolve()),
                provenance_url=UNISAVE_URL.format(accession=record.accession),
            ))
    assignments.sort(key=lambda row: (row.protein_id, row.ec_number, row.accession))
    excluded.sort(key=lambda row: (str(row["accession"]), str(row["stage"]), str(row["reason"])))
    return assignments, excluded, accepted_record_by_protein


def _split_set(value: object) -> Set[str]:
    return {item for item in str(value or "").split(";") if item}


def build_strict(args) -> Dict[str, object]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_dir = args.archive_dir.resolve()
    metadata_path = archive_dir / "unisave_fetch_metadata.json"
    allow_incomplete = getattr(args, "allow_incomplete_archive", False)
    if not metadata_path.exists() and not allow_incomplete:
        raise ValueError(
            "UniSave archive metadata is missing; finish fetch-direct-ec before building"
        )
    if metadata_path.exists():
        fetch_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not fetch_metadata.get("complete", False) and not allow_incomplete:
            raise ValueError(
                "UniSave archive is incomplete; resume fetch-direct-ec before building"
            )

    labels, allowed_anchors, excluded_anchors = common.load_slim(args.slim)
    parents = common.ontology_parents(args.obo)
    assignments, excluded, _accepted_records = load_direct_ec_assignments(
        archive_dir, args.target_fasta
    )
    if not assignments:
        raise ValueError("No exact-sequence direct EC assignments were found")

    snapshot = args.snapshot_dir.resolve()
    required = [
        "uniprot_tcr_reference.tsv", "kegg_enzyme_reaction.tsv",
        "kegg_tcr_genes_by_enzyme.tsv", "kegg_tcr_uniprot.tsv",
        "kegg_reaction_names.tsv",
    ]
    missing = [filename for filename in required if not (snapshot / filename).exists()]
    if missing:
        raise FileNotFoundError("Missing KEGG/UniProt snapshots: " + ", ".join(missing))

    tcr_records = common.parse_uniprot(snapshot / "uniprot_tcr_reference.tsv")
    tcr_by_accession = {record.accession: record for record in tcr_records}
    tcr_by_kegg: MutableMapping[str, List[common.UniProtRecord]] = defaultdict(list)
    for record in tcr_records:
        for kegg_id in record.kegg_ids:
            tcr_by_kegg[kegg_id.removeprefix("tcr:")].append(record)

    ec_reactions: MutableMapping[str, Set[str]] = defaultdict(set)
    for ec, reaction in common.parse_pair_file(
        snapshot / "kegg_enzyme_reaction.tsv", "ec:", "rn:"
    ):
        if common.EC_PATTERN.match(ec):
            ec_reactions[ec].add(reaction)
    ec_tcr_genes: MutableMapping[str, Set[str]] = defaultdict(set)
    for ec, gene in common.parse_pair_file(
        snapshot / "kegg_tcr_genes_by_enzyme.tsv", "ec:", "tcr:"
    ):
        ec_tcr_genes[ec].add(gene)
    kegg_to_uniprot: MutableMapping[str, Set[str]] = defaultdict(set)
    for gene, accession in common.parse_pair_file(
        snapshot / "kegg_tcr_uniprot.tsv", "tcr:", "up:"
    ):
        kegg_to_uniprot[gene].add(accession)
    names = common.reaction_name_map(snapshot / "kegg_reaction_names.tsv")

    s2f_rows = common.read_tsv(args.protein_compartments)
    s2f_by_protein: MutableMapping[str, List[Dict[str, str]]] = defaultdict(list)
    excluded_s2f: List[Dict[str, str]] = []
    for row in s2f_rows:
        go_id = row.get("compartment_id", "")
        if go_id in excluded_anchors:
            excluded_s2f.append(row)
        elif go_id in allowed_anchors:
            s2f_by_protein[row["protein_id"]].append(row)

    known_ec_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    evidence_rows: List[Dict[str, object]] = []
    unresolved_rows: List[Dict[str, object]] = []

    for assignment in assignments:
        reactions = sorted(ec_reactions.get(assignment.ec_number, set()))
        known_ec_rows.append({
            "protein_id": assignment.protein_id,
            "gene_id": assignment.gene_id,
            "uniprot_accession": assignment.accession,
            "uniprot_reviewed": str(assignment.reviewed).lower(),
            "entry_name": assignment.entry_name,
            "entry_version": assignment.entry_version,
            "last_updated": assignment.last_updated,
            "protein_name": assignment.protein_name,
            "ec_number": assignment.ec_number,
            "direct_ec_source": "uniprot_unisave_explicit_ec",
            "sequence_match": "exact",
            "n_kegg_reactions": len(reactions),
            "source_path": assignment.source_path,
            "provenance_url": assignment.provenance_url,
        })
        if not reactions:
            unresolved_rows.append({
                "protein_id": assignment.protein_id,
                "ec_number": assignment.ec_number,
                "reaction_id": "",
                "stage": "ec_to_reaction",
                "reason": "no_kegg_reaction_for_direct_ec",
            })
            continue

        exact_assertions = common.location_assertions(
            assignment.raw_location, labels, allowed_anchors
        )
        for reaction_id in reactions:
            candidates: List[Tuple[str, str, str, bool, str, str, str]] = []
            if exact_assertions:
                for assertion in exact_assertions:
                    candidates.append((
                        "dm28c_archived_exact_protein", assignment.gene_id,
                        assignment.accession, assignment.reviewed,
                        assertion.compartment_id, assertion.evidence_level,
                        assertion.raw_location,
                    ))
            else:
                for tcr_gene in sorted(ec_tcr_genes.get(assignment.ec_number, set())):
                    accessions = sorted(kegg_to_uniprot.get(tcr_gene, set()))
                    records = [tcr_by_accession[value] for value in accessions if value in tcr_by_accession]
                    if not records:
                        records = tcr_by_kegg.get(tcr_gene, [])
                    for record in common.choose_records(records):
                        for assertion in common.location_assertions(
                            record.location, labels, allowed_anchors
                        ):
                            candidates.append((
                                "tcr_cl_brener_ec_transfer", tcr_gene,
                                record.accession, record.reviewed,
                                assertion.compartment_id,
                                assertion.evidence_level,
                                assertion.raw_location,
                            ))

            reaction_locations = {row[4] for row in candidates}
            if not reaction_locations:
                unresolved_rows.append({
                    "protein_id": assignment.protein_id,
                    "ec_number": assignment.ec_number,
                    "reaction_id": reaction_id,
                    "stage": "reaction_to_compartment",
                    "reason": "no_usable_functional_location",
                })
            for source, reference_gene, reference_accession, reviewed, go_id, evidence, raw in candidates:
                evidence_rows.append({
                    "protein_id": assignment.protein_id,
                    "gene_id": assignment.gene_id,
                    "direct_ec_accession": assignment.accession,
                    "ec_number": assignment.ec_number,
                    "reaction_id": reaction_id,
                    "reaction_name": names.get(reaction_id, ""),
                    "compartment_id": go_id,
                    "compartment_label": labels.get(go_id, go_id),
                    "location_source": source,
                    "reference_gene": reference_gene,
                    "reference_uniprot_accession": reference_accession,
                    "reference_uniprot_reviewed": str(reviewed).lower(),
                    "location_evidence": evidence,
                    "raw_functional_location": raw,
                    "strain_caveat": "" if source.startswith("dm28c_") else "CL Brener location transferred to Dm28c by shared EC; isozyme specificity unresolved",
                })

            s2f_locations = {
                row["compartment_id"] for row in s2f_by_protein.get(assignment.protein_id, [])
            }
            status, compatible_pairs = common.compare_compartments(
                reaction_locations, s2f_locations, parents
            )
            comparison_rows.append({
                "protein_id": assignment.protein_id,
                "gene_id": assignment.gene_id,
                "uniprot_accession": assignment.accession,
                "ec_number": assignment.ec_number,
                "direct_ec_source": "uniprot_unisave_explicit_ec",
                "reaction_id": reaction_id,
                "reaction_name": names.get(reaction_id, ""),
                "reaction_compartment_ids": ";".join(sorted(reaction_locations)),
                "reaction_compartment_labels": ";".join(
                    labels.get(go_id, go_id) for go_id in sorted(reaction_locations)
                ),
                "reaction_location_sources": ";".join(sorted({row[0] for row in candidates})),
                "reaction_location_evidence": ";".join(sorted({row[5] for row in candidates})),
                "s2f_compartment_ids": ";".join(sorted(s2f_locations)),
                "s2f_compartment_labels": ";".join(
                    labels.get(go_id, go_id) for go_id in sorted(s2f_locations)
                ),
                "s2f_scores": ";".join(
                    f"{row['compartment_id']}:{row.get('s2f_score', '')}"
                    for row in sorted(
                        s2f_by_protein.get(assignment.protein_id, []),
                        key=lambda value: value["compartment_id"],
                    )
                ),
                "comparison_status": status,
                "compatible_pairs": compatible_pairs,
            })

    catalog_groups: MutableMapping[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in evidence_rows:
        catalog_groups[(str(row["reaction_id"]), str(row["compartment_id"]))].append(row)
    catalog_rows: List[Dict[str, object]] = []
    for (reaction_id, compartment_id), rows in sorted(catalog_groups.items()):
        catalog_rows.append({
            "reaction_id": reaction_id,
            "reaction_name": names.get(reaction_id, ""),
            "compartment_id": compartment_id,
            "compartment_label": labels.get(compartment_id, compartment_id),
            "ec_numbers": ";".join(sorted({str(row["ec_number"]) for row in rows})),
            "dm28c_protein_ids": ";".join(sorted({str(row["protein_id"]) for row in rows})),
            "dm28c_uniprot_accessions": ";".join(sorted({str(row["direct_ec_accession"]) for row in rows})),
            "location_sources": ";".join(sorted({str(row["location_source"]) for row in rows})),
            "reference_genes": ";".join(sorted({str(row["reference_gene"]) for row in rows})),
            "reference_uniprot_accessions": ";".join(sorted({str(row["reference_uniprot_accession"]) for row in rows})),
            "location_evidence": ";".join(sorted({str(row["location_evidence"]) for row in rows})),
            "n_supporting_proteins": len({str(row["protein_id"]) for row in rows}),
            "strain_caveat": ";".join(sorted({str(row["strain_caveat"]) for row in rows if row["strain_caveat"]})),
        })

    known_fields = [
        "protein_id", "gene_id", "uniprot_accession", "uniprot_reviewed",
        "entry_name", "entry_version", "last_updated", "protein_name",
        "ec_number", "direct_ec_source", "sequence_match", "n_kegg_reactions",
        "source_path", "provenance_url",
    ]
    comparison_fields = [
        "protein_id", "gene_id", "uniprot_accession", "ec_number",
        "direct_ec_source", "reaction_id", "reaction_name",
        "reaction_compartment_ids", "reaction_compartment_labels",
        "reaction_location_sources", "reaction_location_evidence",
        "s2f_compartment_ids", "s2f_compartment_labels", "s2f_scores",
        "comparison_status", "compatible_pairs",
    ]
    evidence_fields = [
        "protein_id", "gene_id", "direct_ec_accession", "ec_number",
        "reaction_id", "reaction_name", "compartment_id", "compartment_label",
        "location_source", "reference_gene", "reference_uniprot_accession",
        "reference_uniprot_reviewed", "location_evidence",
        "raw_functional_location", "strain_caveat",
    ]
    catalog_fields = [
        "reaction_id", "reaction_name", "compartment_id", "compartment_label",
        "ec_numbers", "dm28c_protein_ids", "dm28c_uniprot_accessions",
        "location_sources", "reference_genes", "reference_uniprot_accessions",
        "location_evidence", "n_supporting_proteins", "strain_caveat",
    ]
    excluded_fields = ["accession", "stage", "reason", "source_path"]
    unresolved_fields = ["protein_id", "ec_number", "reaction_id", "stage", "reason"]
    common.write_tsv(output / "known_protein_ec.tsv", known_fields, known_ec_rows)
    common.write_tsv(
        output / "protein_reaction_compartment_vs_s2f.tsv",
        comparison_fields, comparison_rows,
    )
    common.write_tsv(output / "reaction_compartment_evidence.tsv", evidence_fields, evidence_rows)
    common.write_tsv(output / "reaction_compartment_pairs.tsv", catalog_fields, catalog_rows)
    common.write_tsv(output / "excluded_direct_ec_records.tsv", excluded_fields, excluded)
    common.write_tsv(
        output / "unresolved_reaction_compartments.tsv",
        unresolved_fields, unresolved_rows,
    )
    common.write_tsv(
        output / "excluded_s2f_compartments.tsv",
        list(s2f_rows[0]) if s2f_rows else ["protein_id"],
        excluded_s2f,
    )

    status_counts = Counter(row["comparison_status"] for row in comparison_rows)
    evaluable_keys = {
        "exact_agreement", "compatible_agreement", "partial_agreement", "conflict"
    }
    evaluable = sum(status_counts[key] for key in evaluable_keys)
    strict_aligned = status_counts["exact_agreement"] + status_counts["compatible_agreement"]
    summary = {
        "schema_version": STRICT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "method": {
            "protein_ec": "explicit complete EC in archived UniProt record plus exact target-sequence match",
            "forbidden_ec_sources": [
                "InterPro", "GO-to-EC", "protein-name parsing", "ortholog EC transfer",
            ],
            "reaction_compartment": "exact archived Dm28c functional location, otherwise KEGG tcr gene to UniProt functional location",
            "raw_go_cc_used": False,
        },
        "inputs": {
            "archive_dir": str(archive_dir),
            "target_fasta": str(args.target_fasta.resolve()),
            "target_fasta_sha256": common.sha256_file(args.target_fasta),
            "protein_compartments": str(args.protein_compartments.resolve()),
            "protein_compartments_sha256": common.sha256_file(args.protein_compartments),
            "snapshot_dir": str(snapshot),
        },
        "counts": {
            "direct_ec_rows": len(known_ec_rows),
            "direct_ec_proteins": len({row["protein_id"] for row in known_ec_rows}),
            "comparison_rows": len(comparison_rows),
            "comparison_status": dict(sorted(status_counts.items())),
            "evaluable_comparisons": evaluable,
            "strict_aligned_comparisons": strict_aligned,
            "strict_alignment_fraction": strict_aligned / evaluable if evaluable else None,
            "reaction_compartment_evidence_rows": len(evidence_rows),
            "unique_reaction_compartment_pairs": len(catalog_rows),
            "unresolved_rows": len(unresolved_rows),
            "excluded_direct_ec_rows": len(excluded),
            "excluded_s2f_rows": len(excluded_s2f),
        },
        "caveats": [
            "An explicit historical TrEMBL EC is a database annotation, not necessarily experimental evidence.",
            "KEGG tcr is CL Brener, so fallback locations are cross-strain EC transfers.",
            "The S2F comparison uses the existing manual 0.1 threshold.",
            "Agreement is corroboration and not biological ground truth.",
        ],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Strict direct-EC reaction-compartment comparison",
        "",
        "Only EC numbers explicitly present in an archived UniProt protein record and attached by exact sequence match were accepted.",
        "",
        f"- Direct protein-EC rows: {len(known_ec_rows):,}",
        f"- Proteins with a direct EC: {summary['counts']['direct_ec_proteins']:,}",
        f"- Protein-reaction comparisons: {len(comparison_rows):,}",
        f"- Evaluable comparisons: {evaluable:,}",
        f"- Strict alignment fraction: {summary['counts']['strict_alignment_fraction'] if evaluable else 'not estimable'}",
        f"- Unique reaction-compartment pairs: {len(catalog_rows):,}",
        "",
        "## Comparison statuses",
        "",
    ]
    report.extend(f"- `{key}`: {value:,}" for key, value in sorted(status_counts.items()))
    report.extend([
        "",
        "The reaction-compartment catalog is independent of S2F and contains no S2F columns.",
        "InterPro, GO-to-EC mappings, protein names, and ortholog EC transfer were not used.",
        "",
    ])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary
