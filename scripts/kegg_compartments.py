#!/usr/bin/env python3
"""Build and compare reaction-compartment hypotheses for T. cruzi.

The bridge follows the biologically meaningful direction described in
``CBM_L02_compartmentsBridge.md``::

    reaction -> EC -> organism gene -> UniProt functional location

For Dm28c, protein-to-reaction links come from exact-gene UniProt EC
annotations.  Reaction compartments are inferred independently from KEGG's
T. cruzi reference organism (CL Brener) and UniProt functional-location
statements.  Exact Dm28c UniProt locations are retained as a clearly labelled
fallback.  S2F Cellular Component predictions are never used to create the
reaction hypothesis; they are compared with it afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


S2F_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLIM = S2F_ROOT / "conf" / "cc_compartments_tcruzi.json"
DEFAULT_OBO = S2F_ROOT / "go.obo"
DM28C_TAXON = 1416333
KEGG_TCR_TAXON = 353153
KEGG_ORGANISM = "tcr"
SCHEMA_VERSION = 1
EC_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
TCDM_PATTERN = re.compile(r"(TCDM_\d+)", re.IGNORECASE)
ECO_EXPERIMENTAL = {"ECO:0000269", "ECO:0007744"}
ECO_INFERRED = {"ECO:0000250", "ECO:0000255", "ECO:0000305"}
ECO_AUTOMATED = {"ECO:0000256"}


SNAPSHOT_URLS = {
    "uniprot_dm28c.tsv": (
        "https://rest.uniprot.org/uniprotkb/stream?"
        + urllib.parse.urlencode({
            "query": f"organism_id:{DM28C_TAXON}",
            "format": "tsv",
            "fields": ",".join([
                "accession", "id", "reviewed", "protein_name", "gene_primary",
                "gene_names", "organism_id", "ec", "xref_kegg",
                "cc_subcellular_location",
            ]),
        })
    ),
    "uniprot_tcr_reference.tsv": (
        "https://rest.uniprot.org/uniprotkb/stream?"
        + urllib.parse.urlencode({
            "query": f"organism_id:{KEGG_TCR_TAXON}",
            "format": "tsv",
            "fields": ",".join([
                "accession", "id", "reviewed", "protein_name", "gene_primary",
                "gene_names", "organism_id", "ec", "xref_kegg",
                "cc_subcellular_location",
            ]),
        })
    ),
    "kegg_enzyme_reaction.tsv": "https://rest.kegg.jp/link/reaction/enzyme",
    "kegg_tcr_genes_by_enzyme.tsv": f"https://rest.kegg.jp/link/{KEGG_ORGANISM}/enzyme",
    "kegg_tcr_uniprot.tsv": f"https://rest.kegg.jp/conv/uniprot/{KEGG_ORGANISM}",
    "kegg_reaction_names.tsv": "https://rest.kegg.jp/list/reaction",
}


# The curated UniProt field is free text.  These aliases map functional
# location statements to the same conservative GO slim used by S2F.  Terms
# scoped as cross-organism validation in the slim are excluded later.
LOCATION_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "GO:0106123": ("reservosome",),
    "GO:0097740": ("paraflagellar rod",),
    "GO:0020016": ("flagellar pocket", "ciliary pocket"),
    "GO:0020015": ("glycosome", "glycosomal"),
    "GO:0020022": ("acidocalcisome",),
    "GO:0020023": ("kinetoplast",),
    "GO:0005783": ("endoplasmic reticulum",),
    "GO:0005794": ("golgi apparatus", "golgi"),
    "GO:0005886": ("plasma membrane", "cell membrane"),
    "GO:0005929": ("flagellum", "flagellar", "cilium", "ciliary"),
    "GO:0005777": ("peroxisome", "peroxisomal"),
    "GO:0005739": ("mitochondrion", "mitochondrial"),
    "GO:0005829": ("cytosol", "cytosolic"),
    "GO:0005737": ("cytoplasm", "cytoplasmic"),
    "GO:0005730": ("nucleolus", "nucleolar"),
    "GO:0005634": ("nucleus", "nuclear"),
    "GO:0005764": ("lysosome", "lysosomal"),
    "GO:0005768": ("endosome", "endosomal"),
    "GO:0005773": ("vacuole", "vacuolar"),
    "GO:0005811": ("lipid droplet",),
    "GO:0005618": ("cell wall",),
    "GO:0005576": ("extracellular", "secreted"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def split_values(value: str) -> List[str]:
    return [item.strip() for item in re.split(r"[;,]", value or "") if item.strip()]


def normalise_reviewed(value: str) -> bool:
    return (value or "").strip().lower() in {"reviewed", "true", "yes", "1"}


def find_column(row: Mapping[str, str], *candidates: str) -> str:
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return row.get(key, "")
    return ""


@dataclass(frozen=True)
class UniProtRecord:
    accession: str
    reviewed: bool
    genes: Tuple[str, ...]
    ecs: Tuple[str, ...]
    kegg_ids: Tuple[str, ...]
    location: str
    protein_name: str


@dataclass(frozen=True)
class LocationAssertion:
    compartment_id: str
    compartment_label: str
    raw_location: str
    evidence_level: str
    eco_codes: str


def parse_uniprot(path: Path) -> List[UniProtRecord]:
    rows = read_tsv(path)
    records: List[UniProtRecord] = []
    for row in rows:
        accession = find_column(row, "Entry", "accession")
        if not accession:
            continue
        gene_text = " ".join([
            find_column(row, "Gene Names (primary)", "gene_primary"),
            find_column(row, "Gene Names", "gene_names"),
        ])
        genes = tuple(sorted(set(TCDM_PATTERN.findall(gene_text.upper())) | set(gene_text.split())))
        ec_text = find_column(row, "EC number", "ec")
        ecs = tuple(sorted({value for value in split_values(ec_text) if EC_PATTERN.match(value)}))
        kegg_text = find_column(row, "KEGG", "xref_kegg")
        kegg_ids = tuple(sorted({value.rstrip(";") for value in split_values(kegg_text)}))
        records.append(UniProtRecord(
            accession=accession,
            reviewed=normalise_reviewed(find_column(row, "Reviewed", "reviewed")),
            genes=genes,
            ecs=ecs,
            kegg_ids=kegg_ids,
            location=find_column(row, "Subcellular location [CC]", "cc_subcellular_location"),
            protein_name=find_column(row, "Protein names", "protein_name"),
        ))
    return records


def location_evidence(raw: str) -> Tuple[str, str]:
    codes = sorted(set(re.findall(r"ECO:\d{7}", raw or "")))
    if set(codes) & ECO_EXPERIMENTAL:
        level = "experimental"
    elif set(codes) & ECO_INFERRED:
        level = "inferred_manual"
    elif set(codes) & ECO_AUTOMATED:
        level = "automated_inferred"
    elif codes:
        level = "other_eco"
    else:
        level = "unspecified"
    return level, ";".join(codes)


def location_assertions(
    raw: str,
    anchor_labels: Mapping[str, str],
    allowed_anchors: Set[str],
) -> List[LocationAssertion]:
    if not raw:
        return []
    text = re.sub(r"\s+", " ", raw).strip()
    # Notes often mention locations unrelated to the primary functional
    # assertion.  Exclude Note= tails before alias matching.
    functional = re.split(r"\bNote=", text, maxsplit=1, flags=re.IGNORECASE)[0].lower()
    evidence, codes = location_evidence(text)
    found: List[LocationAssertion] = []
    for go_id, aliases in LOCATION_ALIASES.items():
        if go_id not in allowed_anchors:
            continue
        if any(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", functional) for alias in aliases):
            found.append(LocationAssertion(
                compartment_id=go_id,
                compartment_label=anchor_labels.get(go_id, go_id),
                raw_location=text,
                evidence_level=evidence,
                eco_codes=codes,
            ))
    return sorted(found, key=lambda item: item.compartment_id)


def load_slim(path: Path) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported slim schema in {path}")
    labels: Dict[str, str] = {}
    allowed: Set[str] = set()
    excluded: Set[str] = set()
    for anchor in config.get("anchors", []):
        go_id = str(anchor["go_id"])
        labels[go_id] = str(anchor.get("label", go_id))
        if anchor.get("scope") == "cross-organism validation":
            excluded.add(go_id)
        else:
            allowed.add(go_id)
    return labels, allowed, excluded


def parse_pair_file(path: Path, left_prefix: str, right_prefix: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            columns = line.split("\t")
            if len(columns) < 2:
                raise ValueError(f"{path}:{line_number}: expected two tab-separated columns")
            left = columns[0].removeprefix(left_prefix)
            right = columns[1].removeprefix(right_prefix)
            pairs.append((left, right))
    return pairs


def reaction_name_map(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            columns = raw.rstrip("\n").split("\t", 1)
            if len(columns) == 2:
                result[columns[0].removeprefix("rn:")] = columns[1]
    return result


def download(url: str, destination: Path, timeout: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "S2F-reaction-compartment-bridge/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(destination.parent)) as temp:
            shutil.copyfileobj(response, temp)
            temp_path = Path(temp.name)
    temp_path.replace(destination)


def fetch_snapshots(snapshot_dir: Path, force: bool = False) -> Dict[str, object]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, object]] = {}
    for filename, url in SNAPSHOT_URLS.items():
        destination = snapshot_dir / filename
        status = "reused"
        if force or not destination.exists() or destination.stat().st_size == 0:
            download(url, destination)
            status = "downloaded"
        files[filename] = {
            "url": url,
            "status": status,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dm28c_taxon": DM28C_TAXON,
        "kegg_reference_organism": KEGG_ORGANISM,
        "kegg_reference_taxon": KEGG_TCR_TAXON,
        "strain_warning": "KEGG tcr is T. cruzi CL Brener, not Dm28c.",
        "files": files,
    }
    (snapshot_dir / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def choose_records(records: Iterable[UniProtRecord]) -> List[UniProtRecord]:
    records = list(records)
    with_location = [record for record in records if record.location]
    reviewed = [record for record in with_location if record.reviewed]
    return reviewed if reviewed else with_location


def canonical_gene(protein_id: str) -> str:
    match = TCDM_PATTERN.search(protein_id)
    return match.group(1).upper() if match else protein_id


def ontology_parents(path: Path) -> Dict[str, Set[str]]:
    parents: Dict[str, Set[str]] = defaultdict(set)
    current = ""
    in_term = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                current = ""
                in_term = True
            elif line.startswith("["):
                in_term = False
            elif in_term and line.startswith("id: "):
                current = line[4:].strip()
            elif in_term and current and line.startswith("is_a: "):
                parents[current].add(line[6:].split(" ! ", 1)[0].strip())
            elif in_term and current and line.startswith("relationship: part_of "):
                parents[current].add(line.split()[2])
    return parents


def ontology_go_ec(path: Path) -> Dict[str, Set[str]]:
    """Return exact GO-to-EC cross-references from the frozen GO ontology."""
    result: Dict[str, Set[str]] = defaultdict(set)
    current = ""
    in_term = False
    obsolete = False
    pending: Dict[str, Set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                if current and not obsolete:
                    result[current].update(pending.get(current, set()))
                current = ""
                obsolete = False
                in_term = True
            elif line.startswith("["):
                if current and not obsolete:
                    result[current].update(pending.get(current, set()))
                current = ""
                in_term = False
            elif in_term and line.startswith("id: "):
                current = line[4:].strip()
            elif in_term and line == "is_obsolete: true":
                obsolete = True
            elif in_term and current and line.startswith("xref: EC:"):
                ec = line[len("xref: EC:"):].split()[0].strip()
                if EC_PATTERN.match(ec):
                    pending[current].add(ec)
    if current and not obsolete:
        result[current].update(pending.get(current, set()))
    return result


def interpro_go_ec_assignments(
    path: Path, go_to_ec: Mapping[str, Set[str]]
) -> Dict[str, Dict[str, Set[str]]]:
    """Map InterProScan GO annotations to complete EC numbers via GO xrefs."""
    assignments: MutableMapping[str, MutableMapping[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            columns = raw.rstrip("\n").split("\t")
            if len(columns) < 14:
                raise ValueError(f"{path}:{line_number}: expected at least 14 InterProScan columns")
            protein_id = columns[0]
            for go_id in set(re.findall(r"GO:\d{7}", columns[13])):
                for ec in go_to_ec.get(go_id, set()):
                    assignments[protein_id][ec].add(go_id)
    return {
        protein: {ec: set(go_ids) for ec, go_ids in ec_rows.items()}
        for protein, ec_rows in assignments.items()
    }


def is_ancestor(ancestor: str, child: str, parents: Mapping[str, Set[str]]) -> bool:
    if ancestor == child:
        return True
    pending = list(parents.get(child, set()))
    seen = set(pending)
    while pending:
        node = pending.pop()
        if node == ancestor:
            return True
        for parent in parents.get(node, set()):
            if parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return False


def compare_compartments(
    reaction: Set[str], s2f: Set[str], parents: Mapping[str, Set[str]]
) -> Tuple[str, str]:
    if not reaction and not s2f:
        return "no_evidence", ""
    if not reaction:
        return "reaction_location_missing", ""
    if not s2f:
        return "s2f_location_missing", ""
    compatible: List[str] = []
    exact = reaction & s2f
    for reaction_id in sorted(reaction):
        for s2f_id in sorted(s2f):
            if is_ancestor(reaction_id, s2f_id, parents) or is_ancestor(s2f_id, reaction_id, parents):
                compatible.append(f"{reaction_id}~{s2f_id}")
    details = ";".join(compatible)
    if reaction == s2f:
        return "exact_agreement", details
    reaction_covered = all(any(pair.startswith(item + "~") for pair in compatible) for item in reaction)
    s2f_covered = all(any(pair.endswith("~" + item) for pair in compatible) for item in s2f)
    if reaction_covered and s2f_covered:
        return "compatible_agreement", details
    if exact or compatible:
        return "partial_agreement", details
    return "conflict", ""


def build(args: argparse.Namespace) -> Dict[str, object]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = args.snapshot_dir.resolve()
    required = list(SNAPSHOT_URLS)
    missing = [name for name in required if not (snapshot / name).exists()]
    if missing:
        raise FileNotFoundError("Missing snapshots: " + ", ".join(missing))

    labels, allowed_anchors, excluded_anchors = load_slim(args.slim)
    parents = ontology_parents(args.obo)
    dm_records = parse_uniprot(snapshot / "uniprot_dm28c.tsv")
    tcr_records = parse_uniprot(snapshot / "uniprot_tcr_reference.tsv")

    dm_by_gene: MutableMapping[str, List[UniProtRecord]] = defaultdict(list)
    for record in dm_records:
        for gene in record.genes:
            match = TCDM_PATTERN.fullmatch(gene.upper())
            if match:
                dm_by_gene[match.group(1).upper()].append(record)

    tcr_by_accession = {record.accession: record for record in tcr_records}
    tcr_by_kegg: MutableMapping[str, List[UniProtRecord]] = defaultdict(list)
    for record in tcr_records:
        for kegg_id in record.kegg_ids:
            tcr_by_kegg[kegg_id.removeprefix("tcr:")].append(record)

    ec_reactions: MutableMapping[str, Set[str]] = defaultdict(set)
    for ec, reaction in parse_pair_file(
        snapshot / "kegg_enzyme_reaction.tsv", "ec:", "rn:"
    ):
        if EC_PATTERN.match(ec):
            ec_reactions[ec].add(reaction)

    ec_tcr_genes: MutableMapping[str, Set[str]] = defaultdict(set)
    for ec, gene in parse_pair_file(
        snapshot / "kegg_tcr_genes_by_enzyme.tsv", "ec:", "tcr:"
    ):
        ec_tcr_genes[ec].add(gene)

    kegg_to_uniprot: MutableMapping[str, Set[str]] = defaultdict(set)
    for kegg_gene, accession in parse_pair_file(
        snapshot / "kegg_tcr_uniprot.tsv", "tcr:", "up:"
    ):
        kegg_to_uniprot[kegg_gene].add(accession)
    names = reaction_name_map(snapshot / "kegg_reaction_names.tsv")

    s2f_rows = read_tsv(args.protein_compartments)
    s2f_by_protein: MutableMapping[str, List[Dict[str, str]]] = defaultdict(list)
    excluded_s2f_rows: List[Dict[str, str]] = []
    for row in s2f_rows:
        go_id = row.get("compartment_id", "")
        if go_id in excluded_anchors:
            excluded_s2f_rows.append(row)
        elif go_id in allowed_anchors:
            s2f_by_protein[row.get("protein_id", "")].append(row)

    all_proteins = set(s2f_by_protein)
    if args.protein_summary and args.protein_summary.exists():
        all_proteins.update(row.get("protein_id", "") for row in read_tsv(args.protein_summary))
    all_proteins.discard("")

    interpro_assignments: Dict[str, Dict[str, Set[str]]] = {}
    if args.interproscan:
        interpro_assignments = interpro_go_ec_assignments(
            args.interproscan, ontology_go_ec(args.obo)
        )

    protein_ec_rows: List[Dict[str, object]] = []
    hypothesis_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    unmapped_rows: List[Dict[str, object]] = []

    for protein_id in sorted(all_proteins):
        gene_id = canonical_gene(protein_id)
        exact_records = dm_by_gene.get(gene_id, [])
        ec_sources: MutableMapping[str, Set[str]] = defaultdict(set)
        ec_details: MutableMapping[str, Set[str]] = defaultdict(set)
        for record in exact_records:
            for ec in record.ecs:
                ec_sources[ec].add("exact_dm28c_gene_uniprot_ec")
                ec_details[ec].add(record.accession)
        for ec, go_ids in interpro_assignments.get(protein_id, {}).items():
            ec_sources[ec].add("interpro_go_exact_ec_xref")
            ec_details[ec].update(go_ids)
        if not ec_sources:
            reason = "no_complete_ec_from_exact_uniprot_or_interpro_go_xref"
            unmapped_rows.append({"protein_id": protein_id, "gene_id": gene_id, "stage": "protein_to_ec", "reason": reason})
            continue

        for ec in sorted(ec_sources):
            matching_records = [record for record in exact_records if ec in record.ecs]
            dm_accessions = sorted({record.accession for record in matching_records})
            reviewed_values = sorted({str(record.reviewed).lower() for record in matching_records})
            protein_ec_evidence = ";".join(sorted(ec_sources[ec]))
            protein_ec_source_details = ";".join(sorted(ec_details[ec]))
            reactions = sorted(ec_reactions.get(ec, set()))
            protein_ec_rows.append({
                "protein_id": protein_id,
                "gene_id": gene_id,
                "uniprot_accession": ";".join(dm_accessions),
                "uniprot_reviewed": ";".join(reviewed_values),
                "ec_number": ec,
                "n_kegg_reactions": len(reactions),
                "protein_ec_evidence": protein_ec_evidence,
                "protein_ec_source_details": protein_ec_source_details,
            })
            if not reactions:
                unmapped_rows.append({"protein_id": protein_id, "gene_id": gene_id, "stage": "ec_to_reaction", "reason": f"no_kegg_reaction_for_ec:{ec}"})
                continue

            for reaction_id in reactions:
                candidates: List[Tuple[str, str, UniProtRecord]] = []
                for tcr_gene in sorted(ec_tcr_genes.get(ec, set())):
                    accessions = sorted(kegg_to_uniprot.get(tcr_gene, set()))
                    records = [tcr_by_accession[acc] for acc in accessions if acc in tcr_by_accession]
                    if not records:
                        records = tcr_by_kegg.get(tcr_gene, [])
                    for record in choose_records(records):
                        candidates.append(("tcr_reference_reaction_enzyme", tcr_gene, record))

                basis = "tcr_reference_reaction_enzyme"
                if not candidates:
                    basis = "dm28c_exact_uniprot_fallback"
                    for record in choose_records(exact_records):
                        candidates.append((basis, gene_id, record))

                reaction_locations: Set[str] = set()
                evidence_levels: Set[str] = set()
                used_sources: Set[str] = set()
                for source, reference_gene, reference_record in candidates:
                    assertions = location_assertions(reference_record.location, labels, allowed_anchors)
                    for assertion in assertions:
                        reaction_locations.add(assertion.compartment_id)
                        evidence_levels.add(assertion.evidence_level)
                        used_sources.add(source)
                        hypothesis_rows.append({
                            "protein_id": protein_id,
                            "gene_id": gene_id,
                            "dm28c_uniprot_accession": ";".join(dm_accessions),
                            "protein_ec_evidence": protein_ec_evidence,
                            "protein_ec_source_details": protein_ec_source_details,
                            "ec_number": ec,
                            "reaction_id": reaction_id,
                            "reaction_name": names.get(reaction_id, ""),
                            "compartment_id": assertion.compartment_id,
                            "compartment_label": assertion.compartment_label,
                            "localization_source": source,
                            "reference_organism": "T. cruzi CL Brener" if source.startswith("tcr_") else "T. cruzi Dm28c",
                            "reference_gene": reference_gene,
                            "reference_uniprot_accession": reference_record.accession,
                            "reference_uniprot_reviewed": str(reference_record.reviewed).lower(),
                            "location_evidence_level": assertion.evidence_level,
                            "hypothesis_tier": ("reviewed_" if reference_record.reviewed else "unreviewed_") + assertion.evidence_level,
                            "eco_codes": assertion.eco_codes,
                            "raw_functional_location": assertion.raw_location,
                            "strain_caveat": "CL Brener evidence transferred by shared EC; isozyme-specificity unresolved" if source.startswith("tcr_") else "exact Dm28c UniProt fallback",
                        })

                s2f_locations = {row["compartment_id"] for row in s2f_by_protein.get(protein_id, [])}
                status, compatible_pairs = compare_compartments(reaction_locations, s2f_locations, parents)
                comparison_rows.append({
                    "protein_id": protein_id,
                    "gene_id": gene_id,
                    "dm28c_uniprot_accession": ";".join(dm_accessions),
                    "protein_ec_evidence": protein_ec_evidence,
                    "protein_ec_source_details": protein_ec_source_details,
                    "ec_number": ec,
                    "reaction_id": reaction_id,
                    "reaction_name": names.get(reaction_id, ""),
                    "reaction_compartment_ids": ";".join(sorted(reaction_locations)),
                    "reaction_compartment_labels": ";".join(labels.get(go_id, go_id) for go_id in sorted(reaction_locations)),
                    "reaction_location_basis": ";".join(sorted(used_sources)) if used_sources else "no_usable_functional_location",
                    "reaction_location_evidence": ";".join(sorted(evidence_levels)),
                    "s2f_compartment_ids": ";".join(sorted(s2f_locations)),
                    "s2f_compartment_labels": ";".join(labels.get(go_id, go_id) for go_id in sorted(s2f_locations)),
                    "s2f_scores": ";".join(f"{row['compartment_id']}:{row.get('s2f_score', '')}" for row in sorted(s2f_by_protein.get(protein_id, []), key=lambda item: item["compartment_id"])),
                    "comparison_status": status,
                    "compatible_pairs": compatible_pairs,
                    "interpretation": "Reaction-derived localization and S2F are independent hypotheses; agreement is corroboration, not ground truth.",
                })

    def aggregate_rows(keys: Sequence[str]) -> List[Dict[str, object]]:
        grouped: MutableMapping[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
        for row in comparison_rows:
            grouped[tuple(str(row[key]) for key in keys)].append(row)
        aggregated: List[Dict[str, object]] = []
        for key_values, rows in sorted(grouped.items()):
            reaction_ids = {str(row["reaction_id"]) for row in rows}
            reaction_locations = {
                go_id for row in rows
                for go_id in str(row["reaction_compartment_ids"]).split(";") if go_id
            }
            s2f_locations = {
                go_id for row in rows
                for go_id in str(row["s2f_compartment_ids"]).split(";") if go_id
            }
            status, compatible_pairs = compare_compartments(reaction_locations, s2f_locations, parents)
            basis = {
                str(row["reaction_location_basis"]) for row in rows
                if row["reaction_location_basis"] != "no_usable_functional_location"
            }
            evidence = {
                item for row in rows
                for item in str(row["reaction_location_evidence"]).split(";") if item
            }
            first = rows[0]
            item: Dict[str, object] = {key: value for key, value in zip(keys, key_values)}
            item.update({
                "gene_id": first["gene_id"],
                "dm28c_uniprot_accession": first["dm28c_uniprot_accession"],
                "protein_ec_evidence": ";".join(sorted({str(row["protein_ec_evidence"]) for row in rows})),
                "protein_ec_source_details": ";".join(sorted({str(row["protein_ec_source_details"]) for row in rows})),
                "ec_numbers": ";".join(sorted({str(row["ec_number"]) for row in rows})),
                "reaction_ids": ";".join(sorted(reaction_ids)),
                "n_reactions": len(reaction_ids),
                "n_reactions_with_location": sum(bool(row["reaction_compartment_ids"]) for row in rows),
                "reaction_compartment_ids": ";".join(sorted(reaction_locations)),
                "reaction_compartment_labels": ";".join(labels.get(go_id, go_id) for go_id in sorted(reaction_locations)),
                "reaction_location_basis": ";".join(sorted(basis)) if basis else "no_usable_functional_location",
                "reaction_location_evidence": ";".join(sorted(evidence)),
                "s2f_compartment_ids": ";".join(sorted(s2f_locations)),
                "s2f_compartment_labels": ";".join(labels.get(go_id, go_id) for go_id in sorted(s2f_locations)),
                "comparison_status": status,
                "compatible_pairs": compatible_pairs,
            })
            aggregated.append(item)
        return aggregated

    protein_ec_comparisons = aggregate_rows(("protein_id", "ec_number"))
    protein_comparisons = aggregate_rows(("protein_id",))

    protein_ec_fields = [
        "protein_id", "gene_id", "uniprot_accession", "uniprot_reviewed",
        "ec_number", "n_kegg_reactions", "protein_ec_evidence",
        "protein_ec_source_details",
    ]
    hypothesis_fields = [
        "protein_id", "gene_id", "dm28c_uniprot_accession",
        "protein_ec_evidence", "protein_ec_source_details", "ec_number",
        "reaction_id", "reaction_name", "compartment_id", "compartment_label",
        "localization_source", "reference_organism", "reference_gene",
        "reference_uniprot_accession", "reference_uniprot_reviewed",
        "location_evidence_level", "hypothesis_tier", "eco_codes", "raw_functional_location",
        "strain_caveat",
    ]
    comparison_fields = [
        "protein_id", "gene_id", "dm28c_uniprot_accession",
        "protein_ec_evidence", "protein_ec_source_details", "ec_number",
        "reaction_id", "reaction_name", "reaction_compartment_ids",
        "reaction_compartment_labels", "reaction_location_basis",
        "reaction_location_evidence", "s2f_compartment_ids",
        "s2f_compartment_labels", "s2f_scores", "comparison_status",
        "compatible_pairs", "interpretation",
    ]
    unmapped_fields = ["protein_id", "gene_id", "stage", "reason"]
    aggregate_fields = [
        "protein_id", "ec_number", "gene_id", "dm28c_uniprot_accession",
        "protein_ec_evidence", "protein_ec_source_details", "ec_numbers",
        "reaction_ids", "n_reactions", "n_reactions_with_location",
        "reaction_compartment_ids", "reaction_compartment_labels",
        "reaction_location_basis", "reaction_location_evidence",
        "s2f_compartment_ids", "s2f_compartment_labels", "comparison_status",
        "compatible_pairs",
    ]
    protein_aggregate_fields = [field for field in aggregate_fields if field != "ec_number"]
    write_tsv(output / "protein_ec.tsv", protein_ec_fields, protein_ec_rows)
    write_tsv(output / "reaction_compartment_hypotheses.tsv", hypothesis_fields, hypothesis_rows)
    write_tsv(output / "protein_reaction_compartment_comparison.tsv", comparison_fields, comparison_rows)
    write_tsv(output / "protein_ec_compartment_comparison.tsv", aggregate_fields, protein_ec_comparisons)
    write_tsv(output / "protein_compartment_comparison_summary.tsv", protein_aggregate_fields, protein_comparisons)
    write_tsv(output / "unmapped_proteins.tsv", unmapped_fields, unmapped_rows)
    write_tsv(output / "excluded_s2f_compartments.tsv", list(s2f_rows[0]) if s2f_rows else ["protein_id"], excluded_s2f_rows)

    status_counts = Counter(row["comparison_status"] for row in comparison_rows)
    evaluable = sum(status_counts[key] for key in ("exact_agreement", "compatible_agreement", "partial_agreement", "conflict"))
    aligned = sum(status_counts[key] for key in ("exact_agreement", "compatible_agreement", "partial_agreement"))

    def alignment_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        counts = Counter(str(row["comparison_status"]) for row in rows)
        denominator = sum(counts[key] for key in ("exact_agreement", "compatible_agreement", "partial_agreement", "conflict"))
        any_alignment = sum(counts[key] for key in ("exact_agreement", "compatible_agreement", "partial_agreement"))
        strict_alignment = sum(counts[key] for key in ("exact_agreement", "compatible_agreement"))
        return {
            "rows": len(rows),
            "status": dict(sorted(counts.items())),
            "evaluable": denominator,
            "any_alignment": any_alignment,
            "any_alignment_fraction": any_alignment / denominator if denominator else None,
            "strict_alignment": strict_alignment,
            "strict_alignment_fraction": strict_alignment / denominator if denominator else None,
        }

    reaction_level = alignment_counts(comparison_rows)
    protein_ec_level = alignment_counts(protein_ec_comparisons)
    protein_level = alignment_counts(protein_comparisons)
    proteins_with_ec = {str(row["protein_id"]) for row in protein_ec_rows}
    proteins_with_reaction = {str(row["protein_id"]) for row in comparison_rows}
    proteins_with_reaction_location = {str(row["protein_id"]) for row in hypothesis_rows}
    evaluable_protein_evidence = Counter(
        str(row["reaction_location_evidence"])
        for row in protein_comparisons
        if row["reaction_compartment_ids"] and row["s2f_compartment_ids"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "inputs": {
            "protein_compartments": str(args.protein_compartments.resolve()),
            "protein_compartments_sha256": sha256_file(args.protein_compartments),
            "protein_summary": str(args.protein_summary.resolve()) if args.protein_summary else "",
            "interproscan": str(args.interproscan.resolve()) if args.interproscan else "",
            "interproscan_sha256": sha256_file(args.interproscan) if args.interproscan else "",
            "snapshot_dir": str(snapshot),
            "slim": str(args.slim.resolve()),
            "obo": str(args.obo.resolve()),
        },
        "method": {
            "protein_reaction": "exact Dm28c UniProt EC and/or InterProScan GO molecular-function -> exact GO EC xref -> KEGG reaction",
            "reaction_compartment": "KEGG reaction EC -> KEGG tcr gene -> UniProt curated functional-location field",
            "fallback": "exact Dm28c UniProt functional-location field when the tcr reference bridge has no usable location",
            "comparison": "reaction-derived GO-slim set versus S2F GO-slim set",
            "raw_go_cc_used_for_reaction_location": False,
        },
        "caveats": [
            "KEGG tcr represents T. cruzi CL Brener (taxon 353153), not Dm28c (taxon 1416333).",
            "A shared EC can have compartment-specific isozymes; transferred reaction locations may be ambiguous.",
            "The S2F assignments use a manually selected 0.1 threshold and are hypotheses, not calibrated ground truth.",
            "Cross-organism validation anchors such as chloroplast and thylakoid are excluded.",
            "Agreement corroborates a hypothesis; conflict does not by itself identify which method is correct.",
            "InterPro-to-EC assignments are only retained when the GO ontology supplies a complete four-level EC cross-reference.",
        ],
        "counts": {
            "input_proteins": len(all_proteins),
            "protein_ec_rows": len(protein_ec_rows),
            "proteins_with_ec": len(proteins_with_ec),
            "proteins_with_kegg_reaction": len(proteins_with_reaction),
            "proteins_with_reaction_location": len(proteins_with_reaction_location),
            "reaction_compartment_hypothesis_rows": len(hypothesis_rows),
            "comparison_rows": len(comparison_rows),
            "excluded_s2f_rows": len(excluded_s2f_rows),
            "unmapped_rows": len(unmapped_rows),
            "unmapped_event_rows": len(unmapped_rows),
            "evaluable_protein_location_evidence": dict(sorted(evaluable_protein_evidence.items())),
            "comparison_status": dict(sorted(status_counts.items())),
            "evaluable_comparisons": evaluable,
            "aligned_comparisons": aligned,
            "alignment_fraction": aligned / evaluable if evaluable else None,
            "reaction_level": reaction_level,
            "protein_ec_level": protein_ec_level,
            "protein_level": protein_level,
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Dm28c reaction-compartment hypotheses versus S2F CC",
        "",
        "This analysis keeps the two localization routes independent. Protein ECs come from exact UniProt annotations and/or InterProScan molecular-function GO terms with exact GO EC cross-references. Reaction locations are then inferred through KEGG T. cruzi genes and UniProt functional-location statements; S2F CC assignments are compared afterwards.",
        "",
        "## Observed counts",
        "",
        f"- Proteins considered: {len(all_proteins):,}",
        f"- Protein-EC rows: {len(protein_ec_rows):,}",
        f"- Proteins with at least one EC hypothesis: {len(proteins_with_ec):,}",
        f"- Proteins linked to at least one KEGG reaction: {len(proteins_with_reaction):,}",
        f"- Proteins with a usable reaction-derived location: {len(proteins_with_reaction_location):,}",
        f"- Reaction-compartment evidence rows: {len(hypothesis_rows):,}",
        f"- Protein-reaction comparisons: {len(comparison_rows):,}",
        f"- Evaluable comparisons with both sources: {evaluable:,}",
        f"- Comparisons with any compatible alignment: {aligned:,}",
        f"- Alignment fraction: {(aligned / evaluable):.3f}" if evaluable else "- Alignment fraction: not estimable",
        f"- Unique proteins evaluable: {protein_level['evaluable']:,}",
        f"- Unique-protein strict alignment (exact or hierarchy-compatible): {protein_level['strict_alignment_fraction']:.3f}" if protein_level["evaluable"] else "- Unique-protein strict alignment: not estimable",
        f"- Unique-protein any alignment (including partial): {protein_level['any_alignment_fraction']:.3f}" if protein_level["evaluable"] else "- Unique-protein any alignment: not estimable",
        "",
        "## Comparison statuses",
        "",
    ]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(status_counts.items()))
    lines.extend([
        "",
        "## Unique-protein comparison statuses",
        "",
    ])
    lines.extend(
        f"- `{key}`: {value:,}"
        for key, value in sorted(protein_level["status"].items())
    )
    lines.extend([
        "",
        f"## Reaction-location evidence among the {protein_level['evaluable']:,} evaluable proteins",
        "",
    ])
    lines.extend(
        f"- `{key}`: {value:,}"
        for key, value in sorted(evaluable_protein_evidence.items())
    )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- KEGG `tcr` is the CL Brener reference strain, so its enzyme localization is transferred to Dm28c by shared EC and is not an exact ortholog assertion.",
        "- UniProt's curated functional-location statement is used; the raw GO CC inventory is deliberately not used to locate reactions.",
        "- S2F results came from the manually selected 0.1 threshold. Agreement is corroboration, while disagreements require protein-level review.",
        "- Cross-organism S2F anchors are excluded before comparison.",
        "- InterProScan-derived ECs are domain/GO-supported hypotheses and remain less direct than an experimentally verified enzyme annotation.",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Freeze UniProt and KEGG source snapshots")
    fetch_parser.add_argument("--snapshot-dir", type=Path, required=True)
    fetch_parser.add_argument("--force", action="store_true")

    for command in ("build", "run"):
        child = subparsers.add_parser(command, help=("Build from frozen snapshots" if command == "build" else "Fetch snapshots and build"))
        child.add_argument("--protein-compartments", type=Path, required=True)
        child.add_argument("--protein-summary", type=Path)
        child.add_argument(
            "--interproscan", type=Path,
            help="Optional InterProScan TSV; GO molecular-function terms are mapped only through exact GO EC xrefs",
        )
        child.add_argument("--snapshot-dir", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--slim", type=Path, default=DEFAULT_SLIM)
        child.add_argument("--obo", type=Path, default=DEFAULT_OBO)
        if command == "run":
            child.add_argument("--force", action="store_true")

    fetch_direct = subparsers.add_parser(
        "fetch-direct-ec",
        help="Freeze the latest archived UniSave record for each Dm28c accession",
    )
    fetch_direct.add_argument("--found-accessions", type=Path, required=True)
    fetch_direct.add_argument("--missing-fasta", type=Path, required=True)
    fetch_direct.add_argument("--archive-dir", type=Path, required=True)
    fetch_direct.add_argument("--workers", type=int, default=4)
    fetch_direct.add_argument("--force", action="store_true")

    for command in ("build-strict", "run-strict"):
        child = subparsers.add_parser(
            command,
            help=(
                "Build the strict direct-EC comparison from frozen inputs"
                if command == "build-strict"
                else "Fetch archived direct EC records and build the strict comparison"
            ),
        )
        child.add_argument("--archive-dir", type=Path, required=True)
        child.add_argument("--target-fasta", type=Path, required=True)
        child.add_argument("--protein-compartments", type=Path, required=True)
        child.add_argument("--protein-summary", type=Path)
        child.add_argument("--snapshot-dir", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--slim", type=Path, default=DEFAULT_SLIM)
        child.add_argument("--obo", type=Path, default=DEFAULT_OBO)
        child.add_argument(
            "--allow-incomplete-archive",
            action="store_true",
            help="Build a clearly partial result even when UniSave downloads failed",
        )
        if command == "run-strict":
            child.add_argument("--found-accessions", type=Path, required=True)
            child.add_argument("--missing-fasta", type=Path, required=True)
            child.add_argument("--workers", type=int, default=4)
            child.add_argument("--force-unisave", action="store_true")
            child.add_argument("--force-snapshots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command == "fetch":
        metadata = fetch_snapshots(args.snapshot_dir, args.force)
        print(json.dumps(metadata["files"], indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        fetch_snapshots(args.snapshot_dir, args.force)
    if args.command in {"fetch-direct-ec", "build-strict", "run-strict"}:
        import strict_kegg_compartments as strict

        if args.command in {"fetch-direct-ec", "run-strict"}:
            metadata = strict.fetch_unisave_archive(
                args.found_accessions,
                args.missing_fasta,
                args.archive_dir,
                workers=args.workers,
                force=(
                    args.force
                    if args.command == "fetch-direct-ec"
                    else args.force_unisave
                ),
            )
            if args.command == "fetch-direct-ec":
                print(json.dumps(metadata, indent=2, sort_keys=True))
                return 0 if metadata["complete"] else 1
        if args.command == "run-strict":
            fetch_snapshots(args.snapshot_dir, args.force_snapshots)
        summary = strict.build_strict(args)
        print(json.dumps(summary["counts"], indent=2, sort_keys=True))
        return 0
    summary = build(args)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
