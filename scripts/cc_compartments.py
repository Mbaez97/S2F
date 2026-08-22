#!/usr/bin/env python3
"""Map S2F Cellular Component predictions to explainable compartments.

The output describes protein-localization hypotheses. It deliberately does not
infer EC numbers, reactions, or reaction compartments from Cellular Component
terms.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import urllib.parse
import urllib.request


S2F_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBO = S2F_ROOT / "go.obo"
DEFAULT_SLIM = S2F_ROOT / "conf" / "cc_compartments_tcruzi.json"
DEFAULT_BENCHMARK_DIR = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data" / "pfp_score_comparison"
DEFAULT_CAFA_ROOT = Path("/run/media/marcelo_baez/HD_Disc1/.S2F/output")
SCHEMA_VERSION = 1
SAFE_RELATIONS = {"is_a", "part_of"}
EXPERIMENTAL_ECO = {"ECO:0000269"}
INFERRED_ECO = {"ECO:0000250", "ECO:0000255", "ECO:0000305"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GoTerm:
    go_id: str
    name: str = ""
    namespace: str = ""
    obsolete: bool = False
    alt_ids: Set[str] = field(default_factory=set)
    replaced_by: List[str] = field(default_factory=list)
    consider: List[str] = field(default_factory=list)
    parents: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class TermResolution:
    requested_id: str
    resolved_id: Optional[str]
    status: str


@dataclass(frozen=True)
class CompartmentPath:
    compartment_id: str
    nodes: Tuple[str, ...]
    relations: Tuple[str, ...]

    @property
    def distance(self) -> int:
        return len(self.relations)


class Ontology:
    def __init__(self, terms: Mapping[str, GoTerm]) -> None:
        self.terms = dict(terms)
        self.alt_ids: Dict[str, str] = {}
        for term in self.terms.values():
            for alt_id in term.alt_ids:
                if alt_id in self.alt_ids and self.alt_ids[alt_id] != term.go_id:
                    raise ValueError(f"Duplicate GO alternate ID: {alt_id}")
                self.alt_ids[alt_id] = term.go_id

    @classmethod
    def from_obo(cls, path: Path) -> "Ontology":
        terms: Dict[str, GoTerm] = {}
        stanza: Dict[str, List[str]] = {}
        stanza_type = ""

        def flush() -> None:
            nonlocal stanza, stanza_type
            if stanza_type != "Term" or "id" not in stanza:
                stanza = {}
                stanza_type = ""
                return
            go_id = stanza["id"][0]
            parents: List[Tuple[str, str]] = []
            for value in stanza.get("is_a", []):
                parents.append(("is_a", value.split(" ! ", 1)[0].strip()))
            for value in stanza.get("relationship", []):
                bits = value.split()
                if len(bits) >= 2 and bits[0] == "part_of":
                    parents.append(("part_of", bits[1]))
            terms[go_id] = GoTerm(
                go_id=go_id,
                name=stanza.get("name", [""])[0],
                namespace=stanza.get("namespace", [""])[0],
                obsolete=stanza.get("is_obsolete", ["false"])[0].lower() == "true",
                alt_ids=set(stanza.get("alt_id", [])),
                replaced_by=[value.split()[0] for value in stanza.get("replaced_by", [])],
                consider=[value.split()[0] for value in stanza.get("consider", [])],
                parents=sorted(set(parents)),
            )
            stanza = {}
            stanza_type = ""

        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("[") and line.endswith("]"):
                    flush()
                    stanza_type = line[1:-1]
                    continue
                if not line or line.startswith("!") or stanza_type != "Term" or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                stanza.setdefault(key, []).append(value)
        flush()
        if not terms:
            raise ValueError(f"No [Term] stanzas found in {path}")
        return cls(terms)

    def resolve(self, go_id: str) -> TermResolution:
        requested = go_id.strip()
        canonical = self.alt_ids.get(requested, requested)
        term = self.terms.get(canonical)
        if term is None:
            return TermResolution(requested, None, "unknown_go_id")
        prefix = "alt_id" if canonical != requested else "current"
        if not term.obsolete:
            return TermResolution(requested, canonical, prefix)
        replacements = [
            self.alt_ids.get(value, value)
            for value in term.replaced_by
            if self.alt_ids.get(value, value) in self.terms
            and not self.terms[self.alt_ids.get(value, value)].obsolete
        ]
        replacements = sorted(set(replacements))
        if len(replacements) == 1:
            return TermResolution(requested, replacements[0], "obsolete_replaced")
        if len(replacements) > 1:
            return TermResolution(requested, None, "obsolete_ambiguous_replacement")
        return TermResolution(requested, None, "obsolete_unresolved")

    def shortest_anchor_paths(
        self,
        go_id: str,
        anchors: Set[str],
        relations: Set[str],
    ) -> List[CompartmentPath]:
        queue: deque[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = deque(
            [(go_id, (go_id,), ())]
        )
        best_node_distance = {go_id: 0}
        best_anchor_distance: Optional[int] = None
        found: Dict[str, CompartmentPath] = {}
        while queue:
            node, nodes, path_relations = queue.popleft()
            distance = len(path_relations)
            if best_anchor_distance is not None and distance > best_anchor_distance:
                continue
            if node in anchors:
                best_anchor_distance = distance
                candidate = CompartmentPath(node, nodes, path_relations)
                current = found.get(node)
                if current is None or (candidate.nodes, candidate.relations) < (current.nodes, current.relations):
                    found[node] = candidate
                continue
            term = self.terms.get(node)
            if term is None:
                continue
            for relation, parent in sorted(term.parents, key=lambda item: (item[1], item[0])):
                if relation not in relations or parent in nodes:
                    continue
                next_distance = distance + 1
                seen = best_node_distance.get(parent)
                if seen is not None and next_distance > seen:
                    continue
                best_node_distance[parent] = next_distance
                queue.append((parent, nodes + (parent,), path_relations + (relation,)))
        return [found[key] for key in sorted(found)]

    def has_ancestor(self, go_id: str, roots: Set[str], relations: Set[str]) -> bool:
        if go_id in roots:
            return True
        queue = deque([go_id])
        seen = {go_id}
        while queue:
            node = queue.popleft()
            term = self.terms.get(node)
            if term is None:
                continue
            for relation, parent in term.parents:
                if relation not in relations or parent in seen:
                    continue
                if parent in roots:
                    return True
                seen.add(parent)
                queue.append(parent)
        return False


class CompartmentMapper:
    def __init__(self, ontology: Ontology, config: Mapping[str, object]) -> None:
        self.ontology = ontology
        self.config = config
        self.relations = set(str(value) for value in config.get("relations", []))
        if not self.relations or not self.relations.issubset(SAFE_RELATIONS):
            raise ValueError("Slim relations must be a non-empty subset of is_a and part_of")
        self.anchor_metadata = {
            str(row["go_id"]): dict(row) for row in config.get("anchors", [])
        }
        self.anchors = set(self.anchor_metadata)
        self.non_spatial_roots = set(str(value) for value in config.get("non_spatial_roots", []))
        self.stage_flags = {
            str(key): value for key, value in dict(config.get("stage_flags", {})).items()
        }
        missing = sorted((self.anchors | self.non_spatial_roots) - set(ontology.terms))
        if missing:
            raise ValueError(f"Slim references GO terms absent from ontology: {', '.join(missing)}")
        non_cc = sorted(
            go_id for go_id in self.anchors
            if ontology.terms[go_id].namespace != "cellular_component"
        )
        if non_cc:
            raise ValueError(f"Slim anchors are not Cellular Component terms: {', '.join(non_cc)}")
        self._mapping_cache: Dict[str, Tuple[TermResolution, str, Tuple[CompartmentPath, ...]]] = {}
        self._lineage_cache: Dict[str, str] = {}

    @classmethod
    def from_files(cls, obo_path: Path, config_path: Path) -> "CompartmentMapper":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported slim schema version in {config_path}")
        return cls(Ontology.from_obo(obo_path), config)

    def map_term(self, requested_id: str) -> Tuple[TermResolution, str, Tuple[CompartmentPath, ...]]:
        cached = self._mapping_cache.get(requested_id)
        if cached is not None:
            return cached
        resolution = self.ontology.resolve(requested_id)
        if resolution.resolved_id is None:
            result = (resolution, resolution.status, ())
        else:
            term = self.ontology.terms[resolution.resolved_id]
            if term.namespace != "cellular_component":
                result = (resolution, "non_cc", ())
            else:
                paths = tuple(self.ontology.shortest_anchor_paths(
                    resolution.resolved_id, self.anchors, self.relations
                ))
                if paths:
                    result = (resolution, "mapped", paths)
                elif self.ontology.has_ancestor(
                    resolution.resolved_id, self.non_spatial_roots, self.relations
                ):
                    result = (resolution, "non_spatial_cc", ())
                else:
                    result = (resolution, "unresolved_cc", ())
        self._mapping_cache[requested_id] = result
        return result

    def lineage(self, anchor_id: str) -> str:
        if anchor_id in self._lineage_cache:
            return self._lineage_cache[anchor_id]
        paths: List[Tuple[int, str]] = [(0, anchor_id)]
        for other in sorted(self.anchors - {anchor_id}):
            candidate = self.ontology.shortest_anchor_paths(
                anchor_id, {other}, self.relations
            )
            if candidate:
                paths.append((candidate[0].distance, other))
        paths.sort(key=lambda item: (item[0], item[1]))
        lineage = " > ".join(value for _, value in paths)
        self._lineage_cache[anchor_id] = lineage
        return lineage


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tsv_writer(path: Path, fieldnames: Sequence[str]) -> Tuple[object, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return handle, writer


def iter_s2f_predictions(path: Path) -> Iterator[Tuple[int, str, str, float]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 tab-separated fields")
            protein_id, go_id, raw_score = fields
            try:
                score = float(raw_score)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid score {raw_score!r}") from exc
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: non-finite score {raw_score!r}")
            yield line_number, protein_id, go_id, score


def read_protein_ids(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    proteins: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        first_nonempty = ""
        for raw_line in handle:
            if raw_line.strip():
                first_nonempty = raw_line
                break
        handle.seek(0)
        if first_nonempty.startswith(">"):
            for raw_line in handle:
                if raw_line.startswith(">"):
                    proteins.add(raw_line[1:].strip().split()[0])
        else:
            for raw_line in handle:
                value = raw_line.strip().split("\t", 1)[0]
                if value:
                    proteins.add(value)
    return proteins


def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    identifier: Optional[str] = None
    chunks: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks).upper()
                identifier = line[1:].split()[0]
                if not identifier:
                    raise ValueError(f"{path}:{line_number}: empty FASTA identifier")
                chunks = []
            else:
                if identifier is None:
                    raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
                chunks.append("".join(line.split()))
    if identifier is not None:
        yield identifier, "".join(chunks).upper()


def threshold_for(policy: Mapping[str, object], compartment_id: str) -> Optional[float]:
    specific = dict(policy.get("compartments", {})).get(compartment_id)
    if isinstance(specific, Mapping) and specific.get("threshold") is not None:
        return float(specific["threshold"])
    value = policy.get("global_threshold")
    return None if value is None else float(value)


def load_s2f_policy(calibration_path: Optional[Path], manual_threshold: Optional[float]) -> Dict[str, object]:
    if calibration_path is not None and manual_threshold is not None:
        raise ValueError("Use either --calibration or --threshold, not both")
    if calibration_path is not None:
        artifact = read_json(calibration_path)
        policies = dict(artifact.get("operational_policies", {}))
        if "s2f_score" not in policies:
            raise ValueError("Calibration artifact has no s2f_score policy")
        return dict(policies["s2f_score"])
    if manual_threshold is None:
        raise ValueError("assign requires --calibration or an explicit --threshold")
    if not 0.0 <= manual_threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    return {
        "global_threshold": manual_threshold,
        "compartments": {},
        "source": "manual",
    }


def read_curated_evidence(path: Optional[Path], anchors: Set[str]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    evidence: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    if path is None:
        return evidence
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "compartment_id", "assertion", "evidence_level"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Curated evidence is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, 2):
            compartment = str(row["compartment_id"]).strip()
            if compartment not in anchors:
                raise ValueError(f"{path}:{line_number}: unknown compartment anchor {compartment}")
            assertion = str(row["assertion"]).strip().lower()
            if assertion not in {"in", "not_in"}:
                raise ValueError(f"{path}:{line_number}: assertion must be in or not_in")
            normalized = {str(key): str(value or "") for key, value in row.items()}
            normalized["assertion"] = assertion
            evidence[(str(row["protein_id"]).strip(), compartment)].append(normalized)
    return evidence


def evidence_tier(score_passes: bool, rows: Sequence[Mapping[str, str]]) -> str:
    if not score_passes:
        return "below_threshold"
    if any(row.get("assertion") == "not_in" for row in rows):
        return "conflict"
    positive = [row for row in rows if row.get("assertion") == "in"]
    levels = {row.get("evidence_level", "").lower() for row in positive}
    if any("experimental" in level for level in levels):
        return "reviewed_experimental_agreement"
    if positive:
        return "reviewed_inferred_agreement"
    return "s2f_only_hypothesis"


def assign_command(args: argparse.Namespace) -> int:
    mapper = CompartmentMapper.from_files(args.obo, args.slim)
    policy = load_s2f_policy(args.calibration, args.threshold)
    curated = read_curated_evidence(args.curated_evidence, mapper.anchors)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_fields = [
        "protein_id", "input_go_id", "resolved_go_id", "go_name", "go_namespace",
        "go_resolution", "s2f_score", "mapping_status", "compartment_id",
        "compartment_label", "distance", "go_path", "relation_path",
        "compartment_lineage", "threshold", "passes_threshold", "stage_flag",
    ]
    evidence_handle, evidence_writer = tsv_writer(
        args.output / "protein_compartment_evidence.tsv", evidence_fields
    )
    proteins = read_protein_ids(args.protein_list)
    protein_stats: Dict[str, MutableMapping[str, int]] = defaultdict(lambda: defaultdict(int))
    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    input_rows = 0
    try:
        for _, protein_id, go_id, score in iter_s2f_predictions(args.prediction):
            input_rows += 1
            proteins.add(protein_id)
            resolution, status, paths = mapper.map_term(go_id)
            if status == "non_cc" and not args.include_non_cc_evidence:
                protein_stats[protein_id]["non_cc"] += 1
                continue
            resolved_term = mapper.ontology.terms.get(resolution.resolved_id or "")
            common = {
                "protein_id": protein_id,
                "input_go_id": go_id,
                "resolved_go_id": resolution.resolved_id or "",
                "go_name": resolved_term.name if resolved_term else "",
                "go_namespace": resolved_term.namespace if resolved_term else "",
                "go_resolution": resolution.status,
                "s2f_score": f"{score:.17g}",
                "mapping_status": status,
            }
            if not paths:
                evidence_writer.writerow({**common, **{
                    "compartment_id": "", "compartment_label": "", "distance": "",
                    "go_path": "", "relation_path": "", "compartment_lineage": "",
                    "threshold": "", "passes_threshold": "", "stage_flag": "",
                }})
                protein_stats[protein_id][status] += 1
                continue
            protein_stats[protein_id]["mapped"] += 1
            for path in paths:
                compartment = path.compartment_id
                threshold = threshold_for(policy, compartment)
                passes = threshold is not None and score >= threshold
                stage_flag = mapper.stage_flags.get(compartment, {})
                evidence_writer.writerow({**common, **{
                    "compartment_id": compartment,
                    "compartment_label": mapper.anchor_metadata[compartment]["label"],
                    "distance": path.distance,
                    "go_path": " > ".join(path.nodes),
                    "relation_path": " > ".join(path.relations),
                    "compartment_lineage": mapper.lineage(compartment),
                    "threshold": "" if threshold is None else f"{threshold:.17g}",
                    "passes_threshold": str(passes).lower(),
                    "stage_flag": json.dumps(stage_flag, sort_keys=True) if stage_flag else "",
                }})
                key = (protein_id, compartment)
                current = best.get(key)
                support = {
                    "score": score,
                    "go_ids": {resolution.resolved_id or go_id},
                    "paths": {" > ".join(path.nodes)},
                    "threshold": threshold,
                }
                if current is None or score > float(current["score"]):
                    best[key] = support
                elif score == float(current["score"]):
                    current["go_ids"].add(resolution.resolved_id or go_id)
                    current["paths"].add(" > ".join(path.nodes))
    finally:
        evidence_handle.close()

    compartment_fields = [
        "protein_id", "compartment_id", "compartment_label", "s2f_score",
        "threshold", "confidence_tier", "supporting_go_ids", "supporting_go_paths",
        "compartment_lineage", "stage_flag", "curated_sources",
    ]
    compartment_handle, compartment_writer = tsv_writer(
        args.output / "protein_compartments.tsv", compartment_fields
    )
    retained: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    try:
        for (protein_id, compartment), value in sorted(best.items()):
            threshold = value["threshold"]
            passes = threshold is not None and float(value["score"]) >= float(threshold)
            if not passes:
                continue
            curated_rows = curated.get((protein_id, compartment), [])
            tier = evidence_tier(True, curated_rows)
            row = {
                "protein_id": protein_id,
                "compartment_id": compartment,
                "compartment_label": mapper.anchor_metadata[compartment]["label"],
                "s2f_score": f"{float(value['score']):.17g}",
                "threshold": f"{float(threshold):.17g}",
                "confidence_tier": tier,
                "supporting_go_ids": ";".join(sorted(value["go_ids"])),
                "supporting_go_paths": ";".join(sorted(value["paths"])),
                "compartment_lineage": mapper.lineage(compartment),
                "stage_flag": json.dumps(mapper.stage_flags.get(compartment, {}), sort_keys=True)
                if compartment in mapper.stage_flags else "",
                "curated_sources": ";".join(sorted({r.get("source", "") for r in curated_rows if r.get("source")})),
            }
            compartment_writer.writerow(row)
            retained[protein_id].append(row)
        for (protein_id, compartment), curated_rows in sorted(curated.items()):
            if not any(row.get("assertion") == "in" for row in curated_rows):
                continue
            if any(row["compartment_id"] == compartment for row in retained.get(protein_id, [])):
                continue
            proteins.add(protein_id)
            row = {
                "protein_id": protein_id,
                "compartment_id": compartment,
                "compartment_label": mapper.anchor_metadata[compartment]["label"],
                "s2f_score": "",
                "threshold": "",
                "confidence_tier": "curated_only",
                "supporting_go_ids": "",
                "supporting_go_paths": "",
                "compartment_lineage": mapper.lineage(compartment),
                "stage_flag": json.dumps(mapper.stage_flags.get(compartment, {}), sort_keys=True)
                if compartment in mapper.stage_flags else "",
                "curated_sources": ";".join(sorted({r.get("source", "") for r in curated_rows if r.get("source")})),
            }
            compartment_writer.writerow(row)
            retained[protein_id].append(row)
    finally:
        compartment_handle.close()

    summary_fields = [
        "protein_id", "status", "compartment_ids", "compartment_labels",
        "confidence_tiers", "scores", "n_high_confidence_compartments",
        "n_mapped_cc_rows", "n_unresolved_cc_rows", "n_non_spatial_cc_rows",
        "n_unknown_or_obsolete_go_rows",
    ]
    summary_handle, summary_writer = tsv_writer(
        args.output / "protein_compartment_summary.tsv", summary_fields
    )
    status_counts: Dict[str, int] = defaultdict(int)
    try:
        for protein_id in sorted(proteins):
            rows = sorted(retained.get(protein_id, []), key=lambda row: str(row["compartment_id"]))
            if rows:
                status = "assigned"
            elif protein_stats[protein_id].get("mapped", 0):
                status = "below_threshold"
            else:
                status = "unresolved"
            status_counts[status] += 1
            summary_writer.writerow({
                "protein_id": protein_id,
                "status": status,
                "compartment_ids": ";".join(str(row["compartment_id"]) for row in rows),
                "compartment_labels": ";".join(str(row["compartment_label"]) for row in rows),
                "confidence_tiers": ";".join(str(row["confidence_tier"]) for row in rows),
                "scores": ";".join(str(row["s2f_score"]) for row in rows),
                "n_high_confidence_compartments": len(rows),
                "n_mapped_cc_rows": protein_stats[protein_id].get("mapped", 0),
                "n_unresolved_cc_rows": protein_stats[protein_id].get("unresolved_cc", 0),
                "n_non_spatial_cc_rows": protein_stats[protein_id].get("non_spatial_cc", 0),
                "n_unknown_or_obsolete_go_rows": sum(
                    protein_stats[protein_id].get(status, 0)
                    for status in (
                        "unknown_go_id", "obsolete_unresolved",
                        "obsolete_ambiguous_replacement",
                    )
                ),
            })
    finally:
        summary_handle.close()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "command": "assign",
        "interpretation": "Protein-localization hypotheses; no reaction or EC inference.",
        "inputs": {
            "prediction": str(args.prediction.resolve()),
            "prediction_sha256": sha256_file(args.prediction),
            "obo": str(args.obo.resolve()),
            "obo_sha256": sha256_file(args.obo),
            "slim": str(args.slim.resolve()),
            "slim_sha256": sha256_file(args.slim),
            "calibration": str(args.calibration.resolve()) if args.calibration else None,
            "curated_evidence": str(args.curated_evidence.resolve()) if args.curated_evidence else None,
        },
        "interpreter": sys.executable,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "input_rows": input_rows,
        "protein_count": len(proteins),
        "status_counts": dict(sorted(status_counts.items())),
        "retained_protein_compartment_pairs": sum(len(rows) for rows in retained.values()),
    }
    write_json(args.output / "run_metadata.json", metadata)
    report = [
        "# S2F CC-to-compartment assignment",
        "",
        (
            "> These are protein-localization hypotheses. Cellular Component "
            "predictions do not establish an EC number, reaction, or reaction compartment."
        ),
        "",
        f"- Proteins: {len(proteins):,}",
        f"- Retained protein-compartment pairs: {metadata['retained_protein_compartment_pairs']:,}",
        f"- Assigned proteins: {status_counts.get('assigned', 0):,}",
        f"- Below-threshold proteins: {status_counts.get('below_threshold', 0):,}",
        f"- Unresolved proteins: {status_counts.get('unresolved', 0):,}",
        "",
        "See `run_metadata.json` for input hashes and provenance.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


def load_score_matrix(path: Path, mapper: CompartmentMapper) -> Tuple[str, List[Dict[str, object]], List[str]]:
    payload = read_json(path)
    columns = [str(value) for value in payload["column_keys"]]
    positions = {name: index for index, name in enumerate(columns)}
    required = {"protein_id", "term_id", "go_domain", "ground_truth", "s2f_score"}
    missing = required - set(positions)
    if missing:
        raise ValueError(f"{path} is missing matrix columns: {', '.join(sorted(missing))}")
    methods = [name for name in columns if name.endswith("_score")]
    collapsed: Dict[Tuple[str, str], Dict[str, object]] = {}
    proteins: Set[str] = set()
    mapped_compartments: Set[str] = set()
    mapped_rows: List[Tuple[str, List[str], List[object]]] = []
    for values in payload["rows"]:
        protein = str(values[positions["protein_id"]])
        proteins.add(protein)
        if str(values[positions["go_domain"]]) != "cellular_component":
            continue
        go_id = str(values[positions["term_id"]])
        _, status, paths = mapper.map_term(go_id)
        if status != "mapped":
            continue
        compartments = sorted({item.compartment_id for item in paths})
        mapped_compartments.update(compartments)
        mapped_rows.append((protein, compartments, values))
    for protein in sorted(proteins):
        for compartment in sorted(mapped_compartments):
            collapsed[(protein, compartment)] = {
                "organism": str(payload["organism"]),
                "protein_id": protein,
                "compartment_id": compartment,
                "ground_truth": 0,
                **{method: 0.0 for method in methods},
            }
    for protein, compartments, values in mapped_rows:
        truth = int(values[positions["ground_truth"]])
        for compartment in compartments:
            row = collapsed[(protein, compartment)]
            row["ground_truth"] = max(int(row["ground_truth"]), truth)
            for method in methods:
                value = values[positions[method]]
                if value is not None:
                    row[method] = max(float(row[method]), float(value))
    return str(payload["organism"]), list(collapsed.values()), methods


def select_global_threshold(
    examples: Sequence[Mapping[str, object]],
    method: str,
    target_precision: float,
) -> Optional[float]:
    candidates = sorted({float(row[method]) for row in examples if float(row[method]) > 0.0})
    best: Optional[Tuple[int, float]] = None
    for threshold in candidates:
        selected = [row for row in examples if float(row[method]) >= threshold]
        if not selected:
            continue
        precision = sum(int(row["ground_truth"]) for row in selected) / len(selected)
        if precision + 1e-15 < target_precision:
            continue
        candidate = (len(selected), -threshold)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else -best[1]


def build_threshold_policy(
    examples: Sequence[Mapping[str, object]],
    method: str,
    target_precision: float,
    min_positives: int,
) -> Dict[str, object]:
    global_threshold = select_global_threshold(examples, method, target_precision)
    compartments: Dict[str, object] = {}
    for compartment in sorted({str(row["compartment_id"]) for row in examples}):
        subset = [row for row in examples if row["compartment_id"] == compartment]
        positives = sum(int(row["ground_truth"]) for row in subset)
        if positives >= min_positives:
            threshold = select_global_threshold(subset, method, target_precision)
            source = "compartment_specific" if threshold is not None else "global_fallback"
        else:
            threshold = None
            source = "global_fallback_limited_calibration"
        compartments[compartment] = {
            "threshold": threshold,
            "source": source,
            "calibration_positive_pairs": positives,
            "limited_calibration": positives < min_positives or threshold is None,
        }
    return {
        "method": method,
        "target_precision": target_precision,
        "min_calibration_positives": min_positives,
        "global_threshold": global_threshold,
        "compartments": compartments,
    }


def policy_prediction(row: Mapping[str, object], method: str, policy: Mapping[str, object]) -> bool:
    threshold = threshold_for(policy, str(row["compartment_id"]))
    return threshold is not None and float(row[method]) >= threshold


def average_precision(examples: Sequence[Mapping[str, object]], method: str) -> float:
    ranked = sorted(
        examples,
        key=lambda row: (-float(row[method]), str(row["protein_id"]), str(row["compartment_id"])),
    )
    positives = sum(int(row["ground_truth"]) for row in ranked)
    if positives == 0:
        return 0.0
    seen_positive = 0
    total = 0.0
    for rank, row in enumerate(ranked, 1):
        if int(row["ground_truth"]):
            seen_positive += 1
            total += seen_positive / rank
    return total / positives


def metrics_from_predictions(
    examples: Sequence[Mapping[str, object]], predictions: Sequence[bool], method: str
) -> Dict[str, float]:
    tp = sum(1 for row, pred in zip(examples, predictions) if pred and int(row["ground_truth"]))
    fp = sum(1 for row, pred in zip(examples, predictions) if pred and not int(row["ground_truth"]))
    fn = sum(1 for row, pred in zip(examples, predictions) if not pred and int(row["ground_truth"]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    truth_by_protein: Dict[str, Set[str]] = defaultdict(set)
    predicted_by_protein: Dict[str, Set[str]] = defaultdict(set)
    proteins: Set[str] = set()
    for row, predicted in zip(examples, predictions):
        protein = str(row["protein_id"])
        compartment = str(row["compartment_id"])
        proteins.add(protein)
        if int(row["ground_truth"]):
            truth_by_protein[protein].add(compartment)
        if predicted:
            predicted_by_protein[protein].add(compartment)
    eligible_proteins: List[str] = []
    macro_f1_values: List[float] = []
    jaccard_values: List[float] = []
    covered: Set[str] = set()
    for protein in sorted(proteins):
        truth = truth_by_protein[protein]
        predicted = predicted_by_protein[protein]
        if predicted:
            covered.add(protein)
        if not truth:
            # Absence of a mapped spatial gold label is not reliable negative
            # evidence for the protein-level metric. Pair-level FP still records
            # predictions made for these proteins.
            continue
        eligible_proteins.append(protein)
        intersection = len(truth & predicted)
        denominator = len(truth) + len(predicted)
        macro_f1_values.append(2 * intersection / denominator if denominator else 0.0)
        union = len(truth | predicted)
        jaccard_values.append(intersection / union if union else 0.0)
    covered_eligible = covered & set(eligible_proteins)
    return {
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "predicted_pairs": float(tp + fp),
        "true_pairs": float(tp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aupr": average_precision(examples, method),
        "protein_macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "protein_macro_jaccard": sum(jaccard_values) / len(jaccard_values) if jaccard_values else 0.0,
        "protein_coverage": len(covered_eligible) / len(eligible_proteins) if eligible_proteins else 0.0,
        "abstention_rate": 1.0 - (len(covered_eligible) / len(eligible_proteins)) if eligible_proteins else 0.0,
        "protein_count": float(len(proteins)),
        "eligible_protein_count": float(len(eligible_proteins)),
    }


def bootstrap_intervals(
    examples: Sequence[Mapping[str, object]],
    predictions: Sequence[bool],
    method: str,
    repetitions: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    if repetitions <= 0:
        return {}
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        grouped[str(row["protein_id"])].append(index)
    proteins = sorted(grouped)
    if not proteins:
        return {}
    rng = random.Random(seed)
    names = ["precision", "recall", "f1", "protein_macro_f1", "protein_macro_jaccard", "protein_coverage"]
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        selected_indices: List[int] = []
        sampled_examples: List[Mapping[str, object]] = []
        sampled_predictions: List[bool] = []
        for draw_index, protein in enumerate(rng.choice(proteins) for _ in proteins):
            for index in grouped[protein]:
                copied = dict(examples[index])
                copied["protein_id"] = f"{protein}#bootstrap{draw_index}"
                sampled_examples.append(copied)
                sampled_predictions.append(predictions[index])
        values = metrics_from_predictions(sampled_examples, sampled_predictions, method)
        for name in names:
            samples[name].append(values[name])
    intervals: Dict[str, Tuple[float, float]] = {}
    for name, values in samples.items():
        values.sort()
        low = values[max(0, int(0.025 * len(values)) - 1)]
        high = values[min(len(values) - 1, int(0.975 * len(values)))]
        intervals[name] = (low, high)
    return intervals


def validation_metric_row(
    fold: str,
    method: str,
    examples: Sequence[Mapping[str, object]],
    predictions: Sequence[bool],
    policy: Mapping[str, object],
    bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    metrics = metrics_from_predictions(examples, predictions, method)
    intervals = bootstrap_intervals(examples, predictions, method, bootstrap, seed)
    row: Dict[str, object] = {
        "fold": fold,
        "method": method,
        "global_threshold": policy.get("global_threshold"),
        **metrics,
    }
    for name, (low, high) in intervals.items():
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    return row


def calibrate_benchmarks(
    mapper: CompartmentMapper,
    benchmark_dir: Path,
    output_dir: Path,
    target_precision: float,
    min_positives: int,
    bootstrap: int,
    seed: int,
    clean_status: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    matrix_paths = sorted(benchmark_dir.glob("*.json"))
    if len(matrix_paths) < 2:
        raise ValueError(f"Need at least two benchmark matrices in {benchmark_dir}")
    by_organism: Dict[str, List[Dict[str, object]]] = {}
    method_sets: List[Set[str]] = []
    for path in matrix_paths:
        organism, examples, methods = load_score_matrix(path, mapper)
        by_organism[organism] = examples
        method_sets.append(set(methods))
    methods = sorted(set.intersection(*method_sets))
    if not methods:
        raise ValueError("Benchmark matrices have no shared score methods")
    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    pooled_by_method: Dict[str, Tuple[List[Dict[str, object]], List[bool]]] = {
        method: ([], []) for method in methods
    }
    fold_policies: Dict[str, object] = {}
    for fold_index, test_organism in enumerate(sorted(by_organism)):
        train_examples = [
            row for organism, rows in by_organism.items()
            if organism != test_organism for row in rows
        ]
        test_examples = by_organism[test_organism]
        fold_policies[test_organism] = {}
        for method_index, method in enumerate(methods):
            policy = build_threshold_policy(
                train_examples, method, target_precision, min_positives
            )
            fold_policies[test_organism][method] = policy
            predictions = [policy_prediction(row, method, policy) for row in test_examples]
            metric_rows.append(validation_metric_row(
                test_organism, method, test_examples, predictions, policy,
                bootstrap, seed + fold_index * 100 + method_index,
            ))
            pooled_by_method[method][0].extend(test_examples)
            pooled_by_method[method][1].extend(predictions)
            for row, predicted in zip(test_examples, predictions):
                threshold = threshold_for(policy, str(row["compartment_id"]))
                prediction_rows.append({
                    "fold": test_organism,
                    "method": method,
                    "protein_id": row["protein_id"],
                    "compartment_id": row["compartment_id"],
                    "score": row[method],
                    "ground_truth": row["ground_truth"],
                    "threshold": threshold,
                    "predicted": int(predicted),
                })
    operational_policies: Dict[str, object] = {}
    all_examples = [row for rows in by_organism.values() for row in rows]
    for method_index, method in enumerate(methods):
        policy = build_threshold_policy(all_examples, method, target_precision, min_positives)
        operational_policies[method] = policy
        examples, predictions = pooled_by_method[method]
        metric_rows.append(validation_metric_row(
            "POOLED_HELD_OUT", method, examples, predictions, policy,
            bootstrap, seed + 1000 + method_index,
        ))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "interpretation": "Thresholds convert GO CC scores to protein-compartment hypotheses only.",
        "target_precision": target_precision,
        "min_calibration_positives": min_positives,
        "benchmark_clean_status": clean_status,
        "benchmark_organisms": sorted(by_organism),
        "methods": methods,
        "fold_policies": fold_policies,
        "operational_policies": operational_policies,
    }
    write_json(output_dir / "calibration.json", artifact)
    metric_fields = [
        "fold", "method", "global_threshold", "true_positive", "false_positive",
        "false_negative", "predicted_pairs", "true_pairs", "precision", "precision_ci_low",
        "precision_ci_high", "recall", "recall_ci_low", "recall_ci_high", "f1",
        "f1_ci_low", "f1_ci_high", "aupr", "protein_macro_f1",
        "protein_macro_f1_ci_low", "protein_macro_f1_ci_high", "protein_macro_jaccard",
        "protein_macro_jaccard_ci_low", "protein_macro_jaccard_ci_high",
        "protein_coverage", "protein_coverage_ci_low", "protein_coverage_ci_high",
        "abstention_rate", "protein_count", "eligible_protein_count",
    ]
    handle, writer = tsv_writer(output_dir / "validation_metrics.tsv", metric_fields)
    try:
        for row in metric_rows:
            writer.writerow(row)
    finally:
        handle.close()
    prediction_fields = [
        "fold", "method", "protein_id", "compartment_id", "score",
        "ground_truth", "threshold", "predicted",
    ]
    handle, writer = tsv_writer(output_dir / "validation_predictions.tsv", prediction_fields)
    try:
        for row in prediction_rows:
            writer.writerow(row)
    finally:
        handle.close()
    return artifact, metric_rows


def scan_prediction_file(path: Path, mapper: CompartmentMapper) -> Dict[str, object]:
    counts: Dict[str, int] = defaultdict(int)
    proteins: Set[str] = set()
    best: Dict[Tuple[str, str], float] = {}
    for _, protein, go_id, score in iter_s2f_predictions(path):
        counts["input_rows"] += 1
        proteins.add(protein)
        _, status, paths = mapper.map_term(go_id)
        counts[status] += 1
        for item in paths:
            key = (protein, item.compartment_id)
            best[key] = max(score, best.get(key, score))
    digest = hashlib.sha256()
    for protein in sorted(proteins):
        digest.update(f"P\t{protein}\n".encode())
    for (protein, compartment), score in sorted(best.items()):
        digest.update(f"C\t{protein}\t{compartment}\t{score:.17g}\n".encode())
    return {
        "path": str(path),
        "protein_count": len(proteins),
        "protein_compartment_pair_count": len(best),
        "digest": digest.hexdigest(),
        **dict(sorted(counts.items())),
    }


def cafa_regression(mapper: CompartmentMapper, root: Path, max_files: int) -> Dict[str, object]:
    paths = sorted(root.glob("cafa3_*/prediction.df"))
    if max_files > 0:
        paths = paths[:max_files]
    rows = [scan_prediction_file(path, mapper) for path in paths]
    repeated_ok: Optional[bool] = None
    repeated_path: Optional[str] = None
    if paths:
        smallest = min(paths, key=lambda value: value.stat().st_size)
        first = next(row for row in rows if row["path"] == str(smallest))
        second = scan_prediction_file(smallest, mapper)
        repeated_ok = first["digest"] == second["digest"]
        repeated_path = str(smallest)
    return {
        "root": str(root),
        "file_count": len(paths),
        "protein_count": sum(int(row["protein_count"]) for row in rows),
        "input_rows": sum(int(row.get("input_rows", 0)) for row in rows),
        "repeat_determinism_file": repeated_path,
        "repeat_determinism_passed": repeated_ok,
        "files": rows,
    }


def benchmark_gate(metric_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    rows = [
        row for row in metric_rows
        if row["fold"] == "POOLED_HELD_OUT" and row["method"] == "s2f_score"
    ]
    if len(rows) != 1:
        return {"passed": False, "reason": "No unique pooled held-out S2F result"}
    row = rows[0]
    passed = (
        float(row["precision"]) >= 0.90
        and float(row["protein_coverage"]) >= 0.10
        and float(row["predicted_pairs"]) >= 20
    )
    return {
        "passed": passed,
        "reason": (
            "No transferable high-precision S2F calls were produced"
            if float(row["predicted_pairs"]) == 0
            else "One or more precision, coverage, or call-count requirements failed"
        ),
        "precision": row["precision"],
        "protein_coverage": row["protein_coverage"],
        "predicted_pairs": row["predicted_pairs"],
        "requirements": {
            "precision_at_least": 0.90,
            "protein_coverage_at_least": 0.10,
            "predicted_pairs_at_least": 20,
        },
    }


def write_validation_report(
    output: Path,
    artifact: Mapping[str, object],
    metrics: Sequence[Mapping[str, object]],
    regression: Optional[Mapping[str, object]],
    gate: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    pooled = [row for row in metrics if row["fold"] == "POOLED_HELD_OUT"]
    pooled.sort(key=lambda row: str(row["method"]))
    lines = [
        "# CC-to-compartment validation report",
        "",
        "## Interpretation boundary",
        "",
        (
            "These results evaluate protein-compartment hypotheses. They do not "
            "infer EC numbers, reactions, or reaction compartments."
        ),
        "",
        "## Scientific-quality gate",
        "",
        f"- Status: **{'PASS' if gate.get('passed') else 'FAIL'}**",
        f"- S2F held-out precision: {float(gate.get('precision', 0.0)):.4f}",
        f"- S2F protein coverage: {float(gate.get('protein_coverage', 0.0)):.4f}",
        f"- S2F predicted pairs: {int(float(gate.get('predicted_pairs', 0))):,}",
        f"- Decision: {gate.get('reason', 'Quality requirements were not met')}",
        f"- Benchmark leakage status: `{artifact['benchmark_clean_status']}`",
        "",
    ]
    if artifact["benchmark_clean_status"] != "clean":
        lines.extend([
            (
                "> The accuracy result is provisional. Existing benchmark configurations "
                "use computed GOA clamping and require a completed donor/target overlap "
                "audit or a clean rerun before supporting an unbiased S2F-quality claim."
            ),
            "",
        ])
    lines.extend([
        "## Pooled organism-held-out results",
        "",
        "| Method | Precision | Recall | F1 | AUPR | Protein coverage | Predicted pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pooled:
        lines.append(
            f"| {row['method']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['aupr']):.4f} | "
            f"{float(row['protein_coverage']):.4f} | {int(float(row['predicted_pairs'])):,} |"
        )
    if regression is not None:
        lines.extend([
            "",
            "## Local CAFA3 regression",
            "",
            f"- Files scanned: {regression['file_count']}",
            f"- Proteins scanned: {int(regression['protein_count']):,}",
            f"- Prediction rows scanned: {int(regression['input_rows']):,}",
            f"- Repeated-run determinism: {'PASS' if regression['repeat_determinism_passed'] else 'FAIL'}",
            "",
            (
                "CAFA3 is used here for parser, ontology, scale, and determinism checks "
                "only; it is not treated as an unbiased biological-accuracy benchmark."
            ),
        ])
    lines.extend([
        "",
        "## T. cruzi readiness",
        "",
        (
            "The Dm28c S2F output currently lacks `prediction.df`. A Dm28c-specific "
            "quality claim additionally requires a clean reviewed-protein holdout run and "
            "the blinded, life-stage-aware expert reference set described in the project plan."
        ),
        "",
        "## Provenance",
        "",
        f"- Interpreter: `{sys.executable}`",
        f"- GO OBO: `{args.obo}`",
        f"- Compartment slim: `{args.slim}`",
        f"- Generated: `{utc_now()}`",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def calibrate_command(args: argparse.Namespace) -> int:
    mapper = CompartmentMapper.from_files(args.obo, args.slim)
    args.output.mkdir(parents=True, exist_ok=True)
    artifact, metrics = calibrate_benchmarks(
        mapper, args.benchmark_dir, args.output, args.target_precision,
        args.min_calibration_positives, args.bootstrap, args.seed, args.s2f_clean_status,
    )
    gate = benchmark_gate(metrics)
    write_validation_report(args.output, artifact, metrics, None, gate, args)
    return 0 if gate["passed"] else 2


def validate_command(args: argparse.Namespace) -> int:
    mapper = CompartmentMapper.from_files(args.obo, args.slim)
    args.output.mkdir(parents=True, exist_ok=True)
    artifact, metrics = calibrate_benchmarks(
        mapper, args.benchmark_dir, args.output, args.target_precision,
        args.min_calibration_positives, args.bootstrap, args.seed, args.s2f_clean_status,
    )
    regression = None
    if args.reuse_regression is not None:
        regression = read_json(args.reuse_regression)
        regression["reused_from"] = str(args.reuse_regression.resolve())
        write_json(args.output / "cafa3_regression.json", regression)
    elif not args.skip_regression:
        regression = cafa_regression(mapper, args.cafa_root, args.max_regression_files)
        write_json(args.output / "cafa3_regression.json", regression)
    gate = benchmark_gate(metrics)
    software_passed = regression is None or (
        int(regression["file_count"]) > 0 and bool(regression["repeat_determinism_passed"])
    )
    validation = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "software_validation_passed": software_passed,
        "scientific_quality_gate": gate,
        "benchmark_clean_status": args.s2f_clean_status,
        "dm28c_quality_claim_ready": False,
        "dm28c_blockers": [
            "Dm28c prediction.df is absent",
            "A clean S2F holdout run on reviewed localized T. cruzi proteins is not yet complete",
            "A blinded, life-stage-aware Dm28c expert reference set is not yet available",
        ],
        "inputs": {
            "obo": str(args.obo.resolve()),
            "obo_sha256": sha256_file(args.obo),
            "slim": str(args.slim.resolve()),
            "slim_sha256": sha256_file(args.slim),
            "benchmark_dir": str(args.benchmark_dir.resolve()),
            "cafa_root": str(args.cafa_root),
        },
        "interpreter": sys.executable,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output / "run_metadata.json", validation)
    write_validation_report(args.output, artifact, metrics, regression, gate, args)
    return 0 if software_passed and gate["passed"] and args.s2f_clean_status == "clean" else 2


def fetch_uniprot_command(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "uniprot_reviewed_tcruzi.tsv"
    metadata_path = args.output / "uniprot_reviewed_tcruzi.metadata.json"
    if (output_path.exists() or metadata_path.exists()) and not args.force:
        raise ValueError(f"Snapshot already exists in {args.output}; use --force to replace it")
    if args.source_tsv is not None:
        shutil.copyfile(args.source_tsv, output_path)
        source = str(args.source_tsv.resolve())
        query_url = None
    else:
        fields = [
            "accession", "id", "protein_name", "gene_names", "organism_name",
            "length", "sequence", "cc_subcellular_location", "go_c",
        ]
        parameters = {
            "compressed": "false",
            "format": "tsv",
            "query": args.query,
            "fields": ",".join(fields),
        }
        query_url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(parameters)
        request = urllib.request.Request(query_url, headers={"User-Agent": "S2F-CC-compartments/1"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response, output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        source = "UniProt REST API"
    counts = {
        "reviewed_entries": 0,
        "with_subcellular_location": 0,
        "with_experimental_location_eco": 0,
        "with_inferred_location_eco": 0,
    }
    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counts["reviewed_entries"] += 1
            location = " ".join(
                str(value or "") for key, value in row.items()
                if "subcellular location" in str(key).lower()
            )
            if location.strip():
                counts["with_subcellular_location"] += 1
            if any(code in location for code in EXPERIMENTAL_ECO):
                counts["with_experimental_location_eco"] += 1
            if any(code in location for code in INFERRED_ECO):
                counts["with_inferred_location_eco"] += 1
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_at_utc": utc_now(),
        "source": source,
        "query": args.query,
        "query_url": query_url,
        "snapshot": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "counts": counts,
        "note": "Raw reviewed evidence snapshot. It is not an automatically mapped Dm28c gold standard.",
    }
    write_json(metadata_path, metadata)
    return 0


def audit_target_command(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    uniprot_by_sequence: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with args.uniprot_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        accession_field = "Entry" if "Entry" in fields else "accession"
        sequence_field = "Sequence" if "Sequence" in fields else "sequence"
        if accession_field not in fields or sequence_field not in fields:
            raise ValueError(
                f"{args.uniprot_tsv} must contain Entry/Sequence or accession/sequence columns"
            )
        for row in reader:
            sequence = str(row.get(sequence_field) or "").replace(" ", "").upper()
            if sequence:
                uniprot_by_sequence[sequence].append({
                    "accession": str(row.get(accession_field) or ""),
                    "entry_name": str(row.get("Entry Name") or row.get("id") or ""),
                    "protein_name": str(row.get("Protein names") or row.get("protein_name") or ""),
                    "subcellular_location": str(
                        row.get("Subcellular location [CC]")
                        or row.get("cc_subcellular_location") or ""
                    ),
                    "go_cellular_component": str(
                        row.get("Gene Ontology (cellular component)") or row.get("go_c") or ""
                    ),
                })
    target_count = 0
    exact_target_count = 0
    matched_accessions: Set[str] = set()
    rows: List[Dict[str, object]] = []
    for target_id, sequence in iter_fasta(args.target_fasta):
        target_count += 1
        matches = uniprot_by_sequence.get(sequence, [])
        if matches:
            exact_target_count += 1
        for match in matches:
            matched_accessions.add(match["accession"])
            rows.append({
                "target_protein_id": target_id,
                "uniprot_accession": match["accession"],
                "uniprot_entry_name": match["entry_name"],
                "sequence_length": len(sequence),
                "match_type": "exact_sequence",
                "match_ambiguity": len(matches),
                "protein_name": match["protein_name"],
                "subcellular_location": match["subcellular_location"],
                "go_cellular_component": match["go_cellular_component"],
            })
    fields = [
        "target_protein_id", "uniprot_accession", "uniprot_entry_name",
        "sequence_length", "match_type", "match_ambiguity", "protein_name",
        "subcellular_location", "go_cellular_component",
    ]
    handle, writer = tsv_writer(args.output / "exact_sequence_matches.tsv", fields)
    try:
        for row in sorted(rows, key=lambda item: (str(item["target_protein_id"]), str(item["uniprot_accession"]))):
            writer.writerow(row)
    finally:
        handle.close()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "target_fasta": str(args.target_fasta.resolve()),
        "target_fasta_sha256": sha256_file(args.target_fasta),
        "uniprot_tsv": str(args.uniprot_tsv.resolve()),
        "uniprot_tsv_sha256": sha256_file(args.uniprot_tsv),
        "target_protein_count": target_count,
        "exactly_matched_target_protein_count": exact_target_count,
        "matched_uniprot_accession_count": len(matched_accessions),
        "exact_match_row_count": len(rows),
        "interpretation": (
            "Exact sequence identity is a conservative cross-database link. "
            "It does not establish that the UniProt evidence is Dm28c-specific or life-stage complete."
        ),
    }
    write_json(args.output / "target_sequence_audit.json", metadata)
    return 0


def add_common_mapping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO)
    parser.add_argument("--slim", type=Path, default=DEFAULT_SLIM)


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_mapping_arguments(parser)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--min-calibration-positives", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--s2f-clean-status",
        choices=["provisional", "clean", "known_overlap"],
        default="provisional",
        help="Use clean only after an explicit donor/target and clamp leakage audit.",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assign_parser = subparsers.add_parser("assign", help="Map an S2F prediction.df to compartments")
    add_common_mapping_arguments(assign_parser)
    assign_parser.add_argument("--prediction", type=Path, required=True)
    assign_parser.add_argument("--output", type=Path, required=True)
    assign_parser.add_argument("--calibration", type=Path)
    assign_parser.add_argument("--threshold", type=float)
    assign_parser.add_argument("--protein-list", type=Path)
    assign_parser.add_argument("--curated-evidence", type=Path)
    assign_parser.add_argument("--include-non-cc-evidence", action="store_true")
    assign_parser.set_defaults(func=assign_command)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate and evaluate on local held-out organisms")
    add_calibration_arguments(calibrate_parser)
    calibrate_parser.set_defaults(func=calibrate_command)

    validate_parser = subparsers.add_parser("validate", help="Run benchmark and CAFA3 regression validation")
    add_calibration_arguments(validate_parser)
    validate_parser.add_argument("--cafa-root", type=Path, default=DEFAULT_CAFA_ROOT)
    validate_parser.add_argument("--skip-regression", action="store_true")
    validate_parser.add_argument(
        "--reuse-regression",
        type=Path,
        help="Reuse a prior cafa3_regression.json when only report/calibration code changed.",
    )
    validate_parser.add_argument("--max-regression-files", type=int, default=0)
    validate_parser.set_defaults(func=validate_command)

    fetch_parser = subparsers.add_parser("fetch-uniprot", help="Freeze reviewed T. cruzi localization evidence")
    fetch_parser.add_argument("--output", type=Path, required=True)
    fetch_parser.add_argument(
        "--query",
        default='(organism_name:"Trypanosoma cruzi") AND (reviewed:true)',
    )
    fetch_parser.add_argument("--source-tsv", type=Path, help="Freeze an existing TSV instead of using the network")
    fetch_parser.add_argument("--timeout", type=int, default=120)
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(func=fetch_uniprot_command)

    audit_parser = subparsers.add_parser(
        "audit-target", help="Audit exact sequence overlap between reviewed UniProt and a target FASTA"
    )
    audit_parser.add_argument("--uniprot-tsv", type=Path, required=True)
    audit_parser.add_argument("--target-fasta", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(func=audit_target_command)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(args, "target_precision") and not 0.0 < args.target_precision <= 1.0:
        raise ValueError("--target-precision must be in (0, 1]")
    if hasattr(args, "min_calibration_positives") and args.min_calibration_positives < 1:
        raise ValueError("--min-calibration-positives must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
