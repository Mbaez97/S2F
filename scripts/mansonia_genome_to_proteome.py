#!/usr/bin/env python3
"""Build S2F protein inputs from the Mansonia genome assemblies by protein-homology gene prediction.

The assemblies in ``data/mansonia/genoma_Mansonia_pos_montagem/`` are nucleotide contigs with no gene
models, while every S2F stage (``[configuration] fasta``, InterPro, HMMER, STRING homology) needs proteins.
This script derives a proteome per species by aligning mosquito reference proteomes (NCBI RefSeq
*Aedes aegypti*, *Culex quinquefasciatus*, *Anopheles gambiae*; one isoform per gene) to each genome
with miniprot, then keeping one non-overlapping, frameshift-free, stop-free gene model per locus.

Stages (each skips work whose output already exists unless ``--force``):

    prepare-refs  longest isoform per reference gene -> refs/mosquito_refs.longest_isoform.faa
    align         miniprot index + spliced protein alignment per species
    select        filter + resolve overlapping models -> <species>/<species>_proteome.faa (+ GFF, TSV, JSON)
    busco         BUSCO protein mode on each proteome (docker image ezlabgva/busco)
    config        write conf/mansonia_<species>_proteome.conf for ``S2F.py predict --run-config``
    all           every stage in order

Genes without a detectable mosquito homolog are not recovered by this approach.
"""

import argparse
import datetime as dt
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

S2F_ROOT = Path(__file__).resolve().parents[1]
MANSONIA_DIR = S2F_ROOT / 'data' / 'mansonia'
GENOME_DIR = MANSONIA_DIR / 'genoma_Mansonia_pos_montagem'
ANNOT_DIR = MANSONIA_DIR / 'annotation'
BIN_DIR = ANNOT_DIR / 'bin'
REFS_DIR = ANNOT_DIR / 'refs'

GENOMES = {
    'humeralis': GENOME_DIR / 'genome_purged_Ma_humeralis_posCBS.fasta',
    'titillans': GENOME_DIR / 'genome_purged_Ma_titillans_posCBS.fasta',
}
ID_PREFIX = {'humeralis': 'Mhum', 'titillans': 'Mtit'}
REFERENCES = {
    'Aedes_aegypti': 'GCF_002204515.2',
    'Culex_quinquefasciatus': 'GCF_015732765.1',
    'Anopheles_gambiae': 'GCF_943734735.2',
}
REF_FASTA = REFS_DIR / 'mosquito_refs.longest_isoform.faa'
REF_TABLE = REFS_DIR / 'reference_proteins.tsv'
BUSCO_IMAGE = 'ezlabgva/busco:v6.1.0_cv3'
BUSCO_LINEAGE = 'diptera_odb12'


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def read_fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    name, header, chunks = None, None, []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if name is not None:
                    yield name, header, ''.join(chunks)
                header = line[1:].rstrip()
                name, chunks = header.split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, header, ''.join(chunks)


def write_fasta(fh, name, seq, width=60):
    fh.write(f'>{name}\n')
    for i in range(0, len(seq), width):
        fh.write(seq[i:i + width] + '\n')


def run(cmd, log_path, stdout_path=None):
    log('running: ' + ' '.join(str(c) for c in cmd))
    with open(log_path, 'a') as err:
        err.write(f'\n# {dt.datetime.now().isoformat()} {" ".join(map(str, cmd))}\n')
        err.flush()
        out = open(stdout_path, 'w') if stdout_path else err
        try:
            subprocess.run([str(c) for c in cmd], stdout=out, stderr=err, check=True)
        finally:
            if stdout_path:
                out.close()


# --------------------------------------------------------------------------- prepare-refs

def prepare_refs(args):
    if REF_FASTA.exists() and not args.force:
        log(f'{REF_FASTA.name} exists, skipping')
        return
    rows = []
    with open(REF_FASTA, 'w') as out:
        for species, acc in REFERENCES.items():
            data = REFS_DIR / species / 'ncbi_dataset' / 'data' / acc
            prot_gene, product = {}, {}
            with open(data / 'genomic.gff') as fh:
                for line in fh:
                    f = line.rstrip('\n').split('\t')
                    if len(f) < 9 or f[2] != 'CDS':
                        continue
                    attrs = dict(kv.split('=', 1) for kv in f[8].split(';') if '=' in kv)
                    pid = attrs.get('protein_id')
                    gid = re.search(r'GeneID:(\d+)', attrs.get('Dbxref', ''))
                    if pid and gid:
                        prot_gene[pid] = gid.group(1)
                        product[pid] = attrs.get('product', '')
            best = {}
            for pid, _, seq in read_fasta(data / 'protein.faa'):
                gene = prot_gene.get(pid)
                if gene is None:
                    continue
                if gene not in best or len(seq) > len(best[gene][1]):
                    best[gene] = (pid, seq)
            for gene, (pid, seq) in sorted(best.items()):
                name = f'{species}|{pid}'
                write_fasta(out, name, seq.rstrip('*'))
                rows.append((name, species, acc, gene, len(seq), product[pid]))
            log(f'{species} ({acc}): {len(prot_gene)} CDS-linked proteins -> {len(best)} genes (longest isoform)')
    with open(REF_TABLE, 'w') as fh:
        fh.write('ref_id\tspecies\tassembly\tgene_id\tlength\tproduct\n')
        for r in rows:
            fh.write('\t'.join(map(str, r)) + '\n')
    log(f'wrote {len(rows)} reference proteins to {REF_FASTA}')


# --------------------------------------------------------------------------- align

def species_dir(sp):
    d = ANNOT_DIR / sp
    d.mkdir(parents=True, exist_ok=True)
    return d


def align(args):
    miniprot = BIN_DIR / 'miniprot'
    for sp in args.species:
        d = species_dir(sp)
        index, gff = d / f'{sp}.mpi', d / f'{sp}.miniprot.gff'
        log_path = ANNOT_DIR / 'logs' / f'{sp}.miniprot.log'
        if not index.exists() or args.force:
            run([miniprot, '-t', args.threads, '-d', index, GENOMES[sp]], log_path)
        if not gff.exists() or args.force:
            tmp = gff.with_suffix('.gff.partial')
            run([miniprot, '-t', args.threads, '-I', '--gff', '--trans', index, REF_FASTA], log_path, tmp)
            tmp.rename(gff)
        else:
            log(f'{gff.name} exists, skipping')


# --------------------------------------------------------------------------- select

def parse_miniprot(path):
    """Yield one dict per alignment from miniprot ``--gff --trans`` output."""
    paf = sta = rec = None
    with open(path) as fh:
        for line in fh:
            if line.startswith('##PAF'):
                if rec:
                    yield rec
                rec, sta = None, None
                f = line.rstrip('\n').split('\t')
                tags = dict(t.split(':', 2)[::2] for t in f[13:] if t.count(':') >= 2)
                paf = {'qlen': int(f[2]), 'qs': int(f[3]), 'qe': int(f[4]),
                       'fs': int(tags.get('fs', 0)), 'st': int(tags.get('st', 0))}
            elif line.startswith('##STA'):
                sta = line.rstrip('\n').split('\t', 1)[1]
            elif line.startswith('#'):
                continue
            else:
                f = line.rstrip('\n').split('\t')
                if len(f) < 9:
                    continue
                if f[2] == 'mRNA':
                    attrs = dict(kv.split('=', 1) for kv in f[8].split(';') if '=' in kv)
                    rec = {'contig': f[0], 'start': int(f[3]), 'end': int(f[4]), 'score': float(f[5]),
                           'strand': f[6], 'mp_id': attrs['ID'], 'rank': int(attrs.get('Rank', 0)),
                           'identity': float(attrs.get('Identity', 0)), 'positive': float(attrs.get('Positive', 0)),
                           'target': attrs['Target'].split()[0], 'cds': [], 'has_stop': False,
                           'protein': sta or '', **paf}
                    rec['ref_coverage'] = (paf['qe'] - paf['qs']) / paf['qlen']
                elif rec and f[2] == 'CDS':
                    rec['cds'].append((int(f[3]), int(f[4]), f[7]))
                elif rec and f[2] == 'stop_codon':
                    rec['has_stop'] = True
    if rec:
        yield rec


def overlaps(accepted, key, intervals):
    starts, ends = accepted[key]
    for s, e in intervals:
        i = bisect_left(starts, s)
        if i > 0 and ends[i - 1] >= s:
            return True
        if i < len(starts) and starts[i] <= e:
            return True
    return False


def add_intervals(accepted, key, intervals):
    starts, ends = accepted[key]
    for s, e in intervals:
        i = bisect_left(starts, s)
        starts.insert(i, s)
        ends.insert(i, e)


def select(args):
    ref_species = {}
    with open(REF_TABLE) as fh:
        next(fh)
        for line in fh:
            f = line.split('\t')
            ref_species[f[0]] = f[1]

    for sp in args.species:
        d = species_dir(sp)
        gff_in = d / f'{sp}.miniprot.gff'
        faa = d / f'{sp}_proteome.faa'
        if faa.exists() and not args.force:
            log(f'{faa.name} exists, skipping')
            continue
        reasons = Counter()
        kept = []
        for rec in parse_miniprot(gff_in):
            reasons['alignments'] += 1
            prot = rec['protein'].rstrip('*')
            if rec['fs'] > 0:
                reasons['rejected_frameshift'] += 1
            elif rec['st'] > 0 or '*' in prot:
                reasons['rejected_internal_stop'] += 1
            elif rec['identity'] < args.min_identity:
                reasons['rejected_identity'] += 1
            elif rec['ref_coverage'] < args.min_ref_coverage:
                reasons['rejected_ref_coverage'] += 1
            elif len(prot) < args.min_length:
                reasons['rejected_short'] += 1
            else:
                rec['protein'] = prot
                kept.append(rec)
        reasons['passed_filters'] = len(kept)

        # Greedy locus resolution: best-scoring model first; a model is dropped if any of its CDS
        # overlaps a CDS already accepted on the same contig and strand.
        kept.sort(key=lambda r: (-r['score'], -r['positive'], r['mp_id']))
        accepted = defaultdict(lambda: ([], []))
        genes = []
        for rec in kept:
            key = (rec['contig'], rec['strand'])
            ivs = [(s, e) for s, e, _ in rec['cds']]
            if overlaps(accepted, key, ivs):
                continue
            add_intervals(accepted, key, ivs)
            genes.append(rec)
        reasons['gene_models'] = len(genes)

        genes.sort(key=lambda r: (r['contig'], r['start']))
        seq_count = Counter(r['protein'] for r in genes)
        with open(faa, 'w') as fa, open(d / f'{sp}_gene_models.gff3', 'w') as gff, \
                open(d / f'{sp}_gene_models.tsv', 'w') as tsv:
            gff.write('##gff-version 3\n')
            tsv.write('protein_id\tcontig\tstart\tend\tstrand\tn_exons\tprotein_len\tref_id\tref_species\t'
                      'identity\tpositive\tref_coverage\tscore\thas_start_M\thas_stop\tidentical_seq_copies\n')
            for i, r in enumerate(genes, 1):
                pid = f'{ID_PREFIX[sp]}_g{i:06d}'
                write_fasta(fa, pid, r['protein'])
                gff.write(f"{r['contig']}\tminiprot\tgene\t{r['start']}\t{r['end']}\t{r['score']:.0f}\t{r['strand']}\t.\t"
                          f"ID={pid}.gene\n")
                gff.write(f"{r['contig']}\tminiprot\tmRNA\t{r['start']}\t{r['end']}\t{r['score']:.0f}\t{r['strand']}\t.\t"
                          f"ID={pid};Parent={pid}.gene;Target={r['target']};Identity={r['identity']}\n")
                for s, e, phase in r['cds']:
                    gff.write(f"{r['contig']}\tminiprot\tCDS\t{s}\t{e}\t.\t{r['strand']}\t{phase}\tParent={pid}\n")
                tsv.write('\t'.join(map(str, [
                    pid, r['contig'], r['start'], r['end'], r['strand'], len(r['cds']), len(r['protein']),
                    r['target'], ref_species.get(r['target'], ''), r['identity'], r['positive'],
                    round(r['ref_coverage'], 4), r['score'], r['protein'].startswith('M'), r['has_stop'],
                    seq_count[r['protein']]])) + '\n')

        lengths = sorted(len(r['protein']) for r in genes)
        summary = {
            'species': sp,
            'date': dt.datetime.now().isoformat(timespec='seconds'),
            'interpreter': sys.executable,
            'genome': str(GENOMES[sp]),
            'references': REFERENCES,
            'reference_fasta': str(REF_FASTA),
            'miniprot_gff': str(gff_in),
            'filters': {'min_identity': args.min_identity, 'min_ref_coverage': args.min_ref_coverage,
                        'min_length': args.min_length, 'frameshifts': 0, 'internal_stops': 0},
            'counts': dict(reasons),
            'median_protein_len': lengths[len(lengths) // 2] if lengths else 0,
            'complete_models_M_and_stop': sum(r['protein'].startswith('M') and r['has_stop'] for r in genes),
            'proteins_with_identical_copy': sum(1 for r in genes if seq_count[r['protein']] > 1),
            'distinct_ref_targets': len({r['target'] for r in genes}),
            'best_ref_species': dict(Counter(ref_species.get(r['target'], '') for r in genes)),
        }
        (d / f'{sp}_proteome.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        log(f'{sp}: ' + json.dumps(summary['counts']))


# --------------------------------------------------------------------------- busco

def busco(args):
    if not shutil.which('docker'):
        sys.exit('docker not found; BUSCO stage needs it')
    uid = f'{os.getuid()}:{os.getgid()}'
    for sp in args.species:
        out_name = f'busco_{sp}_proteome'
        if (ANNOT_DIR / out_name).exists() and not args.force:
            log(f'{out_name} exists, skipping')
            continue
        cmd = ['docker', 'run', '--rm', '-u', uid, '-v', f'{ANNOT_DIR}:/busco_wd', '-w', '/busco_wd', BUSCO_IMAGE,
               'busco', '-i', f'{sp}/{sp}_proteome.faa', '-m', 'proteins', '-l', BUSCO_LINEAGE,
               '-c', args.threads, '-o', out_name, '--download_path', 'busco_downloads', '-f']
        run(cmd, ANNOT_DIR / 'logs' / f'{sp}.busco.log')


# --------------------------------------------------------------------------- config

def config(args):
    for sp in args.species:
        template = S2F_ROOT / 'conf' / f'mansonia_{sp}.conf'
        target = S2F_ROOT / 'conf' / f'mansonia_{sp}_proteome.conf'
        if target.exists() and not args.force:
            log(f'{target.name} exists, skipping')
            continue
        text = template.read_text()
        text = re.sub(r'^alias = .*$', f'alias = mansonia_{sp}_proteome', text, count=1, flags=re.M)
        text = re.sub(r'^fasta = .*$', f'fasta = {ANNOT_DIR / sp / f"{sp}_proteome.faa"}', text, count=1, flags=re.M)
        header = (f'# Generated {dt.date.today()} by scripts/mansonia_genome_to_proteome.py from {template.name}.\n'
                  f'# fasta = miniprot homology-based proteome of {GENOMES[sp].name}\n')
        target.write_text(header + text)
        log(f'wrote {target}')


STAGES = {'prepare-refs': prepare_refs, 'align': align, 'select': select, 'busco': busco, 'config': config}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=[*STAGES, 'all'])
    p.add_argument('--species', nargs='+', choices=list(GENOMES), default=list(GENOMES))
    p.add_argument('--threads', type=int, default=24)
    p.add_argument('--min-identity', type=float, default=0.35)
    p.add_argument('--min-ref-coverage', type=float, default=0.6)
    p.add_argument('--min-length', type=int, default=50)
    p.add_argument('--force', action='store_true', help='recompute outputs that already exist')
    args = p.parse_args()
    (ANNOT_DIR / 'logs').mkdir(parents=True, exist_ok=True)
    for name in (STAGES if args.stage == 'all' else [args.stage]):
        log(f'=== stage {name}')
        STAGES[name](args)


if __name__ == '__main__':
    main()
