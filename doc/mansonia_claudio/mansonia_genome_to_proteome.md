# Mansonia Genome to Proteome for S2F

## Summary

S2F predicts GO terms for **proteins**. Every stage of `S2F.py predict` works on amino-acid sequences:
the run config's `[configuration] fasta`, the InterPro and HMMER seeds, and the BLASTP search against
STRING v12 used to build the homology graph and transfer interactions.

The assemblies we received for *Mansonia humeralis* and *Mansonia titillans* are **nucleotide
contigs with no gene models**. This document describes how a protein FASTA (a proteome) was derived from each
genome by **protein-homology gene prediction**, which quality controls were applied, and what the resulting
proteomes can and cannot support.

| | *Ma. humeralis* | *Ma. titillans* |
|---|---|---|
| Proteins in the S2F FASTA | **12,245** | **13,700** |
| BUSCO `diptera_odb12`, proteome | C:75.2% [S:71.9%, D:3.3%], F:5.4%, M:19.5% | C:82.4% [S:76.9%, D:5.5%], F:3.8%, M:13.8% |
| BUSCO `diptera_odb12`, genome | C:78.1% [S:74.4%, D:3.7%], F:8.7%, M:13.3% | C:83.9% [S:78.1%, D:5.9%], F:5.7%, M:10.4% |
| Earlier 283 candidate proteins present (by sequence) | 224 (79%) | 241 (85%) |
| S2F run config | `conf/mansonia_humeralis_proteome.conf` | `conf/mansonia_titillans_proteome.conf` |

Run date: 2026-09-28. Implementation: `scripts/mansonia_genome_to_proteome.py`. All outputs are under
`data/mansonia/annotation/`, and that directory's `README.md` is the short version of this document.

## Scientific context

The research question is whether S2F can produce useful, high-confidence functional hypotheses for organisms
with almost no experimental annotation. *Mansonia* fits this case well:

- NCBI Datasets has no nuclear genome assembly for any *Mansonia* species (queried 2026-09-28); only
  mitochondrial genomes are public (e.g. *Ma. uniformis*, 15,603 bp).
- Both species have a 2n = 6 karyotype (Rondonia populations, PMC9991104).
- GOA has no experimental GO annotations for them.

The proteomes described here are therefore *predicted* gene products. S2F predictions made on them are
hypotheses to prioritize experimental validation, not a replacement for it.

## Inputs

### Genome assemblies (read only)

| file | md5 | contigs | size | N50 |
|---|---|---|---|---|
| `data/mansonia/genoma_Mansonia_pos_montagem/genome_purged_Ma_humeralis_posCBS.fasta` | `de9419eb974e2ad98feb417149595c55` | 31,795 | 2.80 Gb | 0.14 Mb |
| `data/mansonia/genoma_Mansonia_pos_montagem/genome_purged_Ma_titillans_posCBS.fasta` | `49c6d34911d50c51bc23a550dc082590` | 19,412 | 3.30 Gb | 0.26 Mb |

Characterization of the raw assemblies is in `notebooks/mansonia_genome_s2f_readiness.ipynb`:

- **Assembler.** hifiasm primary contigs (`ptg######l` = linear, `c` = circular), so PacBio HiFi reads.
- **Contigs only.** No gaps (`N` = 0) and no IUPAC or amino-acid characters.
- **Not repeat-masked.** There are no lowercase bases.
- **Processed after assembly.** The "purged" name and the `_N` split suffixes indicate haplotig purging and a
  contamination/adaptor screen. The meaning of `posCBS` has not been confirmed with the data provider.

### Reference proteomes

Downloaded 2026-09-28 from the NCBI Datasets v2 API (`include_annotation_type=PROT_FASTA,GENOME_GFF`) into
`data/mansonia/annotation/refs/<species>/`:

| species | assembly | annotation release | protein-coding genes | `protein.faa` md5 |
|---|---|---|---|---|
| *Aedes aegypti* | GCF_002204515.2 (AaegL5.0) | NCBI Annotation Release 101 | 14,626 | `59a86cb7bf739864387fec83ea89bf95` |
| *Culex quinquefasciatus* | GCF_015732765.1 (VPISU_Cqui_1.0_pri_paternal) | NCBI Annotation Release 100 | 15,094 | `15f879bf187156ad7a91e8875cfc698b` |
| *Anopheles gambiae* | GCF_943734735.2 (idAnoGambNW_F1_1) | GCF_943734735.2-RS_2023_12 | 12,519 | `e61681a070ef4a624d3ef6bd4ebea6e0` |

These are the same three species and RefSeq annotations used to extract the earlier 283-protein candidate
sets in `data/mansonia/mansonia_candidate_proteins/`: 261 of 283 (humeralis) and 257 of 283 (titillans)
`best_hit_qid` accessions in those extraction summaries are the longest-isoform proteins selected here; the rest are other
isoforms of the same releases. *Aedes* and *Culex* belong to Culicinae like *Mansonia*; *Anopheles* is a more distant
outgroup (Anophelinae).

## Why protein-homology gene prediction

No RNA-seq or Iso-Seq exists for either species, and no gene-prediction software was installed. The options considered
were:

| approach | pros | cons |
|---|---|---|
| **Protein homology (miniprot), chosen** | hours instead of days; no repeat masking or training; each model is backed by a mosquito homolog | misses genes with no detectable mosquito homolog, including lineage-specific genes |
| De novo + protein evidence (RepeatModeler2/RepeatMasker, then BRAKER2/GALBA) | finds more genes, including novel ones | several days of compute per 3 Gb genome; heavy install (AUGUSTUS training, GeneMark licence); more false-positive models without RNA-seq |

For S2F the homology bias costs less than it might seem: S2F seeds come from InterPro, HMMER and homology to
annotated proteins, so a protein with no mosquito homolog would usually receive weak or no seeds anyway. It still
has to be stated when interpreting results: **the proteome is not the full gene complement.**

## Software

| tool | version | source |
|---|---|---|
| miniprot | 0.18-r281 | official Linux binary, `github.com/lh3/miniprot` release v0.18 -> `data/mansonia/annotation/bin/miniprot` |
| BUSCO | 6.1.0 (hmmsearch 3.4) | docker image `ezlabgva/busco:v6.1.0_cv3` |
| BUSCO lineage | `diptera_odb12` (created 2026-06-16; 76 genomes; 5,067 BUSCOs) | downloaded by BUSCO into `data/mansonia/annotation/busco_downloads/` |
| Python | 3.9, `/home/marcelo_baez/anaconda3/bin/python3` | standard library only (the pipeline does not import pandas) |

`gffread` 0.12.9 was downloaded to `bin/`, but the pipeline doesn't use it: protein sequences come directly
from miniprot's `--trans` output. No conda environment was created or modified.

Machine: 28 CPUs, 123 GB RAM. Peak memory was below 20 GB.

## Pipeline

Each stage is a subcommand of `scripts/mansonia_genome_to_proteome.py`. A stage skips work whose output already
exists unless `--force` is given, and `all` runs every stage in order.

```text
 NCBI RefSeq proteomes (Aedes, Culex, Anopheles)
        |  prepare-refs: longest isoform per gene
        v
 mosquito_refs.longest_isoform.faa (42,239 proteins)
        |                                   genome FASTA
        |                                        | align: miniprot -d (index)
        +-----------> align: miniprot -I --gff --trans <-----+
                                   |
                        <species>.miniprot.gff (all alignments + translations)
                                   | select: filters + locus resolution
                                   v
          <species>_proteome.faa  +  gene_models.gff3/.tsv  +  summary.json
                 |                                   |
                 | busco (protein mode)              | config
                 v                                   v
          busco_<species>_proteome/        conf/mansonia_<species>_proteome.conf
   (busco-genome: BUSCO genome mode on the raw assembly -> busco_<species>_genome/)
```

### 1. `prepare-refs`: non-redundant reference set

RefSeq `protein.faa` lists every isoform (28,317 / 24,531 / 30,505 proteins). Aligning all of them would place many
near-identical models on the same locus and bias locus resolution toward genes with many isoforms. For each species:

1. Parse `genomic.gff` CDS features and map `protein_id` to gene through `Dbxref=GeneID:<n>`.
2. Keep the **longest protein per GeneID** and strip the terminal `*`.
3. Prefix IDs with the species name: `Aedes_aegypti|XP_001654215.1`.

Result: `refs/mosquito_refs.longest_isoform.faa` with 14,626 + 15,094 + 12,519 = **42,239 proteins**
(md5 `8e023c637543e0775760ba53653daa5c`). The counts equal NCBI's protein-coding gene counts.
`refs/reference_proteins.tsv` maps each ID to species, assembly, GeneID, length and product.

### 2. `align`: spliced protein-to-genome alignment

```bash
miniprot -t 24 -d <species>/<species>.mpi <genome.fasta>
miniprot -t 24 -I --gff --trans <species>/<species>.mpi refs/mosquito_refs.longest_isoform.faa > <species>/<species>.miniprot.gff
```

- `-I` sets the maximum intron length from the genome size (about 3.6 * sqrt(genome size)). The 2.8 to 3.3 Gb assemblies contain long mosquito introns;
  earlier candidate loci spanned 30 to 57 kb.
- `--gff` writes one `mRNA` plus its `CDS`/`stop_codon` features per alignment, preceded by a `##PAF` line with
  frameshift (`fs:i:`) and in-frame stop (`st:i:`) counts.
- `--trans` adds a `##STA` line holding the **translation of the genome along the alignment**. This is the
  *Mansonia* protein sequence. It is not a copy of the reference.
- miniprot defaults were kept for secondary alignments, so one reference protein can align to several paralogous loci.

| | index time | index peak RAM | alignment time | alignment peak RAM | alignments |
|---|---|---|---|---|---|
| humeralis | 158 s | 15.6 GB | 298 s | 16.9 GB | 56,626 |
| titillans | 155 s | 16.8 GB | 359 s | 19.8 GB | 57,687 |

### 3. `select`: filtering and locus resolution

**Per-alignment filters** (defaults, all command-line options):

| rule | default | reason |
|---|---|---|
| no frameshift (`fs = 0`) | fixed | frameshifted models give chimeric proteins; they may be pseudogenes or assembly errors |
| no in-frame stop (`st = 0` and no internal `*`) | fixed | same reason; S2F tools expect clean ORFs |
| identity to reference | `--min-identity 0.35` | removes weak or spurious alignments while keeping distant *Anopheles* orthologs (median *Anopheles* identity in kept models is 0.57) |
| reference coverage (aligned span / reference length) | `--min-ref-coverage 0.6` | removes small fragments; keeps partial models caused by contig breaks |
| protein length | `--min-length 50` aa | removes very short peptides |

**Locus resolution.** After filtering, several alignments usually cover the same locus: orthologs from the three
references plus secondary hits. Models are sorted by miniprot alignment score (ties broken by `Positive`, then ID)
and accepted greedily. A model is **rejected if any of its CDS overlaps a CDS already accepted on the same contig
and strand**. Overlaps on the opposite strand are allowed, since nested and antisense genes exist. This leaves one protein
per locus. Paralogs at different loci are kept because they are separate genes.

**Naming.** Accepted models are sorted by contig and start, then numbered `Mhum_g000001...` / `Mtit_g000001...`.
The IDs are simple tokens with no `|`, spaces or coordinates, so they work with S2F's `fasta_id_parser = entire_id` and
with the InterProScan and BLAST ID handling. Coordinates and evidence stay in the side table.

Filter results:

| | humeralis | titillans |
|---|---|---|
| alignments | 56,626 | 57,687 |
| rejected: frameshift | 2,925 | 2,287 |
| rejected: internal stop | 702 | 784 |
| rejected: reference coverage | 4,617 | 3,327 |
| rejected: identity | 968 | 1,038 |
| rejected: length | 33 | 34 |
| passed filters | 47,381 | 50,217 |
| **gene models after locus resolution** | **12,245** | **13,700** |

### 4. `busco` and `busco-genome`: completeness

```bash
docker run --rm -u $(id -u):$(id -g) -v data/mansonia/annotation:/busco_wd -w /busco_wd ezlabgva/busco:v6.1.0_cv3 \
  busco -i <species>/<species>_proteome.faa -m proteins -l diptera_odb12 -c 24 -o busco_<species>_proteome --download_path busco_downloads -f
# genome mode: the genome directory is mounted read-only at /genomes; -m genome (BUSCO's own miniprot-based pipeline)
```

Protein mode took about 4 minutes per species; genome mode took 26 min (humeralis) and 30 min (titillans).
Comparing the two modes separates **assembly incompleteness** (genes absent from the genome) from
**annotation losses** (genes present in the genome but not recovered by this pipeline).

### 5. `config`: S2F run configs

`conf/mansonia_<species>_proteome.conf` is a copy of the existing `conf/mansonia_<species>.conf`, used for the
283-candidate runs. Only two keys change:

```ini
alias = mansonia_<species>_proteome
fasta = .../data/mansonia/annotation/<species>/<species>_proteome.faa
```

Seeds (`interpro_output = compute`, `hmmer_output = compute`, `alpha = 0.9`, `beta = 0.1`, Foldseek/PLM `skip`),
graphs (`compute`) and every other setting are inherited unchanged, so the whole-proteome and candidate runs stay
comparable. The new alias keeps them in separate output directories under `/run/media/marcelo_baez/HD_Disc1/.S2F/output/`.

## Outputs

`data/mansonia/annotation/<species>/`:

| file | content |
|---|---|
| `<species>_proteome.faa` | **S2F input**. humeralis md5 `2f21110596abaa37ebf1ee9358762ed6`, titillans md5 `2dec03970fc53f32abf7b5551497241a` |
| `<species>_gene_models.gff3` | `gene` / `mRNA` / `CDS` features. The mRNA carries `Target=<reference>` and `Identity` |
| `<species>_gene_models.tsv` | one row per protein (columns below) |
| `<species>_proteome.summary.json` | provenance (date, interpreter, inputs, filters) and all counts |
| `<species>.miniprot.gff` | every raw alignment with its translation; rerun `select` with other thresholds without realigning |
| `<species>.mpi` | miniprot index (6.3 / 7.0 GB); can be deleted and rebuilt in about 3 min |

`<species>_gene_models.tsv` columns: `protein_id`, `contig`, `start`, `end`, `strand`, `n_exons`, `protein_len`,
`ref_id`, `ref_species`, `identity`, `positive`, `ref_coverage`, `score`, `has_start_M`, `has_stop`,
`identical_seq_copies`.

## Quality assessment

| | humeralis | titillans |
|---|---|---|
| distinct reference genes represented | 10,235 | 11,011 |
| median protein length (aa) | 388 | 402 |
| median reference coverage (share >= 0.95) | 0.99 (68%) | 1.00 (73%) |
| models with start M and stop codon | 3,707 (30%) | 4,337 (32%) |
| proteins with an identical copy at another locus | 1,027 | 1,239 |
| best reference: Aedes / Culex / Anopheles | 7,120 / 4,378 / 747 | 7,967 / 4,888 / 845 |
| median identity by best reference (Aedes / Culex / Anopheles) | 0.76 / 0.69 / 0.57 | 0.76 / 0.67 / 0.54 |
| earlier candidate loci overlapping a new model | 233 / 283 | 256 / 283 |
| earlier candidate proteins present by sequence (next section) | 224 / 283 | 241 / 283 |

How to read these numbers:

1. **Annotation losses are small.** Genome-mode BUSCO completeness is only 2.9 (humeralis) and 1.5 (titillans)
   points above protein-mode completeness. Most missing BUSCOs (13.3% / 10.4% in genome mode) are **absent from or
   broken in the assembly**, not lost by this pipeline.
2. **Leftover haplotype duplication is low** (genome-mode D = 3.7% / 5.9%). The large assembly sizes (2.8 / 3.3 Gb,
   more than twice *Ae. aegypti*'s 1.28 Gb) therefore come mostly from repeats, not from uncollapsed haplotypes.
   This corrects the hypothesis raised in the readiness notebook.
3. **Signal of residual base errors or pseudogenes.** In BUSCO genome mode, 12.9% (humeralis) and 13.7% (titillans) of
   complete BUSCO matches contain internal stop codons (`E` in the BUSCO summary). A polished HiFi assembly would
   normally show much less. This pipeline drops such models (the internal-stop and frameshift rejections above),
   so some real genes are missing from the proteomes. Worth raising with the assembly provider.
4. **Many models are partial.** About 30% have both a start M and a stop codon, but median reference coverage is 0.99.
   Most models are therefore near-full-length and only miss terminal residues. The N50 of 0.14 to 0.26 Mb splits some
   genes across contigs, and those produce fragments (coverage 0.6 to 0.9).
5. **Identical copies** (`identical_seq_copies > 1`) are mostly pairs and small families. The largest groups are
   **histone genes**: 45 identical histone H2B copies in humeralis and 29 identical histone H4 copies in titillans.
   These are expected from the tandem-repeated histone cluster and are not an artifact. They are kept in the FASTA.
   Because identical sequences link to each other in S2F's homology graph, consider collapsing them if they dominate a result.
6. **Consistency with the candidate sets, by locus.** 82% (humeralis) and 90% (titillans) of the earlier 283 targeted
   loci overlap a new model. Most of the rest are **chemosensory** loci (29 of 50 in humeralis, 22 of 27 in titillans).
   A locus overlap does not prove that the same protein was recovered, so the next section compares the sequences directly.

## Recovery of the 283 candidate proteins (sequence-level check)

Checked 2026-09-29. The question: is each earlier candidate protein, the S2F input of the `mansonia_<species>` runs,
present in the new predicted proteome? The comparison uses only the two FASTA files per species:

- candidates: `data/mansonia/mansonia_candidate_proteins/<species>/<species>_all_categories_protein.cleaned.faa` (283 proteins)
- proteome: `data/mansonia/annotation/<species>/<species>_proteome.faa` (12,245 / 13,700 proteins)

### Method

1. Strip the terminal `*` from both sets. Candidate IDs are the first header token, e.g.
   `humeralis_4_chemosensory_lcl|ptg...:start-end(strand)`.
2. **Exact and containment checks** in Python: identical sequence; candidate contained as a substring of a predicted
   protein; or a predicted protein (≥ 30 aa) contained in the candidate.
3. **BLASTP** (NCBI BLAST+ 2.17.0, `/usr/local/ncbi-blast-2.17.0+/bin/`) for candidates without an exact or containment match:

   ```bash
   O=notebooks/exports/mansonia_candidates_vs_proteome/blast
   makeblastdb -in data/mansonia/annotation/<species>/<species>_proteome.faa -dbtype prot -out $O/<species>_proteome
   blastp -query $O/<species>_candidates.faa -db $O/<species>_proteome -evalue 1e-5 -max_target_seqs 10 -num_threads 16 \
     -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen" \
     > $O/<species>_candidates_vs_proteome.tsv
   ```

   The best hit per candidate is the one with the highest bitscore. Query coverage = (qend - qstart + 1) / qlen.
4. Each candidate goes into the **first** class it satisfies:

| class | rule | counted as |
|---|---|---|
| 1 | identical sequence | present |
| 2 | candidate is a substring of a longer predicted protein | present (the new model extends it) |
| 3 | a predicted protein is a substring of the candidate | present (the new model is shorter) |
| 4 | best BLAST hit ≥ 95% identity and ≥ 90% query coverage | present (near-identical) |
| 5 | best BLAST hit ≥ 95% identity, < 90% query coverage | not recovered: same sequence, only partly |
| 6 | best BLAST hit < 95% identity | not recovered: only a homolog (paralog or ortholog) |
| 7 | no BLAST hit at e ≤ 1e-5 | not recovered |

### Results

| class | humeralis | titillans |
|---|---|---|
| 1 identical sequence | 127 | 127 |
| 2 candidate contained in a longer predicted protein | 69 | 94 |
| 3 predicted protein is a fragment of the candidate | 3 | 4 |
| 4 near-identical (≥ 95% id, ≥ 90% coverage) | 25 | 16 |
| **present (1-4)** | **224 (79%)** | **241 (85%)** |
| 5 same sequence, partial (≥ 95% id, < 90% coverage) | 5 | 6 |
| 6 only a homolog (< 95% id) | 32 | 26 |
| 7 no BLAST hit | 22 | 10 |
| **not recovered (5-7)** | **59** | **42** |

Present (classes 1-4) by candidate category:

| category | humeralis | titillans |
|---|---|---|
| 1 tracheal / spiracle | 5 / 7 | 9 / 9 |
| 2 cuticle / chitin | 155 / 175 | 155 / 168 |
| 3 hypoxia signaling | 6 / 6 | 5 / 6 |
| 4 chemosensory | **48 / 78** | **59 / 85** |
| 5 mechanosensory | 10 / 17 | 13 / 15 |

### Interpretation

- **Most candidates are in the proteome.** About 45% of candidates are identical to a predicted protein, and another
  24-33% are contained in a longer predicted protein (class 2). There, the miniprot model extends further along the
  gene than the earlier targeted extraction.
- **The chemosensory set is the main gap.** 30 of 78 (humeralis) and 26 of 85 (titillans) chemosensory candidates are not
  recovered, followed by cuticle/chitin (20 and 13). Odorant and gustatory receptors are fast-evolving tandem families.
  They are the hardest case for the identity filter and for locus resolution, where a neighbouring paralog can win the locus.
  The reason each candidate was dropped has not been traced through the raw miniprot alignments.
- **Why the locus check gave higher numbers.** Locus overlap (233 / 256) counts a candidate as recovered whenever any
  model overlaps its coordinates. The sequence check (224 / 241) is stricter: some overlapping models are a different
  protein (class 6).
- **Consequence for S2F.** A whole-proteome S2F run is missing 59 (humeralis) and 42 (titillans) of the proteins
  analysed in the candidate runs. Results for those proteins cannot be compared between the two runs unless the
  missing candidates are added back (see Open questions).

Outputs: `notebooks/exports/mansonia_candidates_vs_proteome/blast/` holds the BLAST databases, the query FASTA files
with `*` stripped (`<species>_candidates.faa`) and the tabular hits (`<species>_candidates_vs_proteome.tsv`).
The classification was run as a one-off Python script, so no per-candidate table was saved.

## Limitations

- **Homology bias.** Genes with no detectable homolog in *Aedes*, *Culex* or *Anopheles* are not recovered, and
  *Mansonia*-specific genes are under-represented. The three references have 12.5k to 15.1k protein-coding genes, so
  12.2k / 13.7k is plausible, but it is not the full gene complement.
- **Structures come from references.** Exon/intron boundaries follow the reference proteins. Isoforms, UTRs and
  species-specific gene structures are not modelled. There is one model per locus.
- **The assembly limits the result.** Genes that are missing, broken or carrying errors in the assembly cannot be recovered.
- **No repeat masking or contamination screen was applied here.** Models on contaminant contigs would receive a
  mosquito homolog only if they are conserved genes. See the GC-outlier and circular-contig checks in the readiness notebook.

## Reproduce

```bash
cd S2F
python3 scripts/mansonia_genome_to_proteome.py all --threads 24          # about 25 min, plus about 55 min for BUSCO genome mode
python3 scripts/mansonia_genome_to_proteome.py select --force \
    --min-identity 0.4 --min-ref-coverage 0.8                            # re-filter without realigning
```

The reference download (stage 0) is not automated by the script. Its commands are:

```bash
cd data/mansonia/annotation/refs
curl -L -o Aedes_aegypti.zip "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_002204515.2/download?include_annotation_type=PROT_FASTA,GENOME_GFF"
# same for GCF_015732765.1 (Culex_quinquefasciatus) and GCF_943734735.2 (Anopheles_gambiae); unzip -d <species>
```

Then run S2F (not yet executed):

```bash
python3 S2F.py predict --run-config conf/mansonia_humeralis_proteome.conf
python3 S2F.py predict --run-config conf/mansonia_titillans_proteome.conf
```

With about 13k proteins instead of 283, the InterProScan, HMMER and STRING-BLAST stages will take much longer than
the candidate runs.

## Open questions and possible extensions

- Confirm with the assembly provider: the meaning of `posCBS`; whether the base errors suggested by BUSCO `E` are
  known (polishing status); any k-mer genome size estimate; whether RNA-seq/Iso-Seq will become available.
- With RNA-seq: BRAKER3 would recover non-homologous genes and correct gene structures. The current homology models
  could then serve as protein evidence.
- Optional filters before S2F: collapse identical-sequence copies; drop models with `ref_coverage < 0.8`; exclude
  proteins on GC-outlier or circular contigs.
- Add back the unrecovered candidate proteins (classes 5-7: 59 humeralis, 42 titillans) to the proteome FASTA, with
  their source recorded, so every candidate-run protein is also in the whole-proteome run. Alternatively, trace why
  each one was dropped (identity, coverage, frameshift, internal stop, or losing its locus to a paralog) in
  `<species>.miniprot.gff` and relax the filters for the chemosensory families.

## References

- Li, H. (2023). Protein-to-genome alignment with miniprot. *Bioinformatics* 39(1): btad014.
- Manni, M. et al. (2021). BUSCO update. *Molecular Biology and Evolution* 38(10): 4647-4654.
- Matthews, B. J. et al. (2018). Improved reference genome of *Aedes aegypti*. *Nature* 563: 501-507.
- Cheng, H. et al. (2021). hifiasm. *Nature Methods* 18: 170-175.
- Torres, M. et al. (2021). Protein function prediction for newly sequenced organisms. *Nature Machine Intelligence*.
- Karyotype of *Mansonia* spp. around the Jirau dam, Rondonia: https://pmc.ncbi.nlm.nih.gov/articles/PMC9991104/
