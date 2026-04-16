#!/usr/bin/env python3
"""
Regenerate a single ortholog cache file for S2F.

Example:
  python3 regenerate_single_ortholog.py \
    --alias cafa3_testset \
    --org-id 1220924 \
    --fasta /path/to/target.fasta \
    --orthologs-dir /media/marcelo_baez/HD_Disc1/.S2F/orthologs \
    --string-dir /media/marcelo_baez/HD_Disc1/.S2F/data/STRINGSequences \
    --protein-format uniprot
"""

import argparse
import os

from graphs.collection import compute_ortholog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate one ortholog pickle file for a specific STRING organism."
    )
    parser.add_argument("--alias", required=True, help="Prediction alias (e.g. cafa3_testset).")
    parser.add_argument("--org-id", required=True, help="STRING organism ID (e.g. 1220924).")
    parser.add_argument("--fasta", required=True, help="Target FASTA used in prediction.")
    parser.add_argument("--orthologs-dir", required=True, help="Ortholog cache directory.")
    parser.add_argument("--string-dir", required=True, help="Directory with STRING *.faa files.")
    parser.add_argument(
        "--protein-format",
        default="uniprot",
        choices=["uniprot", "entire_id"],
        help="Protein ID parser mode used in your run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output file first if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_name = f"{args.alias}_AND_{args.org_id}"
    out_path = os.path.join(args.orthologs_dir, out_name)
    string_fasta = os.path.join(args.string_dir, f"{args.org_id}.faa")

    if not os.path.exists(args.fasta):
        raise SystemExit(f"Target FASTA not found: {args.fasta}")
    if not os.path.exists(string_fasta):
        raise SystemExit(f"STRING FASTA not found: {string_fasta}")
    if not os.path.isdir(args.orthologs_dir):
        raise SystemExit(f"Ortholog directory not found: {args.orthologs_dir}")

    if os.path.exists(out_path):
        if not args.force:
            raise SystemExit(
                f"Output already exists: {out_path}\n"
                "Use --force to remove and recompute it."
            )
        os.remove(out_path)
        print(f"Removed existing file: {out_path}")

    compute_ortholog(
        fasta=args.fasta,
        db=args.fasta,
        string_fasta=string_fasta,
        string_db=string_fasta,
        out=out_name,
        out_dir=args.orthologs_dir,
        protein_format=args.protein_format,
    )

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Regenerated successfully: {out_path}")
    else:
        raise SystemExit(
            "Computation finished but output file was not created or is empty. "
            "Check blastp inputs/output for this organism."
        )


if __name__ == "__main__":
    main()
