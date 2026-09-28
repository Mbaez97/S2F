import os
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import sparse

from diffusion import Diffusion
from diffusion.S2FLabelPropagation import S2FLabelPropagation
from GOTool import GeneOntology
from graphs import collection, combination, homology, Graph
from seeds import hmmer, interpro
from Utils import ColourClass, Configuration, FancyApp, Utilities


class Predict(FancyApp.FancyApp):

    def __init__(self, args):
        super(Predict, self).__init__()
        self.colour = ColourClass.bcolors.OKGREEN
        if args.run_config != "arguments":
            self.run_config = os.path.expanduser(args.run_config)
        else:
            self.run_config = "arguments"
        # load configurations
        if os.path.exists(self.run_config):
            self.tell("Run configuration provided, loading: ", self.run_config)
            Configuration.load_run(self.run_config)
            run_conf = Configuration.RUN_CONFIG

            self.config_file = os.path.expanduser(
                run_conf.get("configuration", "config_file")
            )
            self.alias = run_conf.get("configuration", "alias")
            self.obo = os.path.expanduser(run_conf.get("configuration", "obo"))
            self.fasta = os.path.expanduser(run_conf.get("configuration", "fasta"))
            self.cpu = os.path.expanduser(
                run_conf.get("configuration", "cpu", fallback="infer")
            )

            self.combined_graph = os.path.expanduser(run_conf.get(
                "graphs", "combined_graph", fallback="compute"
            ))
            self.graph_collection = os.path.expanduser(run_conf.get(
                "graphs", "graph_collection", fallback="compute"
            ))
            self.homology_graph = os.path.expanduser(run_conf.get(
                "graphs", "homology_graph", fallback="compute"
            ))
            self.recompute_orthologs = run_conf.getboolean(
                "graphs", "recompute_orthologs", fallback=True
            )
            self.orthologs_alias = run_conf.get(
                "graphs", "orthologs_alias", fallback=self.alias
            )
            self.collection_chunk_size = run_conf.getint(
                "graphs", "collection_chunk_size", fallback=200000
            )

            self.interpro_output = run_conf.get(
                "seeds", "interpro_output", fallback="compute"
            )
            self.hmmer_output = run_conf.get(
                "seeds", "hmmer_output", fallback="compute"
            )
            self.foldseek_output = run_conf.get(
                "seeds", "foldseek_output", fallback="skip"
            )
            self.plm_output = run_conf.get(
                "seeds", "plm_output", fallback="skip"
            )
            self.raw_alpha = run_conf.getfloat("seeds", "alpha", fallback=0.9)
            self.raw_beta = run_conf.getfloat("seeds", "beta", fallback=0.1)
            self.raw_gamma = run_conf.getfloat("seeds", "gamma", fallback=0.0)

            self.hmmer_blacklist = os.path.expanduser(run_conf.get(
                "blacklists", "hmmer_blacklist", fallback="compute"
            ))
            self.transfer_blacklist = os.path.expanduser(run_conf.get(
                "blacklists", "transfer_blacklist", fallback="compute"
            ))

            self.foldseek_target_db = run_conf.get(
                "foldseek", "target_db", fallback=""
            )
            self.foldseek_structure_mode = run_conf.get(
                "foldseek", "structure_mode", fallback="existing"
            )
            self.foldseek_structures_dir = os.path.expanduser(
                run_conf.get("foldseek", "structures_dir", fallback="structures")
            )
            self.foldseek_precomputed_tsv = os.path.expanduser(
                run_conf.get("foldseek", "precomputed_tsv", fallback="")
            )
            self.foldseek_search_recursive = run_conf.getboolean(
                "foldseek", "search_recursive", fallback=False
            )
            self.foldseek_prostt5_model = run_conf.get(
                "foldseek", "prostt5_model", fallback=""
            )
            self.foldseek_gpu = run_conf.get("foldseek", "gpu", fallback="")
            self.foldseek_cuda_visible_devices = run_conf.get(
                "foldseek", "cuda_visible_devices", fallback=""
            )
            self.foldseek_alignment_type = run_conf.getint(
                "foldseek", "alignment_type", fallback=1
            )
            self.foldseek_evalue_max = run_conf.getfloat(
                "foldseek", "evalue_max", fallback=1e-5
            )
            self.foldseek_min_qcov = run_conf.getfloat(
                "foldseek", "min_qcov", fallback=0.70
            )
            self.foldseek_min_tcov = run_conf.getfloat(
                "foldseek", "min_tcov", fallback=0.70
            )
            self.foldseek_min_avg_tm = run_conf.getfloat(
                "foldseek", "min_avg_tm", fallback=0.50
            )
            self.foldseek_max_seqs = run_conf.getint(
                "foldseek", "max_seqs", fallback=0
            )
            self.foldseek_score_mode = run_conf.get(
                "foldseek", "score_mode", fallback="binary"
            )

            has_plm_section = run_conf.has_section("plm")
            plm_get = (
                lambda option, fallback: run_conf.get(
                    "plm", option, fallback=fallback
                )
                if has_plm_section
                else fallback
            )
            plm_getint = (
                lambda option, fallback: run_conf.getint(
                    "plm", option, fallback=fallback
                )
                if has_plm_section
                else fallback
            )
            plm_getfloat = (
                lambda option, fallback: run_conf.getfloat(
                    "plm", option, fallback=fallback
                )
                if has_plm_section
                else fallback
            )
            plm_getboolean = (
                lambda option, fallback: run_conf.getboolean(
                    "plm", option, fallback=fallback
                )
                if has_plm_section
                else fallback
            )
            self.plm_target_fasta = os.path.expanduser(
                plm_get("target_fasta", "")
            )
            self.plm_model_name = plm_get(
                "model_name", "facebook/esm1b_t33_650M_UR50S"
            )
            self.plm_model_dir = os.path.expanduser(plm_get("model_dir", ""))
            self.plm_embeddings_dir = os.path.expanduser(
                plm_get("embeddings_dir", "")
            )
            self.plm_device = plm_get("device", "auto")
            self.plm_knn_k = plm_getint("knn_k", 10)
            self.plm_transfer_strategy = plm_get("transfer_strategy", "knn")
            self.plm_long_sequence_mode = plm_get(
                "long_sequence_mode", "sliding_mean"
            )
            self.plm_long_window_size = plm_getint("long_window_size", 1022)
            self.plm_long_overlap = plm_getint("long_overlap", 128)
            self.plm_score_mode = plm_get("score_mode", "all_ones")
            self.plm_exclude_accessions = os.path.expanduser(
                plm_get("exclude_accessions", "")
            )
            self.plm_kde_bandwidth = plm_getfloat(
                "kde_bandwidth", 0.025409690504535225
            )
            self.plm_kde_weight_floor = plm_getfloat(
                "kde_weight_floor", 1e-6
            )
            self.plm_kde_max_neighbors = plm_getint(
                "kde_max_neighbors", 8192
            )
            self.plm_batch_tokens = plm_getint("batch_tokens", 4096)
            self.plm_query_chunk_size = plm_getint("query_chunk_size", 64)
            self.plm_local_files_only = plm_getboolean(
                "local_files_only", False
            )
            self.plm_precomputed_target_embeddings = os.path.expanduser(
                plm_get("precomputed_target_embeddings", "")
            )
            self.plm_precomputed_query_embeddings = os.path.expanduser(
                plm_get("precomputed_query_embeddings", "")
            )
            self.plm_force_target_embeddings = plm_getboolean(
                "force_target_embeddings", False
            )
            self.plm_force_query_embeddings = plm_getboolean(
                "force_query_embeddings", False
            )

            self.goa_clamp = run_conf.get("functions", "goa_clamp", fallback="compute")
            self.unattended_mode = args.unattended is True or run_conf.getboolean(
                "functions", "unattended", fallback=False
            )
            self.write_collection = run_conf.getboolean(
                "functions", "write_collection", fallback=False
            )
            self.fasta_id_parser = run_conf.get(
                "functions", "fasta_id_parser", fallback="uniprot"
            )
            self.collection_engine = run_conf.get(
                "performance", "collection_engine", fallback="legacy"
            )
            self.homology_engine = run_conf.get(
                "performance", "homology_engine", fallback="legacy"
            )
            self.write_diffusion_tsv = run_conf.getboolean(
                "performance", "write_diffusion_tsv", fallback=True
            )
            self.prediction_chunk_rows = run_conf.getint(
                "performance", "prediction_chunk_rows", fallback=0
            )
            self.prediction_format = run_conf.get(
                "performance", "prediction_format", fallback="tsv"
            )
        else:
            self.config_file = os.path.expanduser(args.config_file)

            Configuration.load_configuration(self.config_file)

            self.alias = args.alias
            self.obo = os.path.expanduser(args.obo)
            self.fasta = os.path.expanduser(args.fasta)
            self.cpu = os.path.expanduser(args.cpu)

            self.combined_graph = os.path.expanduser(args.combined_graph)
            self.graph_collection = os.path.expanduser(args.graph_collection)
            self.homology_graph = os.path.expanduser(args.homology_graph)

            self.interpro_output = os.path.expanduser(args.interpro_output)
            self.hmmer_output = os.path.expanduser(args.hmmer_output)
            self.foldseek_output = os.path.expanduser(args.foldseek_output)
            self.plm_output = os.path.expanduser(args.plm_output)
            self.raw_alpha = args.alpha
            self.raw_beta = args.beta
            self.raw_gamma = args.gamma

            self.recompute_orthologs = not args.skip_orthologs
            self.orthologs_alias = self.alias
            self.collection_chunk_size = 200000
            self.hmmer_blacklist = os.path.expanduser(args.hmmer_blacklist)
            self.transfer_blacklist = os.path.expanduser(args.transfer_blacklist)
            self.foldseek_target_db = os.path.expanduser(args.foldseek_target_db)
            self.foldseek_structure_mode = args.foldseek_structure_mode
            self.foldseek_structures_dir = os.path.expanduser(
                args.foldseek_structures_dir
            )
            self.foldseek_precomputed_tsv = (
                os.path.expanduser(args.foldseek_precomputed_tsv)
                if args.foldseek_precomputed_tsv not in ("", None)
                else ""
            )
            self.foldseek_search_recursive = args.foldseek_search_recursive is True
            self.foldseek_prostt5_model = (
                os.path.expanduser(args.foldseek_prostt5_model)
                if args.foldseek_prostt5_model not in ("", None)
                else ""
            )
            self.foldseek_gpu = args.foldseek_gpu
            self.foldseek_cuda_visible_devices = args.foldseek_cuda_visible_devices
            self.foldseek_alignment_type = args.foldseek_alignment_type
            self.foldseek_evalue_max = args.foldseek_evalue_max
            self.foldseek_min_qcov = args.foldseek_min_qcov
            self.foldseek_min_tcov = args.foldseek_min_tcov
            self.foldseek_min_avg_tm = args.foldseek_min_avg_tm
            self.foldseek_max_seqs = args.foldseek_max_seqs
            self.foldseek_score_mode = args.foldseek_score_mode
            self.plm_target_fasta = os.path.expanduser(args.plm_target_fasta)
            self.plm_model_name = args.plm_model_name
            self.plm_model_dir = os.path.expanduser(args.plm_model_dir)
            self.plm_embeddings_dir = os.path.expanduser(args.plm_embeddings_dir)
            self.plm_device = args.plm_device
            self.plm_knn_k = args.plm_knn_k
            self.plm_transfer_strategy = args.plm_transfer_strategy
            self.plm_long_sequence_mode = args.plm_long_sequence_mode
            self.plm_long_window_size = args.plm_long_window_size
            self.plm_long_overlap = args.plm_long_overlap
            self.plm_score_mode = args.plm_score_mode
            self.plm_exclude_accessions = os.path.expanduser(
                args.plm_exclude_accessions
            )
            self.plm_kde_bandwidth = args.plm_kde_bandwidth
            self.plm_kde_weight_floor = args.plm_kde_weight_floor
            self.plm_kde_max_neighbors = args.plm_kde_max_neighbors
            self.plm_batch_tokens = args.plm_batch_tokens
            self.plm_query_chunk_size = args.plm_query_chunk_size
            self.plm_local_files_only = args.plm_local_files_only is True
            self.plm_precomputed_target_embeddings = os.path.expanduser(
                args.plm_precomputed_target_embeddings
            )
            self.plm_precomputed_query_embeddings = os.path.expanduser(
                args.plm_precomputed_query_embeddings
            )
            self.plm_force_target_embeddings = args.plm_force_target_embeddings is True
            self.plm_force_query_embeddings = args.plm_force_query_embeddings is True

            self.goa_clamp = os.path.expanduser(args.goa_clamp)
            self.unattended_mode = args.unattended is True
            self.write_collection = args.write_collection is True
            self.fasta_id_parser = args.fasta_id_parser
            self.collection_engine = "legacy"
            self.homology_engine = "legacy"
            self.write_diffusion_tsv = True
            self.prediction_chunk_rows = 0
            self.prediction_format = "tsv"

        self.installation_directory = os.path.expanduser(
            Configuration.CONFIG.get("directories", "installation_directory")
        )

        self.interpro = Configuration.CONFIG.get("commands", "interpro")
        self.hmmer = Configuration.CONFIG.get("commands", "hmmer")
        self.blastp = Configuration.CONFIG.get("commands", "blastp")
        self.makeblastdb = Configuration.CONFIG.get("commands", "makeblastdb")

        self.string_links = Configuration.CONFIG.get("databases", "string_links")
        self.string_sequences = Configuration.CONFIG.get(
            "databases", "string_sequences"
        )
        self.string_species = Configuration.CONFIG.get("databases", "string_species")
        self.string_core_only = Configuration.CONFIG.get(
            "databases", "string_core_only"
        )
        self.uniprot_sprot = Configuration.CONFIG.get("databases", "uniprot_sprot")
        self.uniprot_goa = Configuration.CONFIG.get("databases", "uniprot_goa")
        self.filtered_goa = Configuration.CONFIG.get("databases", "filtered_goa")
        self.filtered_sprot = Configuration.CONFIG.get("databases", "filtered_sprot")

        if self.plm_target_fasta in ("", None):
            self.plm_target_fasta = self.filtered_sprot
        if self.plm_model_dir in ("", None):
            self.plm_model_dir = os.path.join(
                self.installation_directory, "data/PLM/models"
            )
        if self.plm_embeddings_dir in ("", None):
            self.plm_embeddings_dir = os.path.join(
                self.installation_directory, "data/PLM/embeddings"
            )

        ec = Configuration.CONFIG.get("options", "evidence_codes")
        self.evidence_codes = ec if ec == "experimental" else ec.split(",")

        if (
            not os.path.exists(self.config_file)
            or self.alias == ""
            or not os.path.exists(self.obo)
            or not os.path.exists(self.fasta)
        ):
            self.warning(
                "The configuration file must include readable "
                + "files and an alias, please revise these"
                + "arguments:\n"
                + "\tconfig_file\n"
                + "\talias\n"
                + "\tobo\n"
                + "\tfasta"
            )
            sys.exit(1)

        self._normalise_seed_weights()

        if not self.unattended_mode:
            self.summary_and_continue()

        self.output_dir = os.path.join(
            self.installation_directory, "output", self.alias
        )
        self.seed_dir_IP = os.path.join(self.installation_directory, "seeds/interpro")
        self.seed_dir_H = os.path.join(self.installation_directory, "seeds/hmmer")
        self.seed_dir_F = os.path.join(self.installation_directory, "seeds/foldseek")
        self.seed_dir_P = os.path.join(self.installation_directory, "seeds/plm")
        self.combination_dir = os.path.join(
            self.installation_directory, "graphs", "combined"
        )
        self.go = GeneOntology.GeneOntology(self.obo, verbose=True)
        self.terms = None
        self.proteins = None
        self.prediction = None
        self.clamp_matrix = None
        self.run_manifest = None

        if self.collection_engine not in ("legacy", "native_filter"):
            raise ValueError(
                "collection_engine must be legacy or native_filter")
        if self.homology_engine not in ("legacy", "vectorized"):
            raise ValueError("homology_engine must be legacy or vectorized")
        if self.prediction_chunk_rows < 0:
            raise ValueError("prediction_chunk_rows cannot be negative")
        if self.prediction_format not in ("tsv", "npz", "both"):
            raise ValueError("prediction_format must be tsv, npz, or both")

        if self.cpu == "infer":
            # https://docs.python.org/3/library/os.html#os.cpu_count
            self.cpu = len(os.sched_getaffinity(0))
        else:
            self.cpu = int(self.cpu)

    def _write_manifest(self):
        if self.run_manifest is None:
            return
        filename = os.path.join(self.output_dir, "run_manifest.json")
        temporary = filename + ".tmp." + str(os.getpid())
        try:
            with open(temporary, "w", encoding="utf-8") as handler:
                json.dump(self.run_manifest, handler, indent=2, sort_keys=True)
                handler.write("\n")
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _initialize_manifest(self):
        self.run_manifest = {
            "alias": self.alias,
            "command": " ".join(sys.argv),
            "run_config": self.run_config,
            "fasta": self.fasta,
            "obo": self.obo,
            "interpreter": sys.executable,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "engines": {
                "collection": self.collection_engine,
                "homology": self.homology_engine,
            },
            "write_diffusion_tsv": self.write_diffusion_tsv,
            "prediction_chunk_rows": self.prediction_chunk_rows,
            "prediction_format": self.prediction_format,
            "seed_weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
            },
            "plm": {
                "transfer_strategy": self.plm_transfer_strategy,
                "knn_k": self.plm_knn_k,
                "score_mode": self.plm_score_mode,
                "kde_bandwidth": self.plm_kde_bandwidth,
                "kde_weight_floor": self.plm_kde_weight_floor,
                "kde_max_neighbors": self.plm_kde_max_neighbors,
                "exclude_accessions": self.plm_exclude_accessions,
                "precomputed_target_embeddings": self.plm_precomputed_target_embeddings,
                "precomputed_query_embeddings": self.plm_precomputed_query_embeddings,
            },
            "events": [],
        }
        self._write_manifest()

    @contextmanager
    def _stage(self, name):
        started = time.monotonic()
        event = {
            "stage": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        self.run_manifest["events"].append(event)
        self._write_manifest()
        try:
            yield
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = repr(exc)
            raise
        else:
            event["status"] = "completed"
        finally:
            event["finished_at"] = datetime.now(timezone.utc).isoformat()
            event["elapsed_seconds"] = round(time.monotonic() - started, 3)
            self._write_manifest()

    def run(self):
        # 1. create output directory
        self.create_output_directory()
        self._initialize_manifest()
        # 2. Exract a list of terms and proteins
        self.make_indices()
        # 3. run (or reuse) interpro
        self.check_progress()
        if self.load_ip_seed:
            with self._stage("interpro_seed"):
                ip_seed = self.run_interpro()
        else:
            ip_seed = self.empty_seed()
        # 4. run (or reuse) hmmer
        if self.load_hmmer_seed:
            with self._stage("hmmer_seed"):
                hmmer_seed = self.run_hmmer()
        else:
            hmmer_seed = self.empty_seed()
        # 5. run (or reuse) foldseek
        if self.load_foldseek_seed:
            with self._stage("foldseek_seed"):
                foldseek_seed = self.run_foldseek()
        else:
            foldseek_seed = self.empty_seed()
        # 6. run (or reuse) PLM
        if self.load_plm_seed:
            with self._stage("plm_seed"):
                plm_seed = self.run_plm()
        else:
            plm_seed = self.empty_seed()
        # 7. run (or reuse) graph collection
        if self.load_graph_collection:
            with self._stage("graph_collection"):
                graph_collection = self.run_graph_collection()
            if self.write_collection:
                for k, g in graph_collection.items():
                    sparse.save_npz(
                        os.path.join(self.output_dir, k + "_collection.npz"), g
                    )
        else:
            graph_collection = None
        # 6. run (or reuse) homology graph
        if self.load_homology_graph:
            with self._stage("homology_graph"):
                graph_homology = self.run_graph_homology()
                self._save_sparse_atomic(
                    os.path.join(self.output_dir, "homology.npz"),
                    graph_homology)
        else:
            graph_homology = None
        # 7. GOA clamp if provided
        if os.path.exists(self.goa_clamp) and self.calculate_diffusion:
            self.load_clamp()
            if self.use_ip_seed:
                self.tell("Clamping InterPro seed...")
                ip_seed = self.clamp(ip_seed)
            if self.use_hmmer_seed:
                self.tell("Clamping HMMER seed...")
                hmmer_seed = self.clamp(hmmer_seed)
            if self.use_foldseek_seed:
                self.tell("Clamping Foldseek seed...")
                foldseek_seed = self.clamp(foldseek_seed)
            if self.use_plm_seed:
                self.tell("Clamping PLM seed...")
                plm_seed = self.clamp(plm_seed)
        # 8. graph combination
        if self.load_combined_graph:
            target_seed = self.combine_seed_matrices(
                ip_seed, hmmer_seed, foldseek_seed, plm_seed
            )
            with self._stage("graph_combination"):
                combined_graph = self.combine_graphs(
                    graph_collection, graph_homology, target_seed
                )
        else:
            combined_graph = None
        # 9. prepare diffusion kernel
        if self.calculate_diffusion:
            kernel_params = {"lambda": 0.1}
            diff = S2FLabelPropagation(combined_graph, self.proteins, self.terms)
            with self._stage("diffusion_kernel"):
                diff.compute_kernel(**kernel_params)
            # 10. diffuse interpro
            if self.use_ip_seed:
                self.tell("Diffusing InterPro seed")
                with self._stage("interpro_diffusion"):
                    ip_diff = diff.diffuse(ip_seed)
                    self._save_sparse_atomic(self.ip_diff_file + ".npz", ip_diff)
                    if self.write_diffusion_tsv:
                        diff.write_results(self.ip_diff_file)
            else:
                ip_diff = self.empty_seed()
            # 11. diffuse hmmer
            if self.use_hmmer_seed:
                self.tell("Diffusing HMMER seed")
                with self._stage("hmmer_diffusion"):
                    hmmer_diff = diff.diffuse(hmmer_seed)
                    self._save_sparse_atomic(
                        self.hmmer_diff_file + ".npz", hmmer_diff)
                    if self.write_diffusion_tsv:
                        diff.write_results(self.hmmer_diff_file)
            else:
                hmmer_diff = self.empty_seed()
            if self.use_foldseek_seed:
                self.tell("Diffusing Foldseek seed")
                foldseek_diff = diff.diffuse(foldseek_seed)
                self._save_sparse_atomic(
                    self.foldseek_diff_file + ".npz", foldseek_diff)
                if self.write_diffusion_tsv:
                    diff.write_results(self.foldseek_diff_file)
            else:
                foldseek_diff = self.empty_seed()
            if self.use_plm_seed:
                self.tell("Diffusing PLM seed")
                plm_diff = diff.diffuse(plm_seed)
                self._save_sparse_atomic(
                    self.plm_diff_file + ".npz", plm_diff)
                if self.write_diffusion_tsv:
                    diff.write_results(self.plm_diff_file)
            else:
                plm_diff = self.empty_seed()
        else:
            self.tell("Diffusion files found, skipping computation...")
            ip_diff = (
                sparse.load_npz(self.ip_diff_file + ".npz")
                if self.use_ip_seed else self.empty_seed()
            )
            hmmer_diff = (
                sparse.load_npz(self.hmmer_diff_file + ".npz")
                if self.use_hmmer_seed else self.empty_seed()
            )
            if self.use_foldseek_seed:
                foldseek_diff = sparse.load_npz(self.foldseek_diff_file + ".npz")
            else:
                foldseek_diff = self.empty_seed()
            if self.use_plm_seed:
                plm_diff = sparse.load_npz(self.plm_diff_file + ".npz")
            else:
                plm_diff = self.empty_seed()
        # 12. output combination
        if os.path.exists(self.goa_clamp):
            if self.use_ip_seed:
                self.tell("Clamping diffused InterPro")
                ip_diff = self.clamp(ip_diff)
            if self.use_hmmer_seed:
                self.tell("Clamping diffused HMMER")
                hmmer_diff = self.clamp(hmmer_diff)
            if self.use_foldseek_seed:
                self.tell("Clamping diffused Foldseek")
                foldseek_diff = self.clamp(foldseek_diff)
            if self.use_plm_seed:
                self.tell("Clamping diffused PLM")
                plm_diff = self.clamp(plm_diff)
        self.tell("Combining diffused seeds")
        self.prediction = self.combine_seed_matrices(
            ip_diff, hmmer_diff, foldseek_diff, plm_diff
        )
        self.prediction = sparse.coo_matrix(self.prediction)
        self.tell("Saving prediction to file")
        with self._stage("prediction_output"):
            if self.prediction_format in ("npz", "both"):
                self._save_sparse_atomic(
                    os.path.join(self.output_dir, "prediction.npz"),
                    self.prediction,
                )
            if self.prediction_format in ("tsv", "both"):
                self.write_prediction(
                    os.path.join(self.output_dir, "prediction.df")
                )

    def check_progress(self):
        self.ip_seed_file = os.path.join(self.seed_dir_IP, self.alias + ".seed.npz")
        self.ip_diff_file = os.path.join(self.output_dir, "ip_seed.diffusion")

        self.hmmer_seed_file = os.path.join(self.seed_dir_H, self.alias + ".seed.npz")
        self.hmmer_evalue_file = os.path.join(self.seed_dir_H, self.alias + ".evalue")
        self.hmmer_diff_file = os.path.join(self.output_dir, "hmmer_seed.diffusion")
        self.foldseek_seed_file = os.path.join(
            self.seed_dir_F, self.alias + ".seed.npz"
        )
        self.foldseek_diff_file = os.path.join(
            self.output_dir, "foldseek_seed.diffusion"
        )
        self.plm_seed_file = os.path.join(
            self.seed_dir_P, self.alias + ".seed.npz"
        )
        self.plm_diff_file = os.path.join(
            self.output_dir, "plm_seed.diffusion"
        )

        self.string_dir = os.path.join(
            self.installation_directory, "data/STRINGSequences"
        )
        self.core_ids = os.path.join(self.installation_directory, "data/coreIds")
        self.orthologs_dir = os.path.join(self.installation_directory, "orthologs")
        self.graphs_dir = os.path.join(self.installation_directory, "graphs/collection")

        self.combined_graph_filename = os.path.join(self.combination_dir, self.alias)
        self.combined_graph_sparse_filename = f"{self.combined_graph_filename}.npz"

        self.load_ip_seed = self.use_ip_seed
        self.load_hmmer_seed = self.use_hmmer_seed
        self.load_foldseek_seed = self.use_foldseek_seed
        self.load_plm_seed = self.use_plm_seed
        self.load_graph_collection = False
        self.load_homology_graph = False
        self.load_combined_graph = False
        active_checkpoints = []
        if self.use_ip_seed:
            active_checkpoints.append(self.ip_diff_file)
        if self.use_hmmer_seed:
            active_checkpoints.append(self.hmmer_diff_file)
        if self.use_foldseek_seed:
            active_checkpoints.append(self.foldseek_diff_file)
        if self.use_plm_seed:
            active_checkpoints.append(self.plm_diff_file)
        self.calculate_diffusion = not all(
            self._diffusion_checkpoint_exists(path)
            for path in active_checkpoints
        )
        if not self.calculate_diffusion:
            self.load_ip_seed = False
            self.load_hmmer_seed = False
            self.load_foldseek_seed = False
            self.load_plm_seed = False
            return

        self.load_combined_graph = True
        if self._has_configured_graph(self.combined_graph, "combined_graph"):
            return
        self.load_graph_collection = True
        self.load_homology_graph = True

    def _diffusion_checkpoint_exists(self, prefix):
        if not os.path.exists(prefix + ".npz"):
            return False
        return not self.write_diffusion_tsv or os.path.exists(prefix)

    @staticmethod
    def _save_sparse_atomic(filename, matrix):
        temporary = filename + ".tmp." + str(os.getpid()) + ".npz"
        try:
            sparse.save_npz(temporary, matrix)
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def write_prediction(self, filename):
        if self.prediction_chunk_rows <= 0:
            Diffusion._write_results(
                self.prediction, self.proteins, self.terms, filename)
            return

        prediction = self.prediction.tocoo()
        protein_ids = np.empty(len(self.proteins), dtype=object)
        protein_ids[self.proteins["protein idx"].to_numpy()] = \
            self.proteins.index.to_numpy()
        term_ids = np.empty(len(self.terms), dtype=object)
        term_ids[self.terms["term idx"].to_numpy()] = self.terms.index.to_numpy()
        temporary = filename + ".tmp." + str(os.getpid())
        try:
            with open(temporary, "w", encoding="utf-8"):
                pass
            for start in range(0, prediction.nnz, self.prediction_chunk_rows):
                stop = min(start + self.prediction_chunk_rows, prediction.nnz)
                chunk = pd.DataFrame({
                    "protein id": protein_ids[prediction.row[start:stop]],
                    "term id": term_ids[prediction.col[start:stop]],
                    "score": prediction.data[start:stop],
                })
                chunk.to_csv(
                    temporary, sep="\t", index=False, header=False, mode="a")
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def create_output_directory(self):
        if not os.path.isdir(self.output_dir):
            self.tell("Creating output directory")
            os.mkdir(self.output_dir)
        os.makedirs(self.seed_dir_IP, exist_ok=True)
        os.makedirs(self.seed_dir_H, exist_ok=True)
        os.makedirs(self.seed_dir_F, exist_ok=True)
        os.makedirs(self.seed_dir_P, exist_ok=True)
        self.tell(self.output_dir, "will be used as output directory")

    def make_indices(self):
        self.tell("Extracting list of proteins from fasta file")
        if self.fasta_id_parser == "uniprot":
            fasta_id_parser = Utilities.keep_uniprot_accession
        else:
            fasta_id_parser = Utilities.keep_entire_prot_id
        self.proteins = Utilities.extract_indices_from_fasta(
            self.fasta, processing_func=fasta_id_parser
        )
        self.proteins.to_pickle(os.path.join(self.output_dir, "proteins.df"))

        self.tell("Building GO structure")
        self.go.build_structure()
        self.tell("Extracting list of GO Terms from structure")
        self.terms = pd.DataFrame(list(enumerate(sorted(self.go.terms.keys()))))
        self.terms.columns = ["term idx", "term id"]
        self.terms.set_index("term id", inplace=True)
        self.terms.to_pickle(os.path.join(self.output_dir, "terms.df"))

    def run_interpro(self):
        if (
            not os.path.exists(self.interpro_output)
            or self.interpro_output == "compute"
        ):
            self.interpro_output = os.path.join(self.output_dir, self.alias + "_IP")
            if not os.path.exists(self.interpro_output):
                self.tell("Running InterPro")
                command = (
                    self.interpro
                    + " -i "
                    + self.fasta
                    + " -goterms -iprlookup -f TSV -o "
                    + self.interpro_output
                )
                subprocess.call(command, shell=True)
            else:
                self.tell("InterPro file found, skipping computation...")
        return self.combine_interpro()

    def combine_interpro(self):
        if not os.path.exists(self.ip_seed_file):
            self.tell("Building InterPro seed file")
            if self._is_assignment_seed_file(self.interpro_output):
                self.tell(
                    "Detected assignment file; converting to seed matrix."
                )
                seed = self._build_assignment_seed(
                    self.interpro_output, "assignment"
                )
            else:
                interpro_seed = interpro.InterProSeed(
                    self.interpro_output,
                    self.proteins,
                    self.terms,
                    self.go,
                    self.fasta_id_parser,
                )
                # TODO: methods is here only to debug stuff, remove
                methods = interpro_seed.process_output()
                for k, s in methods.items():
                    s.to_pickle(os.path.join(self.output_dir, "ipseed_" + k))
                seed = interpro_seed.get_seed()
            sparse.save_npz(self.ip_seed_file, seed)
        else:
            self.tell("InterPro seed file found")
            seed = sparse.load_npz(self.ip_seed_file)
        return seed

    def run_foldseek(self):
        if self.foldseek_output == "skip":
            return self.empty_seed()

        if self.foldseek_output == "compute":
            foldseek_dir = os.path.join(self.output_dir, self.alias + "_FS")
            os.makedirs(foldseek_dir, exist_ok=True)
            assignments_path = os.path.join(foldseek_dir, "assignments.tsv")
            if not os.path.exists(assignments_path):
                self._run_foldseek_script(foldseek_dir)
            else:
                self.tell("Foldseek assignments found, skipping computation...")
            self.foldseek_output = assignments_path
        elif not os.path.exists(self.foldseek_output):
            raise RuntimeError(
                f"Foldseek assignments file {self.foldseek_output} does not exist."
            )

        if not os.path.exists(self.foldseek_seed_file):
            self.tell("Building Foldseek seed file")
            seed = self._build_assignment_seed(self.foldseek_output, "Foldseek")
            sparse.save_npz(self.foldseek_seed_file, seed)
        else:
            self.tell("Foldseek seed file found")
            seed = sparse.load_npz(self.foldseek_seed_file)
        return seed

    def run_plm(self):
        if self.plm_output == "skip" or not self.use_plm_seed:
            return self.empty_seed()

        if self.plm_output == "compute":
            plm_dir = os.path.join(self.output_dir, self.alias + "_PLM")
            os.makedirs(plm_dir, exist_ok=True)
            assignments_path = os.path.join(plm_dir, "assignments.tsv")
            metadata_path = os.path.join(plm_dir, "transfer_metadata.json")
            if not self._plm_output_matches_request(
                assignments_path, metadata_path
            ):
                self._run_plm_script(plm_dir)
            else:
                self.tell(
                    "Compatible PLM assignments found, skipping computation..."
                )
            self.plm_output = assignments_path
        elif not os.path.exists(self.plm_output):
            raise RuntimeError(
                f"PLM assignments file {self.plm_output} does not exist."
            )

        seed_metadata_path = self.plm_seed_file + ".json"
        seed_request = {
            "assignment": self._file_fingerprint(self.plm_output),
            "request": self._plm_request(),
        }
        seed_matches = False
        if os.path.exists(self.plm_seed_file) and os.path.exists(seed_metadata_path):
            try:
                with open(seed_metadata_path, "r", encoding="utf-8") as handler:
                    seed_matches = json.load(handler) == seed_request
            except (OSError, ValueError):
                seed_matches = False
        if not seed_matches:
            self.tell("Building PLM seed file")
            seed = self._build_assignment_seed(self.plm_output, "PLM")
            sparse.save_npz(self.plm_seed_file, seed)
            with open(seed_metadata_path, "w", encoding="utf-8") as handler:
                json.dump(seed_request, handler, indent=2, sort_keys=True)
                handler.write("\n")
        else:
            self.tell("PLM seed file found")
            seed = sparse.load_npz(self.plm_seed_file)
        return seed

    @staticmethod
    def _file_fingerprint(path):
        stat = os.stat(path)
        return {
            "path": os.path.realpath(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _plm_request(self):
        blacklist_path = self._configured_blacklist_path(
            self.hmmer_blacklist, "hmmer_blacklist"
        )
        exclusion_path = (
            os.path.realpath(self.plm_exclude_accessions)
            if self.plm_exclude_accessions not in ("", None)
            else ""
        )
        return {
            "transfer_strategy": self.plm_transfer_strategy,
            "knn_k": self.plm_knn_k,
            "score_mode": self.plm_score_mode,
            "kde_bandwidth": self.plm_kde_bandwidth,
            "kde_weight_floor": self.plm_kde_weight_floor,
            "kde_max_neighbors": self.plm_kde_max_neighbors,
            "blacklist": os.path.realpath(blacklist_path) if blacklist_path else "",
            "exclude_accessions": exclusion_path,
        }

    def _plm_output_matches_request(self, assignments_path, metadata_path):
        if not os.path.exists(assignments_path) or not os.path.exists(metadata_path):
            return False
        try:
            with open(metadata_path, "r", encoding="utf-8") as handler:
                metadata = json.load(handler)
        except (OSError, ValueError):
            return False
        return (
            metadata.get("request") == self._plm_request()
            and metadata.get("input_fingerprints")
            == self._plm_input_fingerprints()
        )

    def _plm_input_fingerprints(self):
        blacklist_path = self._configured_blacklist_path(
            self.hmmer_blacklist, "hmmer_blacklist"
        )
        return {
            "query_fasta": [self._file_fingerprint(self.fasta)],
            "target_fasta": [self._file_fingerprint(self.plm_target_fasta)],
            "goa": [self._file_fingerprint(self.filtered_goa)],
            "go_obo": [self._file_fingerprint(self.obo)],
            "blacklist": [self._file_fingerprint(blacklist_path)] if blacklist_path else [],
            "exclude_accessions": [self._file_fingerprint(self.plm_exclude_accessions)]
            if self.plm_exclude_accessions not in ("", None) else [],
            "precomputed_target_embeddings": self._embedding_cache_fingerprint(
                self.plm_precomputed_target_embeddings
            ),
            "precomputed_query_embeddings": self._embedding_cache_fingerprint(
                self.plm_precomputed_query_embeddings
            ),
        }

    def _embedding_cache_fingerprint(self, path):
        if path in ("", None):
            return {}
        return {
            name: self._file_fingerprint(os.path.join(path, name))
            for name in ("meta.json", "ids.tsv", "embeddings.npy")
            if os.path.isfile(os.path.join(path, name))
        }

    def run_hmmer(self):
        out = os.path.join(self.output_dir, self.alias + ".hmmer")
        if not os.path.exists(self.hmmer_output) or self.hmmer_output == "compute":
            self.hmmer_output = os.path.join(self.output_dir, self.alias + "_H")
            if not os.path.exists(self.hmmer_output):
                self.tell("Running HMMer")
                command = (
                    self.hmmer
                    + " --cpu "
                    + str(self.cpu)
                    + " --noali -o "
                    + out
                    + " --tblout "
                    + self.hmmer_output
                    + " "
                    + self.fasta
                    + " "
                    + self.filtered_sprot
                )
                self.tell(command)
                subprocess.call(command, shell=True)
            else:
                self.tell("HMMer file found, skipping computation...")
        return self.build_hmeer_seed()

    def build_hmeer_seed(self):
        if not os.path.exists(self.hmmer_seed_file):
            self.tell("Building HMMer seed file")
            hmmer_seed = hmmer.HMMerSeed(
                self.hmmer_output,
                self.proteins,
                self.terms,
                self.go,
                self.get_hmmer_blacklist(),
                self.filtered_goa,
                self.fasta_id_parser,
            )
            hmmer_seed.process_output(evalue_file=self.hmmer_evalue_file)
            seed = hmmer_seed.get_seed()
            sparse.save_npz(self.hmmer_seed_file, seed)
        else:
            self.tell("HMMer seed file found, skipping computation...")
            seed = sparse.load_npz(self.hmmer_seed_file)
        return seed

    def get_hmmer_blacklist(self):
        blacklist_path = self._configured_blacklist_path(
            self.hmmer_blacklist, "hmmer_blacklist"
        )
        if blacklist_path is not None:
            return (
                pd.read_csv(blacklist_path, header=None, names=["tax_id"])
                .tax_id.astype("str")
                .tolist()
            )
        return None

    @staticmethod
    def _configured_blacklist_path(configured_path, setting_name):
        """Resolve an explicitly configured taxon blacklist or fail closed.

        Older PFP configurations record removable media below ``/media`` while
        the current mount point is ``/run/media``.  Preserve those
        configurations without allowing a missing explicit blacklist to silently
        turn a leakage-controlled neighbour transfer into an unfiltered run.
        """
        if configured_path in ("", None, "compute"):
            return None
        path = os.path.expanduser(configured_path)
        candidates = [path]
        if path.startswith("/media/"):
            candidates.append("/run" + path)
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise RuntimeError(
            f"Configured {setting_name} does not exist: {path}. Refusing to run "
            "a potentially unfiltered GO-transfer experiment."
        )

    def run_graph_collection(self):
        if self._is_configured_graph(self.graph_collection):
            if not os.path.exists(self.graph_collection):
                raise RuntimeError(
                    f"Configured graph_collection does not exist: {self.graph_collection}"
                )
            self.tell("Loading graph collection from", self.graph_collection)
            graph_df = pd.read_pickle(self.graph_collection)
            Graph.assert_lexicographical_order(
                graph_df, p1="query1", p2="query2"
            )
            graph_collection = {}
            graph_names = [
                "neighborhood",
                "experiments",
                "coexpression",
                "textmining",
                "database",
            ]
            for graph_name in graph_names:
                if graph_name not in graph_df.columns:
                    continue
                graph = (
                    graph_df[["query1", "query2", graph_name]]
                    .merge(self.proteins, left_on="query1", right_index=True)
                    .merge(
                        self.proteins,
                        left_on="query2",
                        right_index=True,
                        suffixes=["1", "2"],
                    )
                )
                graph_collection[graph_name] = sparse.coo_matrix(
                    (
                        graph[graph_name].values,
                        (graph["protein idx1"].values, graph["protein idx2"].values),
                    ),
                    shape=(len(self.proteins), len(self.proteins)),
                )
            return graph_collection

        col = collection.Collection(
            self.fasta,
            self.proteins,
            self.string_dir,
            self.string_links,
            self.core_ids,
            self.output_dir,
            self.orthologs_dir,
            self.graphs_dir,
            self.alias,
            self.cpu,
            self.get_transfer_blacklist(),
            1e-6,
            80.0,
            60.0,
            self.fasta_id_parser,
            self.string_core_only,
            recompute_orthologs=self.recompute_orthologs,
            orthologs_alias=self.orthologs_alias,
            chunk_size=self.collection_chunk_size,
            engine=self.collection_engine,
        )
        col.compute_graph()
        return col.get_graph()

    def get_transfer_blacklist(self):
        blacklist_path = self._configured_blacklist_path(
            self.transfer_blacklist, "transfer_blacklist"
        )
        if blacklist_path is not None:
            return (
                pd.read_csv(blacklist_path, header=None, names=["tax_id"])
                .tax_id.astype("str")
                .tolist()
            )
        return None

    def run_graph_homology(self):
        if self._is_configured_graph(self.homology_graph):
            if not os.path.exists(self.homology_graph):
                raise RuntimeError(
                    f"Configured homology_graph does not exist: {self.homology_graph}"
                )
            self.tell("Loading homology graph from", self.homology_graph)
            if self.homology_graph.endswith(".npz"):
                graph = sparse.load_npz(self.homology_graph).tocoo()
                self._validate_graph_shape(graph, "homology_graph")
                self._validate_graph_protein_index(self.homology_graph)
                return graph
            homology_df = pd.read_pickle(self.homology_graph)
            Graph.assert_lexicographical_order(homology_df)
            h = (
                homology_df.merge(
                    self.proteins, left_on="Protein 1", right_index=True
                )
                .merge(
                    self.proteins,
                    left_on="Protein 2",
                    right_index=True,
                    suffixes=["1", "2"],
                )
            )
            return sparse.coo_matrix(
                (
                    h["weight"].values,
                    (h["protein idx1"].values, h["protein idx2"].values),
                ),
                shape=(len(self.proteins), len(self.proteins)),
            )

        graphs_dir = os.path.join(self.installation_directory, "graphs/homology")
        hom = homology.Homology(
            self.fasta, self.proteins, graphs_dir, self.alias,
            self.fasta_id_parser, cpu=self.cpu,
            engine=self.homology_engine
        )
        hom.compute_graph()
        return hom.get_graph()

    def load_clamp(self):
        self.tell("Loading GOA annotations to clamp...")
        self.go.load_clamp_file(self.goa_clamp, "clamp")
        self.tell("Up-propagating clamp annotations...")
        self.go.up_propagate_annotations("clamp")
        self.tell("Building clamp matrix...")
        clamp_df = self.go.get_annotations("clamp")
        clamp_df = clamp_df.merge(self.proteins, left_on="Protein", right_index=True)
        clamp_df = clamp_df.merge(self.terms, left_on="GO ID", right_index=True)

        p_idx = clamp_df["protein idx"].values
        go_idx = clamp_df["term idx"].values

        self.clamp_matrix = sparse.coo_matrix(
            (np.ones(len(clamp_df)), (p_idx, go_idx)),
            shape=(len(self.proteins), len(self.terms)),
        )

    def clamp(self, seeds):
        clamp_bigger = self.clamp_matrix > seeds
        return (
            seeds
            - seeds.multiply(clamp_bigger)
            + self.clamp_matrix.multiply(clamp_bigger)
        )

    def _run_foldseek_script(self, foldseek_dir):
        self._validate_foldseek_configuration()
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "foldseek.py",
        )
        command = [
            sys.executable,
            script_path,
            "--fastas",
            self.fasta,
            "--goa",
            self.filtered_goa,
            "--go-obo",
            self.obo,
            "--protein-id-mode",
            self.fasta_id_parser,
            "--structure-mode",
            self.foldseek_structure_mode,
            "--structures-dir",
            self.foldseek_structures_dir,
            "--alignment-type",
            str(self.foldseek_alignment_type),
            "--evalue-max",
            str(self.foldseek_evalue_max),
            "--min-qcov",
            str(self.foldseek_min_qcov),
            "--min-tcov",
            str(self.foldseek_min_tcov),
            "--min-avg-tm",
            str(self.foldseek_min_avg_tm),
            "--max-seqs",
            str(self.foldseek_max_seqs),
            "--score-mode",
            self.foldseek_score_mode,
            "--outdir",
            foldseek_dir,
        ]
        if self.foldseek_precomputed_tsv not in ("", None):
            command += ["--precomputed-tsv", self.foldseek_precomputed_tsv]
        else:
            command += ["--target-db", self.foldseek_target_db]
        if self.foldseek_search_recursive:
            command.append("--search-recursive")
        if self.foldseek_prostt5_model not in ("", None):
            command += ["--prostt5-model", self.foldseek_prostt5_model]
        if self.foldseek_gpu not in ("", None):
            command += ["--gpu", self.foldseek_gpu]
        if self.foldseek_cuda_visible_devices not in ("", None):
            command += [
                "--cuda-visible-devices",
                self.foldseek_cuda_visible_devices,
            ]
        blacklist_path = self._configured_blacklist_path(
            self.hmmer_blacklist, "hmmer_blacklist"
        )
        if blacklist_path is not None:
            command += ["--blacklist", blacklist_path]

        self.tell("Running Foldseek GO-transfer script")
        self.tell(" ".join(command))
        subprocess.run(command, check=True)

    def _run_plm_script(self, plm_dir):
        self._validate_plm_configuration()
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "plm.py",
        )
        command = [
            sys.executable,
            script_path,
            "--fastas",
            self.fasta,
            "--target-fasta",
            self.plm_target_fasta,
            "--goa",
            self.filtered_goa,
            "--go-obo",
            self.obo,
            "--protein-id-mode",
            self.fasta_id_parser,
            "--model-name",
            self.plm_model_name,
            "--model-dir",
            self.plm_model_dir,
            "--embeddings-dir",
            self.plm_embeddings_dir,
            "--device",
            self.plm_device,
            "--knn-k",
            str(self.plm_knn_k),
            "--transfer-strategy",
            self.plm_transfer_strategy,
            "--long-sequence-mode",
            self.plm_long_sequence_mode,
            "--long-window-size",
            str(self.plm_long_window_size),
            "--long-overlap",
            str(self.plm_long_overlap),
            "--score-mode",
            self.plm_score_mode,
            "--kde-bandwidth",
            str(self.plm_kde_bandwidth),
            "--kde-weight-floor",
            str(self.plm_kde_weight_floor),
            "--kde-max-neighbors",
            str(self.plm_kde_max_neighbors),
            "--batch-tokens",
            str(self.plm_batch_tokens),
            "--query-chunk-size",
            str(self.plm_query_chunk_size),
            "--evidence-codes",
            ",".join(self.evidence_codes)
            if isinstance(self.evidence_codes, list)
            else self.evidence_codes,
            "--outdir",
            plm_dir,
        ]
        if self.plm_local_files_only:
            command.append("--local-files-only")
        if self.plm_precomputed_target_embeddings not in ("", None):
            command += [
                "--precomputed-target-embeddings",
                self.plm_precomputed_target_embeddings,
            ]
        if self.plm_precomputed_query_embeddings not in ("", None):
            command += [
                "--precomputed-query-embeddings",
                self.plm_precomputed_query_embeddings,
            ]
        if self.plm_force_target_embeddings:
            command.append("--force-target-embeddings")
        if self.plm_force_query_embeddings:
            command.append("--force-query-embeddings")
        if self.plm_exclude_accessions not in ("", None):
            command += ["--exclude-accessions", self.plm_exclude_accessions]
        blacklist_path = self._configured_blacklist_path(
            self.hmmer_blacklist, "hmmer_blacklist"
        )
        if blacklist_path is not None:
            command += ["--blacklist", blacklist_path]

        self.tell("Running PLM GO-transfer script")
        self.tell(" ".join(command))
        subprocess.run(command, check=True)

    def _is_assignment_seed_file(self, path):
        if path in ("", None, "compute"):
            return False
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as handler:
                first_line = handler.readline()
        except (OSError, UnicodeDecodeError):
            return False
        header_tokens = [tok.strip() for tok in first_line.strip().split("\t") if tok]
        header_set = set(header_tokens)
        return {"Protein", "GO ID"}.issubset(header_set)

    def _build_assignment_seed(self, assignments_path, description):
        self.tell("Loading", description, "assignments from", assignments_path)
        try:
            assignments = pd.read_csv(assignments_path, sep="\t")
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read assignments file {assignments_path}"
            ) from exc

        required_columns = {"Protein", "GO ID"}
        if not required_columns.issubset(assignments.columns):
            missing = required_columns - set(assignments.columns)
            raise RuntimeError(
                f"Assignments file {assignments_path} is missing required columns {missing}"
            )

        if "Score" in assignments.columns:
            assignments["Score"] = pd.to_numeric(
                assignments["Score"], errors="coerce"
            ).fillna(0.0)
        else:
            assignments["Score"] = 1.0

        seed_records = (
            assignments[["Protein", "GO ID", "Score"]]
            .groupby(["Protein", "GO ID"], as_index=False)["Score"]
            .max()
        )
        seed_records = seed_records[seed_records["Score"] > 0]

        total_rows = len(seed_records)
        if total_rows == 0:
            self.warning(
                "Assignments file has no positive scores; returning empty seed."
            )
            return self.empty_seed()

        seed_records = seed_records.merge(
            self.proteins, how="inner", left_on="Protein", right_index=True
        )
        matched_proteins = len(seed_records)
        if matched_proteins == 0:
            raise RuntimeError(
                "No proteins in the assignments match the FASTA-derived protein IDs."
            )
        protein_drop = total_rows - matched_proteins
        if protein_drop:
            self.warning(
                f"Dropped {protein_drop} assignments with proteins not present in FASTA index."
            )

        seed_records = seed_records.merge(
            self.terms, how="inner", left_on="GO ID", right_index=True
        )
        matched_terms = len(seed_records)
        if matched_terms == 0:
            raise RuntimeError(
                "No GO terms in the assignments match the GO structure."
            )
        term_drop = matched_proteins - matched_terms
        if term_drop:
            self.warning(
                f"Dropped {term_drop} assignments with GO terms outside the GO structure."
            )

        p_idx = seed_records["protein idx"].values
        go_idx = seed_records["term idx"].values
        scores = seed_records["Score"].values
        return sparse.coo_matrix(
            (scores, (p_idx, go_idx)), shape=(len(self.proteins), len(self.terms))
        )

    def empty_seed(self):
        return sparse.coo_matrix((len(self.proteins), len(self.terms)))

    def combine_seed_matrices(self, ip_seed, hmmer_seed, foldseek_seed, plm_seed):
        combined = (
            ip_seed.multiply(self.alpha)
            + hmmer_seed.multiply(self.beta)
            + foldseek_seed.multiply(self.gamma)
            + plm_seed.multiply(self.delta)
        )
        return combined.tocoo()

    def _validate_foldseek_configuration(self):
        if self.raw_gamma > 0 and self.foldseek_output == "skip":
            raise RuntimeError(
                "Foldseek weight gamma is greater than zero, but foldseek_output is set to skip."
            )
        if self.foldseek_output != "compute":
            return
        if (
            self.foldseek_precomputed_tsv in ("", None)
            and self.foldseek_target_db in ("", None, "compute", "skip")
        ):
            raise RuntimeError(
                "Foldseek compute mode requires a configured Foldseek target database."
            )
        if (
            self.foldseek_precomputed_tsv not in ("", None)
            and not os.path.exists(self.foldseek_precomputed_tsv)
        ):
            raise RuntimeError(
                f"Foldseek precomputed TSV {self.foldseek_precomputed_tsv} does not exist."
            )
        if not os.path.exists(self.filtered_goa):
            raise RuntimeError(
                f"Filtered GOA file {self.filtered_goa} does not exist."
            )

    def _validate_plm_configuration(self):
        if self.delta > 0 and self.plm_output == "skip":
            raise RuntimeError(
                "PLM residual weight is greater than zero, but plm_output is set to skip."
            )
        if self.plm_output != "compute":
            return
        if not os.path.exists(self.plm_target_fasta):
            raise RuntimeError(
                f"PLM target FASTA {self.plm_target_fasta} does not exist."
            )
        if not os.path.exists(self.filtered_goa):
            raise RuntimeError(
                f"Filtered GOA file {self.filtered_goa} does not exist."
            )
        if self.plm_knn_k <= 0:
            raise RuntimeError("PLM knn_k must be greater than zero.")
        if self.plm_transfer_strategy not in ("knn", "kde"):
            raise RuntimeError("PLM transfer_strategy must be knn or kde.")
        if self.plm_score_mode not in ("all_ones", "weighted_support"):
            raise RuntimeError(
                "PLM score_mode must be all_ones or weighted_support."
            )
        if (
            self.plm_transfer_strategy == "kde"
            and self.plm_score_mode != "weighted_support"
        ):
            raise RuntimeError(
                "PLM KDE transfer requires score_mode=weighted_support."
            )
        if self.plm_kde_bandwidth <= 0:
            raise RuntimeError("PLM kde_bandwidth must be greater than zero.")
        if not 0 < self.plm_kde_weight_floor < 1:
            raise RuntimeError(
                "PLM kde_weight_floor must be strictly between zero and one."
            )
        if self.plm_kde_max_neighbors <= 0:
            raise RuntimeError("PLM kde_max_neighbors must be greater than zero.")
        if (
            self.plm_exclude_accessions not in ("", None)
            and not os.path.isfile(self.plm_exclude_accessions)
        ):
            raise RuntimeError(
                "PLM accession exclusion file does not exist: "
                f"{self.plm_exclude_accessions}"
            )
        if self.plm_long_window_size <= 0:
            raise RuntimeError("PLM long_window_size must be greater than zero.")
        if self.plm_long_overlap < 0:
            raise RuntimeError("PLM long_overlap cannot be negative.")
        if self.plm_long_overlap >= self.plm_long_window_size:
            raise RuntimeError(
                "PLM long_overlap must be smaller than long_window_size."
            )
        if self.plm_batch_tokens <= 0:
            raise RuntimeError("PLM batch_tokens must be greater than zero.")
        if self.plm_query_chunk_size <= 0:
            raise RuntimeError("PLM query_chunk_size must be greater than zero.")
        if (
            self.plm_precomputed_target_embeddings not in ("", None)
            and not os.path.isdir(self.plm_precomputed_target_embeddings)
        ):
            raise RuntimeError(
                "PLM precomputed target embeddings directory does not exist: "
                f"{self.plm_precomputed_target_embeddings}"
            )
        if (
            self.plm_precomputed_query_embeddings not in ("", None)
            and not os.path.isdir(self.plm_precomputed_query_embeddings)
        ):
            raise RuntimeError(
                "PLM precomputed query embeddings directory does not exist: "
                f"{self.plm_precomputed_query_embeddings}"
            )

    def _normalise_seed_weights(self):
        if self.raw_gamma > 0 and self.foldseek_output == "skip":
            raise RuntimeError(
                "Foldseek weight gamma is greater than zero, but foldseek_output is set to skip."
            )
        for label, value in (
            ("alpha", self.raw_alpha),
            ("beta", self.raw_beta),
            ("gamma", self.raw_gamma),
        ):
            if value < 0:
                raise RuntimeError(f"Seed weight {label} cannot be negative.")
        total = self.raw_alpha + self.raw_beta + self.raw_gamma
        if total > 1 + 1e-9:
            raise RuntimeError(
                "Seed weights must satisfy alpha + beta + gamma <= 1 "
                "because PLM uses the residual weight."
            )
        self.delta = max(0, 1 - total)
        if self.delta < 1e-9:
            self.delta = 0
        if total <= 0 and self.delta <= 0:
            raise RuntimeError("At least one seed weight must be greater than zero.")
        if self.delta > 0 and self.plm_output == "skip":
            raise RuntimeError(
                "PLM residual weight is greater than zero, but plm_output is set to skip."
            )
        self.alpha = self.raw_alpha
        self.beta = self.raw_beta
        self.gamma = self.raw_gamma
        self.use_ip_seed = self.raw_alpha > 0
        self.use_hmmer_seed = self.raw_beta > 0
        self.use_foldseek_seed = self.raw_gamma > 0 and self.foldseek_output != "skip"
        self.use_plm_seed = self.delta > 0 and self.plm_output != "skip"

    def combine_graphs(self, graph_collection, graph_homology, target_seed):
        if self._is_configured_graph(self.combined_graph):
            combined_graph_path = self._resolve_combined_graph_path(
                self.combined_graph
            )
            self.tell("Loading combined graph from", combined_graph_path)
            graph = sparse.load_npz(combined_graph_path)
            self._validate_graph_shape(graph, "combined_graph")
            graph = Graph.fill_lower_triangle(graph)
            return graph

        signature = self._combined_graph_signature(target_seed)
        metadata_path = self.combined_graph_sparse_filename + ".json"
        cache_matches = False
        if os.path.exists(self.combined_graph_sparse_filename) and os.path.exists(
            metadata_path
        ):
            try:
                with open(metadata_path, "r", encoding="utf-8") as handler:
                    cache_matches = json.load(handler).get("signature") == signature
            except (OSError, ValueError):
                cache_matches = False
        if not cache_matches:
            comb = combination.Combination(
                self.proteins, graph_collection, graph_homology, target_seed
            )
            comb.compute_graph()
            comb.write_graph(self.combined_graph_filename)
            graph = comb.get_graph()
            sparse.save_npz(self.combined_graph_sparse_filename, graph)
            with open(metadata_path, "w", encoding="utf-8") as handler:
                json.dump(
                    {
                        "signature": signature,
                        "protein_count": len(self.proteins),
                        "seed_weights": {
                            "alpha": self.alpha,
                            "beta": self.beta,
                            "gamma": self.gamma,
                            "delta": self.delta,
                        },
                    },
                    handler,
                    indent=2,
                    sort_keys=True,
                )
                handler.write("\n")
        else:
            self.tell("Compatible combined graph found, skipping computation")
            graph = sparse.load_npz(self.combined_graph_sparse_filename)
        self._validate_graph_shape(graph, "combined_graph")
        graph = Graph.fill_lower_triangle(graph)
        return graph

    def _combined_graph_signature(self, target_seed):
        matrix = target_seed.tocoo()
        order = np.lexsort((matrix.col, matrix.row))
        digest = hashlib.sha256()
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.row[order].astype(np.int64, copy=False).tobytes())
        digest.update(matrix.col[order].astype(np.int64, copy=False).tobytes())
        digest.update(matrix.data[order].astype(np.float64, copy=False).tobytes())
        digest.update("\n".join(map(str, self.proteins.index)).encode("utf-8"))
        return digest.hexdigest()

    def _validate_graph_shape(self, graph, description):
        expected = (len(self.proteins), len(self.proteins))
        if graph.shape != expected:
            raise RuntimeError(
                f"Configured {description} has shape {graph.shape}, expected "
                f"{expected} for the current ordered FASTA protein index."
            )

    def _validate_graph_protein_index(self, graph_path):
        index_path = os.path.join(os.path.dirname(graph_path), "proteins.df")
        if not os.path.isfile(index_path):
            self.warning(
                f"No sibling proteins.df was found for {graph_path}; graph "
                "reuse is validated by shape only."
            )
            return
        saved = pd.read_pickle(index_path)
        if list(saved.index) != list(self.proteins.index):
            raise RuntimeError(
                f"Graph protein index {index_path} does not match the current "
                "ordered FASTA protein index. Refusing to reuse the graph."
            )

    def _is_configured_graph(self, path):
        return path not in ("", None, "compute", "skip")

    def _has_configured_graph(self, path, description):
        if not self._is_configured_graph(path):
            return False
        if description == "combined_graph":
            self._resolve_combined_graph_path(path)
        elif not os.path.exists(path):
            raise RuntimeError(f"Configured {description} does not exist: {path}")
        return True

    def _resolve_combined_graph_path(self, path):
        if os.path.exists(path) and path.endswith(".npz"):
            return path
        npz_path = path + ".npz"
        if os.path.exists(npz_path):
            return npz_path
        raise RuntimeError(
            "Configured combined_graph must point to a sparse .npz file "
            f"or to a base path with a .npz companion: {path}"
        )

    def summary_and_continue(self):
        summary = "These are the loaded values \n"
        summary += ColourClass.coloured_string(
            ColourClass.bcolors.HEADER, "\t[INSTALLATION CONFIGURATION]\n"
        )
        summary += "\tInstallation directory:\t\t"
        summary += self.installation_directory + "\n\n"
        summary += "\tPath to InterPro executable:\t" + self.interpro + "\n"
        summary += "\tPath to HMMer:\t\t\t" + self.hmmer + "\n"
        summary += "\tPath to blastp:\t\t\t" + self.blastp + "\n"
        summary += "\tPath to makeblastdb:\t\t" + self.makeblastdb + "\n\n"

        # TODO: These warnings are extremely unlikely to occur because
        # the installation should put all of these in the
        # right values. This code could be simplified.
        dbs = [
            ("STRING interaction database:\t", self.string_links),
            ("STRING sequences database:\t", self.string_sequences),
            ("STRING species database:\t", self.string_species),
            ("UniProt SwissProt:\t\t", self.uniprot_sprot),
            ("UniProt GOA:\t\t\t", self.uniprot_goa),
        ]
        for desc, database in dbs:
            summary += "\t" + desc
            if database == "download":
                summary += FancyApp.FancyApp.warning_text(database) + "\n"
            else:
                summary += database + "\n"

        summary += "\n\tFiltered UniProt GOA:\t\t" + self.filtered_goa + "\n"
        summary += "\tFiltered UniProt SwissProt:\t" + self.filtered_sprot
        summary += "\n\n"

        summary += "\tEvidence Codes:\t\t\t" + str(self.evidence_codes)
        summary += "\n\n"

        summary += ColourClass.coloured_string(
            ColourClass.bcolors.HEADER, "\t[RUN CONFIGURATION]\n"
        )
        # summary += '\tInstallation configuration file: ' +\
        #            self.config_file + '\n'
        summary += "\tAlias:\t\t\t\t" + self.alias + "\n"
        summary += "\tOBO file:\t\t\t" + self.obo + "\n"
        summary += "\tFasta file:\t\t\t" + self.fasta + "\n\n"

        summary += "\tCombined Graph:\t\t\t" + self.combined_graph + "\n"
        summary += "\tGraph Collection:\t\t" + self.graph_collection + "\n"
        summary += "\tHomology Graph:\t\t\t" + self.homology_graph + "\n"
        summary += "\tRecompute orthologs:\t\t" + str(self.recompute_orthologs) + "\n"
        summary += (
            "\tCollection chunk size:\t\t" + str(self.collection_chunk_size) + "\n\n"
        )

        summary += "\tInterPro seed:\t\t\t" + self.interpro_output + "\n"
        summary += "\tHMMer seed:\t\t\t" + self.hmmer_output + "\n"
        summary += "\tFoldseek seed:\t\t\t" + self.foldseek_output + "\n"
        summary += "\tPLM seed:\t\t\t" + self.plm_output + "\n"
        summary += "\tSeed weights:\t\t\t"
        summary += (
            f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
            f"gamma={self.gamma:.3f}, delta={self.delta:.3f}\n"
        )
        summary += "\tFoldseek target DB:\t\t" + self.foldseek_target_db + "\n"
        summary += "\tFoldseek precomputed TSV:\t" + self.foldseek_precomputed_tsv + "\n"
        summary += "\tFoldseek structure mode:\t" + self.foldseek_structure_mode + "\n"
        summary += "\tFoldseek structures dir:\t" + self.foldseek_structures_dir + "\n"
        summary += "\tFoldseek score mode:\t\t" + self.foldseek_score_mode + "\n"
        summary += "\tFoldseek max seqs:\t\t" + str(self.foldseek_max_seqs) + "\n"
        summary += "\tPLM target FASTA:\t\t" + self.plm_target_fasta + "\n"
        summary += "\tPLM model:\t\t\t" + self.plm_model_name + "\n"
        summary += "\tPLM model dir:\t\t\t" + self.plm_model_dir + "\n"
        summary += "\tPLM embeddings dir:\t\t" + self.plm_embeddings_dir + "\n"
        summary += "\tPLM device:\t\t\t" + self.plm_device + "\n"
        summary += "\tPLM transfer strategy:\t\t" + self.plm_transfer_strategy + "\n"
        summary += "\tPLM KNN k:\t\t\t" + str(self.plm_knn_k) + "\n"
        summary += "\tPLM long sequence mode:\t\t" + self.plm_long_sequence_mode + "\n"
        summary += "\tPLM score mode:\t\t" + self.plm_score_mode + "\n"
        summary += "\tPLM exclude accessions:\t" + self.plm_exclude_accessions + "\n"
        summary += "\tPLM KDE bandwidth:\t\t" + str(self.plm_kde_bandwidth) + "\n"

        summary += "\tHMMer black list:\t\t" + self.hmmer_blacklist + "\n"
        summary += "\tTransfer black list:\t\t"
        summary += self.transfer_blacklist + "\n"
        summary += "\tGOA clamp:\t\t\t" + self.goa_clamp + "\n"
        summary += "\tFASTA handling:\t\t\t" + self.fasta_id_parser + "\n"

        summary += "Do you want to continue with these settings?"

        if Utilities.query_yes_no(summary):
            return
        sys.exit()
