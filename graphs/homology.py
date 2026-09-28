import os
import subprocess

import numpy as np
import pandas as pd
from scipy import sparse

from graphs import Graph
from Utils import Utilities


def vectorized_homology_graph(homology):
    """Build the legacy homology weights without Python pairwise loops."""
    proteins = sorted(homology.keys())
    protein_index = {protein: index for index, protein in enumerate(proteins)}
    size = len(proteins)
    evalues = np.full((size, size), 10.0, dtype=np.float64)
    for query, subjects in homology.items():
        query_index = protein_index[query]
        for subject, evalue in subjects.items():
            subject_index = protein_index.get(subject)
            if subject_index is not None:
                evalues[query_index, subject_index] = evalue

    pair_evalues = np.maximum(evalues, evalues.T)
    graph = np.zeros_like(pair_evalues)
    diagonal = np.eye(size, dtype=bool)
    ones = (pair_evalues == 0) | diagonal
    graph[ones] = 1.0
    transform = (~ones) & (pair_evalues > 0) & (pair_evalues < 11)
    graph[transform] = -np.log(pair_evalues[transform] / 11.0)
    maxi = graph[transform].max(initial=-1.0)
    if maxi <= 0:
        raise RuntimeError('Unable to normalise an empty homology graph')
    graph[graph != 1] *= 1.0 / maxi
    return proteins, graph

class Homology(Graph):

    def __init__(self, fasta, proteins, graphs_dir, alias, protein_format,
                 cpu='infer', engine='legacy'):
        super(Homology, self).__init__()
        self.fasta = fasta
        self.proteins = proteins
        self.graphs_dir = graphs_dir
        self.alias = alias
        self.homology_graph = None
        self.protein_format = protein_format
        self.cpu = cpu
        self.engine = engine
        if self.engine not in ('legacy', 'vectorized'):
            raise ValueError('Unknown homology engine: ' + self.engine)
        if self.cpu == 'infer':
            # https://docs.python.org/3/library/os.html#os.cpu_count
            self.cpu = len(os.sched_getaffinity(0))

    def get_graph(self, **kwargs):
        if sparse.issparse(self.homology_graph):
            return self.homology_graph.tocoo()
        h = self.homology_graph.merge(self.proteins, left_on='Protein 1',
                                      right_index=True)
        h = h.merge(self.proteins, left_on='Protein 2', right_index=True,
                    suffixes=['1', '2'])
        p1_idx = h['protein idx1'].values
        p2_idx = h['protein idx2'].values
        return sparse.coo_matrix((h['weight'], (p1_idx, p2_idx)),
                                 shape=(len(self.proteins),
                                        len(self.proteins)))

    def write_graph(self, filename):
        if sparse.issparse(self.homology_graph):
            sparse.save_npz(filename, self.homology_graph)
            return
        Graph.assert_lexicographical_order(self.homology_graph)
        self.homology_graph.to_csv(filename, sep='\t')

    @staticmethod
    def _atomic_sparse(matrix, filename):
        temporary = filename + '.tmp.' + str(os.getpid()) + '.npz'
        try:
            sparse.save_npz(temporary, matrix)
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _parse_blast(self, filename):
        homology = {}
        for line in open(filename):
            fields = line.strip().split('\t')
            query_id = fields[0]
            subject_id = fields[1]
            if self.protein_format == 'uniprot':
                query_id = Utilities.extract_uniprot_accession(query_id)
                subject_id = Utilities.extract_uniprot_accession(subject_id)
            evalue = float(fields[10])
            if query_id not in homology:
                homology[query_id] = {}
            if subject_id in homology[query_id]:
                homology[query_id][subject_id] = np.min([
                    homology[query_id][subject_id], evalue])
            else:
                homology[query_id][subject_id] = evalue
        return homology

    def _vectorized_sparse_graph(self, homology):
        ordered_proteins, graph = vectorized_homology_graph(homology)
        rows, cols = np.triu_indices(len(ordered_proteins))
        values = graph[rows, cols]
        mapping = self.proteins['protein idx'].to_dict()
        order_to_target = np.fromiter(
            (mapping[name] for name in ordered_proteins), dtype=np.int64,
            count=len(ordered_proteins))
        protein_rows = order_to_target[rows]
        protein_cols = order_to_target[cols]
        return sparse.coo_matrix(
            (values, (protein_rows, protein_cols)),
            shape=(len(self.proteins), len(self.proteins)))

    def compute_graph(self):
        homology_graph = os.path.join(self.graphs_dir, self.alias)
        sparse_graph = homology_graph + '.npz'
        if self.engine == 'vectorized' and os.path.exists(sparse_graph):
            self.tell('Vectorized homology graph found, skipping computation...')
            self.homology_graph = sparse.load_npz(sparse_graph).tocoo()
            return
        if not os.path.exists(homology_graph):
            self.tell('Computing homology graph...')
            # compute the homology graph
            blast_command = "blastp -query {fasta} -db {blastdb} " +\
                            "-out {out}" + ' -outfmt "6 std qlen"' +\
                            " -num_threads {cpu}"
            out = os.path.join(self.graphs_dir, self.alias + '_homology.blast')
            if not os.path.exists(out):
                self.tell(blast_command.format(fasta=self.fasta,
                                               blastdb=self.fasta,
                                               out=out,
                                               cpu=self.cpu))
                subprocess.call(blast_command.format(fasta=self.fasta,
                                                     blastdb=self.fasta,
                                                     out=out,
                                                     cpu=self.cpu), shell=True)

            self.tell('Parsing BLAST output...')
            homology = self._parse_blast(out)

            if self.engine == 'vectorized':
                self.tell('Building vectorized homology graph...')
                self.homology_graph = self._vectorized_sparse_graph(homology)
                self._atomic_sparse(self.homology_graph, sparse_graph)
                return

            self.tell('Building graph...')
            hom_graph = {'Protein 1': [], 'Protein 2': [], 'weight': []}
            proteins = sorted(homology.keys())
            graph = np.zeros((len(proteins), len(proteins)))
            maxi = -1.0
            for i, p1 in enumerate(proteins):
                for j in range(i, len(proteins)):
                    p2 = proteins[j]
                    e12 = 10.0
                    e21 = 10.0

                    if p2 in homology[p1]:
                        e12 = homology[p1][p2]
                    if p1 in homology[p2]:
                        e21 = homology[p2][p1]

                    val = np.max([e12, e21])
                    transformed = 0.0

                    if val == 0 or p1 == p2:
                        transformed = 1.0
                    elif 0.0 < val < 11.0:
                        transformed = -1.0 * np.log(val/11.0)
                        maxi = np.max([maxi, transformed])
                    graph[i, j] = transformed
                    graph[j, i] = transformed

            inv_max = 1.0 / maxi
            mask_neg = graph != 1

            graph[mask_neg] *= inv_max
            self.tell('Normalising homology graph...')
            for i, p1 in enumerate(proteins):
                for j in range(i, len(proteins)):
                    val = graph[i, j]
                    p2 = proteins[j]
                    hom_graph['Protein 1'].append(p1)
                    hom_graph['Protein 2'].append(p2)
                    hom_graph['weight'].append(val)
            self.homology_graph = pd.DataFrame.from_dict(hom_graph)

            # we make sure that lexicographical order is maintained so that
            # all values are kept on the upper triangle in the matrix
            Graph.assert_lexicographical_order(self.homology_graph)

            self.homology_graph.to_pickle(homology_graph)
        else:
            self.tell('Homology graph found, skipping computation...')
            self.homology_graph = pd.read_pickle(homology_graph)
            Graph.assert_lexicographical_order(self.homology_graph)
