import math
from enum import Enum, auto
import numpy as np
import networkx as nx
from itertools import combinations
from networkx.classes.graph import Graph
from typing import List, Optional, Dict
from collections import defaultdict
import copy
from itertools import product
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset

from utils.classes import EncoderType


def generate_per_n_disjoint_paths(n: int, k: int) -> List[Graph]:
    graph_list = []
    hash_set = set()
    for num_zeros in range(k):
        num_ones = n - 1 - num_zeros
        start_pos_of_edges = np.array(list(combinations(range(n - 1), num_ones))).T  # (|E|, |G|)
        edge_tensor = np.stack((start_pos_of_edges, start_pos_of_edges + 1), axis=1)  # (|E|, 2, |G|)
        for i in range(edge_tensor.shape[2]):
            graph = nx.Graph()  # undirected
            graph.add_nodes_from(range(n))
            graph.add_edges_from(edge_tensor[:, :, i])
            hash = nx.weisfeiler_lehman_graph_hash(graph)  # disjoint_paths
            if hash not in hash_set:
                hash_set.add(hash)
                graph_list.append(graph)

    return graph_list


class UniqueGraphs(Enum):
    """
        an object for the different activation types
    """
    disjoint_paths = auto()
    binary_trees = auto()
    all = auto()
    otter = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return UniqueGraphs[s]
        except KeyError:
            raise ValueError()

    def generate_per_n(self, n: int, k: int, generated_graphs: Optional[Dict[int, List[Graph]]]) -> List[Graph]:
        assert (k is not None and self.has_disjoint_paths()) or (k is None and not self.has_disjoint_paths()),\
            f"k is an irrelavent argument for UniqueGraphs {self.name}"
        if self is UniqueGraphs.disjoint_paths:
            return generate_per_n_disjoint_paths(n=n, k=k)
        if self is UniqueGraphs.all:
            graph_list = []
            # Generate all possible edges
            nodes = list(range(n))
            all_possible_edges = list(combinations(nodes, 2))
            # Iterate over all subsets of edges
            hash_set = set()
            for i in range(2 ** len(all_possible_edges)):
                graph = nx.Graph()  # undirected
                graph.add_nodes_from(nodes)
                edges = [all_possible_edges[j] for j in range(len(all_possible_edges)) if (i >> j) & 1]
                graph.add_edges_from(edges)
                hash = nx.weisfeiler_lehman_graph_hash(graph)  # all graphs
                if hash not in hash_set:
                    hash_set.add(hash)
                    graph_list.append(graph)
            return graph_list
        elif self is UniqueGraphs.binary_trees:
            return generated_graphs[n]
        elif self is UniqueGraphs.otter:
            assert n % 2 == 0, "otter construction is defined for even size graphs solely"
            half_n = n // 2
            disjoint_paths_list = generate_per_n_disjoint_paths(n=half_n, k=k)  # undirected
            path_graph = nx.path_graph(half_n, create_using=nx.Graph)  # undirected

            # Loop over each graph in the input list
            merged_graphs = []
            hash_set = set()
            for graph in disjoint_paths_list:
                # For every possible combination of nodes from the path graph and the current graph
                for path_node, graph_node in product(path_graph.nodes, graph.nodes):
                    # Create a copy of both graphs to avoid modifying the originals
                    new_graph = nx.disjoint_union(path_graph, graph)
                    new_graph.add_edge(graph_node, path_node + half_n)
                    new_graph.add_edge(path_node + half_n, graph_node)

                    # Append the merged graph to the list
                    hash = nx.weisfeiler_lehman_graph_hash(new_graph)  # otter graphs
                    if hash not in hash_set:
                        hash_set.add(hash)
                        merged_graphs.append(new_graph)
            return merged_graphs
        else:
            raise ValueError(f'UniqueGraphs {self.name} not supported')

    def generate_forall_n(self, graph_size_list: List[int]) -> Optional[Dict[int, List[Graph]]]:
        if self is UniqueGraphs.binary_trees:
            max_n = max(graph_size_list)
            graph_list_forall_n = defaultdict(list)

            assert len([n for n in graph_size_list if n % 2 == 0]) == 0,\
                "two_child_tree construction is defined for odd size graphs solely"
            base_graph = nx.Graph()  # undirected
            base_graph.add_nodes_from([0, 1, 2])
            base_graph.add_edge(0, 1)
            base_graph.add_edge(0, 2)

            iterations = int((max_n - 3) / 2)
            graph_list = [[base_graph, 0]]
            for idx in range(iterations):  # we don\'t actually need the last iteration
                hash_set = set()
                num_nodes = 5 + 2 * idx
                new_graph_list = []
                for graph, graph_idx in graph_list:
                    leaf_generator = (x for x in graph.nodes() if graph.degree(x) == 1)
                    for leaf_idx in leaf_generator:
                        new_graph = copy.deepcopy(graph)
                        new_graph.add_nodes_from([num_nodes, num_nodes + 1])
                        new_graph.add_edge(leaf_idx, num_nodes)
                        new_graph.add_edge(leaf_idx, num_nodes + 1)

                        hash = nx.weisfeiler_lehman_graph_hash(new_graph)  # binary_trees
                        if hash not in hash_set:
                            hash_set.add(hash)
                            graph_idx += 1
                            new_graph_list.append([new_graph, graph_idx])
                graph_list = copy.deepcopy(new_graph_list)

                if num_nodes in graph_size_list:
                    graph_list_forall_n[num_nodes] = list(map(lambda x: x[0], graph_list))
            return graph_list_forall_n
        else:
            return None

    def has_disjoint_paths(self) -> bool:
        return self in [UniqueGraphs.disjoint_paths, UniqueGraphs.otter]

    def is_otter(self) -> bool:
        return self is UniqueGraphs.otter

    def get_title(self) -> str:
        if self is UniqueGraphs.all:
            return r'$\mathcal{G}_n$'
        elif self is UniqueGraphs.otter:
            return r'$\mathcal{F}_n$'
        elif self is UniqueGraphs.binary_trees:
            return r'$\mathcal{T}_n^{(2)}$'
        else:
            raise ValueError(f'get_title is not supported for {self.name}')


class RealGraphs(Enum):
    """
        an object for the different activation types
    """
    MOLHIV = auto()
    MUTAG = auto()
    NCI1 = auto()
    MCF_7H = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return RealGraphs[s]
        except KeyError:
            raise ValueError()

    def load(self):
        if self is RealGraphs.MOLHIV:
            return PygGraphPropPredDataset(name='ogbg-molhiv', root='datasets/MOLHIV')
        elif self in [RealGraphs.MUTAG, RealGraphs.NCI1, RealGraphs.MCF_7H]:
            return TUDataset(root='datasets/' + self.name, name=self.name.replace('_', '-'), use_node_attr=True)
        else:
            raise ValueError(f'RealGraphs {self.name} not supported')

    def get_encoder_type(self) -> EncoderType:
        if self in [RealGraphs.MUTAG, RealGraphs.NCI1, RealGraphs.MCF_7H]:
            return EncoderType.NONE
        elif self is RealGraphs.MOLHIV:
            return EncoderType.MOL
        else:
            raise ValueError(f'RealGraphs {self.name} not supported in dataloader')

    def our_bound(self, m_n: int, radius: float) -> Optional[float]:
        k = math.floor(radius / 4)
        k = 1 if k < 1 else k
        bound = m_n / (k + 1)
        return bound
