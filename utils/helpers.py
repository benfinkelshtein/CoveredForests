from argparse import ArgumentParser
import networkx as nx
from torch import Tensor


def print_args(args: ArgumentParser):
    """
        a print of the arguments

        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()


def nx_add_features(nx_graph: nx.Graph, features: Tensor):
    """
    Combine node features into a string representation to initialize WL hashing.
    """
    num_nodes, feature_dim = features.shape
    for node in nx_graph.nodes:
        if node >= num_nodes:
            feature_vector = [0] * feature_dim
        else:
            feature_vector = features[node].tolist()
        nx_graph.nodes[node]['label'] = str(feature_vector)
    return nx_graph
