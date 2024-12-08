from argparse import ArgumentParser
from torch.cuda import set_device
import networkx as nx
import neptune
from neptune.utils import stringify_unsupported
from collections import defaultdict

from utils.helpers import print_args
from utils.constants import API_TOKEN
from utils.set_cover import solve_set_cover
from utils.distances import adj_dist_mat
from utils.graph_classes import RealGraphs
from utils.distances import DistanceType


if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--project", dest="project", default="CoveredForestsArxiv/Sandbox", type=str, required=False)
    parser.add_argument("--graph_family", dest="graph_family", default=RealGraphs.MUTAG,
                        type=RealGraphs.from_string, choices=list(RealGraphs), required=False)
    parser.add_argument("--distance_type", dest="distance_type", default=DistanceType.FD,
                        type=DistanceType.from_string, choices=list(DistanceType), required=False)
    parser.add_argument(
        "--graph_size_list",
        dest="graph_size_list",
        nargs="+",  # "+" allows one or more arguments
        type=int,  # Ensure each item is an integer
        required=True  # Make it required if you want to ensure it’s always provided
    )
    parser.add_argument(
        "--radius_list",
        dest="radius_list",
        nargs="+",  # "+" allows one or more arguments
        type=float,  # Ensure each item is an integer
        required=True  # Make it required if you want to ensure it’s always provided
    )
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    args = parser.parse_args()
    if args.gpu is not None:
        set_device(args.gpu)
    print_args(args=args)
    neptune_logger = neptune.init_run(project=args.project, api_token=API_TOKEN)  # your credentials
    neptune_logger["params"] = stringify_unsupported({arg: getattr(args, arg) for arg in vars(args)})

    # calculate the coverings given a radius and graph size
    num_nodes = len(args.graph_size_list)
    num_radiuses = len(args.radius_list)
    dataset = args.graph_family.load()
    covers = defaultdict(list)
    m_graph_size_list = []
    for idx, n in enumerate(args.graph_size_list):
        # pad all graphs with size < n  to n
        str_prefix = f'Node idx: {idx+1}/{num_nodes}'
        padded_graphs = []
        hash_set = set()
        for data in dataset:
            current_num_nodes = data.num_nodes
            graph = nx.Graph()
            graph.add_nodes_from(list(range(current_num_nodes)))
            graph.add_edges_from(data.edge_index.t().tolist())
            if current_num_nodes < n:
                graph.add_nodes_from(list(range(n)))
            elif current_num_nodes > n:
                continue
            hash = nx.weisfeiler_lehman_graph_hash(graph)  # real-world datasets
            if hash not in hash_set:
                hash_set.add(hash)
                padded_graphs.append(graph)
        m_n = len(padded_graphs)
        m_graph_size_list.append(m_n)
        neptune_logger[f'm_{n}'] = m_n

        # calculate covers per n for varying radii
        dist_mat = adj_dist_mat(elements=padded_graphs, distance_type=args.distance_type)
        for idx, radius in enumerate(args.radius_list):
            cover = solve_set_cover(dist_mat=dist_mat, radius=radius)
            covers[radius].append(cover)
            print(str_prefix + f', radius idx: {idx + 1}/{num_radiuses}, cover: {cover}')
            neptune_logger[f'cover_n_{n}_r_{radius}'] = cover
