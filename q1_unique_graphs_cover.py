from torch.cuda import set_device
import neptune
from neptune.utils import stringify_unsupported
from argparse import ArgumentParser
from collections import defaultdict

from utils.graph_classes import UniqueGraphs
from utils.distances import DistanceType
from utils.distances import adj_dist_mat
from utils.constants import API_TOKEN
from utils.helpers import print_args
from utils.set_cover import solve_set_cover


if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--project", dest="project", default="CoveredForestsArxiv/Sandbox", type=str,
                        required=False)
    parser.add_argument("--graph_family", dest="graph_family", default=UniqueGraphs.all,
                        type=UniqueGraphs.from_string, choices=list(UniqueGraphs), required=False)
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
    parser.add_argument("--k", dest="k", default=None, type=int, required=False)
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
    generated_graphs = args.graph_family.generate_forall_n(graph_size_list=args.graph_size_list)  # generate binary_trees
    covers = defaultdict(list)
    m_graph_size_list = []
    for idx, n in enumerate(args.graph_size_list):
        str_prefix = f'Node idx: {idx+1}/{num_nodes}'
        # generate datasets per n
        graph_list = args.graph_family.generate_per_n(n=n, k=args.k, generated_graphs=generated_graphs)
        m_n = len(graph_list)
        m_graph_size_list.append(m_n)
        neptune_logger[f'm_{n}'] = m_n

        # calculate covers per n for varying radii
        dist_mat = adj_dist_mat(elements=graph_list, distance_type=DistanceType.FD)
        for idx, radius in enumerate(args.radius_list):
            cover = solve_set_cover(dist_mat=dist_mat, radius=radius)
            covers[radius].append(cover)
            print(str_prefix + f', radius idx: {idx + 1}/{num_radiuses}, cover: {cover}')
            neptune_logger[f'cover_n_{n}_r_{radius}'] = cover
