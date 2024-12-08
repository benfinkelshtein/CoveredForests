from argparse import ArgumentParser
from torch.cuda import set_device
import neptune
from neptune.utils import stringify_unsupported
import torch
from collections import defaultdict
import random
import networkx as nx

from utils.distances import DistanceType
from utils.graph_classes import RealGraphs
from utils.helpers import print_args
from utils.constants import API_TOKEN
from utils.trainer import GINModel

BATCH_SIZE = 128
LR = 0.001
EPOCHS = 500
SAMPLES = 1000

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--project", dest="project", default="CoveredForestsArxiv/Sandbox", type=str, required=False)
    parser.add_argument("--graph_family", dest="graph_family", default=RealGraphs.MUTAG,
                        type=RealGraphs.from_string, choices=list(RealGraphs), required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=None, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=None, type=int, required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    args = parser.parse_args()
    if args.gpu is not None:
        set_device(args.gpu)
    print_args(args=args)
    neptune_logger = neptune.init_run(project=args.project, api_token=API_TOKEN)  # your credentials
    neptune_logger["params"] = stringify_unsupported({arg: getattr(args, arg) for arg in vars(args)})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # partition the graphs in the datasets per size
    dataset = args.graph_family.load()
    graphs_by_size = defaultdict(list)
    for idx, data in enumerate(dataset):
        graphs_by_size[data.x.shape[0]].append(idx)
    keys_to_remove = [k for k, v in graphs_by_size.items() if len(v) < 2]
    for k in keys_to_remove:
        graphs_by_size.pop(k)
    num_keys = len(graphs_by_size.keys())

    # sample pairs of graphs
    encoder_type = args.graph_family.get_encoder_type()
    for idx in range(SAMPLES):
        # sample a pair of graphs
        graph_size = random.sample(list(graphs_by_size.keys()), 1)[0]
        graph_indices = graphs_by_size[graph_size]
        g1_idx, g2_idx = random.sample(graph_indices, 2)
        data1, data2 = dataset[g1_idx], dataset[g2_idx]

        # convert to the pair to nx
        graph1, graph2 = nx.Graph(), nx.Graph()
        graph1.add_nodes_from(list(range(graph_size)))
        graph2.add_nodes_from(list(range(graph_size)))
        graph1.add_edges_from(data1.edge_index.t().tolist())
        graph2.add_edges_from(data2.edge_index.t().tolist())

        # calculate the graph distance
        A, B = nx.adjacency_matrix(graph1), nx.adjacency_matrix(graph2)
        graph_dist = DistanceType.FD.adj_dist(A=A, B=B, x_A=data1.x, x_B=data2.x, num_layers=args.num_layers)
        neptune_logger[f"stochastic_{idx}"] = graph_dist

        # calculate the mpnn distance
        if args.num_layers > 0:
            model = GINModel(in_dim=dataset[0].x.shape[1], hidden_dim=args.hidden_dim, out_dim=1,
                             num_layers=args.num_layers, encoder_type=encoder_type).to(device)
        else:
            model = lambda x, edge_index: x
        out1 = model(x=data1.x.to(device), edge_index=data1.edge_index.to(device))
        out2 = model(x=data2.x.to(device), edge_index=data2.edge_index.to(device))
        mpnn_dist = torch.norm(out1.float() - out2.float(), p='fro')
        neptune_logger[f"mpnn_{idx}"] = mpnn_dist
        print(f'Sample {idx + 1}/{SAMPLES}, MPNN: {mpnn_dist}, Graph: {graph_dist}')
