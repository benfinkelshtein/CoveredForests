from argparse import ArgumentParser
from torch.cuda import set_device
import neptune
from neptune.utils import stringify_unsupported
import torch
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import networkx as nx

from utils.graph_classes import RealGraphs
from utils.distances import DistanceType, adj_dist_mat
from utils.helpers import print_args, nx_add_features
from utils.constants import API_TOKEN
from utils.trainer import GINModel, trainer
from utils.set_cover import solve_set_cover

BATCH_SIZE = 128
NUM_SEEDS = 5
LR = 0.001

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--project", dest="project", default="CoveredForestsArxiv/Sandbox", type=str, required=False)
    parser.add_argument("--graph_family", dest="graph_family", default=RealGraphs.MUTAG,
                        type=RealGraphs.from_string, choices=list(RealGraphs), required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=3, type=int, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=None, type=int, required=False)
    parser.add_argument("--epochs", dest="epochs", default=500, type=int, required=False)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train model NUM_SEEDS times
    dataset = args.graph_family.load()
    encoder_type = args.graph_family.get_encoder_type()
    loss_list, metric_list = [], []
    for seed in range(NUM_SEEDS):
        print(f"Seed {seed}")
        model = GINModel(in_dim=dataset[0].x.shape[1], hidden_dim=args.hidden_dim, out_dim=1,
                         num_layers=args.num_layers, encoder_type=encoder_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model, losses_n_metrics = \
            trainer(dataset=dataset, model=model, optimizer=optimizer, batch_size=BATCH_SIZE, epochs=args.epochs,
                    device=device, neptune_logger=neptune_logger)
        loss_list.append(losses_n_metrics.get_losses())
    loss_tensor = torch.stack(loss_list, dim=1)
    mean_loss, std_loss = loss_tensor.mean(dim=1), loss_tensor.std(dim=1)

    # neptune record
    neptune_logger[f"train_loss_mean"] = mean_loss[0].item()
    neptune_logger[f"test_loss_mean"] = mean_loss[2].item()
    neptune_logger[f"diff_loss_mean"] = (loss_tensor[0] - loss_tensor[2]).mean()
    neptune_logger[f"train_loss_std"] = std_loss[0].item()
    neptune_logger[f"test_loss_std"] = std_loss[2].std()
    neptune_logger[f"diff_loss_std"] = (loss_tensor[0] - loss_tensor[2]).std()
    print(f'mean std: {mean_loss}')
    print(f'loss std: {std_loss}')

    # the list of graphs which are 1-wl distinguishable with their features
    max_nodes = max([data.num_nodes for data in dataset])
    hash_dict = {}
    feature_tensor = []
    for idx, data in enumerate(dataset):
        # m_n_d_l
        graph = nx.Graph()
        graph.add_nodes_from(list(range(max_nodes)))
        graph.add_edges_from(data.edge_index.t().tolist())
        graph_with_features = nx_add_features(nx_graph=graph, features=data.x)
        hash = weisfeiler_lehman_graph_hash(graph_with_features, iterations=args.num_layers)
        if hash not in hash_dict:
            hash_dict[hash] = graph

        # feature tensor
        if max_nodes > data.x.shape[0]:
            zero_features = torch.zeros(size=(max_nodes - data.x.shape[0], data.x.shape[1]), dtype=data.x.dtype)
            padded_features = torch.cat((data.x, zero_features), dim=0)
        else:
            padded_features = data.x
        feature_tensor.append(padded_features)
    padded_graphs = list(hash_dict.values())
    m_n_d_l = len(padded_graphs)
    feature_tensor = torch.stack(feature_tensor, dim=0)
    neptune_logger[f"max_nodes"] = max_nodes
    neptune_logger[f'm_n_d_{args.num_layers}'] = m_n_d_l
    print(f'max_nodes: {max_nodes}')
    print(f'm_n_d_{args.num_layers}: {m_n_d_l}')

    # uncomment if you would like to calculate the set cover
    # num_radiuses = len(args.radius_list)
    # dist_mat = adj_dist_mat(elements=padded_graphs, distance_type=DistanceType.FD, num_layers=args.num_layers,
    #                         feature_tensor=feature_tensor)
    # for idx, radius in enumerate(args.radius_list):
    #     print(f'radius idx: {idx + 1}/{num_radiuses}')
    #     cover = solve_set_cover(dist_mat=dist_mat, radius=radius)
    #     neptune_logger[f'cover_r_{radius}'] = cover
    #     print(f'max_nodes: {max_nodes}, radius idx: {idx + 1}/{num_radiuses}, cover: {cover}')
