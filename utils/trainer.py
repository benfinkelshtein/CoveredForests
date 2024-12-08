import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch.nn import ModuleList, Linear, BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from typing import Tuple, List
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torchmetrics import Accuracy

from utils.classes import EncoderType, LossesAndMetrics

TASK_LOSS = BCEWithLogitsLoss()
METRIC = Accuracy(task='binary')


# Define the GIN model class
class GINModel(torch.nn.Module):
    def __init__(self, encoder_type: EncoderType, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super(GINModel, self).__init__()
        self.out_dim = out_dim
        self.node_enc = encoder_type.node_encoder(in_dim=in_dim, emb_dim=hidden_dim)
        if self.node_enc is None:
            dim_list = [in_dim]
        else:
            dim_list = [hidden_dim]
        dim_list = dim_list + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = [GINConv(nn=Linear(in_dim_i, out_dim_i), train_eps=True)
                  for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        self.layers = ModuleList(layers)

    def forward(self, x: Tensor, edge_index: Adj, batch: OptTensor = None):
        if self.node_enc is not None:
            x = self.node_enc(x)
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        x = global_add_pool(x, batch)  # Pooling over the graph
        return x


def train(model, loader, optimizer, device) -> float:
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        if data.y.dim() < out.dim():
            out = out.squeeze(dim=-1)
        loss = TASK_LOSS(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def test(model, loader, device) -> Tuple[float, float]:
    model.eval()

    total_y = None
    total_loss = 0
    total_scores = torch.empty(size=(0, model.out_dim))
    for data in loader:
        data = data.to(device)
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)

        # metric
        with torch.no_grad():
            total_scores = torch.cat((total_scores, out.detach().cpu()), dim=0)
            if total_y is None:
                total_y = data.y.cpu()
            else:
                total_y = torch.cat((total_y, data.y.cpu()), dim=0)

        # loss
        if data.y.dim() < out.dim():
            out = out.squeeze(dim=-1)
        loss = TASK_LOSS(out, data.y.float())
        total_loss += loss.item() * data.num_graphs

    if total_y.dim() < total_scores.dim():
        total_scores = total_scores.squeeze(dim=-1)
    metric = METRIC(total_scores, total_y)
    loss = total_loss / len(loader.dataset)
    return loss, metric


def trainer(dataset: List[Data], model: GINModel, optimizer, batch_size: int, epochs: int, device,
            neptune_logger) -> Tuple[GINModel, LossesAndMetrics]:
    dataset = dataset[torch.randperm(len(dataset))]
    split_point1 = (len(dataset) // 10)
    split_point2 = 2 * (len(dataset) // 10)
    test_dataset = dataset[:split_point1]
    val_dataset = dataset[split_point1:split_point2]
    train_dataset = dataset[split_point2:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_losses_n_metrics = LossesAndMetrics(train_loss=float('inf'), val_loss=float('inf'),
                                             test_loss=float('inf'), train_metric=float('-inf'),
                                             val_metric=float('-inf'), test_metric=float('-inf'),)
    best_model_dict = model.state_dict()
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, device)
        train_loss, train_metric = test(model, train_loader, device)
        val_loss, val_metric = test(model, val_loader, device)
        test_loss, test_metric = test(model, test_loader, device)
        losses_n_metrics = \
            LossesAndMetrics(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                             train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)
        for name in losses_n_metrics._fields:
            neptune_logger[name].append(getattr(losses_n_metrics, name))

        if losses_n_metrics.val_metric > best_losses_n_metrics.val_metric:
            best_losses_n_metrics = losses_n_metrics
            best_model_dict = model.state_dict()
        print(f"Epoch: {epoch: 03d}, Loss: {loss: .4f}, Train met: {train_metric: .4f}, Test met: {test_metric: .4f}")
    model.load_state_dict(best_model_dict)
    return model, best_losses_n_metrics

