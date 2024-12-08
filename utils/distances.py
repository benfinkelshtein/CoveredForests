import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.classes.graph import Graph
from typing import List, Optional
from enum import Enum, auto
import torch
from torch_geometric.typing import OptTensor
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

from utils.tmd import TMD


class DistanceType(Enum):
    """
        an object for the different activation types
    """
    L1 = auto()
    FD = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DistanceType[s]
        except KeyError:
            raise ValueError()

    def adj_dist(self, A: sp.csr_matrix, B: sp.csr_matrix, x_A: OptTensor = None, x_B: OptTensor = None,
                 num_layers: Optional[int] = 3):
        """
        Compute the distance between two adjacency matrices A and B:
        d(A,B) = min_{S doubly stochastic} ||AS - SB||_norm_type
        """
        if self is DistanceType.L1:
            if x_A is None or x_B is None:
                L = None
            else:
                L = sp.csr_matrix((torch.cdist(x_A.float(), x_B.float(), p=2).numpy()))
            S = cp.Variable((A.shape[0], A.shape[0]))
            objective = cp.sum(cp.norm(A @ S - S @ B, p=1))
            if L is not None:
                objective = objective + cp.trace(S.T @ L)
            objective = cp.Minimize(objective)
            problem = cp.Problem(objective, [cp.sum(S, axis=0) == 1, cp.sum(S, axis=1) == 1, S >= 0])
            problem.solve()
            return problem.value
        elif self is DistanceType.FD:
            edge_index_A, _ = from_scipy_sparse_matrix(A)
            edge_index_B, _ = from_scipy_sparse_matrix(B)
            if x_A is None:
                x_A = torch.ones(size=(A.shape[0], 1), dtype=torch.float)
            else:
                x_A = torch.tensor(x_A, dtype=torch.float32)
            data_A = Data(x=x_A, edge_index=edge_index_A)

            if x_B is None:
                x_B = torch.ones(size=(A.shape[0], 1), dtype=torch.float)
            else:
                x_B = torch.tensor(x_B, dtype=torch.float32)
            data_B = Data(x=x_B, edge_index=edge_index_B)
            return TMD(g1=data_A, g2=data_B, w=1, L=num_layers)
        else:
            raise ValueError(f'DistanceType {self.name} not supported')


def adj_dist_mat(elements: List[Graph], distance_type: DistanceType, num_layers: Optional[int] = 3,
                 feature_tensor: OptTensor = None) -> np.ndarray:
    n = len(elements)
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        csr_adj_i = nx.adjacency_matrix(elements[i])
        for j in range(i + 1, n):
            csr_adj_j = nx.adjacency_matrix(elements[j])
            x_A, x_B = None, None
            if feature_tensor is not None:
                x_A = feature_tensor[i]
                x_B = feature_tensor[j]
            dist_mat[i, j] = distance_type.adj_dist(A=csr_adj_i, B=csr_adj_j, num_layers=num_layers,
                                                    x_A=x_A, x_B=x_B)
    dist_mat = dist_mat + dist_mat.T
    return dist_mat

