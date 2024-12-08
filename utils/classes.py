from enum import Enum, auto
from typing import NamedTuple
from torch.nn import Linear
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder

from utils.zinc_encoders import TypeDictNodeEncoder


class LossesAndMetrics(NamedTuple):
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_metrics(self):
        return torch.tensor([self.train_metric, self.val_metric, self.test_metric])

    def get_losses(self):
        return torch.tensor([self.train_loss, self.val_loss, self.test_loss])


class EncoderType(Enum):
    """
        an object for the different encoders
    """
    NONE = auto()
    MOL = auto()
    ZINC = auto()
    QM9 = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()

    def node_encoder(self, in_dim: int, emb_dim: int):
        if self is EncoderType.NONE:
            return None
        elif self is EncoderType.QM9:
            return Linear(in_dim, emb_dim)
        elif self is EncoderType.MOL:
            return AtomEncoder(emb_dim)
        elif self is EncoderType.ZINC:
            return TypeDictNodeEncoder(emb_dim)
        else:
            raise ValueError(f'EncoderType {self.name} not supported')
