import os
import numpy as np
from typing import Callable, List, Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import stochastic_blockmodel_graph


class StochasticBlockModelBlobDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            block_sizes: Union[List[int], Tensor],
            edge_probs: Union[List[List[float]], Tensor],
            num_channels: Optional[int] = None,
            centers=None,
            is_undirected: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            flip_y: float = 0.01,
            **kwargs,
    ):
        if not isinstance(block_sizes, torch.Tensor):
            block_sizes = torch.tensor(block_sizes, dtype=torch.long)
        if not isinstance(edge_probs, torch.Tensor):
            edge_probs = torch.tensor(edge_probs, dtype=torch.float)

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.num_channels = num_channels
        self.is_undirected = is_undirected
        self.flip_y = flip_y
        self.kwargs = {
            'centers': centers,
            'shuffle': False,
        }
        self.kwargs.update(kwargs)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        from sklearn.datasets import make_blobs

        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)

        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)

        x = None
        if self.num_channels is not None:

            x, y_not_sorted, c = make_blobs(n_samples=num_samples, n_features=self.num_channels, return_centers=True,
                                            **self.kwargs)
            x = x[np.argsort(y_not_sorted)]
            x = torch.from_numpy(x).to(torch.float)

        y = torch.arange(num_classes).repeat_interleave(self.block_sizes)
        if self.flip_y >= 0.0:
            flip_mask = torch.bernoulli(self.flip_y * torch.ones_like(y)).type(torch.bool)
            y[flip_mask] = torch.randint(num_classes, size=(int(flip_mask.sum().item()),))
        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def load_sbm_dataset(data_dir:str, snapshot:int):
    assert snapshot < len(next(os.walk(data_dir))[1]) and snapshot >= 0
    data, _ = torch.load(os.path.join(data_dir, str(snapshot), "StochasticBlockModelBlobDataset", "processed", "data.pt"))
    return data