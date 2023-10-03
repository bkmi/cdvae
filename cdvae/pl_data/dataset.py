from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal
import joblib
import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


def preprocess_or_load(
        path: Path | str,
        num_workers: int, 
        niggli: bool, 
        primitive: bool, 
        graph_method: Literal["crystalnn", "none"],
        prop_list: list,
    ) -> list:
    preproc_path = Path(path).with_suffix('.joblib')
    if preproc_path.exists():
        # cached_data = pickle.loads(preproc_path.read_bytes())
        cached_data = joblib.load(preproc_path)
    else:
        cached_data = preprocess(
            path,
            num_workers=num_workers,
            niggli=niggli,
            primitive=primitive,
            graph_method=graph_method,
            prop_list=prop_list)
        print("Writing pickle file...", preproc_path)
        # preproc_path.write_bytes(pickle.dumps(cached_data))
        joblib.dump(cached_data, preproc_path)
    return cached_data


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, cond: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.cond = cond
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_or_load(
            path=self.path,
            num_workers=preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop, cond],
        )
        # # preproc_path = Path(self.path).with_suffix('.pkl')
        # preproc_path = Path(self.path).with_suffix('.joblib')
        # if preproc_path.exists():
        #     # self.cached_data = pickle.loads(preproc_path.read_bytes())
        #     self.cached_data = joblib.load(preproc_path)
        # else:
        #     self.cached_data = preprocess(
        #         self.path,
        #         preprocess_workers,
        #         niggli=self.niggli,
        #         primitive=self.primitive,
        #         graph_method=self.graph_method,
        #         prop_list=[prop, cond])
        #     print("Writing pickle file...", preproc_path)
        #     # preproc_path.write_bytes(pickle.dumps(self.cached_data))
        #     joblib.dump(self.cached_data, preproc_path)
        
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None
        self.cond_scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        # cond = self.cond_scaler.transform(data_dict[self.cond])
        cond = torch.tensor([data_dict[self.cond]]).float()
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            cond=cond.view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)
    cond_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.cond)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    dataset.cond_scaler = cond_scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
