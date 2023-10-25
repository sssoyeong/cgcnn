# https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py

from __future__ import print_function, division
from typing import List

import csv
import functools
import json
import os
from pathlib import Path
import random
import warnings
import glob
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


class collate_pool_for_paired:
    """
    Collate() a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """

    def __init__(self, device=None, non_blocking=True):
        self.device_kwargs = {"device": device, "non_blocking": non_blocking}

    def __call__(self, dataset_list):
        batch_atom_fea_A, batch_nbr_fea_A, batch_nbr_fea_idx_A, crystal_atom_idx_A = [], [], [], []
        batch_atom_fea_B, batch_nbr_fea_B, batch_nbr_fea_idx_B, crystal_atom_idx_B = [], [], [], []
        base_idx_A, base_idx_B = 0, 0

        batch_target, batch_cif_ids = [], []

        for (
            ((atom_fea_A, nbr_fea_A, nbr_fea_idx_A), (atom_fea_B, nbr_fea_B, nbr_fea_idx_B)),
            target,
            (cif_id_A, cif_id_B),
        ) in dataset_list:
            # collate for A pair
            n_i_A = atom_fea_A.shape[0]  # number of atoms for this crystal
            batch_atom_fea_A.append(atom_fea_A)
            batch_nbr_fea_A.append(nbr_fea_A)
            batch_nbr_fea_idx_A.append(nbr_fea_idx_A + base_idx_A)
            new_idx_A = torch.LongTensor(np.arange(n_i_A) + base_idx_A)
            crystal_atom_idx_A.append(new_idx_A)
            base_idx_A += n_i_A

            # collate for B pair
            n_i_B = atom_fea_B.shape[0]  # number of atoms for this crystal
            batch_atom_fea_B.append(atom_fea_B)
            batch_nbr_fea_B.append(nbr_fea_B)
            batch_nbr_fea_idx_B.append(nbr_fea_idx_B + base_idx_B)
            new_idx_B = torch.LongTensor(np.arange(n_i_B) + base_idx_B)
            crystal_atom_idx_B.append(new_idx_B)
            base_idx_B += n_i_B

            batch_target.append(target)
            batch_cif_ids.append((cif_id_A, cif_id_B))

        input_A = (
            torch.cat(batch_atom_fea_A, dim=0).to(**self.device_kwargs),
            torch.cat(batch_nbr_fea_A, dim=0).to(**self.device_kwargs),
            torch.cat(batch_nbr_fea_idx_A, dim=0).to(**self.device_kwargs),
            [crys_idx.to(**self.device_kwargs) for crys_idx in crystal_atom_idx_A],
            # torch.nested.nested_tensor(crystal_atom_idx_A).to(**self.device_kwargs),
        )
        input_B = (
            torch.cat(batch_atom_fea_B, dim=0).to(**self.device_kwargs),
            torch.cat(batch_nbr_fea_B, dim=0).to(**self.device_kwargs),
            torch.cat(batch_nbr_fea_idx_B, dim=0).to(**self.device_kwargs),
            [crys_idx.to(**self.device_kwargs) for crys_idx in crystal_atom_idx_B],
            # torch.nested.nested_tensor(crystal_atom_idx_B).to(**self.device_kwargs),
        )
        return (
            (*input_A, *input_B),
            torch.stack(batch_target, dim=0).to(**self.device_kwargs),
            batch_cif_ids,
        )


class Data_Preprocessor:
    def __init__(
        self,
        csv_filepath,
        random_seed=0,
        csv_headers=OrderedDict(
            [
                ("id_charge", str),
                ("id_discharge", str),
                ("average_voltage", float),
            ]
        ),
        *args,
        **kwargs,
    ):
        # average_voltage(target) will have dummy value for inference situation.
        assert os.path.exists(
            csv_filepath
        ), f"{csv_filepath} that contains id-property info does not exist!"

        with open(csv_filepath) as f:
            reader = csv.reader(f)
            id_prop_header = next(reader)
            assert tuple(id_prop_header) == tuple(
                csv_headers.keys()
            ), f"first row of csv file should have column name {csv_headers.keys()}"
        id_prop_data = []
        dataframe = pd.read_csv(csv_filepath, sep=",", index_col=False, header=0, dtype=csv_headers)
        for i in dataframe.index:
            id_prop_data.append(dataframe.iloc[i].to_list())
        if random_seed:
            random.seed(random_seed)
            random.shuffle(id_prop_data)
        self.csv_headers = csv_headers
        self.id_prop_data = id_prop_data
        self.id_prop_train, self.id_prop_val, self.id_prop_test = [], [], []

    def get_total_list(self):
        return self.id_prop_data

    def get_train_list(self):
        return self.id_prop_train

    def get_val_list(self):
        return self.id_prop_val

    def get_test_list(self):
        return self.id_prop_test

    def save_each_list(self, save_path):
        for data_list, savename in zip(
            [self.id_prop_train, self.id_prop_val, self.id_prop_test],
            ["train.csv", "eval.csv", "test.csv"],
        ):
            df = pd.DataFrame(data_list, columns=list(self.csv_headers.keys())).astype(
                self.csv_headers
            )
            df.to_csv(Path(save_path / savename), index=False)

    def split_data(
        self,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        **kwargs,
    ):
        id_prop_data = self.id_prop_data
        total_size = len(id_prop_data)
        if kwargs["train_size"] is None:
            if train_ratio is None:
                assert val_ratio + test_ratio < 1
                train_ratio = 1 - val_ratio - test_ratio
                print(
                    f"[Warning] train_ratio is None, using 1 - val_ratio - "
                    f"test_ratio = {train_ratio} as training data."
                )
            else:
                assert train_ratio + val_ratio + test_ratio <= 1
        if kwargs["train_size"]:
            train_size = kwargs["train_size"]
        else:
            train_size = int(train_ratio * total_size)
        if kwargs["test_size"]:
            test_size = kwargs["test_size"]
        else:
            test_size = int(test_ratio * total_size)
        if kwargs["val_size"]:
            valid_size = kwargs["val_size"]
        else:
            valid_size = int(val_ratio * total_size)

        id_prop_train, id_prop_val, id_prop_test = [], [], []
        indices = list(range(total_size))
        if train_size != 0:
            id_prop_train = [id_prop_data[i] for i in indices[:train_size]]
        if valid_size != 0:
            id_prop_val = [id_prop_data[i] for i in indices[-(valid_size + test_size) : -test_size]]
        if test_size != 0:
            id_prop_test = [id_prop_data[i] for i in indices[-test_size:]]
        self.id_prop_train, self.id_prop_val, self.id_prop_test = (
            id_prop_train,
            id_prop_val,
            id_prop_test,
        )


class Paired_CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    data_dir
    ├── atom_init.json  - shared for any dataset
    ├── {data_name}.csv  - id_prop matched data
    ├── {data_name}
        ├── id0.cif
        ├── id1.cif
        ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    data_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(
        self,
        id_prop_data,
        data_dir="./",
        atom_init_file="./mcgcnn/atom_init.json",
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        is_cif_preload=False,
        *args,
        **kwargs,
    ):
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.atom_feature_inputdim = self.ari.check_embedding_len()
        self.nbr_feature_inputdim = len(self.gdf.filter)
        # nbr_idx_inputdim = max_num_nbr
        self.id_prop_data = id_prop_data

        self.cif_files_dir = Path(data_dir)
        assert os.path.exists(self.cif_files_dir), "cif files' directory does not exist!"

        self.is_cif_preload = is_cif_preload

        # preload cif as Structure for IO reduce
        if self.is_cif_preload:
            self.cif_preload = {}
            for i in glob.glob(str(self.cif_files_dir / "*.cif")):
                i_path = Path(i)
                self.cif_preload[i_path.stem] = Structure.from_file(i_path)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        data_row = self.id_prop_data[idx]
        (charge_cif, discharge_cif), target = data_row[:-1], data_row[-1]

        output_feature_list = []
        output_cif_id_list = []

        for cif_id in [charge_cif, discharge_cif]:
            if self.is_cif_preload:
                crystal = self.cif_preload[cif_id]
            else:
                if cif_id.endswith(".cif"):
                    crystal = Structure.from_file(self.cif_files_dir / cif_id)
                else:
                    crystal = Structure.from_file(self.cif_files_dir / f"{cif_id}.cif")

            atom_fea = np.vstack([self.ari.get_atom_fea(atom.specie.number) for atom in crystal])
            nbr_fea, nbr_fea_idx = self.make_neighbor_features(crystal, cif_id=cif_id)

            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

            output_feature_list.append((atom_fea, nbr_fea, nbr_fea_idx))
            output_cif_id_list.append(cif_id)

        target = torch.Tensor([target])
        return output_feature_list, target, output_cif_id_list

    def make_neighbor_features(self, cif_structure, cif_id=""):
        nbr_fea_idx, nbr_fea = [], []

        all_nbrs = cif_structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius."
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        return nbr_fea, nbr_fea_idx


class Paired_CIFData_input_ready(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    data_dir
    ├── atom_init.json  - shared for any dataset
    ├── {data_name}.csv  - id_prop matched data
    ├── {data_name}
        ├── id0.cif
        ├── id1.cif
        ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    data_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(
        self,
        charged_cif,
        discharged_cif,
        atom_init_file="./mcgcnn/atom_init.json",
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        is_cif_preload=False,
        *args,
        **kwargs,
    ):
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.atom_feature_inputdim = self.ari.check_embedding_len()
        self.nbr_feature_inputdim = len(self.gdf.filter)
        # nbr_idx_inputdim = max_num_nbr
        self.id_prop_data = id_prop_data

        self.cif_files_dir = Path(data_dir)
        assert os.path.exists(self.cif_files_dir), "cif files' directory does not exist!"

        self.is_cif_preload = is_cif_preload

        # preload cif as Structure for IO reduce
        if self.is_cif_preload:
            self.cif_preload = {}
            for i in glob.glob(str(self.cif_files_dir / "*.cif")):
                i_path = Path(i)
                self.cif_preload[i_path.stem] = Structure.from_file(i_path)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        data_row = self.id_prop_data[idx]
        cif_ids, target = data_row[:-1], data_row[-1]

        output_feature_list = []
        output_cif_id_list = []

        for cif_id in cif_ids:
            if self.is_cif_preload:
                crystal = self.cif_preload[cif_id]
            else:
                crystal = Structure.from_file(self.cif_files_dir / f"{cif_id}.cif")

            atom_fea = np.vstack([self.ari.get_atom_fea(atom.specie.number) for atom in crystal])
            nbr_fea, nbr_fea_idx = self.make_neighbor_features(crystal, cif_id=cif_id)

            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

            output_feature_list.append((atom_fea, nbr_fea, nbr_fea_idx))
            output_cif_id_list.append(cif_id)

        target = torch.Tensor([target])
        return output_feature_list, target, output_cif_id_list

    def make_neighbor_features(self, cif_structure, cif_id=""):
        nbr_fea_idx, nbr_fea = [], []

        all_nbrs = cif_structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius."
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        return nbr_fea, nbr_fea_idx


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        eps = 1e-8
        self.filter = np.arange(dmin, dmax + eps, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

    def check_embedding_len(self):
        embedding_len = None
        for id in self._embedding.keys():
            assert len(self._embedding[id].shape) == 1
            if embedding_len is None:
                embedding_len = len(self._embedding[id])
            assert len(self._embedding[id]) == embedding_len
            pass
        return embedding_len


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class Paired_CIFData_for_Inference(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    data_dir
    ├── atom_init.json  - shared for any dataset
    ├── {data_name}.csv  - id_prop matched data
    ├── {data_name}
        ├── id0.cif
        ├── id1.cif
        ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    data_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(
        self,
        data_dir,
        charge_discharge_pair,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        *args,
        **kwargs,
    ):
        assert os.path.exists(data_dir), "data_dir does not exist!"

        atom_init_file = Path(data_dir) / "atom_init.json"
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"

        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.charge_discharge_pair = charge_discharge_pair
        self.crystal_pair_list = []
        for ch_cif, disch_cif in charge_discharge_pair:
            crystal_charged = Structure.from_file(ch_cif)
            crystal_discharged = Structure.from_file(disch_cif)
            self.crystal_pair_list.append([crystal_charged, crystal_discharged])

    def __len__(self):
        return len(self.id_prop_data)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        charge_cif, discharge_cif = self.charge_discharge_pair[idx]

        output_feature_list = []
        output_cif_id_list = []

        for cif_path in [charge_cif, discharge_cif]:
            crystal = Structure.from_file(cif_path)

            atom_fea = np.vstack(
                [self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
            )
            atom_fea = torch.Tensor(atom_fea)
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    warnings.warn(
                        "{} not find enough neighbors to build graph. "
                        "If it happens frequently, consider increase "
                        "radius.".format(cif_path)
                    )
                    nbr_fea_idx.append(
                        list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                    )
                    nbr_fea.append(
                        list(map(lambda x: x[1], nbr))
                        + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                    )
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

            output_feature_list.append((atom_fea, nbr_fea, nbr_fea_idx))
            output_cif_id_list.append(cif_path)

        target = torch.Tensor([0.0])
        return output_feature_list, target, output_cif_id_list
