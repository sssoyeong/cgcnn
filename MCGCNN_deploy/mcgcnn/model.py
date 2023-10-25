# https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py

from __future__ import print_function, division

from pathlib import Path

import omegaconf
import hydra

import torch
import torch.nn as nn
from abc import abstractmethod

PROJECT_ROOT = Path(__file__).parents[1]


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(
            N, M, self.atom_fea_len * 2
        )
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class Base_CGCNN_Module(nn.Module):
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        # assert sum([idx_map.shape[0] for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        # # TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
        # # TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


class Paired_CGCNN_separated(Base_CGCNN_Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        model_cfg,
        *args,
        **kwargs,
    ):
        super(Paired_CGCNN_separated, self).__init__()
        """

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """

        atom_fea_len = model_cfg.atom_feature_size
        n_conv = model_cfg.num_convlayers
        h_fea_len = model_cfg.hidden_feature_size
        n_h = model_cfg.num_hiddenlayers
        dropout_rate = model_cfg.dropout_rate
        combine_features = model_cfg.combine_features
        assert combine_features in ["subtract_abs", "mean", "subtract", "add"]

        # average of embedding pair
        self.combine_features = None
        if combine_features == "subtract_abs":
            self.combine_features = lambda x, y: torch.abs(x - y)
        elif combine_features == "subtract":
            self.combine_features = lambda x, y: torch.subtract(x, y)
        elif combine_features == "add":
            self.combine_features = lambda x, y: torch.add(x, y)
        elif combine_features == "mean":
            self.combine_features = lambda x, y: torch.mean(torch.stack([x, y], dim=0), dim=0)
        else:
            raise Exception

        # self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding_A = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding_B = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # self.convs = nn.ModuleList(
        #     [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        # )
        self.convs_A = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.convs_B = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )

        self.dense_feature_A = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.Softplus(),
        )
        self.dense_feature_B = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.Softplus(),
        )

        self.feature_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_fea_len, h_fea_len),
                    nn.Softplus(),
                    nn.Dropout(p=dropout_rate),
                )
                for _ in range(n_h)
            ]
        )

        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(
        self,
        atom_fea_A,
        nbr_fea_A,
        nbr_fea_idx_A,
        crystal_atom_idx_A,
        atom_fea_B,
        nbr_fea_B,
        nbr_fea_idx_B,
        crystal_atom_idx_B,
    ):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        # atom_fea_A, atom_fea_B = self.embedding(atom_fea_A),  self.embedding(atom_fea_B)
        # for conv_func in self.convs:
        #     atom_fea_A = conv_func(atom_fea_A, nbr_fea_A, nbr_fea_idx_A)
        #     atom_fea_B = conv_func(atom_fea_B, nbr_fea_B, nbr_fea_idx_B)

        atom_fea_A, atom_fea_B = self.embedding_A(atom_fea_A), self.embedding_B(atom_fea_B)
        for conv_func_A, conv_func_B in zip(self.convs_A, self.convs_B):
            atom_fea_A = conv_func_A(atom_fea_A, nbr_fea_A, nbr_fea_idx_A)
            atom_fea_B = conv_func_B(atom_fea_B, nbr_fea_B, nbr_fea_idx_B)
        crys_fea_A = self.dense_feature_A(self.pooling(atom_fea_A, crystal_atom_idx_A))
        crys_fea_B = self.dense_feature_B(self.pooling(atom_fea_B, crystal_atom_idx_B))

        #################################
        # average of embedding pair
        crys_fea = self.combine_features(crys_fea_A, crys_fea_B)

        #################################

        for _layer in self.feature_ff:
            crys_fea = _layer(crys_fea)

        out = self.fc_out(crys_fea)

        return out


class Single_CGCNN(Base_CGCNN_Module):
    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        model_cfg,
        *args,
        **kwargs,
    ):
        super(Single_CGCNN, self).__init__()

        """

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """

        atom_fea_len = model_cfg.atom_feature_size
        n_conv = model_cfg.num_convlayers
        h_fea_len = model_cfg.hidden_feature_size
        n_h = model_cfg.num_hiddenlayers

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(
        self,
        atom_fea_A,
        nbr_fea_A,
        nbr_fea_idx_A,
        crystal_atom_idx_A,
        atom_fea_B,
        nbr_fea_B,
        nbr_fea_idx_B,
        crystal_atom_idx_B,
    ):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = (
            atom_fea_B,
            nbr_fea_B,
            nbr_fea_idx_B,
            crystal_atom_idx_B,
        )

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)

        return out


class Paired_CGCNN_weightshare(Base_CGCNN_Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        model_cfg,
        *args,
        **kwargs,
    ):
        super(Paired_CGCNN_weightshare, self).__init__()
        """

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """

        atom_fea_len = model_cfg.atom_feature_size
        n_conv = model_cfg.num_convlayers
        h_fea_len = model_cfg.hidden_feature_size
        n_h = model_cfg.num_hiddenlayers
        dropout_rate = model_cfg.dropout_rate
        self.combine_features = model_cfg.combine_features
        assert self.combine_features in ["subtract_abs", "mean", "subtract", "add"]

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # self.embedding_A = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # self.embedding_B = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        # self.convs_A = nn.ModuleList(
        #     [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        # )
        # self.convs_B = nn.ModuleList(
        #     [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        # )

        self.dense_feature = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.Softplus(),
        )
        # self.dense_feature_A = nn.Sequential(
        #    nn.Linear(atom_fea_len, h_fea_len),
        #    nn.Softplus(),
        # )
        # self.dense_feature_B = nn.Sequential(
        #    nn.Linear(atom_fea_len, h_fea_len),
        #    nn.Softplus(),
        # )

        self.feature_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_fea_len, h_fea_len),
                    nn.Softplus(),
                    nn.Dropout(p=dropout_rate),
                )
                for _ in range(n_h)
            ]
        )

        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(
        self,
        atom_fea_A,
        nbr_fea_A,
        nbr_fea_idx_A,
        crystal_atom_idx_A,
        atom_fea_B,
        nbr_fea_B,
        nbr_fea_idx_B,
        crystal_atom_idx_B,
    ):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        # atom_fea_A, atom_fea_B = self.embedding(atom_fea_A),  self.embedding(atom_fea_B)
        # for conv_func in self.convs:
        #     atom_fea_A = conv_func(atom_fea_A, nbr_fea_A, nbr_fea_idx_A)
        #     atom_fea_B = conv_func(atom_fea_B, nbr_fea_B, nbr_fea_idx_B)

        atom_fea_A, atom_fea_B = self.embedding(atom_fea_A), self.embedding(atom_fea_B)
        for conv_func in self.convs:
            atom_fea_A = conv_func(atom_fea_A, nbr_fea_A, nbr_fea_idx_A)
            atom_fea_B = conv_func(atom_fea_B, nbr_fea_B, nbr_fea_idx_B)
        crys_fea_A = self.dense_feature(self.pooling(atom_fea_A, crystal_atom_idx_A))
        crys_fea_B = self.dense_feature(self.pooling(atom_fea_B, crystal_atom_idx_B))

        #################################
        # average of embedding pair
        if self.combine_features == "subtract_abs":
            crys_fea = torch.abs(crys_fea_A - crys_fea_B)
            # crys_fea = torch.sqrt(torch.pow(crys_fea_A - crys_fea_B, exponent=2))
        elif self.combine_features == "subtract":
            crys_fea = torch.subtract(crys_fea_A, crys_fea_B)
        elif self.combine_features == "add":
            crys_fea = torch.add(crys_fea_A, crys_fea_B)
        elif self.combine_features == "mean":
            crys_fea = torch.mean(torch.stack([crys_fea_A, crys_fea_B], dim=0), dim=0)
        else:
            return Exception

        #################################

        for _layer in self.feature_ff:
            crys_fea = _layer(crys_fea)

        out = self.fc_out(crys_fea)

        return out
