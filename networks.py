from functools import partial

import torch

import schnetpack as spk

from e3nn.point.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.point.radial import CosineBasisModel

from e3nn.non_linearities.gated_block import GatedBlock


def convolution(cutoff, n_bases, n_neurons, n_layers, act):
    RadialModel = partial(
        CosineBasisModel,
        max_radius=cutoff,
        number_of_basis=n_bases,
        h=n_neurons,
        L=n_layers,
        act=act
    )
    K = partial(Kernel, RadialModel=RadialModel)
    return partial(Convolution, K)


class Network(torch.nn.Module):
    def __init__(self, conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act):
        super().__init__()

        Rs = [[(embed, 0)]]
        Rs_mid = [(mul, l) for l, mul in enumerate([l0, l1, l2, l3])]
        Rs += [Rs_mid] * L
        self.Rs = Rs

        qm9_max_z = 10
        self.layers = torch.nn.ModuleList([torch.nn.Embedding(qm9_max_z, embed, padding_idx=0)])
        self.layers += [
            GatedBlock(
                partial(conv, rs_in),
                rs_out,
                scalar_act,
                gate_act
            ) for rs_in, rs_out in zip(Rs, Rs[1:])
        ]

    def forward(self, batch):
        features, geometry, mask = batch[spk.Properties.Z], batch[spk.Properties.R], batch[spk.Properties.atom_mask]
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        for layer in self.layers[1:]:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features


def gate_error(x):
    raise ValueError("There should be no L>0 components in a scalar network.")


class OutputScalarNetwork(torch.nn.Module):
    def __init__(self, conv, previous_Rs, scalar_act):
        super(OutputScalarNetwork, self).__init__()
        Rs = [previous_Rs]
        Rs += [[(1, 0)]]
        self.Rs = Rs

        self.layers = torch.nn.ModuleList([
            GatedBlock(
                partial(conv, rs_in),
                rs_out,
                scalar_act,
                gate_error) for rs_in, rs_out in zip(Rs, Rs[1:])
        ])

    def forward(self, batch):
        features, geometry, mask = batch["representation"], batch[spk.Properties.R], batch[spk.Properties.atom_mask]
        batchwise_num_atoms = mask.sum(dim=-1)
        for layer in self.layers:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features


if __name__ == '__main__':
    pass
