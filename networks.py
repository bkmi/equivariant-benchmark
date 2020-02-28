from functools import partial

import torch

import schnetpack as spk

from e3nn.point.kernelconv import KernelConv
from e3nn.radial import CosineBasisModel, GaussianRadialModel

from e3nn.non_linearities.gated_block import GatedBlock


def create_kernel_conv(cutoff, n_bases, n_neurons, n_layers, act, radial_model):
    if radial_model == "cosine":
        RadialModel = partial(
            CosineBasisModel,
            max_radius=cutoff,
            number_of_basis=n_bases,
            h=n_neurons,
            L=n_layers,
            act=act
        )
    elif radial_model == "gaussian":
        RadialModel = partial(
            GaussianRadialModel,
            max_radius=cutoff,
            number_of_basis=n_bases,
            h=n_neurons,
            L=n_layers,
            act=act
        )
    else:
        raise ValueError("radial_model must be either cosine or gaussian")
    K = partial(KernelConv, RadialModel=RadialModel)
    return K


def constants(batch):
    one_hot, geometry, mask = batch[spk.Properties.Z], batch[spk.Properties.R], batch[spk.Properties.atom_mask]
    rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
    ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
    r = rb - ra
    radii = r.norm(2, dim=-1)
    # y = o3.spherical_harmonics_xyz(set_of_l_filters, geometry)
    return one_hot, geometry, mask, r, radii


class Network(torch.nn.Module):
    def __init__(self, kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act):
        super().__init__()

        Rs = [[(embed, 0)]]
        Rs_mid = [(mul, l) for l, mul in enumerate([l0, l1, l2, l3])]
        Rs += [Rs_mid] * L
        self.Rs = Rs

        qm9_max_z = 10

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_act)
            kc = kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([torch.nn.Embedding(qm9_max_z, embed, padding_idx=0)])
        self.layers += [make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])]

    def forward(self, batch):
        features, _, mask, r, radii = constants(batch)
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        for kc, act in self.layers[1:]:
            features = kc(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), r, mask, radii=radii)
            features = act(features)
            features = features * mask.unsqueeze(-1)
        return features


class ResNetwork(Network):
    def __init__(self, kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act):
        super(ResNetwork, self).__init__(kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act)

    def forward(self, batch):
        features, _, mask, r, radii = constants(batch)
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        kc, act = self.layers[1]
        features = kc(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), r, mask, radii=radii)
        features = act(features)
        for kc, act in self.layers[2:]:
            new_features = kc(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), r, mask, radii=radii)
            new_features = act(new_features)
            new_features = new_features * mask.unsqueeze(-1)
            features = features + new_features
        return features


def gate_error(x):
    raise ValueError("There should be no L>0 components in a scalar network.")


class OutputScalarNetwork(torch.nn.Module):
    def __init__(self, kernel_conv, previous_Rs, scalar_act):
        super(OutputScalarNetwork, self).__init__()
        Rs = [previous_Rs]
        Rs += [[(1, 0)]]
        self.Rs = Rs

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_error)
            kc = kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])])

    def forward(self, batch):
        _, _, mask, r, radii = constants(batch)
        features = batch["representation"]
        batchwise_num_atoms = mask.sum(dim=-1)
        for kc, act in self.layers:
            features = kc(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), r, mask, radii=radii)
            features = act(features)
            features = features * mask.unsqueeze(-1)
        return features


if __name__ == '__main__':
    pass
