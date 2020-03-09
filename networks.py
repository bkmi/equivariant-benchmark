from functools import partial

import torch

import schnetpack as spk

from e3nn.point.kernelconv import KernelConv
from e3nn.radial import CosineBasisModel, GaussianRadialModel

from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.o3 import spherical_harmonics_xyz


CUSTOM_BACKWARD = True


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
    diff_geo = (rb - ra).detach()
    radii = diff_geo.norm(2, dim=-1).detach()
    return one_hot, geometry, mask, diff_geo, radii


class Network(torch.nn.Module):
    def __init__(self, kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act, avg_n_atoms):
        super().__init__()
        self.avg_n_atoms = avg_n_atoms

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
        features, _, mask, diff_geo, radii = constants(batch)
        embedding = self.layers[0]
        features = embedding(features)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        for kc, act in self.layers[1:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            features = kc(
                features.div(self.avg_n_atoms ** 0.5),
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            features = act(features)
            features = features * mask.unsqueeze(-1)
        return features


class ResNetwork(Network):
    def __init__(self, kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act, avg_n_atoms):
        super(ResNetwork, self).__init__(kernel_conv, embed, l0, l1, l2, l3, L, scalar_act, gate_act, avg_n_atoms)

    def forward(self, batch):
        features, _, mask, diff_geo, radii = constants(batch)
        embedding = self.layers[0]
        features = embedding(features)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        kc, act = self.layers[1]
        features = kc(
            features.div(self.avg_n_atoms ** 0.5),
            diff_geo,
            mask,
            y=y,
            radii=radii,
            custom_backward=CUSTOM_BACKWARD
        )
        features = act(features)
        for kc, act in self.layers[2:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            new_features = kc(
                features.div(self.avg_n_atoms ** 0.5),
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            new_features = act(new_features)
            new_features = new_features * mask.unsqueeze(-1)
            features = features + new_features
        return features


def gate_error(x):
    raise ValueError("There should be no L>0 components in a scalar network.")


class OutputScalarNetwork(torch.nn.Module):
    def __init__(self, kernel_conv, previous_Rs, scalar_act, avg_n_atoms):
        super(OutputScalarNetwork, self).__init__()
        self.avg_n_atoms = avg_n_atoms

        Rs = [previous_Rs]
        Rs += [[(1, 0)]]
        self.Rs = Rs

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_error)
            kc = kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])])

    def forward(self, batch):
        _, _, mask, diff_geo, radii = constants(batch)
        features = batch["representation"]
        for kc, act in self.layers:
            features = kc(features.div(self.avg_n_atoms ** 0.5), diff_geo, mask, radii=radii)
            features = act(features)
            features = features * mask.unsqueeze(-1)
        return features


class NormVarianceLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(NormVarianceLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.rand(out_features) - 0.5)

    def forward(self, x):
        size = x.size(-1)
        return x @ (self.weight.t() / size ** 0.5) + self.bias


class OutputMLPNetwork(torch.nn.Module):
    def __init__(self, kernel_conv, previous_Rs, l0, scalar_act, avg_n_atoms):
        super(OutputMLPNetwork, self).__init__()
        self.avg_n_atoms = avg_n_atoms

        Rs = [previous_Rs]
        Rs += [[(l0, 0)]]
        self.Rs = Rs

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_error)
            kc = kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])])
        self.mlp = torch.nn.ModuleList([
            NormVarianceLinear(l0, l0),
            torch.nn.ReLU(),
            NormVarianceLinear(l0, 1),
            torch.nn.ReLU()
        ])

    def forward(self, batch):
        _, _, mask, diff_geo, radii = constants(batch)
        features = batch["representation"]
        for kc, act in self.layers:
            features = kc(features.div(self.avg_n_atoms ** 0.5), diff_geo, mask, radii=radii)
            features = act(features)
            features = features * mask.unsqueeze(-1)
        features = features.sum(dim=1)
        new_features = features
        for layer in self.mlp:
            new_features = layer(new_features)
        return (features + new_features).unsqueeze(1)


if __name__ == '__main__':
    pass
