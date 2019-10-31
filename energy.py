from functools import partial

import torch

from schnetpack.data import train_test_split, AtomsLoader
from schnetpack.datasets import QM9

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel

data = QM9("/home/ben/science/data/qm9.db")

train, val, test = train_test_split(data, num_train=100, num_val=100)
loader = AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = AtomsLoader(val)

def convolution(args):
    radial_act = rescaled_act.ShiftedSoftplus(beta=5.0)
    RadialModel = partial(CosineBasisModel,
                          max_radius=args.rad_maxr, number_of_basis=args.rad_nb,
                          h=args.rad_h, L=args.rad_L, act=radial_act)
    K = partial(Kernel, RadialModel=RadialModel)
    return partial(Convolution, K)


class Network(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        rs = [(0, 6)]
        rs += [
            (mul, l)
            for l, mul in enumerate([args.l0, args.l1, args.l2, args.l3])
        ] * args.L
        rs += [[(1, 1)]]

        c = convolution(args)
        sp = rescaled_act.Softplus(beta=5.0)

        self.layers = torch.nn.ModuleList([
            GatedBlock(rs_in, rs_out, sp, rescaled_act.sigmoid, c)
            for rs_in, rs_out in zip(rs, rs[1:])
        ])

    def forward(self, features, geometry, batchwise_num_atoms=None):
        if batchwise_num_atoms is None:
            batchwise_num_atoms = geometry.size(1)

        for layer in self.layers:
            features = layer(features.div(batchwise_num_atoms ** 0.5), geometry)
        return features

net = Network(args)
