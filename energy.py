from functools import partial

import argparse

import torch

from schnetpack.data import train_test_split, AtomsLoader
from schnetpack.datasets import QM9

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pickle", type=str, required=True, help="File for saving args and results.")
    # parser.add_argument("--nte", type=int, required=True, help="Number of test examples.")
    # parser.add_argument("--ntr", type=int, required=True, help="Number of training examples.")
    # parser.add_argument("--data_seed", type=int, default=0, help="Random seed for organizing data.")
    # parser.add_argument("--init_seed", type=int, default=0, help="Random seed for initializing network.")
    # parser.add_argument("--batch_seed", type=int, default=0, help="Random seed for batch distribution.")
    # parser.add_argument("--wall", type=float, required=True, help="If calculation time is too long, break.")
    parser.add_argument("-d", "--db", type=str, required=True, help="Path to qm9 database.")
    parser.add_argument("--bs", type=int, default=5, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    # parser.add_argument("--nnei", type=float, default=12, help="Average number of atoms convolved together.")
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--l0", type=int, default=4)
    parser.add_argument("--l1", type=int, default=4)
    parser.add_argument("--l2", type=int, default=4)
    parser.add_argument("--l3", type=int, default=4)
    parser.add_argument("--L", type=int, default=4, help="How many layers to create.")
    parser.add_argument("--rad_nb", type=int, default=30, help="Radial number of bases.")
    parser.add_argument("--rad_maxr", type=float, default=1.6, help="Max radius.")
    parser.add_argument("--rad_h", type=int, default=100, help="Size of radial weight parameters.")
    parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
    # parser.add_argument("--save_state", type=int, default=0)
    # parser.add_argument("--parity", type=int, default=0)
    return parser


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

        rs = [[(6, 0)]]
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


def main():
    parser = get_parser()
    args = parser.parse_args()
    data = QM9(args.db)

    train, val, test = train_test_split(data, num_train=100, num_val=100)
    loader = AtomsLoader(train, batch_size=100, num_workers=4)
    val_loader = AtomsLoader(val)

    net = Network(args)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = lambda b, p: F.mse_loss(p["y"], b[QM9.U0])
    trainer = spk.train.Trainer("output/", model, loss, opt, loader, val_loader)

    for batch in loader:
        output = net(loader['elements'], loader['_positions'], loader['mask'].sum(dim=-1))




if __name__ == '__main__':
    main()
