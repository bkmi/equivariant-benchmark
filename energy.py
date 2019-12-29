from functools import partial
from time import perf_counter

import argparse

import torch
import torch.nn.functional as F

from schnetpack.data import train_test_split, AtomsLoader
from schnetpack.datasets import QM9

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel


torch.set_default_dtype(torch.float64)


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pickle", type=str, required=True, help="File for saving args and results.")
    # parser.add_argument("--nte", type=int, required=True, help="Number of test examples.")
    # parser.add_argument("--ntr", type=int, required=True, help="Number of training examples.")
    # parser.add_argument("--data_seed", type=int, default=0, help="Random seed for organizing data.")
    # parser.add_argument("--init_seed", type=int, default=0, help="Random seed for initializing network.")
    # parser.add_argument("--batch_seed", type=int, default=0, help="Random seed for batch distribution.")
    parser.add_argument("--wall", type=float, required=True, help="If calculation time is too long, break.")
    parser.add_argument("--db", type=str, required=True, help="Path to qm9 database.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--gpu", type=bool, default=True, help="Use gpu.")
    parser.add_argument("--bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    # parser.add_argument("--nnei", type=float, default=12, help="Average number of atoms convolved together.")
    parser.add_argument("--embed", type=int, default=32)
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

        rs = [[(args.embed, 0)]]
        rs_mid = [(mul, l) for l, mul in enumerate([args.l0, args.l1, args.l2, args.l3])]
        rs += [rs_mid] * args.L
        rs += [[(1, 0)]]

        c = convolution(args)
        sp = rescaled_act.Softplus(beta=5.0)

        qm9_max_z = 10
        self.layers = torch.nn.ModuleList([torch.nn.Embedding(qm9_max_z, args.embed, padding_idx=0)])
        self.layers += [GatedBlock(rs_in, rs_out, sp, rescaled_act.sigmoid, c) for rs_in, rs_out in zip(rs, rs[1:])]

    def forward(self, features, geometry, mask):
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        for layer in self.layers[1:]:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.gpu = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    data = QM9(args.db)

    train, val, test = train_test_split(data, num_train=100, num_val=100)
    loader = AtomsLoader(train, batch_size=16, num_workers=4)
    val_loader = AtomsLoader(val)

    net = Network(args).to(args.gpu)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # trainer = spk.train.Trainer("output/", model, loss, opt, loader, val_loader)

    wall_start = perf_counter()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {
                k: v.to(device=args.gpu, dtype=torch.float64) if v.dtype is torch.float32 else v.to(device=args.gpu)
                for k, v in batch.items()
            }

            output = net(batch['_atomic_numbers'], batch['_positions'], batch['_atom_mask'])
            pooled = output.sum(dim=1)
            loss = F.mse_loss(pooled, batch[QM9.U0])
            print(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wall = perf_counter() - wall_start

            if wall > args.wall:
                break


if __name__ == '__main__':
    main()
