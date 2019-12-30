import torch
import torch.nn.functional as F

from functools import partial

import schnetpack as spk
from schnetpack import Properties
from schnetpack.datasets import QM9

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel

from arguments import qm9_energy_parser


torch.set_default_dtype(torch.float64)


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

    def forward(self, batch):
        features, geometry, mask = batch[Properties.Z], batch[Properties.R], batch[Properties.atom_mask]
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        for layer in self.layers[1:]:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features.sum(dim=1)


def main():
    parser = qm9_energy_parser()
    args = parser.parse_args()

    # load qm9 dataset and download if necessary
    data = QM9("/home/ben/science/data/qm9.db")

    # split in train and val
    train, val, test = data.create_splits(num_train=args.ntr, num_val=args.nva)
    loader = spk.data.AtomsLoader(train, batch_size=args.bs, num_workers=4)
    val_loader = spk.data.AtomsLoader(val)

    # create model
    model = Network(args)

    # create trainer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = lambda b, p: F.mse_loss(p, b[QM9.U0])
    trainer = spk.train.Trainer("output/", model, loss, opt, loader, val_loader)

    # start training
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    # TODO I need to make it so qm9 is saved as a double, it is required by se3cnn now
    trainer.train(device, args.epochs)


if __name__ == '__main__':
    main()
