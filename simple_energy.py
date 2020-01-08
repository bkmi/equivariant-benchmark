import logging
import os
import sys
from shutil import rmtree
from functools import partial
from time import perf_counter
from itertools import count

import torch

import schnetpack as spk
from schnetpack import Properties
from schnetpack.datasets import QM9

from e3nn.non_linearities import rescaled_act
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.point.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.point.radial import CosineBasisModel

from arguments import qm9_energy_parser


torch.set_default_dtype(torch.float32)


def convolution(cutoff, n_bases, n_neurons, n_layers, beta):
    radial_act = rescaled_act.ShiftedSoftplus(beta=beta)
    RadialModel = partial(
        CosineBasisModel,
        max_radius=cutoff,
        number_of_basis=n_bases,
        h=n_neurons,
        L=n_layers,
        act=radial_act
    )
    K = partial(Kernel, RadialModel=RadialModel)
    return partial(Convolution, K)


class Net(torch.nn.Module):
    def __init__(self, conv, embed, l0, l1, l2, l3, L, scalar_act):
        super().__init__()
        Rs = [[(embed, 0)]]
        Rs_mid = [(mul, l) for l, mul in enumerate([l0, l1, l2, l3])]
        Rs += [Rs_mid] * L
        Rs += [[(1, 0)]]
        self.Rs = Rs

        qm9_max_z = 10
        self.layers = torch.nn.ModuleList([torch.nn.Embedding(qm9_max_z, embed, padding_idx=0)])
        self.layers += [
            GatedBlock(
                partial(conv, rs_in),
                rs_out,
                scalar_act,  # TODO In this section, you are trying to predict a negative number with a relu
                rescaled_act.sigmoid
            ) for rs_in, rs_out in zip(Rs, Rs[1:])
        ]

    def forward(self, features, geometry, mask):
        batchwise_num_atoms = mask.sum(dim=-1)
        embedding = self.layers[0]
        features = embedding(features)
        for layer in self.layers[1:]:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    parser = qm9_energy_parser()
    parser.add_argument("--wall", type=float, required=True, help="If calculation time is too long, break.")
    args = parser.parse_args()

    # basic settings
    try:
        os.makedirs(args.model_dir)
    except FileExistsError:
        rmtree(args.model_dir, ignore_errors=True)
        os.makedirs(args.model_dir)
    torch.save(args, args.model_dir + "args.pkl")
    properties = [QM9.U0]

    # data preparation
    logging.info("get dataset")
    dataset = QM9(args.db)
    train, val, test = spk.train_test_split(
        dataset,
        num_train=args.ntr,
        num_val=args.nva,
        split_file=os.path.join(args.model_dir, "split.npz")
    )
    train_loader = spk.AtomsLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    # val_loader = spk.AtomsLoader(val, batch_size=args.bs, num_workers=args.num_workers)

    # model build
    logging.info("build model")
    conv = convolution(
        cutoff=args.rad_maxr,
        n_bases=args.rad_nb,
        n_neurons=args.rad_h,
        n_layers=args.rad_L,
        beta=5.0
    )
    sp = rescaled_act.Softplus(beta=5.0)
    net = Net(
        conv=conv,
        embed=args.embed,
        l0=args.l0,
        l1=args.l1,
        l2=args.l2,
        l3=args.l3,
        L=args.L,
        scalar_act=sp
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    dynamics = []
    wall_start = perf_counter()

    # run training
    logging.info("training")
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    logging.info(f"device: {device}")

    for batch in train_loader:
        pass
    net = net.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    for step in count():
        # batch = {k: v.to(device) for k, v in batch.items()}
        out = net(features=batch[Properties.Z], geometry=batch[Properties.R], mask=batch[Properties.atom_mask])
        pool = out.squeeze(dim=-1).sum(dim=-1)
        loss = (batch[QM9.U0] - pool).norm(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wall = perf_counter() - wall_start

        dynamics.append({
            'step': step,
            'wall': wall,
            'loss': loss.item(),
        })

        print("[wall={:.1f} step={}] [loss={:.1f}]".format(wall, step, loss.item()))
        sys.stdout.flush()

        if wall > args.wall:
            break
    return {
        'args': args,
        'dynamics': dynamics,
    }


if __name__ == '__main__':
    main()
