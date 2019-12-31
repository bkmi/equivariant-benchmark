import logging
import os
from functools import partial

import torch

import schnetpack as spk
from schnetpack import Properties
from schnetpack.datasets import QM9

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel

from arguments import qm9_energy_parser


torch.set_default_dtype(torch.float32)


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
        self.rs = rs

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
        return features


class OutputNetwork(torch.nn.Module):
    def __init__(self, args, last_Rs):
        super(OutputNetwork, self).__init__()
        rs = [last_Rs]
        rs += [[(1, 0)]]
        self.rs = rs

        c = convolution(args)
        sp = rescaled_act.Softplus(beta=5.0)

        self.layers = torch.nn.ModuleList([
            GatedBlock(rs_in, rs_out, sp, rescaled_act.sigmoid, c) for rs_in, rs_out in zip(rs, rs[1:])
        ])

    def forward(self, batch):
        features, geometry, mask = batch["representation"], batch[Properties.R], batch[Properties.atom_mask]
        batchwise_num_atoms = mask.sum(dim=-1)
        for layer in self.layers:
            features = layer(features.div(batchwise_num_atoms.reshape(-1, 1, 1) ** 0.5), geometry)
            features = features * mask.unsqueeze(-1)
        return features


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    parser = qm9_energy_parser()
    args = parser.parse_args()

    # basic settings
    os.makedirs(args.model_dir)
    properties = [QM9.U0]

    # data preparation
    logging.info("get dataset")
    dataset = QM9(args.db, load_only=[QM9.U0])
    train, val, test = spk.train_test_split(
        dataset,
        num_train=args.ntr,
        num_val=args.nva,
        split_file=os.path.join(args.model_dir, "split.npz")
    )
    train_loader = spk.AtomsLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = spk.AtomsLoader(val, batch_size=args.bs, num_workers=args.num_workers)

    # statistics
    atomrefs = dataset.get_atomref(properties)
    means, stddevs = train_loader.get_statistics(
        properties, divide_by_atoms=True, single_atom_ref=atomrefs
    )

    # model build
    logging.info("build model")
    # representation = spk.SchNet(n_interactions=6)
    # output_modules = [
    #     spk.atomistic.Atomwise(
    #         n_in=representation.n_atom_basis,
    #         property=QM9.U0,
    #         mean=means[QM9.U0],
    #         stddev=stddevs[QM9.U0],
    #         atomref=atomrefs[QM9.U0],
    #     )
    # ]
    # model = spk.AtomisticModel(representation, output_modules)

    net = Network(args)
    outnet = OutputNetwork(args, net.rs[-1])
    output_modules = [
        spk.atomistic.Atomwise(
            n_in=0,
            property=QM9.U0,
            mean=means[QM9.U0],
            stddev=stddevs[QM9.U0],
            atomref=atomrefs[QM9.U0],
            outnet=outnet
        )
    ]
    model = spk.AtomisticModel(net, output_modules)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # hooks
    logging.info("build trainer")
    metrics = [spk.train.metrics.MeanAbsoluteError(p, p) for p in properties]
    hooks = [spk.train.CSVHook(log_path=args.model_dir, metrics=metrics),
             spk.train.ReduceLROnPlateauHook(optimizer)]

    # trainer
    loss = spk.train.build_mse_loss(properties)
    trainer = spk.train.Trainer(
        args.model_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    # run training
    logging.info("training")
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    trainer.train(device=device, n_epochs=args.epochs)
    # Problem serializing the partial trainer.py ln 241


if __name__ == '__main__':
    main()
