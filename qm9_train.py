import logging
import os
import sys
import argparse
from shutil import rmtree
from time import perf_counter
from datetime import date

import torch
import numpy as np

import schnetpack as spk
from schnetpack.datasets import QM9

from e3nn.non_linearities import rescaled_act
from e3nn.rs import dim

from arguments import train_parser, qm9_property_selector, fix_old_args_with_defaults
from networks import create_kernel_conv, Network, OutputScalarNetwork, ResNetwork, OutputMLPNetwork
from evaluation import evaluate, record_versions


def configuration(args):
    torch.set_default_dtype(torch.float32)
    log_level = os.environ.get("LOGLEVEL", "INFO")
    print(f"Logging level {log_level}")
    logging.basicConfig(level=log_level)
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def create_or_load_directory(args):
    args_file = os.path.join(args.model_dir, "args.pkl")
    try:
        os.makedirs(args.model_dir)
        torch.save(args, args_file)
    except FileExistsError:
        logging.warning(f"Model directory {args.model_dir} exists.")
        if args.overwrite:
            logging.warning("Overwriting.")
            rmtree(args.model_dir)
            os.makedirs(args.model_dir)
            torch.save(args, args_file)
        else:
            logging.warning("Loading from checkpoint, continuing using SAVED args.")
            args = torch.load(args_file)
    return args


def get_data(args, properties):
    split_file = os.path.join(args.model_dir, "split.npz") if not args.split_file else args.split_file
    dataset = QM9(args.db, load_only=properties)
    train, val, test = spk.train_test_split(
        dataset,
        num_train=args.ntr,
        num_val=args.nva,
        split_file=split_file
    )
    assert len(train) == args.ntr
    assert len(val) == args.nva
    train_loader = spk.AtomsLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = spk.AtomsLoader(val, batch_size=args.bs, num_workers=args.num_workers)
    test_loader = spk.AtomsLoader(test, batch_size=args.bs, num_workers=args.num_workers)
    return dataset, split_file, train_loader, val_loader, test_loader


def get_statistics(dataset, split_file, properties, train_loader):
    atomrefs = dataset.get_atomref(properties)
    with np.load(split_file) as split_npz:
        split_npz_dict = {k: v for k, v in split_npz.items()}

    try:
        avg_n_atoms = torch.from_numpy(split_npz_dict['avg_n_atoms'])
    except KeyError:
        n_atoms = 0
        molecules = 0
        for batch in train_loader:
            mask = batch[spk.Properties.atom_mask]
            molecules += mask.size(0)
            n_atoms += mask.sum().item()
        avg_n_atoms = n_atoms / molecules

    try:
        means = {prop: torch.from_numpy(split_npz_dict[f'{prop}_mean']) for prop in properties}
        stddevs = {prop: torch.from_numpy(split_npz_dict[f'{prop}_stddev']) for prop in properties}
        logging.info(f"statistics loaded from {split_file}")
    except KeyError:
        means, stddevs = train_loader.get_statistics(
            properties, divide_by_atoms=True, single_atom_ref=atomrefs
        )

        np.savez(
            split_file,
            **{'avg_n_atoms': avg_n_atoms},
            **{f'{prop}_mean': means[prop] for prop in properties},
            **{f'{prop}_stddev': stddevs[prop] for prop in properties},
            **{
                k: v for k, v in split_npz_dict.items()
                if k not in [f"{prop}_mean" for prop in properties]
                and k not in [f"{prop}_stddev" for prop in properties]
                and k != 'avg_n_atoms'
            }
        )
        logging.info(f"statistics saved to {split_file}")
    return atomrefs, means, stddevs, avg_n_atoms


def create_model(args, atomrefs, means, stddevs, properties, avg_n_atoms):
    ssp = rescaled_act.ShiftedSoftplus(beta=args.beta)
    kernel_conv = create_kernel_conv(
        cutoff=args.rad_maxr,
        n_bases=args.rad_nb,
        n_neurons=args.rad_h,
        n_layers=args.rad_L,
        act=ssp,
        radial_model=args.radial_model
    )

    sp = rescaled_act.Softplus(beta=args.beta)
    if args.res:
        net = ResNetwork(
            kernel_conv=kernel_conv,
            embed=args.embed,
            l0=args.l0,
            l1=args.l1,
            l2=args.l2,
            l3=args.l3,
            L=args.L,
            scalar_act=sp,
            gate_act=rescaled_act.sigmoid,
            avg_n_atoms=avg_n_atoms
        )
    else:
        net = Network(
            kernel_conv=kernel_conv,
            embed=args.embed,
            l0=args.l0,
            l1=args.l1,
            l2=args.l2,
            l3=args.l3,
            L=args.L,
            scalar_act=sp,
            gate_act=rescaled_act.sigmoid,
            avg_n_atoms=avg_n_atoms
        )

    ident = torch.nn.Identity()

    if args.mlp_out:
        outnet = OutputMLPNetwork(
            kernel_conv=kernel_conv,
            previous_Rs=net.Rs[-1],
            l0=args.outnet_l0,
            l1=args.outnet_l1,
            l2=args.outnet_l2,
            l3=args.outnet_l3,
            L=args.outnet_L,
            scalar_act=sp,
            gate_act=rescaled_act.sigmoid,
            mlp_h=args.outnet_neurons,
            mlp_L=args.outnet_layers,
            avg_n_atoms=avg_n_atoms
        )
    else:
        outnet = OutputScalarNetwork(
            kernel_conv=kernel_conv,
            previous_Rs=net.Rs[-1],
            scalar_act=ident,
            avg_n_atoms=avg_n_atoms
        )

    output_modules = [
        spk.atomistic.Atomwise(
            property=prop,
            mean=means[prop],
            stddev=stddevs[prop],
            atomref=atomrefs[prop],
            outnet=outnet,
            # aggregation_mode='sum'
        ) for prop in properties
    ]
    model = spk.AtomisticModel(net, output_modules)
    return model


def train(args, model, properties, means, stddevs, wall, device, train_loader, val_loader):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError("Optimizer must be either adam or sgd.")

    # hooks
    logging.info("build trainer")
    metrics = [spk.train.metrics.MeanAbsoluteError(p, p) for p in properties]
    hooks = [
        spk.train.CSVHook(log_path=args.model_dir, metrics=metrics),
        spk.train.ReduceLROnPlateauHook(optimizer, patience=args.reduce_lr_patience),
        WallHook(wall),
        spk.train.EarlyStoppingHook(patience=args.early_stop_patience),
        NoValidationDerivativeHook()
    ]
    if not args.cpu and logging.root.level <= logging.DEBUG:
        hooks += [MemoryProfileHook(device)]

    # trainer
    if args.mlp_out:
        means = {k: v.to(device) for k, v in means.items()}
        stddevs = {k: v.to(device) for k, v in stddevs.items()}
        loss = build_standardized_mse_loss(properties, means, stddevs)
    else:
        loss = spk.train.build_mse_loss(properties)
    trainer = spk.train.Trainer(
        args.model_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        keep_n_checkpoints=args.keep_n_checkpoints,
        checkpoint_interval=args.checkpoint_interval
    )  # This has the side effect of loading the last checkpoint's state_dict

    # run training
    logging.info("training")
    logging.info(f"device: {device}")
    n_epochs = args.epochs if args.epochs else sys.maxsize
    logging.info(f"Max epochs {n_epochs}")
    trainer.train(device=device, n_epochs=n_epochs)
    # import cProfile
    # cProfile.runctx("trainer.train(device=device, n_epochs=n_epochs)", globals(), locals(), sort="tottime")


def build_standardized_mse_loss(properties, means, stddevs, factors=None):
    """The mean squared error loss function with normalized variance and unit expectation at initialization.
    This assumes that the result from the model is being shifted and scaled by the mean and variance of the data.
    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        factors (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function
    """
    if factors is None:
        factors = [1] * len(properties)
    if len(properties) != len(factors):
        raise ValueError("factors must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, factors):
            batch_rescaled = (batch[prop] - means[prop]) / stddevs[prop]
            result_rescaled = (result[prop] - means[prop]) / stddevs[prop]
            diff = batch_rescaled - result_rescaled
            diff = diff ** 2
            err_sq = factor * torch.mean(diff)
            loss += err_sq
        return loss

    return loss_fn


class WallHook(spk.hooks.Hook):
    def __init__(self, wall_max: float):
        super(WallHook, self).__init__()
        self.wall_start = None
        self.wall_max = wall_max

    def on_train_begin(self, trainer):
        self.wall_start = perf_counter()

    def on_epoch_begin(self, trainer):
        wall = perf_counter() - self.wall_start
        if wall > self.wall_max:
            trainer._stop = True


class NoValidationDerivativeHook(spk.hooks.Hook):
    def on_validation_begin(self, trainer):
        trainer._model.requires_grad_(False)

    def on_validation_end(self, trainer, val_loss):
        trainer._model.requires_grad_(True)


class MemoryProfileHook(spk.hooks.Hook):
    def __init__(self, device_to_profile):
        super(MemoryProfileHook, self).__init__()
        self.device = device_to_profile
        torch.cuda.reset_peak_memory_stats(device=self.device)

    def on_batch_begin(self, trainer, train_batch):
        logging.debug(f"epoch: {trainer.epoch}")
        logging.debug(f"batch position shape: {train_batch[spk.Properties.R].shape}")
        torch.cuda.reset_peak_memory_stats(device=self.device)

    def on_batch_end(self, trainer, train_batch, result, loss):
        memory = {
            "batch": sum([v.element_size() * v.nelement() for k, v in train_batch.items()]),
            "allocated": torch.cuda.max_memory_allocated(device=self.device),
            "cached": torch.cuda.max_memory_cached(device=self.device),
            # "reserved": torch.cuda.max_memory_reserved(device=self.device),
        }

        unit, factor = "mb", 1e-6
        memory = {k: round(v * factor, 1) for k, v in memory.items()}

        logging.debug(f"batch memory: {memory['batch']}")
        logging.debug(
            f"Training Max Stats ({unit}): "
            f"allocated: {memory['allocated']}, "
            f"cached: {memory['cached']}, "
            # f"reserved: {memory['reserved']}"
        )

    def on_validation_batch_begin(self, trainer):
        torch.cuda.reset_peak_memory_stats(device=self.device)

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        logging.debug(f"validation batch position shape: {val_batch[spk.Properties.R].shape}")
        memory = {
            "batch": sum([v.element_size() * v.nelement() for k, v in val_batch.items()]),
            "allocated": torch.cuda.max_memory_allocated(device=self.device),
            "cached": torch.cuda.max_memory_cached(device=self.device),
            # "reserved": torch.cuda.max_memory_reserved(device=self.device),
        }

        unit, factor = "mb", 1e-6
        memory = {k: round(v * factor, 1) for k, v in memory.items()}

        logging.debug(f"validation batch memory: {memory['batch']}")
        logging.debug(
            f"Validation Max Stats ({unit}): "
            f"allocated: {memory['allocated']}, "
            f"cached: {memory['cached']}, "
            # f"reserved: {memory['reserved']}"
        )


def main():
    # Setup script
    parser = argparse.ArgumentParser(parents=[train_parser(), qm9_property_selector()])
    args = parser.parse_args()
    wall = args.wall
    device = configuration(args)
    args = create_or_load_directory(args)
    args = fix_old_args_with_defaults(args)
    properties = [vars(QM9)[k] for k, v in vars(args).items() if k in QM9.properties and v]
    if not properties:
        raise ValueError("No properties were selected to train on.")

    dataset, split_file, train_loader, val_loader, test_loader = get_data(args, properties)
    atomrefs, means, stddevs, avg_n_atoms = get_statistics(dataset, split_file, properties, train_loader)
    model = create_model(args, atomrefs, means, stddevs, properties, avg_n_atoms)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train(args, model, properties, means, stddevs, wall, device, train_loader, val_loader)
    record_versions(os.path.join(args.model_dir, f"versions_{os.uname()[1]}_{date.today()}.txt"))

    if args.evaluate != "False":
        evaluation_file = f"{args.evaluate}_{os.uname()[1]}_{date.today()}.csv"
        logging.info(f"Evaluating test set to file {evaluation_file}")
        metrics = [spk.train.metrics.MeanAbsoluteError(p, p) for p in properties]
        metrics += [spk.train.metrics.RootMeanSquaredError(p, p) for p in properties]
        evaluate(
            args.model_dir,
            model,
            {"test": test_loader},
            device,
            metrics,
            csv_file=evaluation_file
        )


if __name__ == '__main__':
    main()
