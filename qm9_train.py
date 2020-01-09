import logging
import os
import argparse
from shutil import rmtree

import torch

import schnetpack as spk
from schnetpack.datasets import QM9

from e3nn.non_linearities import rescaled_act

from arguments import train_parser, qm9_property_selector
from networks import convolution, Network, OutputScalarNetwork

# Setup script
torch.set_default_dtype(torch.float32)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
parser = argparse.ArgumentParser(parents=[train_parser(), qm9_property_selector()])
args = parser.parse_args()
properties = [k for k, v in vars(args).items() if k in QM9.properties and v]
if not properties:
    raise ValueError("No properties were selected to train on.")

# Build directory
try:
    os.makedirs(args.model_dir)
except FileExistsError:
    logging.warning(f"Model directory {args.model_dir} exists.")
    if args.overwrite:
        logging.warning("Overwriting.")
        rmtree(args.model_dir, ignore_errors=True)
        os.makedirs(args.model_dir)
    else:
        logging.warning("Loading from checkpoint.")

torch.save(args, args.model_dir + "args.pkl")

# data preparation
logging.info("get dataset")
split_file = os.path.join(args.model_dir, "split.npz") if not args.split_file else args.split_file
dataset = QM9(args.db, load_only=[QM9.U0])
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

# statistics
atomrefs = dataset.get_atomref(properties)
means, stddevs = train_loader.get_statistics(
    properties, divide_by_atoms=True, single_atom_ref=atomrefs
)

# model build
logging.info("build model")

ssp = rescaled_act.ShiftedSoftplus(beta=args.beta)
conv = convolution(
    cutoff=args.rad_maxr,
    n_bases=args.rad_nb,
    n_neurons=args.rad_h,
    n_layers=args.rad_L,
    act=ssp
)

sp = rescaled_act.Softplus(beta=args.beta)
net = Network(
    conv=conv,
    embed=args.embed,
    l0=args.l0,
    l1=args.l1,
    l2=args.l2,
    l3=args.l3,
    L=args.L,
    scalar_act=sp,
    gate_act=rescaled_act.sigmoid
)

ident = torch.nn.Identity()
outnet = OutputScalarNetwork(conv=conv, previous_Rs=net.Rs[-1], scalar_act=ident)

output_modules = [
    spk.atomistic.Atomwise(
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
         spk.train.ReduceLROnPlateauHook(optimizer),
         ShoutHook()]

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
device = torch.device("cpu") if args.cpu else torch.device("cuda")
logging.info(f"device: {device}")
trainer.train(device=device, n_epochs=args.epochs)