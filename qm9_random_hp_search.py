import os
import subprocess
from argparse import ArgumentParser

import numpy as np


def randomized_hps():
    feat_min = 64
    feat_max = 192
    total_components = np.random.randint(feat_min, feat_max)
    percent_l0 = np.random.rand(1)

    outnet_L = np.random.randint(1, 2)
    L = np.random.randint(4, 6) - outnet_L

    total_outnet_components = np.random.randint(feat_min, feat_max)
    percent_outnet_l0 = np.random.rand(1)

    return {
        "bs": int(np.random.randint(2, 25)),
        "lr": float(np.random.rand(1) * 1e-1 + 1e-6),
        "radial_model": ["cosine", "gaussian", "bessel"][np.random.randint(0, 2)],
        "rad_nb": int(np.random.randint(10, 100)),
        "rad_maxr": float(np.random.rand(1) * 28.8 + 1.2),
        "rad_h": int(np.random.randint(feat_min, feat_max)),
        "rad_L": int(np.random.randint(1, 4)),
        "embed": int(np.random.randint(feat_min, feat_max)),
        "l0": int(np.round(percent_l0 * total_components)),
        "l1": int(np.round((1 - percent_l0) * total_components / 3)),
        "L": L,
        "outnet_L": outnet_L,
        "outnet_l0": int(np.round(percent_outnet_l0 * total_outnet_components)),
        "outnet_l1": int(np.round((1 - percent_outnet_l0) * total_outnet_components / 3)),
        "outnet_neurons": int(np.random.randint(feat_min, feat_max)),
        "outnet_layers": int(np.random.randint(1, 3)),
        "beta": float(np.random.rand(1) * 10 + 0.5),
    }


def dict_to_statement(d):
    return " ".join([
        f"--{k} {v}" for k, v in d.items()
    ])


def main(args):
    prefix = "python qm9_train.py"
    fixed_args = f"--split_file {args.split_file} --db {args.db} --mlp_out"
    targets = "--mu --alpha --homo --lumo --gap --r2 --zpve --U0 --U --H --G --Cv"

    try:
        os.mkdir(args.contain_dir)
    except FileExistsError:
        pass

    try:
        num_dir = int(sorted(os.listdir(args.contain_dir))[-1]) + 1
    except IndexError:
        num_dir = 0

    model_dir = os.path.join(args.contain_dir, f'{num_dir:04d}')
    while os.path.exists(model_dir):
        num_dir += 1
        model_dir = os.path.join(args.contain_dir, f'{num_dir:04d}')

    rand_hps = randomized_hps()
    hps = {
        "wall": args.wall,
        "epochs": args.epochs,
        "min_lr": 1e-7,
        "model_dir": model_dir,
        "ntr": args.ntr,
        "nva": args.nva
    }

    statement = " ".join([
        prefix, dict_to_statement(rand_hps), dict_to_statement(hps), fixed_args, targets, "--evaluate False"
    ])
    subprocess.run(
        statement.split(),
    )

    print(model_dir)
    with open(os.path.join(model_dir, 'call.txt'), 'w') as f:
        f.write(statement)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("contain_dir", help="Path to the directory where to put all the model directories.")
    parser.add_argument("db", type=str, help="Path to database.")
    parser.add_argument("split_file", type=str, help="A split.npz file. Loads if exists, writes if not.")
    parser.add_argument("--wall", type=float, default=18000, help="If calculation time is too long, break. One day.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs. 55 is about a day.")
    parser.add_argument("--ntr", type=int, default=109000, help="Number of training examples.")
    parser.add_argument("--nva", type=int, default=1000, help="Number of validation examples.")
    args = parser.parse_args()
    main(args)
