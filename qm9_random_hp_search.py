import os
import subprocess
from shutil import rmtree
from argparse import ArgumentParser

import numpy as np


def randomized_hps():
    feat_min = 80
    feat_max = 144
    total_components = np.random.randint(feat_min, feat_max)
    percent_l0 = np.random.rand(1)

    min_radial = 5.0
    max_radial = 30.0
    width_radial = max_radial - min_radial

    outnet_L = np.random.randint(2, 3)
    L = np.random.randint(4, 7) - outnet_L

    outnet_feat_smaller_by = 16
    outnet_feat_min = feat_min - outnet_feat_smaller_by
    outnet_feat_max = feat_max - outnet_feat_smaller_by
    total_outnet_components = np.random.randint(outnet_feat_min, outnet_feat_max)
    percent_outnet_l0 = np.random.rand(1)

    return {
        "bs": int(np.random.randint(8, 25)),
        "lr": float(np.random.rand(1) * 3e-1 + 1e-5),
        "radial_model": ["cosine", "gaussian", "bessel"][np.random.randint(0, 3)],
        "rad_nb": int(np.random.randint(25, 100)),
        "rad_maxr": float(np.random.rand(1) * width_radial + min_radial),
        "rad_h": int(np.random.randint(feat_min, feat_max)),
        "rad_L": int(np.random.randint(1, 3)),
        "embed": int(np.random.randint(feat_min, feat_max)),
        "l0": int(np.round(percent_l0 * total_components)),
        "l1": int(np.round((1 - percent_l0) * total_components / 3)),
        "L": L,
        "outnet_L": outnet_L,
        "outnet_l0": int(np.round(percent_outnet_l0 * total_outnet_components)),
        "outnet_l1": int(np.round((1 - percent_outnet_l0) * total_outnet_components / 3)),
        "outnet_neurons": int(np.random.randint(outnet_feat_min, outnet_feat_max)),
        "outnet_layers": int(np.random.randint(1, 3)),
        "beta": float(np.random.rand(1) * 10 + 0.5),
    }


def dict_to_statement(d):
    return " ".join([
        f"--{k} {v}" for k, v in d.items()
    ])


def main(args):
    prefix = f"python {os.path.join(os.path.dirname(__file__), 'qm9_train.py')}"
    fixed_args = f"--split_file {args.split_file} " \
                 f"--db {args.db} " \
                 f"--mlp_out " \
                 f"--keep_n_checkpoints 1 " \
                 f"--checkpoint_interval 1"
    targets = "--mu --alpha --homo --lumo --gap --r2 --zpve --U0 --U --H --G --Cv"

    try:
        os.mkdir(args.contain_dir)
    except FileExistsError:
        pass

    num_dir = np.random.randint(0, 999999)
    model_dir = os.path.join(args.contain_dir, f'{num_dir:06d}')
    while os.path.exists(model_dir):
        num_dir += 1
        model_dir = os.path.join(args.contain_dir, f'{num_dir:06d}')

    hps = {
        "wall": args.wall,
        "epochs": args.epochs,
        "min_lr": 1e-7,
        "model_dir": model_dir,
        "ntr": args.ntr,
        "nva": args.nva,
        "evaluate": args.evaluate
    }

    success = False
    while not success:
        rand_hps = randomized_hps()
        statement = " ".join([prefix, dict_to_statement(rand_hps), dict_to_statement(hps), fixed_args, targets])
        r = subprocess.run(statement, shell=True)
        success = True if r.returncode == 0 else False
        if not success:
            rmtree(model_dir)

    with open(os.path.join(model_dir, 'call.txt'), 'w') as f:
        f.write(statement)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("contain_dir", help="Path to the directory where to put all the model directories.")
    parser.add_argument("db", type=str, help="Path to database.")
    parser.add_argument("split_file", type=str, help="A split.npz file. Loads if exists, writes if not.")
    parser.add_argument("--wall", type=float, default=36000, help="If calculation time is too long, break. One day.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs. 55 is about a day.")
    parser.add_argument("--ntr", type=int, default=109000, help="Number of training examples.")
    parser.add_argument("--nva", type=int, default=1000, help="Number of validation examples.")
    parser.add_argument("--evaluate", type=str, default="eval", help="Use False to stop evaluation.")
    args = parser.parse_args()
    main(args)
