import logging
import os
import argparse
from datetime import date

import torch

import schnetpack as spk
from schnetpack.datasets import QM9

from arguments import fix_old_args_with_defaults
from qm9_train import configuration, get_data, get_statistics, create_model
from evaluation import evaluate


def load_directory_and_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to the model directory to evaluate.")
    parser.add_argument('--file', type=str, default="eval", help="Name of evaluation file.")
    parser.add_argument('--cpu', action='store_true', help="When true, force cpu usage.")
    parser.add_argument('--bs', type=int, default=0, help="Evaluate at a different batch size.")
    parser.add_argument('--load_recent', action='store_true', help="Load most recent checkpoint, otherwise best.")
    args = parser.parse_args()
    loaded_args = torch.load(os.path.join(args.model_dir, "args.pkl"))
    loaded_args.model_dir = args.model_dir
    loaded_args.cpu = args.cpu
    loaded_args.evaluate = args.file
    loaded_args.load_recent = args.load_recent
    loaded_args.bs = args.bs if args.bs != 0 else loaded_args.bs
    return fix_old_args_with_defaults(loaded_args)


def main():
    args = load_directory_and_args()
    device = configuration(args)
    properties = [vars(QM9)[k] for k, v in vars(args).items() if k in QM9.properties and v]
    if not properties:
        raise ValueError("No properties were selected to train on.")

    dataset, split_file, train_loader, val_loader, test_loader = get_data(args, properties)
    atomrefs, means, stddevs, avg_n_atoms = get_statistics(dataset, split_file, properties, train_loader)
    model = create_model(args, atomrefs, means, stddevs, properties, avg_n_atoms)

    if args.load_recent:
        checkpoints_dir = os.path.join(args.model_dir, "checkpoints")
        epoch = max(
            [
                int(f.split(".")[0].split("-")[-1])
                for f in os.listdir(checkpoints_dir)
                if f.startswith("checkpoint")
            ]
        )
        chkpt = os.path.join(checkpoints_dir, "checkpoint-" + str(epoch) + ".pth.tar")
        model.load_state_dict(torch.load(chkpt)['model'])
    else:
        chkpt = os.path.join(args.model_dir, 'best_model_state_dict.pth.tar')
        model.load_state_dict(torch.load(chkpt))
    logging.info(f'Loading checkpoint: {chkpt}')

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
        file=evaluation_file
    )


if __name__ == '__main__':
    main()
