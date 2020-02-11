import argparse

import torch

from qm9_train import configuration


def load_directory_and_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', required=True, help="Path to the model directory to evaluate.")
    parser.add_argument('--cpu', action='store_true', help="When true, force cpu usage.")
    args = parser.parse_args()
    loaded_args = torch.load(args.model_dir + "args.pkl")
    loaded_args.cpu = args.cpu
    return loaded_args


def main():
    args = load_directory_and_args()
    device = configuration(args)


if __name__ == '__main__':
    main()
