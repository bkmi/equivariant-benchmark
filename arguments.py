from argparse import ArgumentParser


def directory_format(string: str):
    string = str(string)
    if string[-1] is "/":
        pass
    elif string[-1] is "\\":
        raise TypeError("Can't use pc format.")
    else:
        string += "/"
    return string


def qm9_energy_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=directory_format, required=True, help="Directory to save model.")

    parser.add_argument("--ntr", type=int, default=1000, help="Number of training examples.")
    parser.add_argument("--nva", type=int, default=100, help="Number of validation examples.")

    # parser.add_argument("--init_seed", type=int, default=0, help="Random seed for initializing network.")
    # parser.add_argument("--data_seed", type=int, default=0, help="Random seed for organizing data.")
    # parser.add_argument("--batch_seed", type=int, default=0, help="Random seed for batch distribution.")

    parser.add_argument("--db", type=str, required=True, help="Path to qm9 database.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    # parser.add_argument("--wall", type=float, required=True, help="If calculation time is too long, break.")
    parser.add_argument("--cpu", action='store_true', help="Only allow cpu.")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers for data loader.")

    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--l0", type=int, default=64)
    parser.add_argument("--l1", type=int, default=0)
    parser.add_argument("--l2", type=int, default=0)
    parser.add_argument("--l3", type=int, default=0)
    parser.add_argument("--L", type=int, default=4, help="How many layers to create.")

    parser.add_argument("--rad_nb", type=int, default=25, help="Radial number of bases.")
    parser.add_argument("--rad_maxr", type=float, default=5.0, help="Max radius.")
    parser.add_argument("--rad_h", type=int, default=128, help="Size of radial weight parameters.")
    parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
    return parser


if __name__ == '__main__':
    pass
