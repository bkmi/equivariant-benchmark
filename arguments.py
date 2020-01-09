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
    parser = ArgumentParser(add_help=False)
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


def qm9_property_selector():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(
        title="Select qm9 property to train on.",
        description="Choose from options in schnetpack/src/schnetpack/datasets/qm9.py",
    )
    group.add_argument("--A", action='store_true', help="rotational_constant_A")
    group.add_argument("--B", action='store_true', help="rotational_constant_B")
    group.add_argument("--C", action='store_true', help="rotational_constant_C")
    group.add_argument("--mu", action='store_true', help="dipole_moment")
    group.add_argument("--alpha", action='store_true', help="isotropic_polarizability")
    group.add_argument("--homo", action='store_true', help="homo")
    group.add_argument("--lumo", action='store_true', help="lumo")
    group.add_argument("--gap", action='store_true', help="gap")
    group.add_argument("--r2", action='store_true', help="electronic_spatial_extent")
    group.add_argument("--zpve", action='store_true', help="zpve")
    group.add_argument("--U0", action='store_true', help="energy_U0")
    group.add_argument("--U", action='store_true', help="energy_U")
    group.add_argument("--H", action='store_true', help="enthalpy_H")
    group.add_argument("--G", action='store_true', help="free_energy")
    group.add_argument("--Cv", action='store_true', help="heat_capacity")
    return parser


def train_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model_dir", type=directory_format, required=True, help="Directory to save model.")
    parser.add_argument("--split_file", type=str, default="", help="A split.npz file. Loads if exists, writes if not.")
    parser.add_argument("--overwrite", action='store_true', help="When set, overwrite content of model_dir.")
    parser.add_argument("--db", type=str, required=True, help="Path to database.")
    parser.add_argument("--wall", type=float, required=True, help="If calculation time is too long, break.")
    parser.add_argument("--cpu", action='store_true', help="Only allow cpu.")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers for data loader.")

    parser.add_argument("--ntr", type=int, default=1000, help="Number of training examples.")
    parser.add_argument("--nva", type=int, default=100, help="Number of validation examples.")

    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--bs", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

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

    parser.add_argument("--beta", type=float, default=5.0, help="Softplus and ShiftedSoftplus rescale parameter.")
    return parser


if __name__ == '__main__':
    pass
