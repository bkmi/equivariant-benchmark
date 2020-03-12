import os

import matplotlib.pyplot as plt
import pandas as pd
# import matplotlib as mpl
# mpl.use("pgf")
# plt.rcParams.update({"pgf.rcfonts": False})


TARGETS = sorted("A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv".split())
SCHNET = """
https://pubs.acs.org/doi/10.1021/acs.jctc.8b00908
property    unit	    model   MAE     RMSE    time
U0          kcal mol–1	SchNet	0.26	0.54	12 h
U0	        kcal mol–1	ACSF	0.49	0.92	8 h
U0	        kcal mol–1	wACSF	0.43	0.81	6 h

mu      	Debye   	SchNet	0.020	0.038	13 h
mu      	Debye   	ACSF	0.064	0.100	8 h
mu      	Debye   	wACSF	0.064	0.095	8 h
"""
SCHNET2 = """
Name                       MAE        unit
dipole_moment              0.021     Debye
electronic_spatial_extent  0.158   Bohr**2
energy_U                   0.012        eV
energy_U0                  0.012        eV
enthalpy_H                 0.012        eV
free_energy                0.013        eV
gap                        0.074        eV
heat_capacity              0.034  Kcal/mol
homo                       0.047        eV
isotropic_polarizability   0.124   Bohr**3
lumo                       0.039        eV
zpve                       1.616       meV
"""

SCHNET_BEST = {"U0_MAE": 0.012, "mu_MAE": 0.021}


def pair_targets_directories(parent: str):
    assert os.path.isdir(parent)
    children = [os.path.join(parent, child) for child in os.listdir(parent)]
    children = [child for child in children if os.path.isdir(child)]
    children = sorted(children, key=lambda x: x.split("/")[-1].split("_")[1])
    pairs = {
        target: directory for target, directory in zip(TARGETS, children)
        if directory.split("/")[-1].split("_")[1] == target
    }
    return pairs


def collect_columns(column, target_directory_pairs):
    def collect_column(directory):
        log = os.path.join(directory, "log.csv")
        df = pd.read_csv(log)
        if column not in df.columns and column == "MAE":
            idx = [x[:3] for x in df.columns].index("MAE")
            return df[df.columns[idx]]
        else:
            return df[column]
    return {target: collect_column(directory).to_numpy() for target, directory in target_directory_pairs.items()}


def populate_page(column, parent_directory, axes, label):
    pairs = pair_targets_directories(parent_directory)
    data = collect_columns(column, pairs)
    lines = []

    for axis, target in zip(axes.flatten(), data.keys()):
        # if column == "MAE":
        #     if target == "U0":
        #         axis.axhline(SCHNET_BEST["U0_MAE"])
        #     elif target == "mu":
        #         axis.axhline(SCHNET_BEST["mu_MAE"])
        # line, = axis.plot(data[target],  label=label)
        line, = axis.semilogy(data[target], label=label)
        lines.append(line)
        axis.set_title(target)
        # axis.xaxis.set_major_locator(plt.MaxNLocator(3))
    # fig.suptitle(column)
    # fig.tight_layout()
    # fig.show()
    return lines


def all_targets():
    columns = ["Train loss", "Validation loss", "MAE"]
    for column in columns:
        fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10, 8), dpi=200)
        l0_lines = populate_page(column, "big_swift", axes, "L0")
        l0l1_lines = populate_page(column, "big_l1_tesseract", axes, "L0 & L1")
        l0_deep3_lines = populate_page(column, "big_3layer_tesseract", axes, "L0 shallow")
        l0l1_deep3_lines = populate_page(column, "big_3layer_l1_tesseract", axes, "L0 & L1 shallow")
        fig.legend(
            (l0_lines[0], l0l1_lines[0], l0_deep3_lines[0], l0l1_deep3_lines[0]),
            ("L0", "L0 & L1", "L0 shallow", "L0 & L1 shallow")
        )
        fig.tight_layout()
        fig.savefig(column.lower().replace(' ', '_') + '.png')
        fig.show()


def u0(figsize=(10/2, 8/2), dpi=200, format=".pdf"):
    prefix = "u0"
    log = "log.csv"
    dfd = {
        "cos_bs12": "20200220_U0",
        "cos_l1_bs12": "20200220_U0_l1",
        "cos_l1l2_bs12": "20200220_U0_l1l2",
        "gau_bs20": "20200301_U0_gauss",
        "gau_l1_bs20": "20200301_U0_gauss_l1",
        "cos_l1_bs16": "u0_res_l1_swift",
        "cos_bs16": "u0_res_swift",
    }

    dfd = {k: pd.read_csv(os.path.join(prefix, v, log)) for k, v in dfd.items()}
    columns = ["Train loss", "Validation loss", "MAE_energy_U0"]
    for column in columns:
        fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
        for k, v in dfd.items():
            axis.semilogy(v[column], label=k)
        fig.legend()
        fig.tight_layout()
        fig.savefig("u0_" + column.lower().replace(' ', '_') + format)
        fig.show()


def mu(figsize=(10/2, 8/2), dpi=200, format=".pdf"):
    prefix = "mu"
    log = "log.csv"
    dfd = {}
    dfd.update({
        "cos_bs12": "20200220_mu",
        "cos_bs12_l1": "20200220_mu_l1",
        "cos_bs12_l1l2": "20200220_mu_l1l2",
    })

    dfd.update({
        "gau_bs20": "20200301_mu_gauss",
        "gau_bs20_l1": "20200301_mu_gauss_l1",
        "gau_bs30": "mu_gauss_bs30",
        "gau_bs30_l1": "mu_l1_gauss_bs30",
    })

    dfd.update({
        "cos_bs16_r50": "mu_bs16_r50",
        "cos_bs16_l1_r50": "mu_l1_bs16_r50",
    })

    dfd.update({
        "gau_bs16": "mu_gauss_bs16",
        "gau_bs16_r50": "mu_gauss_bs16_r50",
        "gau_bs16_r50_shallow": "mu_gauss_bs16_r50_shallow",
        "gau_bs16_l1": "mu_l1_gauss_bs16",
        "gau_bs16_l1_r50": "mu_l1_gauss_bs16_r50",
        "gau_bs16_l1_r50_shallow": "mu_l1_gauss_bs16_r50_shallow",
        "gau_bs16_l1_shallow": 'mu_l1_gauss_bs16_shallow',
    })

    dfd = {k: pd.read_csv(os.path.join(prefix, v, log)) for k, v in dfd.items()}
    columns = ["Train loss", "Validation loss", "MAE_dipole_moment"]
    for column in columns:
        fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
        for k, v in dfd.items():
            axis.semilogy(v[column], label=k)
        # fig.legend()
        fig.tight_layout()
        fig.savefig("mu_" + column.lower().replace(' ', '_') + format)
        fig.show()


def main():
    mu(format='.png')
    u0(format='.png')


if __name__ == '__main__':
    # all_targets()
    main()
