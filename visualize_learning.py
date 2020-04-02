import os
import glob
from functools import partial, reduce
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import torch

import schnetpack as spk
from schnetpack.datasets import QM9


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
LOG = "log.csv"
MAE_COLUMNS = {
    "dipole_moment": "MAE_dipole_moment",
    "electronic_spatial_extent": "MAE_electronic_spatial_extent",
    "energy_U": "MAE_energy_U",
    "energy_U0": "MAE_energy_U0",
    "enthalpy_H": "MAE_enthalpy_H",
    "free_energy": "MAE_free_energy",
    "gap": "MAE_gap",
    "heat_capacity": "MAE_heat_capacity",
    "homo": "MAE_homo",
    "isotropic_polarizability": "MAE_isotropic_polarizability",
    "lumo": "MAE_lumo",
    "zpve": "MAE_zpve",
}


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


def read_log_files(prefix, name_filename_dicts, log_file="log.csv"):
    df_dicts = []
    for nfd in name_filename_dicts:
        df_dicts.append({k: pd.read_csv(os.path.join(prefix, v, log_file)) for k, v in nfd.items()})
    return df_dicts


def semilogy_df_dict(df_dict, column, axis):
    for k, v in df_dict.items():
        axis.semilogy(v[column], label=k)
    return None


def several_axes_semilog_df_dict(columns, axes, dfds, titles, xlabels, ylabels):
    assert len(axes) == len(dfds)
    assert len(axes) == len(titles)
    assert len(axes) == len(ylabels)
    assert len(axes) == len(xlabels)

    for col, ax, dfd, title, xlabel, ylabel in zip(columns, axes, dfds, titles, xlabels, ylabels):
        semilogy_df_dict(dfd, col, ax)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    return None


compare_two_mu_mae_semilogs_by_order = partial(
    several_axes_semilog_df_dict,
    columns=["MAE_dipole_moment"] * 2,
    titles=["$Y^l_m$ order = {0}", "$Y^l_m$ order = {0, 1}"],
    xlabels=["epochs"] * 2,
    ylabels=["MAE $\mu$", None]
)


def mu_u0_compare_order(figsize=(3, 5), dpi=200, format=".pdf"):
    dfd_mu = {
        "cos_bs12": "20200220_mu",
        "cos_bs12_l1": "20200220_mu_l1",
        "cos_bs12_l1l2": "20200220_mu_l1l2",
    }
    dfd_u0 = {
        "cos_bs12": "20200220_U0",
        "cos_l1_bs12": "20200220_U0_l1",
        "cos_l1l2_bs12": "20200220_U0_l1l2",
    }
    dfd_mu, = read_log_files("mu", [dfd_mu])
    dfd_u0, = read_log_files("u0", [dfd_u0])
    dfds = [dfd_mu, dfd_u0]
    fig, axis = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi)
    several_axes_semilog_df_dict(
        ["MAE_dipole_moment", "MAE_energy_U0"],
        axis.flatten(),
        dfds,
        # titles=["$\mu$", "u0"],
        titles=[None, None],
        xlabels=[None, "epochs"],
        ylabels=["MAE $\mu$", "MAE u0"]
    )
    axis[1].legend(["$Y^0_m$", "$Y^{0, 1}_m$", "$Y^{0, 1, 2}_m$"])
    fig.tight_layout()
    fig.savefig("mu_u0_compare_orders" + format)
    fig.show()


def mu_bs16_r50_compare_basis(figsize=(5, 3), dpi=200, format=".pdf"):
    dfd = {
        "cos_bs16_r50": "mu_bs16_r50",
        "gau_bs16_r50": "mu_gauss_bs16_r50",
        "bes_bs16_r50": "mu_bs16_r50_bessel",
    }
    dfd_l1 = {
        "cos_bs16_l1_r50": "mu_l1_bs16_r50",
        "gau_bs16_l1_r50": "mu_l1_gauss_bs16_r50",
        "bes_bs16_l1_r50": "mu_l1_bs16_r50_bessel"
    }

    dfds = read_log_files("mu", [dfd, dfd_l1])
    fig, axis = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize, dpi=dpi)
    compare_two_mu_mae_semilogs_by_order(axes=axis, dfds=dfds)
    axis[1].legend(["Cosine", "Gaussian", "Bessel"])
    fig.tight_layout()
    fig.savefig("mu_bs16_r50_compare_basis" + format)
    fig.show()


def mu_r25_compare_bs(figsize=(5, 3), dpi=200, format=".pdf"):
    dfd = {
        "cos_bs12": "20200220_mu",
        "gau_bs16": "mu_gauss_bs16",
        "gau_bs20": "20200301_mu_gauss",
        "gau_bs30": "mu_gauss_bs30",
    }
    dfd_l1 = {
        "cos_bs12_l1": "20200220_mu_l1",
        "gau_bs16_l1": "mu_l1_gauss_bs16",
        "gau_bs20_l1": "20200301_mu_gauss_l1",
        "gau_bs30_l1": "mu_l1_gauss_bs30",
    }

    dfds = read_log_files("mu", [dfd, dfd_l1])
    fig, axis = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize, dpi=dpi)
    compare_two_mu_mae_semilogs_by_order(axes=axis, dfds=dfds)
    axis[1].legend(["cos bs12", "gauss bs16", "gauss bs20", "gauss bs30"])
    fig.tight_layout()
    fig.savefig("mu_r25_compare_bs" + format)
    fig.show()


def mu_compare_model_size(figsize=(5, 3), dpi=200, format=".pdf"):
    dfd = {
        "cos_bs12": "20200220_mu",
        "gau_bs16": "mu_gauss_bs16",
        "cos_bs16_r50": "mu_bs16_r50",
        "gau_bs16_r50": "mu_gauss_bs16_r50",
        "gau_bs16_r50_shallow": "mu_gauss_bs16_r50_shallow",
    }
    dfd_l1 = {
        "cos_bs12_l1": "20200220_mu_l1",
        "gau_bs16_l1": "mu_l1_gauss_bs16",
        "cos_bs16_l1_r50": "mu_l1_bs16_r50",
        "gau_bs16_l1_r50": "mu_l1_gauss_bs16_r50",
        "gau_bs16_l1_r50_shallow": "mu_l1_gauss_bs16_r50_shallow",
    }

    dfds = read_log_files("mu", [dfd, dfd_l1])
    fig, axis = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize, dpi=dpi)
    compare_two_mu_mae_semilogs_by_order(axes=axis, dfds=dfds)
    axis[1].legend(["cos bs12 r25", "gauss r25", "cos", "gauss", "gauss shallow"])
    fig.tight_layout()
    fig.savefig("mu_compare_model_size" + format)
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
        "bes_bs16_r50": "mu_bs16_r50_bessel",
        "bes_bs16_l1_r50": "mu_l1_bs16_r50_bessel"
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
        "cos_bs16_r50": "u0_bs16_r50",
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


def partition_polar_molecules():
    db = "qm9.db"
    dataset = QM9(db)
    _, _, test = spk.train_test_split(
        dataset,
        num_train=109000,
        num_val=1000,
        split_file="pst.npz"
    )
    dipoles = [t["dipole_moment"].item() for t in test]
    print(any(dipoles == 0.))  # Only gave 8!!


def gleichzeitig(figsize=(10, 8), dpi=200, format=".pdf"):
    dfds = {
        "cos_bs16": "20200310_gleichzeitig",
        "cos_bs16_l1": "20200310_gleichzeitig_l1",
    }
    dfds = {k: pd.read_csv(os.path.join(v, LOG)) for k, v in dfds.items()}
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=figsize, dpi=dpi)
    for hps, dfd in dfds.items():
        for ax, (k, v) in zip(axes.flatten(), MAE_COLUMNS.items()):
            ax.semilogy(dfd[v], label=hps)
            ax.set_title(k)
        ax.set_xlabel('epochs')
        ax.legend()
    fig.tight_layout()
    fig.savefig("gleichzeitig" + format)
    fig.show()


def random_hp(figsize=(10, 8), dpi=200, format=".pdf"):
    prefix = 'randsearch'
    dfs = [i for i in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, i))]
    dfds = {d: pd.read_csv(os.path.join(prefix, d, LOG)) for d in dfs}
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=figsize, dpi=dpi)
    for hps, dfd in dfds.items():
        for ax, (k, v) in zip(axes.flatten(), MAE_COLUMNS.items()):
            ax.semilogy(dfd[v], label=hps)
            ax.set_title(k)

    # j = len(axes[-1, :].flatten())
    for i, ax in enumerate(axes[-1, :].flatten()):
        ax.set_xlabel('epochs')
        # ymin, ymax = ax.get_ylim()
        # ymax = 10 if ymax > 10 else ymax
        # print(ymax)
        # ax.set_ylim(None, ymax)
        # ax.set_ylim(10, 10e-1)
        # ax.legend()
        # if i == j-1:
        #     ax.legend()
    fig.tight_layout()
    # fig.legend()
    fig.savefig("randsearch" + format)
    fig.show()


def fix_eval_csv_strings(parent):
    evals = glob.glob(os.path.join(parent, '*/eval*.csv'))
    for ev in evals:
        with open(ev) as fin, tempfile.NamedTemporaryFile(mode="w", delete=False) as fout:
            for line in fin:
                if line[:2] == '"[' and line[-3:-1] == ']"':
                    fout.write(line[2:-3])
                else:
                    fout.write(line)
            os.rename(fout.name, os.path.abspath(ev))


def random_hp_table(parent):
    evals = glob.glob(os.path.join(parent, '*/eval*.csv'))
    names = [os.path.dirname(ev).split("/")[-1] for ev in evals]
    dfs = [pd.read_csv(ev).set_index(pd.Index([name])) for ev, name in zip(evals, names)]
    df = reduce(lambda x, y: x.append(y), dfs)
    return df.dropna().apply(pd.to_numeric).drop([col for col in df.columns if not "MAE" in col], axis=1)


def table_from_folder_of_model_dirs(parent):
    evals = glob.glob(os.path.join(parent, '*/eval*.csv'))
    names = [ev.split('/')[1].split("_")[-1] for ev in evals]
    dfs = [pd.read_csv(ev) for ev in evals]
    mae = {}
    for name, df in zip(names, dfs):
        item = df.iloc[0, 0]
        try:
            item[0]
            mae[name] = eval(item)[0]
        except IndexError:
            mae[name] = item
    return mae


def logs_from_folder_of_model_dirs(parent, column_prefix_or_title="MAE"):
    logs = glob.glob(os.path.join(parent, '*/log.csv'))
    names = [ev.split('/')[1].split("_")[-1] for ev in logs]
    dfs = [pd.read_csv(log) for log in logs]
    columns = [col for df in dfs for col in df.columns if column_prefix_or_title in col]
    return list(zip(*sorted(zip(names, dfs, columns))))


def plot_all_targets(names, dfs, columns, axes, label):
    lines = []
    for name, df, column, axis in zip(names, dfs, columns, axes):
        axis.set_title(name)
        line, = axis.semilogy(df[column], label=label)
        # line, = axis.loglog(df[column], label=label)
        lines.append(line)
    return lines


def fig_axis_all_targets(parents, savefile="", alternative_labels=None):
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(10, 8), dpi=200)
    for i, parent in enumerate(parents):
        lines = plot_all_targets(
            *logs_from_folder_of_model_dirs(parent),
            axes.flatten(),
            parent if alternative_labels is None else alternative_labels[i]
        )

    for axis in axes.flatten():
        axis.set_xlim(None, 60)
    axes.flatten()[-1].legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("epochs")

    fig.tight_layout()
    if savefile:
        fig.savefig(savefile)
    fig.show()


def schnet_alpha_small_batches(figsize=(3, 3), dpi=200, format=".pdf"):
    df = pd.read_csv('/home/ben/sci/equivariant-benchmark/alpha_schnet/log.csv')
    fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
    axis.semilogy(df['MAE_isotropic_polarizability'][:300], label="SchNet bs 64")
    axis.set_ylim(0, None)
    axis.legend()
    axis.set_ylabel(r'MAE $\alpha$')
    axis.set_xlabel('epochs')
    fig.tight_layout()
    fig.savefig("schnet_alpha_small_batches" + format)
    fig.show()


def l0net_l1net_mu_mae_correlation(figsize=(10, 8), dpi=200, format=".pdf"):
    us = torch.load("us.pkl")
    mus = torch.load("mus.pkl")
    l0net = torch.abs(torch.load("all/20200308_U/results.pkl")['energy_U'].cpu() - us)
    l1net = torch.abs(torch.load("all_l1/20200308_U/results.pkl")['energy_U'].cpu() - us)
    diff = l1net - l0net
    fig, axis = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=figsize, dpi=dpi)
    axis[0].scatter(mus.cpu().numpy(), l0net.cpu().numpy())
    axis[1].scatter(mus.cpu().numpy(), l1net.cpu().numpy())
    # axis[0].scatter(mus, diff)
    fig.show()


def main():
    # fix_eval_csv_strings("randsearch")
    # df = random_hp_table("randsearch")
    # df.idxmin().value_counts()

    # 851941
    # 072052
    # fig_axis_all_targets(["all", "all_l1"], "all_l0l1_compare.pdf", alternative_labels=["L0Net", "L1Net"])

    # mae = table_from_folder_of_model_dirs("851941")
    # order = "mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    # # alphabetical_order = [y for x, y in sorted(zip(order.lower().split(), order.split()))]
    # for name in order.split():
    #     try:
    #         val = mae[name] if name in "mu alpha r2 Cv".split() else 1000 * mae[name]
    #         # val = mae[name]
    #         print(name, round(val, 3))
    #     except KeyError:
    #         print(name)

    # mu(format='.png')
    # u0(format='.png')

    # Main small plots
    # schnet_alpha_small_batches()
    # mu_bs16_r50_compare_basis(format='.pdf')
    # mu_r25_compare_bs(format='.pdf')
    # mu_compare_model_size(format='.pdf')
    # mu_u0_compare_order(format='.pdf')
    l0net_l1net_mu_mae_correlation()

    # done with naive multi target bad
    # gleichzeitig(format='.png')
    # random_hp(format='.png')

    # partition_polar_molecules()
    print("done")


if __name__ == '__main__':
    # all_targets()
    main()
