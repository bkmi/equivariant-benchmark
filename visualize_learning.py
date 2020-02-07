import os

import matplotlib.pyplot as plt
import pandas as pd


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
SCHNET_BEST = {"U0_MAE": 0.26, "U0_RMSE": 0.54, "mu_MAE": 0.020, "mu_RMSE": 0.038}


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


def main():
    columns = ["Train loss", "Validation loss", "MAE"]
    for column in columns:
        fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10, 8), dpi=200)
        l0_lines = populate_page(column, "big", axes, "L0")
        l0l1_lines = populate_page(column, "from_tesseract", axes, "L0 & L1")
        fig.legend((l0_lines[0], l0l1_lines[0]), ("L0", "L0 & L1"))
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    main()
