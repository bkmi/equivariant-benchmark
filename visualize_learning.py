import os

import matplotlib.pyplot as plt
import pandas as pd


TARGETS = sorted("A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv".split())


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
        axis.set_title(target)
        # axis.plot(data[target])
        line, = axis.semilogy(data[target], label=label)
        lines.append(line)
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
