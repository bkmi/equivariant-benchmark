# pylint: disable=C, R, not-callable, no-member, arguments-differ
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection

import torch
import numpy as np

from e3nn.non_linearities import GatedBlock
from e3nn.point.operations import Convolution
from e3nn.non_linearities.rescaled_act import relu, sigmoid
from e3nn.kernel import Kernel
from e3nn.radial import CosineBasisModel
from e3nn.o3 import rand_rot
from e3nn.rs import dim


torch.set_default_dtype(torch.float64)
torch.manual_seed(44)
LABELS = ["chiral_shape_1", "chiral_shape_2", "square", "line", "corner", "T", "zigzag", "L"]


def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],   # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],   # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],   # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],   # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],   # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],   # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]   # L
    tetris = torch.tensor(tetris, dtype=torch.get_default_dtype())
    labels = torch.arange(len(tetris))

    # apply random rotation
    tetris = torch.stack([torch.einsum("ij,nj->ni", (rand_rot(), x)) for x in tetris])

    return tetris, labels


class AvgSpacial(torch.nn.Module):
    def forward(self, features):
        return features.mean(1)


class SE3Net(torch.nn.Module):
    def __init__(self, num_classes, representations):
        super().__init__()

        R = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=50, act=relu)
        K = partial(Kernel, RadialModel=R)

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, relu, sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)
            return torch.nn.ModuleList([conv, act])

        self.firstlayers = torch.nn.ModuleList([
            make_layer(Rs_in, Rs_out)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])
        self.lastlayers = torch.nn.Sequential(AvgSpacial(), torch.nn.Linear(64, num_classes))

    def forward(self, features, geometry):
        for conv, act in self.firstlayers:
            features = conv(features, geometry, n_norm=4)
            features = act(features)

        return self.lastlayers(features)


def train(tetris, labels, f):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tetris = tetris.to(device)
    labels = labels.to(device)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.4)

    feature = tetris.new_ones(tetris.size(0), tetris.size(1), 1)

    training = {"step": [], "loss": [], "accuracy": [], "elementwise_accuracy": []}
    for step in range(120):
        out = f(feature, tetris)
        loss = torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        element_wise_acc = out.softmax(dim=-1).diag().detach().cpu().numpy()
        acc = out.argmax(1).eq(labels).double().mean().item()
        training["step"].append(step)
        training["loss"].append(loss.item())
        training["accuracy"].append(acc)
        training["elementwise_accuracy"].append(element_wise_acc)
        print("step={} loss={} accuracy={}".format(step, loss.item(), acc))

    out = f(feature, tetris)

    r_tetris, _ = get_dataset()
    r_tetris = r_tetris.to(device)
    r_out = f(feature, r_tetris)
    test_accuracy = r_out.argmax(1).eq(labels).double().mean().item()

    print('equivariance error={}'.format((out - r_out).pow(2).mean().sqrt().item()))
    return training, test_accuracy


def plot_training(data, sections, figsize=(8, 4)):
    fig, ax = plt.subplots(2, len(data), sharex="col", sharey="row", figsize=figsize)
    for i, d in enumerate(data):
        cmap=plt.get_cmap("plasma")

        acc = np.array([np.mean(i) for i in np.split(np.array(d["accuracy"]), sections)])
        points = np.stack([np.linspace(0, len(acc), len(acc)), acc], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(acc.min(), acc.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(acc)
        # lc.set_linewidth(2)
        ax[0, i].add_collection(lc)

        # acc = np.array([np.mean(i) for i in np.split(np.array(d["accuracy"]), sections)])
        # ax[0, i].plot(acc)
        ax[0, i].set_ylim(0.0, 1.09)
        if i == 0:
            ax[0, i].set_ylabel("Mean Argmax Accuracy")

        if i == 0:
            ax[0, i].set_title(f"$Y^l_m$ order = {list(range(i+1))}")
        else:
            ax[0, i].set_title(f"{list(range(i+1))}")

        color_grid = np.stack(d["elementwise_accuracy"], axis=-1)
        color_grid = np.stack([np.mean(i, axis=1) for i in np.split(color_grid, sections, axis=1)], axis=-1)
        ax[1, i].pcolormesh(
            color_grid,
            cmap=cmap
        )
        ax[1, i].set_yticklabels(LABELS)
        ax[1, i].set_xlabel("training")

    plt.tight_layout()
    plt.savefig("tetris.pdf", dpi=200)
    plt.show()


def main():
    representations1 = [(1,), (3, 4, 0, 0), (8, 8, 0, 0), (8, 6, 0, 0), (64,)]
    representations1 = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations1]
    representations2 = [(1,), (2, 3, 2, 0), (6, 5, 5, 0), (6, 4, 4, 0), (64,)]
    representations2 = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations2]
    representations3 = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]
    representations3 = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations3]
    representations0 = [[(mul, 0)] for l, mul in enumerate([dim(r) for r in representations3])]

    tetris, labels = get_dataset()
    data = []
    for i, reps in enumerate([representations0, representations1, representations2, representations3]):
        f = SE3Net(len(tetris), reps)
        training, _ = train(tetris, labels, f)
        data.append(training)
    return data


if __name__ == '__main__':
    data = main()
    plot_training(data, 24)
