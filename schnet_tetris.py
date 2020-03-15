# pylint: disable=C, R, not-callable, no-member, arguments-differ
from functools import partial

import matplotlib.pyplot as plt

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

    feature = tetris.new_ones(tetris.size(0), tetris.size(1), 1)

    training = {"step": [], "loss": [], "accuracy": [], "elementwise_accuracy": []}
    for step in range(100):
        out = f(feature, tetris)
        loss = torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def plot_training(training, schnet_training):
    color_grid = np.stack(training["elementwise_accuracy"]).T
    schnet_color_grid = np.stack(schnet_training["elementwise_accuracy"]).T

    fig, ax = plt.subplots(2, 2, sharex="col", sharey="row")
    ax[0, 0].pcolor(color_grid)
    ax[0, 0].set_yticklabels(LABELS)
    ax[0, 0].set_title("Y^l_m$ order = {0, 1, 2, 3}")

    ax[0, 1].pcolor(schnet_color_grid)
    ax[0, 1].set_title("Y^l_m$ order = {0}")

    ax[1, 0].plot(training["accuracy"])
    ax[1, 0].set_xlabel("epochs")
    ax[1, 0].set_ylabel("Mean Argmax Accuracy")

    ax[1, 1].plot(schnet_training["accuracy"])
    ax[1, 1].set_xlabel("epochs")
    plt.show()


def main():
    representations = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]
    representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]
    schnet_reps = [[(mul, 0)] for l, mul in enumerate([dim(r) for r in representations])]

    tetris, labels = get_dataset()
    f = SE3Net(len(tetris), representations)
    schnet = SE3Net(len(tetris), schnet_reps)
    training, test_acc = train(tetris, labels, f)
    schnet_training, schnet_acc = train(tetris, labels, schnet)
    plot_training(training, schnet_training)
    # plot_training(schnet_training)
    print('Ok')


if __name__ == '__main__':
    main()
