# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
import matplotlib.pyplot as plt
import numpy as np


def plot_std_mean_error_dist(RAs, RBs, origA, origB, secret, Q, n_brute_force):
    plt.subplots(1, 3, figsize=(12, 3 * 1))
    plt.subplot(1, 3, 1)
    plt.plot(np.abs(RAs).mean(0), label="RA")
    plt.plot(np.abs(origA).mean(0), label="origA")
    for x in secret.nonzero()[0]:
        plt.axvline(x, color="r", alpha=0.2)
    plt.axvline(n_brute_force, color="g", alpha=0.5)
    plt.title("means")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(RAs.std(0), label="RA")
    plt.plot(origA.std(0), label="origA")
    plt.title("stds")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.hist(
        (RAs @ secret - RBs) % Q,
        bins=min(100, Q),
        alpha=0.5,
        label="reduced",
        density=True,
    )
    plt.hist(
        (origA @ secret - origB) % Q,
        bins=min(100, Q),
        alpha=0.5,
        label="orig",
        density=True,
    )
    plt.hist(
        (RAs @ np.random.permutation(secret) - RBs) % Q,
        bins=min(100, Q),
        alpha=0.5,
        label="random",
        density=True,
    )
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_partial(start, end, RAs, RBs, secret, Q):
    bins = 50
    range_ = (0, Q - 1)
    plt.hist(
        (RAs[:, start:end] @ secret[start:end] - RBs) % Q,
        bins=bins,
        alpha=0.5,
        range=range_,
        label="partial secret",
    )
    plt.hist(
        (RAs[:, start:end] @ np.random.permutation(secret[start:end]) - RBs) % Q,
        bins=bins,
        range=range_,
        alpha=0.5,
        label="random",
    )
    plt.legend(loc="lower right")
    plt.show()
