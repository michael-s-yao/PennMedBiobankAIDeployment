"""
Simple script to plot sums of gaussian distributions.

Author(s):
    Allison Chae
    Michael S Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Union


def gaussian(
    x: Union[float, np.ndarray], mu: float, sig: float
) -> Union[float, np.ndarray]:
    """
    Returns the value of a gaussian distribution with mean `mu` and spread
    `sigma` at a specified value (or array of values) x.
    Input:
        x: input values.
        mu: mean of the gaussian distribution.
        sigma: standard deviation of the gaussian distribution.
    Returns:
        N(x; mu, sigma).
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def figure(
    figsize: Tuple[int, int] = (10, 6), savepath: Union[Path, str] = None
) -> None:
    """
    Function to reproduce the distribution shift figure in the manuscript.
    Input:
        figsize: figure size. Default 10 by 6.
        savepath: optional path to save the figure plot. Default not saved.
    Returns:
        None.
    """
    plt.figure(figsize=figsize)
    mu = [[1, 3, 6], [-10, -4, -2], [4, 6, 7]]
    sigma = [[1, 1, 1], [2, 3, 1], [1, 1, 1]]
    colors = ["#E64B35", "#00A087", "#3C5488"]
    xmin = min([min(series) for series in mu]) - (
        3.0 * max([max(series) for series in sigma])
    )
    xmax = max([max(series) for series in mu]) + (
        1.0 * max([max(series) for series in mu])
    )
    X = np.linspace(xmin, xmax, num=1000)
    Y = [
        np.sum(
            np.array([gaussian(X, u, s) for u, s in zip(dmu, dsigma)]), axis=0
        )
        for dmu, dsigma in zip(mu, sigma)
    ]
    for co, dist in zip(colors, Y):
        plt.plot(X, dist, alpha=0.5, color=co)
        plt.fill_between(X, dist, alpha=0.25, color=co)
    plt.axis("off")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(
            savepath,
            dpi=600,
            transparent=True,
            bbox_inches="tight"
        )
    return


if __name__ == "__main__":
    figsize = (10, 5)
    savepath = os.path.join(os.path.curdir, str(Path(__file__).stem) + ".png")
    figure(figsize=figsize, savepath=savepath)
