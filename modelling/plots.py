"""
Module with some reusable plotting functions. Taken from:
https://github.com/SebastianCallh/blog-spectral-mixture-kernel/blob/master/src/smk/plots.py
"""

from os import path
from pathlib import Path
import matplotlib as mpl
import numpy as np
import torch

from torch.distributions import Categorical, Independent, Normal
from torch.distributions.mixture_same_family import MixtureSameFamily

font = {"family": "DejaVu Sans", "size": 18}
mpl.rc("font", **font)

ROOT_DIR = Path(path.dirname(path.abspath(__file__)))
PLOTS_DIR = ROOT_DIR / "plots"


def save_plot(fig, name: str, format: str = "svg") -> None:
    fig.savefig(PLOTS_DIR / (name + f".{format}"), format=format)


def plot_cov_mat(kernel, ax, xx):
    ax.matshow(kernel(xx, xx).numpy())


def plot_kernel(kernel, ax, xx=torch.linspace(-0.1, 0.1, 1000), col="tab:blue"):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), color=col)


def plot_density(freq, density, ax):
    x = freq.numpy().flatten()
    y = density.numpy().flatten()
    ax.plot(x, y, color="tab:blue", lw=3)
    ax.fill_between(x, y, np.ones_like(x) * y.min(), color="tab:blue", alpha=0.5)
    ax.set_title("Kernel spectral density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Log Density")


def plot_components(smk, ax, nyquist):

    iter = zip(
        smk.mixture_weights.detach(),
        smk.mixture_means.detach(),
        smk.mixture_scales.detach(),
    )

    freqs = torch.linspace(0, nyquist, 1000)
    for w, mu, sigma in iter:
        dist = Normal(mu % nyquist, sigma)
        ax.plot(
            freqs,
            w * dist.log_prob(freqs).flatten().exp(),
            label=f"Compoment at {dist.mean.item():.3}",
        )

    ax.set_title("'Kernel spectral density")
    ax.legend(fontsize=11)
    return ax
