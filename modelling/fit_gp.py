"""
Script for fitting a GP to the cyg-X1 data.
Author: Ryan-Rhys Griffiths
"""

import torch
from matplotlib import pyplot as plt

from data.dataloader import load_data
from modelling.smk_models import SMKernelGP
from modelling.train import train
from modelling.plots import plot_kernel, plot_density, save_plot

if __name__ == "__main__":

    times, counts, uncertainties, states = load_data(path='../data/cyg_data.txt')

    times = torch.tensor(times, dtype=torch.float32)
    counts = torch.tensor(counts, dtype=torch.float32)
    noise = torch.tensor(uncertainties, dtype=torch.float32)

    model = SMKernelGP(times, counts, noise, num_mixtures=10)
    loss = train(model, times, counts, num_iters=50, lr=0.05)

    grid = torch.linspace(times.min().item(), times.max().item(), 2000)
    pred, lower, upper = model.predict(grid)


    def plot_model_fit():
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(14, 12))
        ax_t.set_title("Model fit")
        ax_t.set_xlabel("t")
        ax_t.set_ylabel("y(t)")
        ax_t.plot(grid, pred.numpy().flatten(), label="Predicted mean")
        ax_t.fill_between(grid, lower.numpy(), upper.numpy(), alpha=0.5, label="Confidence")
        ax_t.scatter(times, counts, label="Observations", s=5)
        ax_t.legend()

        density = model.spectral_density()
        freq = torch.linspace(0, 4000, 10000).reshape(-1, 1)
        plot_density(freq, density.log_prob(freq).exp(), ax=ax_f)
        ax_f.set_ylabel("Density")
        fig.tight_layout()
        save_plot(fig, "toy_data_model_fit")


    def plot_model_kernel():
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_kernel(model.cov, xx=torch.linspace(-2, 2, 1000), ax=ax, col="tab:blue")
        ax.set_title("Learned kernel")
        save_plot(fig, "toy_data_model_kernel")


    plot_model_fit()
    plot_model_kernel()