"""Redraw Figure 4 using the saved HCHO excitation energies and gradients."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullLocator, ScalarFormatter
import numpy as np

root = Path(__file__).resolve().parent
samples = [*range(1, 11), *range(20, 101, 10), *range(200, 1001, 100)]
grids = ["(15,86)", "(40,194)", "(60,302)", "(75,302)", "(90,590)",
         "(105,770)", "(120,974)", "(135,1202)", "(150,1202)"]
colors = {"SVWN": "red", "PBE": "blue", "B3LYP": "green", "TPSS": "magenta"}
hartree_to_ev = 27.2114  # Conversion used for the published energy panels.

plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 12})
fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.65), height_ratios=[0.83, 1.0])
fig.subplots_adjust(left=0.104, right=0.984, bottom=0.18, top=0.975,
                    wspace=0.30, hspace=0.34)

for row, (sweep, energy_file) in enumerate([("quadrature", "Energy.npy"),
                                             ("spatial-grid", "ExcitationEnergy.npy")]):
    x = samples if row == 0 else np.arange(len(grids))
    for functional, color in colors.items():
        folder = root / sweep / functional
        energies = np.load(folder / energy_file)
        gradients = np.load(folder / "Gradient.npy")
        # The last setting is the reference. Average energies over the three states;
        # average gradients over states, atoms and Cartesian components.
        energy_mae = np.abs(energies[:-1] - energies[-1]).mean(axis=1) * hartree_to_ev
        gradient_mae = np.abs(gradients[:-1] - gradients[-1]).mean(axis=(1, 2, 3))
        for ax, mae in zip(axes[row], [energy_mae, gradient_mae]):
            ax.plot(x, mae, "o-", color=color, linewidth=1.5, markersize=4.5, label=functional)

for i, ax in enumerate(axes.flat):
    ax.text(-0.158, 1.025, f"({'abcd'[i]})", transform=ax.transAxes,
            ha="left", va="top", fontsize=13)
    ax.set_ylabel(["MAE of Energies (eV)", "MAE of Gradients (a.u.)"][i % 2])
    ax.set_yscale("log")
    ax.tick_params(direction="in", which="both")
    ax.legend(loc="upper right", fontsize=10.8, framealpha=1, fancybox=True,
              edgecolor="black", borderpad=0.35, handlelength=2, labelspacing=0.28)
    if i < 2:
        ax.set_xscale("log")
        ax.set_xlabel("Gauss-Legendre Sampling Points")
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8)))
        ax.set_ylim((1e-14, 0.9) if i == 0 else (1e-16, 0.009))
        ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-14 if i == 0 else -16, 0, 2)))
    else:
        ax.set_xlabel("Size of Spatial Grids")
        ax.set_xticks(range(len(grids)), grids, rotation=90)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))

fig.savefig(root / "figure4-corrected.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
