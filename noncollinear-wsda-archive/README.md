# Noncollinear generalization of nonlocal pure exchange–correlation functionals

Wang et al., *J. Chem. Phys.* **162**, 214104 (2025). [Article](https://doi.org/10.1063/5.0260762)

## Files

Energies are in hartree (Ha). Figures 1 and 4 are schematics.

| Result | Files |
| --- | --- |
| [Table I · Li₃](table-I/) | `GridsTest0220`, `GridsTest0221` and `GridsTest0224` are the three original job batches. [Grids.xlsx](table-I/Grids.xlsx) collects the energies. |
| [Figure 2 · Na](fig-2/) | One folder per spin-sample count. `GET.npy` and `GMO.npy` hold total and orbital energies. [SpinSamples.xlsx](fig-2/SpinSamples.xlsx) lists the errors. The lowest total-energy errors are plotted at approximately **1e-14 Ha**. |
| [Figure 3 · Li₃](fig-3/) | The `x`, `y` and `z` folders hold the input and energy log for each rotation axis. The plotting floor was 1e-15 Ha. |
| [Table II · Li₂](table-II/) | `para` and `oppo` label parallel and antiparallel spins. The suffix gives the bond length in Å. Each folder contains the input and job log. |
| [Table III · H₃](table-III/) | `PySCF/` groups jobs by basis set and bond length. `Octopus/10A/` and `Octopus/25A/` hold the OEP jobs at **1.0 Å** and **2.5 Å**, respectively. |
| [Figure 5 · Cr₃](fig-5/) | `SCF/` holds the density calculation. The panel data and plotting notebooks are in `magnetization-field/` for (a,b) and `torque/` for (c). |

## Notes on the published results

**Table III.** The Octopus OEP calculation used a grid spacing of 0.300 bohr (0.159 Å). The article gives 0.3 Å. At a bond length of 1.0 Å, the noncollinear SCF reached the iteration limit without converging.

## Software

**Source files.** `src/Na/` holds the WSDA source for Figure 2, whose jobs used a pruned spatial grid. `src/WSDA/` holds the later implementation. Figure 5 field routines are in the panel folders. `src/Tools/` contains the rotation and torque routines.

**Software versions.** Python 3.8.19, PySCF 2.5.0, NumPy 1.24.4, SciPy 1.10.1, Libxc 6.1.0 and MCfun 0.2.2. The Cr₃ SCF used the group’s modified pySD (base commit `03581f6b10c180066efb0d865851fdd0576aad24`). Octopus 15.1 reported commit `27390fd694aed19e6b59b38239c62765391ba37a`.
