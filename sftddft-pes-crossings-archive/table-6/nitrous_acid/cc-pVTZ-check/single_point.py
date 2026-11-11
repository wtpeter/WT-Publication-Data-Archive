"""Recalculate the Table 6 vertical gap at an S4 geometry (2026 archive check)."""
import platform
import sys
from pathlib import Path

import h5py
import mcfun
import numpy as np
import pyscf
import scipy
from pyscf import dft, gto, lib, sftda
from pyscf.data.nist import HARTREE2EV
from pyscf.sftda.tools_td import spin_square
from pyscf.sftda.uhf_sf import get_ab_sf
from scipy.linalg import eig

geometry = Path(sys.argv[1])
basis = sys.argv[2]
print("XYZ input:", geometry.name, "\n" + geometry.read_text().strip())
print("Python", platform.python_version(), "PySCF", pyscf.__version__,
      "NumPy", np.__version__, "SciPy", scipy.__version__, "h5py", h5py.__version__,
      "Libxc", dft.libxc.__version__, "MCfun", mcfun.__version__)
print("PySCF-forge 6fcda45, MCfun 65a75f9. Threads:", lib.num_threads())
mol = gto.M(atom="\n".join(geometry.read_text().splitlines()[2:]),
            basis=basis, spin=2, charge=0, unit="Angstrom", max_memory=11000)
mf = mol.UKS(xc="B3LYP")
print("Basis:", basis, "XC:", mf.xc, "grid level:", mf.grids.level, "spin samples: 50")
mf.kernel()
if not mf.converged:
    raise RuntimeError("Triplet UKS reference did not converge")

# The historical notebook builds the full spin-flip Casida matrix in this order.
(a_up, a_down), (b_up, b_down) = get_ab_sf(mf, collinear_samples=50)
n_beta, nv_alpha = a_up.shape[:2]
n_alpha, nv_beta = a_down.shape[:2]
down = n_alpha * nv_beta
up = n_beta * nv_alpha
response = np.block([[a_down.reshape(down, down), b_down.reshape(down, up)],
                     [-b_up.reshape(up, down), -a_up.reshape(up, up)]])
print("Casida dimension:", response.shape[0], "positive-norm sector:", down)
roots, vectors = eig(response, check_finite=False)
# In the triplet reference the last two occupied alpha MOs are singly occupied.
# The DES puts both electrons in the upper frontier orbital: alpha(lower) -> beta(upper).
overlap = mf.mo_coeff[0][:, n_alpha-2:n_alpha].T @ mf.get_ovlp() @ mf.mo_coeff[1][:, n_beta:n_beta+2]
print("Overlap of the two frontier alpha and beta orbitals:\n", overlap)
target_occupied = n_alpha - 2
target_virtual = np.argmax(abs(overlap[1]))
print("DES orbital pair (1-based occupied alpha, virtual beta):", target_occupied+1, target_virtual+1)
print("root   E-Eref/eV       S^2      |X_DES|^2   main pair (occ,vir)  |X_max|^2")
states = []
positive_roots = 0
for k in np.argsort(roots.real):
    v = vectors[:, k]
    norm = np.vdot(v[:down], v[:down]).real - np.vdot(v[down:], v[down:]).real
    if norm <= 1.e-8 or abs(roots[k].imag) > 1.e-7:
        continue
    energy = roots[k].real * HARTREE2EV
    if states and energy > states[0][0] + 15:
        break
    x = (v[:down] / np.sqrt(norm)).reshape(n_alpha, nv_beta)
    y = (v[down:] / np.sqrt(norm)).reshape(n_beta, nv_alpha)
    s2 = spin_square(mf, ((0, x), (y, 0)), extype=1, tdtype="TDDFT").real
    weights = abs(x)**2
    weight = weights[target_occupied, target_virtual]
    i, j = np.unravel_index(weights.argmax(), weights.shape)
    print(f"{positive_roots:4d} {energy:12.7f} {s2:9.5f} {weight:12.7f}   ({i+1},{j+1}) {weights[i,j]:13.7f}")
    if 0 <= s2 <= 0.7:
        states.append((energy, weight, positive_roots))
    positive_roots += 1

ground = min(states)
doubly_excited = max(states, key=lambda state: state[1])
print(f"S0 root {ground[2]}, DES root {doubly_excited[2]}")
print(f"Vertical gap (DES - S0) = {doubly_excited[0]-ground[0]:.7f} eV")
print(f"DES partition |X|^2 = {doubly_excited[1]:.7f}")
