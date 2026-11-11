#!/usr/bin/env python

from pyscf import gto, dft
import numpy as np
from scipy.linalg import block_diag
import sys
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT/src')
sys.path.append('/home/peter/Research/NCNLDFT/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/Tools')
sys.path.append('/home/peter/Research/Tools')
from gks_nl_new import GKSMCNL_nscf
from rotate import rotate_spin_dm

vec = (1, 0, 0)
vec = (0, 1, 0)
bas = 'cc-pvqz'
l = 4.0
a = l / np.sqrt(3)
atom=f'''
Li 0.0 {a:.10f} 0.0;
Li {-np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
Li {np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
'''
mol = gto.M(atom=atom, charge=0, spin=3, basis=bas, verbose=4)

gwsda = GKSMCNL_nscf(mol, xc='', epsilon=1e-11)
gwsda.spin_samples = 110
gwsda.grids.prune = None
gwsda.grids.atom_grid = (15, 86)
gwsda.max_memory = 300000

theta_list = np.linspace(0, 180, 13)
energy_list = []
for theta in theta_list:
    ang = np.deg2rad(theta)
    dm0 = np.load('../GHF.npy')
    dm1 = rotate_spin_dm(dm0, vec, ang)
    wsdae = gwsda.energy_tot(dm1)
    print('theta:', theta, 'energy:', wsdae)
    energy_list.append(wsdae)

    np.save('energy', np.array(energy_list))
