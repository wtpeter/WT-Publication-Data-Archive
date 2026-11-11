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
from time import time

bas = 'cc-pvqz'
l = 4.0
a = l / np.sqrt(3)

atom=f'''
Li 0.0 {a:.10f} 0.0;
Li {-np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
Li {np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
'''

mol = gto.M(atom=atom, charge=0, spin=3, basis=bas)

grid_list = [(15, 86),
             (30, 146),
             (45, 194),
             (60, 230),
             (75, 302)
]

energy_list = []

for grid in grid_list:
    gwsda = GKSMCNL_nscf(mol, xc='', epsilon=1e-11)
    gwsda.spin_samples = 110
    gwsda.grids.prune = None
    gwsda.grids.atom_grid = grid
    gwsda.max_memory = 300000
    t0 = time()
    gwsdae = gwsda.energy_tot(np.load('GHF.npy'))
    t1 = time()
    print('grid:', grid, 'energy:', gwsdae, 'time:', t1-t0)
    energy_list.append(gwsdae)
