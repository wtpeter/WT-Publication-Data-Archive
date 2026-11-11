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
from uks_nl_new import UKSNL

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
             (15, 146),
             (30, 146),
             (30, 194),
             (45, 194)
]

ncol_list = []
col_list = []
delta_list = []

for grid in grid_list:
    gwsda = GKSMCNL_nscf(mol, xc='', epsilon=1e-11)
    gwsda.spin_samples = 110
    gwsda.grids.prune = None
    gwsda.grids.atom_grid = grid
    gwsda.max_memory = 300000
    gwsdae = gwsda.energy_tot(np.load('GHF.npy'))
    print('grid:', grid, 'NCOL:', gwsdae, end=' ')
    
    uwsda = UKSNL(mol, xc='', epsilon=1e-11)
    uwsda.grids.prune = None
    uwsda.grids.atom_grid = grid
    uwsda.max_memory = 300000
    uwsdae = uwsda.energy_tot(np.load('UHF.npy'))
    print('COL:', uwsdae, end=' ')

    print('Delta:', uwsdae - gwsdae)

    ncol_list.append(gwsdae)
    col_list.append(uwsdae)
    delta_list.append(uwsdae - gwsdae)

    np.save('NCOL', np.array(ncol_list))
    np.save('COL', np.array(col_list))
    np.save('Delta', np.array(delta_list))

