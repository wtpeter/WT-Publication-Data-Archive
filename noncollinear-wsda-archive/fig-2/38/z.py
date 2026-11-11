#!/usr/bin/env python

from pyscf import gto, dft
import numpy as np
from scipy.linalg import block_diag, eigh
import sys
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT')
sys.path.append('/public1/home/scg0213/Jobs_WT/Tools')
sys.path.append('/home/peter/Research/Tools')
sys.path.append('/home/peter/Research/NCNLDFT/code')
from uks_nl_new import UKSNL
from gks_nl_new import GKSMCNL
from rotate import rotate_spin_dm, rotate_spatial_dm

s = 38

udm = np.load(r'../udmi.npy')

mol = gto.M(atom='Na 0.0 0.0 0.0', charge=0, spin=1, basis='cc-pvqz', verbose=4)

gdm = block_diag(udm[0], udm[1])
gks = GKSMCNL(mol, xc='', epsilon=1e-11)
gks.conv_tol = 1e-8
gks.grids.atom_grid = (15, 86)
gks.spin_samples = s
gks.max_memory = 300000
ge = gks.kernel(gdm)
np.save('GMO', gks.mo_energy)
np.save('GET', np.array([ge]))
np.save('GDM', gks.make_rdm1())