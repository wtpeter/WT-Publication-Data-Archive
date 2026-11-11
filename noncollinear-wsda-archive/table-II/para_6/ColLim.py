#!/usr/bin/env python

from pyscf import gto, scf, dft
import numpy as np
from scipy.linalg import block_diag
import sys
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/Tools')
sys.path.append('/home/peter/Research/Tools')
sys.path.append('/home/peter/Research/NCNLDFT/code/src')
from rotate import rotate_total_dm
from uks_nl_new import UKSNL
from gks_nl_new import GKSMCNL_nscf, GKSNCNL

bas = 'cc-pvqz'
mol1 = gto.M(atom='Li 0.0 0.0 0.0', charge=0, spin=1, basis=bas)
mf1 = mol1.UHF().run()
dm1 = mf1.make_rdm1()

theta = 0.0000000000
gdm1 = block_diag(dm1[0], dm1[1])
gdm2 = rotate_total_dm(mol1, gdm1, (0, 1, 0), theta)
dmps = [gdm1, gdm2]
dmaa, dmab, dmba, dmbb = np.zeros((4, 2*mol1.nao_nr(), 2*mol1.nao_nr()), dtype=complex)
for i in range(2):
    nao = mol1.nao_nr()
    dmaa[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, :nao]
    dmab[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, nao:]
    dmba[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, :nao]
    dmbb[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, nao:]
udmi = np.array([dmaa, dmbb]).real

r = 6.0
s = 2
mol2 = gto.M(atom=f'Li 0.0 0.0 0.0; Li 0.0 {r:.1f} 0.0', charge=0, spin=s, basis=bas)
mf2 = mol2.UHF()
mf2.max_cycle = 50
mf2.kernel(udmi)

udm = mf2.make_rdm1()
dm = block_diag(udm[0], udm[1])

np.save('dm.npy', dm)

print('Collinear: ', end='')
unl = UKSNL(mol2, xc='')
unl.grids.atom_grid = (15, 86)
unl.max_memory = 300000
print(unl.energy_tot(udm))

print('LocallyCollinear: ', end='')
gncnl = GKSNCNL(mol2, xc='')
gncnl.grids.atom_grid = (15, 86)
gncnl.max_memory = 300000
print(gncnl.energy_tot(dm))

print('MultiCollinear: ', end='')
gmcnl = GKSMCNL_nscf(mol2, xc='')
gmcnl.grids.atom_grid = (15, 86)
gmcnl.spin_samples = 110
gmcnl.max_memory = 300000
print(gmcnl.energy_tot(dm))