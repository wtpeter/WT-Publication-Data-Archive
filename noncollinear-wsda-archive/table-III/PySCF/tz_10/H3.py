#!/usr/bin/env python

from pyscf import gto, dft
import numpy as np
from scipy.linalg import block_diag
import sys
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT/src')
sys.path.append('/home/peter/Research/NCNLDFT/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/Tools')
sys.path.append('/home/peter/Research/Tools')
from uks_nl_new import UKSNL
from gks_nl_new import GKSMCNL
from rotate import rotate_total_dm

bas = 'cc-pvtz'
molH = gto.M(atom='H 0,0 0.0 0.0', charge=0, spin=1, basis=bas)
mfH = molH.UKS(xc='B88,')
mfH.kernel()
dmH = mfH.make_rdm1()
dmH = block_diag(dmH[0], dmH[1])

dmp1 = rotate_total_dm(molH, dmH, (1, 0, 0), -np.pi/2)
dmp2 = rotate_total_dm(molH, dmp1, (0, 0, 1), 2*np.pi/3)
dmp3 = rotate_total_dm(molH, dmp2, (0, 0, 1), 2*np.pi/3)
dmps = [dmp1, dmp2, dmp3]

l = 1.0
a = l / np.sqrt(3)
atom=f'''
H 0.0 {a:.10f} 0.0;
H {-np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
H {np.sqrt(3)*a/2:.10f} {-0.5*a:.10f} 0.0;
'''
mol = gto.M(atom=atom, charge=0, spin=3, basis=bas)
dmaa, dmab, dmba, dmbb = np.zeros((4, mol.nao_nr(), mol.nao_nr()), dtype=complex)
for i in range(3):
    nao = dmps[i].shape[-1]//2
    dmaa[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, :nao]
    dmab[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, nao:]
    dmba[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, :nao]
    dmbb[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, nao:]
dmi = np.block([[dmaa, dmab], [dmba, dmbb]])

uhf = mol.UKS(xc='HF,')
print('UHF', end=' ')
print(uhf.kernel())
np.save('UHF', uhf.make_rdm1())

ghf = mol.GKS(xc='HF,')
print('GHF', end=' ')
print(ghf.kernel(dmi))
np.save('GHF', ghf.make_rdm1())

uslater = mol.UKS(xc='Slater,')
print('USlater', end=' ')
uslater.grids.atom_grid = (15, 86)
uslater.grids.prune = None
print(uslater.kernel(np.load('USlater.npy')))
np.save('USlater', uslater.make_rdm1())

gslater = mol.GKS(xc='Slater,')
print('GSlater', end=' ')
gslater.collinear = 'mcol'
gslater.grids.atom_grid = (15, 86)
gslater.grids.prune = None
print(gslater.kernel(np.load('GSlater.npy')))
np.save('GSlater', gslater.make_rdm1())

ub88 = mol.UKS(xc='B88,')
ub88.grids.atom_grid = (15, 86)
ub88.grids.prune = None
print('UB88', end=' ')
print(ub88.kernel(np.load('UB88.npy')))
np.save('UB88', ub88.make_rdm1())

gb88 = mol.GKS(xc='B88,')
gb88.collinear = 'mcol'
gb88.grids.atom_grid = (15, 86)
gb88.grids.prune = None
gb88.spin_samples = 110
gb88._numint.collinear_thrd = None
gb88.conv_tol = 1e-8
print('GB88', end=' ')
print(gb88.kernel(np.load('GB88.npy')))
np.save('GB88', gb88.make_rdm1())

upbe = mol.UKS(xc='PBE,')
upbe.grids.atom_grid = (15, 86)
upbe.grids.prune = None
print('UPBE', end=' ')
print(upbe.kernel(np.load('UPBE.npy')))
np.save('UPBE', upbe.make_rdm1())

gpbe = mol.GKS(xc='PBE,')
gpbe.collinear = 'mcol'
gpbe._numint.collinear_thrd = None
gpbe.grids.prune = None
gpbe.grids.atom_grid = (15, 86)
gpbe.spin_samples = 110
gpbe.conv_tol = 1e-8
print('GPBE', end=' ')
print(gpbe.kernel(np.load('GPBE.npy')))
np.save('GPBE', gpbe.make_rdm1())

uwsda = UKSNL(mol, xc='', epsilon=1e-11)
uwsda.max_memory = 300000
uwsda.grids.prune = None
uwsda.grids.atom_grid = (15, 86)
uwsda.conv_tol = 1e-8
uwsda.grids.level = 0
print('UWSDA', end=' ')
print(uwsda.kernel(np.load('UWSDA.npy')))
np.save('UWSDA', uwsda.make_rdm1())

gwsda = GKSMCNL(mol, xc='', epsilon=1e-11)
gwsda.spin_samples = 110
gwsda.grids.prune = None
gwsda.grids.atom_grid = (15, 86)
gwsda.max_memory = 300000
gwsda.conv_tol = 1e-8
gwsda.max_cycle = 25
print('GWSDA', end=' ')
print(gwsda.kernel(np.load('GWSDA.npy')))
np.save('GWSDA', gwsda.make_rdm1())
