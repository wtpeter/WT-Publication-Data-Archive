import numpy as np
from pyscf import gto, sftda, scf, ao2mo
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.sftda.tools_td import transition_analyze
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/data/home/db100401/db100401/Research/Tools/SFTD')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

fun = 'BLYP'
bas = 'cc-pvdz'

omega_list = [0, 15, 30, 45, 60, 75, 80, 85, 88, 89, 90, 91, 92, 95, 100, 105, 120, 135, 150, 165, 180]
eref_list = np.zeros_like(omega_list, dtype=float)
e1_list = np.zeros_like(omega_list, dtype=float)
e2_list = np.zeros_like(omega_list, dtype=float)
e3_list = np.zeros_like(omega_list, dtype=float)
e4_list = np.zeros_like(omega_list, dtype=float)

for i in range(len(omega_list)):
    omega = omega_list[i]
    atom = f'''
            C
            C 1 1.35520
            H 1 1.07000 2 120.0
            H 1 1.07000 2 120.0 3 180.0
            H 2 1.07000 1 120.0 3 {0.0+omega}
            H 2 1.07000 1 120.0 5 180.0
            '''
    mol = gto.M(atom=atom, basis=bas, charge=0, spin=2)
    mf = mol.UKS(xc=fun)
    mf.kernel()
    gse = mf.e_tot

    na, nb = mf.mol.nelec
    oae1, oae2 = mf.mo_energy[0][na-2], mf.mo_energy[0][na-1]
    vbe1, vbe2 = mf.mo_energy[1][nb], mf.mo_energy[1][nb+1]
    e1, e2, e3, e4 = vbe1-oae1, vbe1-oae2, vbe2-oae1, vbe2-oae2
    e1, e2, e3, e4 = sorted([e1, e2, e3, e4])
    print(np.array([e1, e2, e3, e4]) * 27.2114)

    eref_list[i] = gse
    e1_list[i] = gse + e1
    e2_list[i] = gse + e2
    e3_list[i] = gse + e3
    e4_list[i] = gse + e4

    np.save('Ref.npy', eref_list)
    np.save('E1.npy', e1_list)
    np.save('E2.npy', e2_list)
    np.save('E3.npy', e3_list)
    np.save('E4.npy', e4_list)