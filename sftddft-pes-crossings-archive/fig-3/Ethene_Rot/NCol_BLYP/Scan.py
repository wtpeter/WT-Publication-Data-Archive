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

    a, b = get_ab_sf(mf, collinear_samples=50)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b,n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b,n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a,n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a,n_occ_b*n_virt_a), order='C')

    Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                            [-B_baab_2d,-A_baba_2d]])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

    idxt = eigenvals.real.argsort()
    eigenvals = eigenvals[idxt].real
    eigenvecs = eigenvecs[:, idxt]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

    sfd_eigenvals = eigenvals[norms>0]
    sfd_eigenvecs = eigenvecs[:, norms>0].transpose(1,0)

    def norm_xy(z):
        x = z[:n_occ_a*n_virt_b].reshape(n_occ_a,n_virt_b)
        y = z[n_occ_a*n_virt_b:].reshape(n_occ_b,n_virt_a)
        norm = np.linalg.norm(x)**2 - np.linalg.norm(y)**2
        norm = np.sqrt(1./norm)
        return ((0,x*norm), (y*norm,0))

    mftd1 = sftda.TDA_SF(mf)
    mftd1.nstates = len(sfd_eigenvals)
    mftd1.extype = 1
    mftd1.collinear_samples = 50
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]

    S = extract_state(mf, mftd1, Smin=0.0, Smax=0.7)
    target1 = S[0]
    target2 = S[1]
    target3 = S[2]

    eref_list[i] = gse
    e1_list[i] = gse + mftd1.e[target1]
    e2_list[i] = gse + mftd1.e[target2]
    e3_list[i] = gse + mftd1.e[target3]

np.save('Ref', eref_list)
np.save('S0', e1_list)
np.save('S1', e2_list)
np.save('S2', e3_list)
