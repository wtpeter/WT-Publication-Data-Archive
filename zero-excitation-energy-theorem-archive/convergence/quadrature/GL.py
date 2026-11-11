import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.grad import tduks_sf
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

fun = 'SVWN'

atom = '''
 O                 -0.00000600   -0.67763800    0.00000000
 C                 -0.00000600    0.52895800    0.00000000
 H                 -0.93784100    1.12375900    0.00000000
 H                  0.93792600    1.12359500    0.00000000
'''
mol = gto.M(atom=atom, basis='cc-pvtz', charge=0, spin=2)

sam_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]

e_list = []
g_list = []
mf = mol.UKS(xc=fun)
mf.max_memory = 100000
mf.grids.level = 0
mf.kernel()
for samp in sam_list:
    print(f'Calculating for sample size {samp}...')
    a, b = get_ab_sf(mf, collinear_samples=samp)
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
    eigenvals = eigenvals[idxt]
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
    mftd1.collinear_samples = samp
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]

    e0 = mftd1.e[0]
    e1 = mftd1.e[1]
    e2 = mftd1.e[2]
    e_list.append(np.array([e0, e1, e2]))

    mftdg1 = mftd1.Gradients()
    g0 = mftdg1.kernel(state=1)
    g1 = mftdg1.kernel(state=2)
    g2 = mftdg1.kernel(state=3)
    g_list.append(np.array([g0, g1, g2]))

    np.save('Energy', np.array(e_list))
    np.save('Gradient', np.array(g_list))
    print('---------------')
