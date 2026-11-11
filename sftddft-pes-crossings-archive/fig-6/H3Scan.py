import numpy as np
from pyscf import gto, sftda
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

b_list = np.arange(-0.02, 0.02, 0.002) + 1.0
theta_list = np.arange(-2, 2, 0.2) + 60.0

b_grid, theta_grid = np.meshgrid(b_list, theta_list)

e1 = np.zeros_like(b_grid)
e2 = np.zeros_like(b_grid)

for i in range(b_grid.shape[0]):
    for j in range(b_grid.shape[1]):
        bond = b_grid[i, j]
        theta = theta_grid[i, j]
        atom = f'H; H 1 1.0; H 1 {bond} 2 {theta}'
        mol = gto.M(atom=atom, charge=0, spin=3, basis='6-31G*')
        mf = mol.UKS(xc='B3LYP')
        mf.kernel()

        a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=50)
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
        mftd1.collinear_samples = 50
        mftd1.e = sfd_eigenvals
        mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]
        D = extract_state(mf, mftd1, tdtype='TDDFT', Smin=0.5, Smax=1.0)
        e1[i, j] = mftd1.e[D[0]] + mf.e_tot
        e2[i, j] = mftd1.e[D[1]] + mf.e_tot

np.save('b_grid.npy', b_grid)
np.save('theta_grid.npy', theta_grid)
np.save('E1.npy', e1)
np.save('E2.npy', e2)