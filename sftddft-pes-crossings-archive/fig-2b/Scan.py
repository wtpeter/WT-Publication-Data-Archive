import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

basis = '6-31+g**'
fun = 'B3LYP'

omega_list = np.linspace(-3, 3, 21)
tau_list = np.linspace(-3, 3, 21)
Omega, Tau = np.meshgrid(omega_list, tau_list)
e1_list = np.zeros_like(Omega)
e2_list = np.zeros_like(Omega)

for i in range(len(omega_list)):
    for j in range(len(tau_list)):
        omega = omega_list[i]
        tau = tau_list[j]
        atom = f'''
        C
        H 1 1.09556
        H 1 1.10338 2 112.30086
        C 1 1.40553 2 121.79226 3 -174.86582
        H 4 1.15234 1  {90.00955+omega} 2  {122.80031+tau}
        H 4 1.11282 1 110.46180 2 {-146.91019+tau}
        '''
        mol = gto.M(atom=atom, basis=basis, spin=2)

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

        S = extract_state(mf, mftd1)
        target1 = S[0]
        target2 = S[1]

        e1_list[i, j] = gse + sfd_eigenvals[target1]
        e2_list[i, j] = gse + sfd_eigenvals[target2]

np.save('omega.npy', omega_list)
np.save('tau.npy', tau_list)
np.save('e1.npy', e1_list)
np.save('e2.npy', e2_list)
