import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.sftda.tools_td import transition_analyze, spin_square, extract_state

r_list = np.concatenate((np.linspace(0.4, 3.0, 53), np.linspace(3.2, 6.0, 15)))
bas = '6-311g*'

s2_list = []
td_list = []
for r in r_list:
    mol = gto.M(atom=f'H 0.0 0.0 0.0; F 0.0 0.0 {r}',
                basis=bas, charge=0, spin=2, symmetry='C2v')
    mf = mol.UKS(xc='TPSS')
    mf.irrep_nelec={'A1':(4,2),'B1':(1,1),'B2':(1,1)}
    mf.kernel()

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

    target = extract_state(mf, mftd1, tdtype='TDDFT', Smin=0.0, Smax=0.9)[0]

    print(transition_analyze(mf, mftd1, mftd1.e[target], mftd1.xy[target], tdtype='TDDFT'))
    td_list.append(mftd1.e[target] + mf.e_tot)
    s2_list.append(spin_square(mf, mftd1.xy[target], extype=1))
np.save('r.npy', r_list)
np.save(f'TPSS_e.npy', td_list)
np.save(f'TPSS_s2.npy', s2_list)
print('-----------------------')