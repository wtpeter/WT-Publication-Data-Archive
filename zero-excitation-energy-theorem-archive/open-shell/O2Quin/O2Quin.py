from pyscf import gto, dft
from pyscf.sftda.uhf_sf import get_ab_sf
import numpy as np
from scipy.linalg import eig

atom = '''
8  0.000000    0.000000    0.947370
8  0.000000    0.000000   -0.947370
'''
mol = gto.M(atom=atom, charge=0, spin=4, basis='cc-pVTZ')

uhf = mol.UHF()
uhf.kernel()
mo = uhf.stability(external=True)

import os

fun_list = ['SVWN', 'BLYP', 'PBE', 'TPSS', 'B3LYP', 'PBE0']
for fun in fun_list:
    print('----------')
    print(fun)

    os.mkdir(fun)

    mf = mol.UKS(xc=fun)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel(uhf.make_rdm1())

    a, b = get_ab_sf(mf)
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
    
    min_index = np.argmin(np.abs(sfd_eigenvals))

    print(sfd_eigenvals[min_index: min_index+5] * 27.21138602)
    np.save(f'{fun}/TDE', sfd_eigenvals * 27.21138602)
    np.save(f'{fun}/Casida', Casida_matrix)

    tda_eigenvals, tda_eigenvecs = np.linalg.eig(A_abab_2d)
    idxt = tda_eigenvals.real.argsort()
    tda_eigenvals = tda_eigenvals[idxt]
    print(tda_eigenvals[min_index: min_index+5] * 27.21138602)
    np.save(f'{fun}/TDAE', tda_eigenvals * 27.21138602)