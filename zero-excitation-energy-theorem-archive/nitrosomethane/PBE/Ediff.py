import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.grad import tduks_sf # this import is necessary.
try:
    import mcfun # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None
from pyscf.data.nist import BOHR
import os
from pyscf.sftda.tools_td import transition_analyze

basis = 'cc-pvtz'
fun = 'PBE'

target_list = [2, 3]
for target in target_list:
    geom = np.load(f'{target}/geom.npy')[-1]
    atom = '''
    C                  1.10383100   -0.13217700    0.00000000
    H                  1.88053300    0.63390800   -0.00001900
    H                  1.17943200   -0.76563300   -0.89021600
    H                  1.17944600   -0.76559900    0.89023900
    N                 -0.23969200    0.53942400    0.00000000
    O                 -1.14806900   -0.26069700    0.00000000
    '''
    mol_a = gto.M(atom=atom, charge=0, spin=2, basis=basis)
    atom_b = [[mol_a.atom_symbol(i), geom[i]] for i in range(len(geom))]
    mol = gto.M(atom=atom_b, charge=0, spin=2, basis=basis, unit='Bohr')

    mf = mol.UKS(xc=fun)
    mf.kernel()
    e1 = mf.e_tot

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

    Casida0 = np.load(f'{target}/Casida.npy')
    if np.max(np.abs(Casida0 - Casida_matrix)) > 1e-6:
        raise ValueError(f'Casida matrix not equal to {target}/Casida.npy')

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
    mftd1.collinear_samples = 200
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]

    print(transition_analyze(mf, mftd1, mftd1.e[0], mftd1.xy[0], tdtype='TDDFT'))
    print(transition_analyze(mf, mftd1, mftd1.e[1], mftd1.xy[1], tdtype='TDDFT'))
    print(transition_analyze(mf, mftd1, mftd1.e[2], mftd1.xy[2], tdtype='TDDFT'))
    print(transition_analyze(mf, mftd1, mftd1.e[3], mftd1.xy[3], tdtype='TDDFT'))

    np.save(f'{target}/ExE', mftd1.e)