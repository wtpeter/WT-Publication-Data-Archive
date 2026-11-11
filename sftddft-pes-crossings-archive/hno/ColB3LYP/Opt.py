import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.grad import tduks_sf
import sys
sys.path.append('/data/home/db100401/db100401/Research/Tools/SFTD')
from tools import extract_state

basis = 'cc-pvtz'
fun = 'B3LYP'

def tdgrad(mol):
    print(mol.tostring())

    mf = mol.UKS(xc=fun)
    mf.kernel()
    gse = mf.e_tot

    a, b = get_ab_sf(mf, collinear_samples=-1)
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
    mftd1.collinear_samples = -1
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]

    S = extract_state(mf, mftd1)[0]
    T = extract_state(mf, mftd1, Smin=1.9, Smax=2.7)[0]

    e1 = mftd1.e[S] + gse
    e2 = mftd1.e[T] + gse

    mftdg1 = mftd1.Gradients()
    g1 = mftdg1.kernel(state=S+1)
    g2 = mftdg1.kernel(state=T+1)

    if e1 > e2:
        e1, e2 = e2, e1
        g1, g2 = g2, g1

    return e1, e2, g1, g2

atom = '''
O      0.63265879   -0.03758723    0.00000000
N     -0.53048232    0.10408252    0.00000000
H     -1.34789414   -0.42787980    0.00000000
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis=basis)

R_list = []
f_list = []
u_list = []
f_R_list = []
e1_list = []
e2_list = []
sigma = 3.5
alpha = 0.02
tol_step = 1e-6
tol_grad = 0.005

def Fmidium(R):
    coords = R.reshape((mol.natm, 3))
    atom_new = [[mol.atom_symbol(i), coords[i]] for i in range(mol.natm)]
    mol_new = gto.M(atom=atom_new, charge=0, spin=2, basis=basis, unit='Bohr')
    e1, e2, g1, g2 = tdgrad(mol_new)
    g = (e1 - e2)**2 / (e2 - e1 + alpha)
    f = 0.5 * (e1 + e2) + sigma * g

    g1 = g1.flatten()
    g2 = g2.flatten()
    f_R = 0.5 * (g1 + g2)
    u = ((e1 - e2)**2 + 2 * alpha * (e2 - e1)) / (e2 - e1 + alpha)**2
    u *= (g2 - g1)
    f_R += sigma * u
    f_R_list.append(f_R)
    f_list.append(f)
    u_list.append(u / np.linalg.norm(u))
    R_list.append(R)
    e1_list.append(e1)
    e2_list.append(e2)
    return f, f_R, u

def F(R):
    f, f_R, u = Fmidium(R)
    return f, f_R

def callback(R):
    f0 = f_list[-2]
    f = f_list[-1]
    f_R = f_R_list[-1]
    u = u_list[-1]
    tol1 = np.abs(f0 - f) < tol_step
    tol2 = (f_R @ u / sigma) < tol_grad
    tol3 = np.linalg.norm(f_R - f_R @ u * u) < tol_grad
    print(f'Delta F = {np.abs(f0 - f):.2e},  Delta E = {np.abs(e1_list[-1] - e2_list[-1]):.2e}.')
    np.save('R', R)
    if tol1 and tol2 and tol3:
        raise StopIteration(f"Converged at given sigma={sigma:.2f}.")


sigma = 3.5
R0 = mol.atom_coords().flatten()
from scipy.optimize import minimize
minimize(fun=F, x0=R0, method='BFGS', jac=True, tol=1e-6, callback=callback)
