import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.grad import tduks_sf # this import is necessary.
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

basis = '6-31g(d,p)'
fun = 'B3LYP'

def f(R, mf, collinear_samples=20):
    mol = mf.mol
    mol.set_geom_(R.reshape(-1, 3), unit='Bohr')
    print(mol.tostring())
    mf = mol.UKS(xc=mf.xc)
    mf.kernel()
    gse = mf.e_tot

    td = mf.TDDFT_SF()
    td.extype = 1
    td.nstates = 5
    td.collinear_samples = 50
    td.positive_eig_threshold = -0.1
    td.kernel()

    S = extract_state(mf, td, Smin=0.0, Smax=0.7)
    target1 = S[1]
    T = extract_state(mf, td, Smin=1.9, Smax=2.6)
    target2 = T[0]

    tdg1 = td.Gradients()
    g1 = tdg1.kernel(state=target1+1)
    g2 = tdg1.kernel(state=target2+1)

    g1 = g1.flatten()
    g2 = g2.flatten()

    e1 = gse + td.e[target1]
    e2 = gse + td.e[target2]

    return e1, e2, g1, g2

def BFGS_Hessian(B0, s, y):
    B1 = B0 + np.outer(y, y) / (y.T @ s) - B0 @ np.outer(s, s) @ B0.T / (s.T @ B0 @ s)
    return B1

def SR1_Hessian(B0, s, y):
    B1 = B0 + np.outer(y - B0 @ s, y - B0 @ s) / (y - B0 @ s).T @ s
    return B1

def Bofill_Hessian(B0, s, y):
    B_SR1 = SR1_Hessian(B0, s, y)
    B_BFGS = BFGS_Hessian(B0, s, y)
    phi = ((-y + B0 @ s).T @ s) ** 2
    phi /= ((-y + B0 @ s).T @ (-y + B0 @ s)) * (s.T @ s)
    phi = np.sqrt(phi)
    B1 = phi * B_SR1 + (1 - phi) * B_BFGS
    B1 = 0.5 * (B1 + B1.T)
    return B1

def search(gmean, gdiff, PHP, alpha):
    e, v = np.linalg.eigh(PHP)
    idx = np.argsort(np.abs(e))[2:]
    e, v = e[idx], v[:, idx]
    H_inv = v @ np.diag(1/e) @ v.T
    s = - H_inv @ gmean

    s1 = - gdiff / 2 / alpha
    s = 0.25 * s + s1
    return s

def optimize(f, R0, mf, collinear_samples=50, max_iter=200):
    R = R0.copy()
    Hmean = np.eye(len(R))
    e1, e2, g1, g2 = f(R, mf, collinear_samples)
    DGV = g1 - g2
    alpha = np.linalg.norm(DGV)
    x = DGV / np.linalg.norm(DGV)
    y = 0.5 * (g1 + g2)
    y = y - (x.T @ y) * x
    y = y / np.linalg.norm(y)
    P = 1 - np.outer(x, x) - np.outer(y, y)
    gdiff = 2 * (e1 - e2) * x
    gmean = 0.5 * (g1 + g2)
    g = gdiff + P @ gmean

    for i in range(max_iter):
        s = search(gmean, gdiff, Hmean, alpha)

        verbose = (
            f"Iter: {i}, "
            f"Loss: {0.5*(e1+e2)+(e1-e2)**2/alpha:.6f}, "
            f"Emean: {0.5*(e1+e2):.4f}, "
            f"Ediff: {e2-e1:.2e}, "
            f"RMS_G: {np.sqrt(np.mean(g**2)):.2e}, "
            f"RMS_s: {np.sqrt(np.mean(s**2)):.2e}"
        )
        print(verbose)

        tol4 = np.sqrt(np.mean(s**2)) < 1e-3 / 0.53
        tol3 = np.max(s) < 1.5e-3 / 0.53
        tol2 = np.sqrt(np.mean(g**2)) < 2e-4 * 0.53
        tol1 = np.max(g) < 3e-4 * 0.53
        tol0 = np.abs(e2-e1) < 0.1 / 2625.5
        if tol4 and tol3 and tol2 and tol1 and tol0:
            print("Conical intersection search converged at iter", i)
            break

        R += s

        e1, e2, g1, g2 = f(R, mf, collinear_samples)
        DGV = g1 - g2
        alpha = np.linalg.norm(DGV)

        x_prev = x.copy()
        x = DGV / np.linalg.norm(DGV)
        y_prev = y.copy()
        y = (y_prev.T @ x) * x_prev - (x_prev.T @ x) * y_prev
        y /= np.linalg.norm(y)

        gmean_prev = gmean.copy()

        P = 1 - np.outer(x, x) - np.outer(y, y)
        gdiff = 2 * (e1 - e2) * x
        gmean = 0.5 * (g1 + g2)
        g = gdiff + P @ gmean
        Hmean = BFGS_Hessian(Hmean, s, gmean - gmean_prev)
    return R, x, y

atom = '''
 C                  0.05013000   -0.03161000   -2.64108000
 C                  1.28831000   -0.14882000   -2.08654000
 C                  1.44310000   -0.14952000   -0.62841000
 C                  0.37729000    0.05096000    0.10717000
 C                 -0.94509000    0.15282000   -0.43379000
 C                 -1.10593000    0.10024000   -1.82004000
 H                 -0.06150000   -0.02302000   -3.70996000
 H                  2.17160000   -0.20712000   -2.69740000
 H                  2.40055000   -0.30036000   -0.16347000
 H                 -2.08139000    0.23673000   -2.25298000
 O                 -1.90274000    0.29126000    0.48315000
 H                 -1.36651000   -0.41529000    1.51601000
 N                  0.42971000   -0.06540000    1.56496000
 O                  1.10365000    0.66563000    2.22293000
 O                 -0.47570000   -0.83549000    2.01663000
'''
mol = gto.M(atom=atom, basis=basis, charge=0, spin=2)
mf = mol.UKS(xc=fun)
R0 = mf.mol.atom_coords().flatten()

optimize(f, R0, mf, collinear_samples=50, max_iter=10)
