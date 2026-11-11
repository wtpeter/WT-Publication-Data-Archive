#!/usr/bin/env python
'''
Module Name: nl_utils_serial.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240702
Last Modified Date: 20240711

Description: utilities of nonlocal WSDA funcional (sequential)
'''

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist
from pyscf import gto, dft
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
mask_coef = 0.4

def _prange(start, end, step):
    '''
    Generates sagments.
    This function refers to Zhichen Pu.
    '''
    if start<end:
        for i in range(start, end, step):
            yield i, min(i + step, end)

def zero_mask(r12, weights1):
    '''
    Masks the zero components in r12 with mask_coef * weight^{1/3}
    Args:
        r12: (n2, n1)
        weights1: (n1,)
    Returns:
        r12_n0: (n2, n1)
    '''
    # method1: reff
    zero_m = (r12==0.0)
    reff = mask_coef * np.power(np.abs(weights1), 1/3) # (n,)
    r12_n0 = r12.copy()
    r12_n0[zero_m] = np.where(zero_m, reff[:, np.newaxis].T, 0)[zero_m]
    r12_n0 += (r12_n0==0.0)

    # method2: negelect
    # r12_n0 = r12 + (r12==0.0) * 1e6
    return r12_n0

def calc_radius_sp(x1, rho, coords, weights, p1):
    '''
    Calculates the formula below:
    int dr2 rho(r2) e^{-[r12/x1]^{3/2}} - 1

    Args:
        x1: (n,)
        rho: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        p1 (list): p1[1]-p1[0] = n
    
    Returns:
        formula's value (n,)
    '''
    coords1 = coords[p1[0]: p1[1]]
    coords2, weights2 = coords, weights
    r12 = cdist(coords2, coords1) # (ng, n)
    x1 = x1.reshape(1, -1) # (1, n)
    exp = np.exp(-np.power(r12/x1, 1.5)) # (ng, n)
    res = (weights2 * rho) @ exp - 1 # (n,)
    return res

def radius_deriv_sp(x1, rho, coords, weights, p1):
    '''
    Args:
        x1: (n,)
        rho: (ng,)
    '''
    coords1 = coords[p1[0]: p1[1]]
    coords2, weights2 = coords, weights
    r12 = cdist(coords2, coords1)
    x = x1.reshape(1, -1) # (1, n)
    exp = np.power(r12, 1.5) * np.exp(-np.power(r12/x, 1.5)) # (ng, n)
    res = (1.5 * weights2 * rho) @ exp # (n,)
    res /= np.power(x1, 2.5)
    return res

def bisection_sp(rho, coords, weights, p1, epsilon=1e-6, max_iter=50):
    '''
    Args:
        rho: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        p1 (list): p1[1]-p1[0] = n
        epsilon (float)
        max_iter (int)
    
    Returns:
        (n,)
    '''
    n = p1[1] - p1[0]
    xl = np.ones((n,)) * 1.0e-14
    xr = np.ones((n,)) * 1.0e4
    epsilon = np.ones((n,)) * epsilon
    f_l = calc_radius_sp(xl, rho, coords, weights, p1)
    f_r = calc_radius_sp(xr, rho, coords, weights, p1)
    if np.any(f_l>0) or np.any(f_r<0):
        raise ValueError("Density is too weird.")
    for i in range(max_iter):
        xm = (xl + xr) * 0.5
        f_m = calc_radius_sp(xm, rho, coords, weights, p1)
        xl = np.where(f_m<=0, xm, xl)
        xr = np.where(f_m>0, xm, xr)
        cond1 = np.all(np.abs(xl-xr)<epsilon)
        cond2 = np.all(np.abs(f_m)<epsilon)
        if cond1 and cond2:
            return xm
    # return xm
    raise ValueError('The bisection method does not converage.')

def newton_sp(x10, rho, coords, weights, p1, epsilon=1e-6, max_iter=200):
    '''
    Args:
        x10: initial guess of x1 (n,)
        rho: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        p1 (list): p1[1] - p1[0] = n
    Returns:
        (n,)
    '''
    epsilon = np.ones(x10.shape) * epsilon
    for i in range(max_iter):
        f = calc_radius_sp(x10, rho, coords, weights, p1)
        f1 = radius_deriv_sp(x10, rho, coords, weights, p1)
        x11 = x10 - f / f1
        if np.any(x11<=0):
            raise ValueError('The initial guess for Newton\'s method is too bad!')
        cond1 = np.all(np.abs(x10-x11)<epsilon)
        cond2 = np.all(np.abs(f)<epsilon)
        if cond1 and cond2:
            return x11
        x10 = x11 * 1.0
    # return x11
    raise ValueError('Newton\'s method does not converage.')

def search_sp(rho, coords, weights, p1, epsilon=1e-7, max_iter=1000):
    '''
    Serial process for searching rx.
    '''
    # mkl.set_num_threads(1)

    # epsilon1 = 1e-2
    # if epsilon >= epsilon1:
    #     x1 = bisection_sp(rho, coords, weights, p1, epsilon, max_iter)
    #     return x1
    # x1 = bisection_sp(rho, coords, weights, p1, epsilon1, max_iter)
    # x2 = newton_sp(x1, rho, coords, weights, p1, epsilon, max_iter)
    x2 = bisection_sp(rho, coords, weights, p1, epsilon, max_iter)
    return x2

def pdv1(r12, rx1):
    r'''
    Calculates \pdv{e^{-(r12/rx1)^{3/2}}}{rx1}
    3/2 * r12^{3/2} / rx1^{5/2} e^{-(r12/rx1)^{3/2}}
    
    Args:
        r12 (np.array): shape (n2, n1)
        rx1 (np.array): shape (n1,)
    
    Returns:
        (np.array) pdv values shape (n2, n1)
    '''
    rx1 = rx1.reshape(1, -1) # (1, n1)
    exp = np.exp(-np.power(r12/rx1, 1.5)) # (n2, n1)
    res = 1.5 * r12**1.5 * exp / rx1**2.5
    return res

def pdv2(r12, rx1):
    r'''
    Calculates \pdv[2]{e^{-(r12/rx1)^{3/2}}}{rx1}
    (9/4 * r12^3/rx1^5 - 15/4 * r12^{3/2}/rx1^{7/2}) * e^{-(r12/rx1)^{3/2}}
    '''
    rx1 = rx1.reshape(1, -1)
    exp = np.exp(-np.power(r12/rx1, 1.5))
    fact = 2.25 * r12**3 / rx1**5
    fact -= 3.75 * r12**1.5 / rx1**3.5
    return fact * exp

def eval_FGHI_sp(rho, rx, coords, weights, p1, with_HI=True):
    '''
    Calculates F, G, H and I

    Args:
        rx: (ng,)
        rho: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        p1 (list): p1[1]-p1[0] = n
    Returns:
        (4, n) or (2, n) depending on the value of with_HI
    '''
    # mkl.set_num_threads(1)

    coords1, weights1 = coords[p1[0]: p1[1]], weights[p1[0]: p1[1]]
    rx1 = rx[p1[0]: p1[1]]
    coords2, weights2 = coords, weights
    r12 = cdist(coords2, coords1)
    r12_n0 = zero_mask(r12, weights1)
    p1 = pdv1(r12, rx1) # (ng, n)
    wrho = weights2 * rho
    F = wrho @ p1 # (n,)
    G = wrho @ (p1 / r12_n0) # (n,)
    if with_HI==True:
        p2 = pdv2(r12, rx1)
        H = wrho @ p2
        I = wrho @ (p2 / r12_n0)
        return np.vstack((F, G, H, I))
    return np.vstack((F, G))

def eval_evx_sp(rho, rx, F, G, coords, weights, p1, small=False):
    coords1, weights1 = coords[p1[0]: p1[1]], weights[p1[0]: p1[1]]
    coords2, weights2 = coords, weights
    rho2 = rho
    r12 = cdist(coords2, coords1)
    r12_n0 = zero_mask(r12, weights1)
    wrho2 = weights2 * rho2
    if small==True:
        ex = wrho2 @ (1/r12_n0)
        vx = -wrho2 @ (1/r12_n0)
        return np.vstack((ex, vx))
    rx1 = rx[p1[0]: p1[1]].reshape(1, -1)
    rx2 = rx.reshape(-1, 1)
    exp1 = np.exp(-np.power(r12/rx1, 1.5)) # (ng, n)
    exp2 = np.exp(-np.power(r12/rx2, 1.5)) # (ng, n)
    vx = -0.5 * wrho2 @ ((exp1+exp2) / r12_n0)
    vx -= -0.5 * (wrho2 * G / F) @ exp2
    ex = wrho2 @ (exp1 / r12_n0)
    return np.vstack((ex, vx))

def eval_evx(rho, rx, F, G, coords, weights, small=False,
             max_memory=40000, nproc=32):
    '''
    Evaluates vx.
    Args:
        rho, rx, F, G: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        small (bool): whether the electron number is less than 1
    
    Returns:
        (Ex, vx)
        Ex (float): exchange energy
        vx (np.array): exchange potential on grids
    '''
    gpt = min(2e4*max_memory/len(coords)/nproc, len(weights)/nproc+10)
    gpt = int(gpt)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = []
        for p1 in _prange(0, len(coords), gpt):
            f = ex.submit(eval_evx_sp, rho, rx, F, G, coords, weights, p1,
                          small)
            futures.append(f)
        res = [f.result() for f in futures]
    res = np.hstack(res)
    ex = res[0]
    vx = res[1]
    Ex = -0.5 * (weights * rho * ex).sum()
    return Ex, vx

def eval_evfx_sp(rho, rx, F, G, H, I, coords, weights, p1, p2, small=False):
    '''
    Evaluates WSDA exchange energy, potential and kernel.

    Args:
        rho, rx, F, G, H, I: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        p1 (list): p1[1] - p1[0] = n1
        p2 (list): p2[1] - p2[0] = n2
        small (bool): whether occupation number is smaller than 1

    Returns:
        ex(r1, r2) (n1, n2)
        vx(r1, r2) (n1, n2)
        fx(r1, r2) (n1, n2)
    '''
    coords1, weights1 = coords[p1[0]: p1[1]], weights[p1[0]: p1[1]]
    coords2, weights2 = coords[p2[0]: p2[1]], weights[p2[0]: p2[1]]
    coords3, weights3 = coords, weights
    r12 = cdist(coords1, coords2)
    r12_n0 = zero_mask(r12, weights2)
    rho1 = rho[p1[0]: p1[1], None]
    rho2 = rho[None, p2[0]: p2[1]]
    if small==True:
        return np.stack([-0.5*rho1*rho2/r12_n0, -rho2/r12_n0, -1/r12_n0])
    r13 = cdist(coords3, coords1)
    r13_n0 = zero_mask(r13, weights1)
    r13 = (r13.T)[:, None, :] # (n1, 1, ng)
    r13_n0 = (r13_n0.T)[:, None, :]
    r23 = cdist(coords2, coords3)
    r23_n0 = zero_mask(r23, weights3)
    r23 = r23[None, :, :] # (1, n2, n3)
    r23_n0 = r23_n0[None, :, :]

    rx1 = rx[p1[0]: p1[1], None, None] # (n1, 1, 1)
    rx2 = rx[None, p2[0]: p2[1], None]
    rx3 = rx[None, None, :] # (1, 1, ng)

    exp21 = np.exp(-np.power(r12/rx1[:, :, 0], 1.5))
    exp12 = np.exp(-np.power(r12/rx2[:, :, 0], 1.5))
    exp13 = np.exp(-np.power(r13/rx3, 1.5)) # (n1, 1, ng)
    exp23 = np.exp(-np.power(r23/rx3, 1.5)) # (1, n2, ng)

    p13 = pdv1(r13[:, 0, :], rx3[0, 0, :])[:, None, :] # (n1, 1, ng)
    p23 = pdv1(r23[0, :, :], rx3[0, 0, :])[None, :, :] # (1, n2, ng)
    p31 = (pdv1(r13[:, 0, :].T, rx1[:, 0, 0]).T)[:, None, :] # (n1, 1, ng)
    p32 = (pdv1(r23[0, :, :].T, rx2[0, :, 0]).T)[None, :, :] # (1, n2, ng)

    tmpng = weights * rho

    tmp1 = -(exp23*p13+exp13*p23-exp13*exp23*H[None,None,:]/F[None,None,:])
    tmp1 *= G[None, None, :] / F[None, None, :]
    tmp1 += p13 / r13_n0 * exp23 + p23 / r23_n0 * exp13
    tmp1 -= I[None, None, :] * exp13 * exp23 / F[None, None, :]

    f = -(exp12 + exp21) / r12_n0
    f += ((p31 / r13_n0) @ tmpng) * exp21 / F[p1[0]: p1[1], None]
    f += ((p32 / r23_n0) @ tmpng) * exp12 / F[None, p2[0]: p2[1]]
    f += tmp1 @ (tmpng / F)

    e = -0.5 * rho1 * rho2 * exp21 / r12_n0
    v = -0.5 * rho2 * (exp21 + exp12) / r12_n0
    v -= -0.5 * rho2 * exp12 * G[None, p2[0]: p2[1]] / F[None, p2[0]: p2[1]]
    return np.stack([e, v, 0.5 * f])


if __name__=='__main__':
    from memory_profiler import memory_usage
    mol = gto.M(atom='Li 0.0 0.0 0.0', basis='cc-pvdz', charge=0, spin=1)
    mf = mol.UKS(xc='LDA, VWN')
    mf.grids.level = 3
    mf.kernel()
    dm = mf.make_rdm1()
    coords = mf.grids.coords
    weights = mf.grids.weights
    print(coords.shape)
    ao = dft.numint.eval_ao(mol, coords)
    rhoa = dft.numint.eval_rho(mol, ao, dm[0])
    rhob = dft.numint.eval_rho(mol, ao, dm[1])
    
    # print(rhoa @ weights)
    # print(rhob @ weights)
    # print((dm[0] @ mol.intor('int1e_ovlp')).trace())
    # print((dm[1] @ mol.intor('int1e_ovlp')).trace())
    # search_sp(rhoa, coords, weights, [0,1000])
    # print(1)
    # rxa = search(rhoa, coords, weights, gpt=200)
    # print(rxa)
    # print(np.max(calc_radius_sp(rxa, rhoa, coords, weights, [0, len(coords)])))
    # rxb = bisection_sp(rhob, coords, weights, [0, len(coords)])
    # print(rxb)
    