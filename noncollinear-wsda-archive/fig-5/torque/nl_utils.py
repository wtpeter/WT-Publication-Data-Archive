#!/usr/bin/env python
'''
Module Name: nl_utils.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240704
Last Modified Date: 20240814

Description: utilities of nonlocal WSDA funcional (parallel)
'''

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pyscf import gto, dft
from nl_utils_serial import search_sp, eval_FGHI_sp, eval_evx_sp, eval_evfx_sp
from mcfun.eval_xc import _make_sph_samples

def eval_B(mol, target_coords, grids, dm, epsilon, max_memory,
                nproc, spin_samples, hermi=1):
    '''
    evaluates the effective exchange potential at given coordinates
    '''
    coords = grids.coords
    weights = grids.weights
    coords = np.vstack((coords, target_coords))
    weights = np.hstack((weights, np.zeros((len(target_coords),))))

    ao = dft.numint.eval_ao(mol, coords)
    ac, aw = make_sph_samples(spin_samples)
    nao = dm.shape[-1] // 2
    dmaa = dm[:nao, :nao]
    dmab = dm[:nao, nao:]
    dmba = dm[nao:, :nao]
    dmbb = dm[nao:, nao:]
    dmx = dmab + dmba
    dmy = 1j * dmab - 1j * dmba
    dmz = dmaa - dmbb
    dmrho = dmaa + dmbb

    E10 = np.zeros_like(weights)
    E01 = np.zeros_like(coords.T)

    for i in range(len(ac)):
        print(i, end=' ')
        dmw = dmx * ac[i][0] + dmy * ac[i][1] + dmz * ac[i][2]
        dma, dmb = 0.5 * (dmrho + dmw), 0.5 * (dmrho - dmw)
        rhoa = dft.numint.eval_rho(mol, ao, dma)
        rhob = dft.numint.eval_rho(mol, ao, dmb)
        na = (mol.intor('int1e_ovlp') @ dma).trace()
        nb = (mol.intor('int1e_ovlp') @ dmb).trace()
        nelec = ((weights * rhoa).sum(), (weights *rhob).sum())
        smalla = na < (1 + 1e-4)
        smallb = nb < (1 + 1e-4)
        rxa = search(rhoa, coords, weights, epsilon, small=smalla,
                     max_memory=max_memory, nproc=nproc)
        Fa, Ga, Ha, Ia = eval_FGHI(rhoa, rxa, coords, weights, True,
                                   smalla, max_memory, nproc)
        rxb = search(rhob, coords, weights, epsilon, small=smallb,
                     max_memory=max_memory, nproc=nproc)
        Fb, Gb, Hb, Ib = eval_FGHI(rhob, rxb, coords, weights, True,
                                   smallb, max_memory, nproc)
        rhoas = (rhoa, rxa, Fa, Ga, Ha, Ia)
        rhobs = (rhob, rxb, Fb, Gb, Hb, Ib)
        Eeff10, Eeff01 = eval_mc_vx(rhoas, rhobs, len(target_coords), coords,
                                    weights, (smalla, smallb), max_memory,
                                    nproc)
        E10 += Eeff10 * aw[i]
        E01[0] += Eeff01 * aw[i] * ac[i][0]
        E01[1] += Eeff01 * aw[i] * ac[i][1]
        E01[2] += Eeff01 * aw[i] * ac[i][2]

    B = - E01
    return B


def _prange(start, end, step):
    '''
    Generates sagments.
    This function refers to Zhichen Pu.
    '''
    if start<end:
        for i in range(start, end, step):
            yield i, min(i + step, end)

def search(rho, coords, weights, epsilon=1e-7, max_iter=200, 
           small=False, max_memory=40000, nproc=32):
    '''
    Uses bisection and newton's methods to get characteristic radius rx.
    
    Args:
        rho (np.array): density on grids (ng,)
        coords (np.array): coordinates of grids (ng, 3)
        weights (np.array): weights of grids (ng,)
        epsilon (float): convergence threshold
        max_iter (int): max iteration numbers
        grids_per_task (int): number of grids in each processions
        nproc (int): number of processions
    
    Returns:
        rx (np.array): characteristic radius (ng,2)
    '''
    if small==True:
        return None
    # print(len(coords))
    gpt = min(3.2e4*max_memory/len(coords)/nproc, len(weights)/nproc+10)
    gpt = int(gpt)
    # print(gpt)

    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = []
        for p1 in _prange(0, len(coords), gpt):
            f = ex.submit(search_sp, rho, coords, weights, p1,
                          epsilon, max_iter)
            futures.append(f)
        res = [f.result() for f in futures]
    res = np.hstack(res)
    return res

def eval_FGHI(rho, rx, coords, weights, with_HI=True, small=False,
              max_memory=40000, nproc=32):
    '''
    Calculates F, G, H and I
    Args:
        rx: (ng,)
        rho: (ng,)
        coords: (ng, 3)
        weights: (ng,)
        with_HI (bool): whether to calculate H and I
    
    Returns:
        (F, G, H, I) or (F, G) with shape (ng,)
    '''
    if small==True:
        if with_HI==True:
            return None, None, None, None
        return None, None
    gpt = min(1.8e4*max_memory/len(coords)/nproc, len(weights)/nproc+10)
    gpt = int(gpt)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = []
        for p1 in _prange(0, len(coords), gpt):
            f = ex.submit(eval_FGHI_sp, rho, rx, coords, weights, p1, with_HI)
            futures.append(f)
        res = [f.result() for f in futures]
    res = np.hstack(res)
    resF = res[0]
    resG = res[1]
    if with_HI==True:
        resH = res[2]
        resI = res[3]
        return resF, resG, resH, resI
    return resF, resG


def make_sph_samples(spin_samples):
    '''
    Gives half of Lebedev angular grids.
    '''
    ac, aw = _make_sph_samples(spin_samples)
    uac = []
    uaw = []
    for i in range(len(ac)):
        x, y, z = ac[i]
        if z>0 or (z==0 and y>0) or (z==0 and y==0 and x>0):
            uac.append(ac[i])
            uaw.append(2 * aw[i])
    return np.array(uac), np.array(uaw) 


def eval_mc_vx(rhoas, rhobs, target_len, coords, weights, smalls, max_memory,
                nproc):
    '''
    calculate effective potential at target coordinates
    target coordinates are at the end of coords
    '''
    rhoa, rxa, Fa, Ga, Ha, Ia = rhoas
    rhob, rxb, Fb, Gb, Hb, Ib = rhobs
    smalla, smallb = smalls

    gpt2 = np.sqrt(4.3e4*max_memory/len(coords)/nproc)
    gpt2 = int(min(gpt2, len(coords)/nproc+1))
    gpt1 = int(4.3e4*max_memory/len(coords)/gpt2/nproc)
    gpt1 = min(gpt1, len(coords))

    E10 = np.zeros((len(coords), ))
    E01 = np.zeros((len(coords), ))

    for p1 in _prange(len(coords)-target_len, len(coords), gpt1):
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = []
            for p2 in _prange(0, len(coords)-target_len, gpt2):
                f = ex.submit(eval_evfx_sp, rhoa, rxa, Fa, Ga, Ha, Ia,
                            coords, weights, p1, p2, smalla)
                futures.append(f)
            res = [f.result() for f in futures]
        exa, vxa, fxa = np.concatenate(res, axis=2)
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = []
            for p2 in _prange(0, len(coords)-target_len, gpt2):
                f = ex.submit(eval_evfx_sp, rhob, rxb, Fb, Gb, Hb, Ib,
                            coords, weights, p1, p2, smallb)
                futures.append(f)
            res = [f.result() for f in futures]
        exb, vxb, fxb = np.concatenate(res, axis=2)
        vcola, vcolb = vxa @ weights[:len(coords)-target_len], vxb @ weights[:len(coords)-target_len]
        E10[p1[0]: p1[1]] = (
            0.5 * (vcola + vcolb) + 
            0.25 * (fxa - fxb) @ ((rhoa - rhob) * weights)[:len(coords)-target_len]
        )
        E01[p1[0]: p1[1]] = (
            vcola - vcolb + 
            0.25 * (fxa + fxb) @ ((rhoa - rhob) * weights)[:len(coords)-target_len]
        )
    return E10, E01

# def eval_mc_evx(rhoas, rhobs, coords, weights, smalls, max_memory, nproc):
#     '''
#     Evaluates multicollinear effective energy and potential.

#     Args:
#         rhos (tuple): (rho, rx, F, G, H, I) each one has shape (ng,)
#         coords: (ng, 3)
#         weights: (ng,)
#         smalls (tuple): (smalla, smallb) each one is a bool
    
#     Returns:
#         Eeff: effective energy
#         E10, E01: effective potential (ng,)
#     '''
#     rhoa, rxa, Fa, Ga, Ha, Ia = rhoas
#     rhob, rxb, Fb, Gb, Hb, Ib = rhobs
#     smalla, smallb = smalls

#     gpt2 = np.sqrt(4.3e4*max_memory/len(coords)/nproc)
#     gpt2 = int(min(gpt2, len(coords)/nproc+1))
#     gpt1 = int(4.3e4*max_memory/len(coords)/gpt2/nproc)
#     gpt1 = min(gpt1, len(coords))
    
#     Eeff = np.zeros((len(coords), ))
#     E10 = np.zeros((len(coords), ))
#     E01 = np.zeros((len(coords), ))
#     for p1 in _prange(0, len(coords), gpt1):
#         with ProcessPoolExecutor(max_workers=nproc) as ex:
#             futures = []
#             for p2 in _prange(0, len(coords), gpt2):
#                 f = ex.submit(eval_evfx_sp, rhoa, rxa, Fa, Ga, Ha, Ia,
#                             coords, weights, p1, p2, smalla)
#                 futures.append(f)
#             res = [f.result() for f in futures]
#         exa, vxa, fxa = np.concatenate(res, axis=2)
#         with ProcessPoolExecutor(max_workers=nproc) as ex:
#             futures = []
#             for p2 in _prange(0, len(coords), gpt2):
#                 f = ex.submit(eval_evfx_sp, rhob, rxb, Fb, Gb, Hb, Ib,
#                             coords, weights, p1, p2, smallb)
#                 futures.append(f)
#             res = [f.result() for f in futures]
#         exb, vxb, fxb = np.concatenate(res, axis=2)
#         vcola, vcolb = vxa @ weights, vxb @ weights
#         Eeff[p1[0]: p1[1]] = (
#             (exa + exb) @ weights +
#             0.5 * (rhoa[p1[0]: p1[1]] - rhob[p1[0]: p1[1]]) * (vcola - vcolb)
#         )
#         E10[p1[0]: p1[1]] = (
#             0.5 * (vcola + vcolb) + 
#             0.25 * (fxa - fxb) @ ((rhoa - rhob) * weights)
#         )
#         E01[p1[0]: p1[1]] = (
#             vcola - vcolb + 
#             0.25 * (fxa + fxb) @ ((rhoa - rhob) * weights)
#         )
#     Eeff = Eeff @ weights
#     return Eeff, E10, E01

