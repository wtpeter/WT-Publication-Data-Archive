#!/usr/bin/env python
'''
Module Name: nl_utils.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240704
Last Modified Date: 20240711

Description: utilities of nonlocal WSDA funcional (parallel)
'''

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pyscf import gto, dft
from nl_utils_serial import search_sp, eval_FGHI_sp, eval_evx_sp, eval_evfx_sp
from mcfun.eval_xc import _make_sph_samples

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
    gpt = min(2.6e4*max_memory/len(coords)/nproc, len(weights)/nproc+10)
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

# def eval_mc_evx_sp(rhoas, rhobs, coords, weights, smalls, p1, nproc2, gpt):
#     rhoa, rxa, Fa, Ga, Ha, Ia = rhoas
#     rhob, rxb, Fb, Gb, Hb, Ib = rhobs
#     smalla, smallb = smalls
#     gpt = p1[1] - p1[0]
#     with ProcessPoolExecutor(max_workers=nproc2) as ex:
#         futures = []
#         for p2 in _prange(0, len(coords), gpt):
#             f = ex.submit(eval_evfx_sp, rhoa, rxa, Fa, Ga, Ha, Ia,
#                           coords, weights, p1, p2, smalla)
#             futures.append(f)
#         res = [f.result() for f in futures]
#     exa, vxa, fxa = np.concatenate(res, axis=2)
#     with ProcessPoolExecutor(max_workers=nproc2) as ex:
#         futures = []
#         for p2 in _prange(0, len(coords), gpt):
#             f = ex.submit(eval_evfx_sp, rhob, rxb, Fb, Gb, Hb, Ib,
#                           coords, weights, p1, p2, smallb)
#             futures.append(f)
#         res = [f.result() for f in futures]
#     exb, vxb, fxb = np.concatenate(res, axis=2)
#     Ecol = (exa + exb) @ weights
#     vcola, vcolb = vxa @ weights, vxb @ weights
#     E10 = 0.5 * (vcola + vcolb)
#     E10 += 0.25 * (fxa - fxb) @ ((rhoa - rhob) * weights)
#     E01 = vcola - vcolb
#     E01 += 0.25 * (fxa + fxb) @ ((rhoa - rhob) * weights)
#     return np.stack([Ecol, vcola, vcolb, E10, E01])

# def eval_mc_evx(rhoas, rhobs, coords, weights, smalls,
#                 max_memory, nproc, nproc1, gpt):
#     rhoa, rhob = rhoas[0], rhobs[0]
#     nproc2 = nproc // nproc1
#     with ProcessPoolExecutor(max_workers=nproc1) as ex:
#         futures = []
#         for p1 in _prange(0, len(coords), gpt):
#             f = ex.submit(eval_mc_evx_sp, rhoas, rhobs, coords,
#                           weights, smalls, p1, nproc2, gpt)
#             futures.append(f)
#         res = [f.result() for f in futures]
#     Ecol, vcola, vcolb, E10, E01 = np.concatenate(res, axis=1)
#     Eeff = (Ecol @ weights) + 0.5 * (weights * (rhoa - rhob)) @ (vcola - vcolb)
#     return Eeff, E10, E01

def eval_mc_evx(rhoas, rhobs, coords, weights, smalls, max_memory, nproc):
    rhoa, rxa, Fa, Ga, Ha, Ia = rhoas
    rhob, rxb, Fb, Gb, Hb, Ib = rhobs
    smalla, smallb = smalls

    gpt2 = np.sqrt(4.3e4*max_memory/len(coords)/nproc)
    gpt2 = int(min(gpt2, len(coords)/nproc+1))
    gpt1 = int(4.3e4*max_memory/len(coords)/gpt2/nproc)
    gpt1 = min(gpt1, len(coords))
    
    Eeff = np.zeros((len(coords), ))
    E10 = np.zeros((len(coords), ))
    E01 = np.zeros((len(coords), ))
    for p1 in _prange(0, len(coords), gpt1):
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = []
            for p2 in _prange(0, len(coords), gpt2):
                f = ex.submit(eval_evfx_sp, rhoa, rxa, Fa, Ga, Ha, Ia,
                            coords, weights, p1, p2, smalla)
                futures.append(f)
            res = [f.result() for f in futures]
        exa, vxa, fxa = np.concatenate(res, axis=2)
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = []
            for p2 in _prange(0, len(coords), gpt2):
                f = ex.submit(eval_evfx_sp, rhob, rxb, Fb, Gb, Hb, Ib,
                            coords, weights, p1, p2, smallb)
                futures.append(f)
            res = [f.result() for f in futures]
        exb, vxb, fxb = np.concatenate(res, axis=2)
        vcola, vcolb = vxa @ weights, vxb @ weights
        Eeff[p1[0]: p1[1]] = (
            (exa + exb) @ weights +
            0.5 * (rhoa[p1[0]: p1[1]] - rhob[p1[0]: p1[1]]) * (vcola - vcolb)
        )
        E10[p1[0]: p1[1]] = (
            0.5 * (vcola + vcolb) + 
            0.25 * (fxa - fxb) @ ((rhoa - rhob) * weights)
        )
        E01[p1[0]: p1[1]] = (
            vcola - vcolb + 
            0.25 * (fxa + fxb) @ ((rhoa - rhob) * weights)
        )
    Eeff = Eeff @ weights
    return Eeff, E10, E01

def numint_nl_uks(mol, grids, dm, epsilon, max_memory, nproc):
    '''
    Calculates WSDA exchange energy and potential matrix.

    Args:
        mol: an instance of class `Mole`
        grids: an instance of class `Grids`
        dm (np.array): UKS density matrix (2, nao, nao)
        max_memory (float): maximum size of cache to use (in MB)
        nproc (int): number of parallel processes

    Returns:
        nelec, exsum, vxmat
        nelec (tuple): number of (a, b) electrons calculated by numerical integration
        exsum (float): WSDA exchange energy
        vxmat (np.array): WSDA exchange potential matrix (2, nao, nao)
    '''
    dma, dmb = dm
    coords = grids.coords
    weights = grids.weights
    ao = dft.numint.eval_ao(mol, coords)
    rhoa = dft.numint.eval_rho(mol, ao, dma)
    rhob = dft.numint.eval_rho(mol, ao, dmb)
    na = (mol.intor('int1e_ovlp') @ dma).trace()
    nb = (mol.intor('int1e_ovlp') @ dmb).trace()
    nelec = ((weights * rhoa).sum(), (weights *rhob).sum())

    if na <= (1 + 1e-4):
        Exa, vxa = eval_evx(rhoa, None, None, None, coords, weights, True,
                            max_memory, nproc)
    else:
        rxa = search(rhoa, coords, weights, epsilon=epsilon,
                     max_memory=max_memory, nproc=nproc)
        Fa, Ga = eval_FGHI(rhoa, rxa, coords, weights, with_HI=False,
                           max_memory=max_memory, nproc=nproc)
        Exa, vxa = eval_evx(rhoa, rxa, Fa, Ga, coords, weights,
                            max_memory=max_memory, nproc=nproc)
    if nb <= (1 + 1e-4):
        Exb, vxb = eval_evx(rhob, None, None, None, coords, weights, True,
                            max_memory=max_memory, nproc=nproc)
    else:
        rxb = search(rhob, coords, weights, epsilon=epsilon,
                     max_memory=max_memory, nproc=nproc)
        Fb, Gb = eval_FGHI(rhob, rxb, coords, weights, with_HI=False,
                           max_memory=max_memory, nproc=nproc)
        Exb, vxb = eval_evx(rhob, rxb, Fb, Gb, coords, weights,
                            max_memory=max_memory, nproc=nproc)
    vmata = dft.numint.eval_mat(mol, ao, weights, rhoa, vxa)
    vmatb = dft.numint.eval_mat(mol, ao, weights, rhob, vxb)
    return nelec, Exa+Exb, np.array([vmata, vmatb])

def numint2c_nl_gks(mol, grids, dm, epsilon, max_memory,
                    nproc, spin_samples, hermi):
    coords = grids.coords
    weights = grids.weights
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

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc

    Ex = 0
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
        Eeff, Eeff10, Eeff01 = eval_mc_evx(rhoas, rhobs, coords, weights,
                                           (smalla, smallb), 
                                           max_memory, nproc)
        Ex += Eeff * aw[i]
        E10 += Eeff10 * aw[i]
        E01[0] += Eeff01 * aw[i] * ac[i][0]
        E01[1] += Eeff01 * aw[i] * ac[i][1]
        E01[2] += Eeff01 * aw[i] * ac[i][2]
        
    vx = np.array([[E10], [E01[0]], [E01[1]], [E01[2]]])
    vmat = dft.numint2c._mcol_lda_vxc_mat(mol, ao, weights, None, vx, None,
                                          shls_slice, ao_loc, hermi)
    if hermi:
        vmat = vmat + vmat.conj().transpose()
    print()
    return nelec, Ex, vmat


if __name__=='__main__':
    mol = gto.M(atom='''
    Ne 0.0 0.0 0.0''',
    basis='6-31G', charge=0, spin=0)
    mf1 = mol.GKS(xc='HF,')
    mf1.grids.level = 0
    mf1.kernel()
    # coords = mf1.grids.coords
    # weights = mf1.grids.weights
    # ao = dft.numint.eval_ao(mol, coords)
    # rhoa = dft.numint.eval_rho(mol, ao, mf1.make_rdm1()[0])
    # rxa = search(rhoa, coords, weights, nproc=32)
    # Fa, Ga, Ha, Ia = eval_FGHI(rhoa, rxa, coords, weights, True, False)
    # eval_evfx_sp(rhoa, rxa, Fa, Ga, Ha, Ia, coords, weights, [0, 10], [0, 20], small=False)
    numint2c_nl_gks(mol, mf1.grids, mf1.make_rdm1(), 1e-12, 30000, 32, 4, 6, hermi=1)
