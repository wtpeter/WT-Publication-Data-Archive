#!/usr/bin/env python
'''
Module Name: torque.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20250419
Last Modified Date: 20250421

Description: XC torque calculation.
'''

import numpy as np
from pyscf import gto, dft, lib

def eval_B(mol, dm, coords, xc='PBE', spin_samples=110):
    mf = mol.GKS(xc=xc)
    mf.collinear = 'mcol'
    mf.spin_samples = spin_samples
    mf._numint.collinear_thrd = None
    ni = mf._numint
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    rho = dft.numint2c.eval_rho(mol, ao, dm, xctype='GGA', with_lapl=False, hermi=1)
    eval_xc = ni.mcfun_eval_xc_adapter('B88')
    xc = eval_xc('B88,', rho, deriv=2, xctype='GGA')
    exc, vxc, fxc = xc

    aos = dft.numint.eval_ao(mol, coords, deriv=2)
    ao_0 = aos[0] # (ng, nao)
    ao_1 = aos[1:4] # (3, ng, nao)
    ao_2 = aos[[[4, 5, 6], [5, 7, 8], [6, 8, 9]]] # (3, 3, ng, nao)

    nao = mol.nao_nr()
    dmaa = dm[:nao, :nao]
    dmab = dm[:nao, nao:]
    dmba = dm[nao:, :nao]
    dmbb = dm[nao:, nao:]
    dms = np.array([dmaa+dmbb, dmab+dmba, 1j*dmab-1j*dmba, dmaa-dmbb])

    rho_2 = lib.einsum('xyiu,iv,mvu->mxyi', ao_2, ao_0, dms)
    rho_2 += lib.einsum('xiu,yiv,mvu->mxyi', ao_1, ao_1, dms)
    rho_2 += lib.einsum('yiu,xiv,mvu->mxyi', ao_1, ao_1, dms)
    rho_2 += lib.einsum('iu,xyiv,mvu->mxyi', ao_0, ao_2, dms)

    rho_2 = 0.5 * (rho_2 + rho_2.conj()).real # (4, 3, 3, ng)

    B = vxc[:, 0, :]
    B -= lib.einsum('mnxi,nxi->mi', fxc[:, 0, :, 1:4, :], rho[:, 1:4, :])
    B -= lib.einsum('mxnyi,nxyi->mi', fxc[:, 1:4, :, 1:4, :], rho_2)

    return - B[1:4].T


def eval_m(mol, dm, coords):
    ao = dft.numint.eval_ao(mol, coords, deriv=0)
    rho = dft.numint2c.eval_rho(mol, ao, dm, xctype='LDA', with_lapl=False, hermi=1)
    return rho[1:4].T


def eval_torque(mol, dm, coords, xc='PBE', spin_samples=110):
    '''
    torque defined as vec(m) x vec(B)
    '''
    B = eval_B(mol, dm, coords, xc=xc, spin_samples=spin_samples).T
    M = eval_m(mol, dm, coords).T
    torquex = M[1] * B[2] - M[2] * B[1]
    torquey = M[2] * B[0] - M[0] * B[2]
    torquez = M[0] * B[1] - M[1] * B[0]
    return np.array([torquex, torquey, torquez]).T
