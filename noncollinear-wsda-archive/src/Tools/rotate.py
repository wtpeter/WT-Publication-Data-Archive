#!/usr/bin/env python
'''
Module Name: rotate.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240626
Last Modified Date: 20240627

Description: Unitary transformations of density matrix.
'''


import numpy as np
from pyscf import gto
from pyscf.symm.Dmatrix import Dmatrix
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag

# Pauli matrices
s0 = np.array([[1, 0], [0, 1]], dtype=complex)
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)

# Rotatation matrix in 3D Euclidean space with (n, theta) representation
def rm_3d(n, theta):
    nx, ny, nz = n / np.linalg.norm(n)
    c = np.cos(theta)
    s = np.sin(theta)
    r = np.zeros((3, 3))
    r[0, 0] = c + nx**2 * (1 - c)
    r[0, 1] = nx * ny * (1 - c) - nz * s
    r[0, 2] = nx * nz * (1 - c) + ny * s
    r[1, 0] = ny * nx * (1 - c) + nz * s
    r[1, 1] = c + ny**2 * (1 - c)
    r[1, 2] = ny * nz * (1 - c) - nx * s
    r[2, 0] = nz * nx * (1 - c) - ny * s
    r[2, 1] = nz * ny * (1 - c) + nx * s
    r[2, 2] = c + nz**2 * (1 - c)
    return r

# Rotation matrix in spin-1/2
def rm_spin(n, theta):
    nx, ny, nz = n / np.linalg.norm(n)
    m = nx * s1 + ny * s2 + nz * s3
    m *= -1j * np.sin(theta/2)
    return m + np.cos(theta/2) * s0

# Rotate density matrix in spin-1/2
def rotate_spin_dm(dm, n, theta):
    '''
    Rotates the density matrix in spin space.

    Args:
        dm (np.array): GKS density matrix (2*nao, 2*nao)
        n (np.array): rotation axis (3,)
        theta (float): rotation angle in radical system
    
    Return:
        density matrix after rotation (2*nao, 2*nao)
    '''
    nao = dm.shape[-1] // 2
    M = rm_spin(n, theta)
    dm_block = np.empty((2, 2), dtype=object)
    dm_block[0, 0] = dm[:nao, :nao]
    dm_block[0, 1] = dm[:nao, nao:]
    dm_block[1, 0] = dm[nao:, :nao]
    dm_block[1, 1] = dm[nao:, nao:]
    dm1_block = M @ dm_block @ np.conjugate(M.T)
    dm1 = np.block([[dm1_block[0, 0], dm1_block[0, 1]],
                    [dm1_block[1, 0], dm1_block[1, 1]]])
    return dm1

# Rotate spatial basis function
def rotate_spatial_dm(mol, dm, n, theta, gks=False):
    '''
    Rotates the density matrix in terms of spatial basis functions.

    Args:
        mol (pyscf.gto.mole): mole
        dm (np.array): RKS density matrix (nao, nao)
        n (np.array): rotation axis (3,)
        theta (float): rotation angle in radical system
        GKS (bool): if True, dm's shape is (2*nao, 2*nao)
    
    Return:
        new density matrix (nao, nao) or (2*nao, 2*nao)
    '''
    if gks == False:
        # r = Rotation.from_matrix(rm_3d(n, theta))
        n = np.array(n)
        r = Rotation.from_rotvec(n * theta)
        a, b, c = r.as_euler('ZYZ', degrees=False)
        aolist = mol.ao_labels()
        block = []
        num0 = -1
        shell0 = 0
        for i in aolist:
            num, atom, shell = i.split()
            if num != num0 or shell[:2]!=shell0:
                if shell[1] == 's':
                    block.append(Dmatrix(0, a, b, c, reorder_p=True))
                elif shell[1] == 'p':
                    block.append(Dmatrix(1, a, b, c, reorder_p=True))
                elif shell[1] == 'd':
                    block.append(Dmatrix(2, a, b, c, reorder_p=True))
                elif shell[1] == 'f':
                    block.append(Dmatrix(3, a, b, c, reorder_p=True))
                elif shell[1] == 'g':
                    block.append(Dmatrix(4, a, b, c, reorder_p=True))
            num0 = num
            shell0 = shell[:2]
        bigD = block_diag(*block)
        if dm.shape == bigD.shape:
            return bigD @ dm @ np.conjugate(bigD.T)
        else:
            raise ValueError("The shape of density matrix is wrong!")
    else:
        nao = dm.shape[-1] // 2
        rdmaa = rotate_spatial_dm(mol, dm[:nao, :nao], n, theta, gks=False)
        rdmab = rotate_spatial_dm(mol, dm[:nao, nao:], n, theta, gks=False)
        rdmba = rotate_spatial_dm(mol, dm[nao:, :nao], n, theta, gks=False)
        rdmbb = rotate_spatial_dm(mol, dm[nao:, nao:], n, theta, gks=False)
        return np.block([[rdmaa, rdmab], [rdmba, rdmbb]])

def rotate_total_dm(mol, dm, n, theta):
    dm1 = rotate_spatial_dm(mol, dm, n, theta, gks=True)
    dm2 = rotate_spin_dm(dm1, n, theta)
    dm3 = rotate_spin_dm(dm, n, theta)
    dm4 = rotate_spatial_dm(mol, dm3, n, theta, gks=True)
    if np.allclose(dm2, dm4):
        return dm2

if __name__=='__main__':
    mol = gto.M(atom='H 0.0 0.0 0.0; F 0.0 0.0 1.0', charge=0, spin=0, basis='def2tzvp')
    mf = mol.RKS(xc='LDA, VWN').run()
    dm = mf.make_rdm1()
    dm1 = rotate_spatial_dm(mol, dm, (0, 1, 0), 1.0)
    print(mf.energy_elec(dm1), mf.energy_elec(dm))
    # n = (0.3, 0.4, 0.5)
    # theta = 0.8
    # R = rm_3d(n, theta)
    # print(np.allclose(R.T @ R, np.identity(3)))
    # M = rm_spin(n, theta)
    # print(np.allclose(np.conjugate(M.T) @ M, np.identity(2)))
    # pm = np.empty(3, dtype=object)
    # pm[0] = s1
    # pm[1] = s2
    # pm[2] = s3
    # pm = pm.T
    # rpm2 = R @ pm
    # rpm1x = np.conjugate(M.T) @ s1 @ M
    # rpm1y = np.conjugate(M.T) @ s2 @ M
    # rpm1z = np.conjugate(M.T) @ s3 @ M
    # rpm1 = np.empty(3, dtype=object)
    # rpm1[0] = rpm1x
    # rpm1[1] = rpm1y
    # rpm1[2] = rpm1z
    # rpm1 = rpm1.T
    # r = rpm1 - rpm2
    # r = np.array(r.tolist())
    # print(np.allclose(r, np.zeros_like(r)))

    # molH = gto.M(atom='H 0.0 0.0 0.0', charge=0, spin=1, basis='def2tzvp')
    # mfH = molH.GKS(xc='LDA, VWN')
    # mfH.kernel()
    # dmH = mfH.make_rdm1()
    # dmH1 = rotate_dm(dmH, (0, 1, 0), np.pi/2)
    # dmH2 = rotate_dm(dmH1, (0, 0, 1), 2*np.pi/3)
    # dmH3 = rotate_dm(dmH2, (0, 0, 1), 2*np.pi/3)
    # dmH4 = rotate_dm(dmH3, (0, 0, 1), 2*np.pi/3)
    # print(np.allclose(dmH1, dmH4))
    pass