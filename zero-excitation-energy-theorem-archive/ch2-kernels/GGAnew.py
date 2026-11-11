import numpy as np
from pyscf import gto, dft, lib
from scipy.special import roots_legendre

coef = 1.0

def eval_xc(xc_code, rho, *args, deriv = 1, **kwargs):
    '''
    rho: ((4, N), (4, N)) where N is the number of grid points
    '''
    rhoas, rhobs = rho
    rhoa, xa, ya, za = rhoas
    rhob, xb, yb, zb = rhobs
    gaa = xa ** 2 + ya ** 2 + za ** 2
    gbb = xb ** 2 + yb ** 2 + zb ** 2
    gab = xa * xb + ya * yb + za * zb
    rho = rhoa + rhob

    exc = np.power(rhoa - rhob, 2) * (gaa + gbb - 2*gab) / rho
    exc = coef * exc

    vrhoa = 2 * (rhoa - rhob) * (gaa + gbb - 2*gab)
    vrhob = -2 * (rhoa - rhob) * (gaa + gbb - 2*gab)
    vsigmaaa = np.power(rhoa - rhob, 2)
    vsigmabb = np.power(rhoa - rhob, 2)
    vsigmaab = -2 * np.power(rhoa - rhob, 2)
    vrho = np.column_stack((vrhoa, vrhob))
    vsigma = np.column_stack((vsigmaaa, vsigmaab, vsigmabb))
    vxc = [vrho, vsigma]
    vxc = [coef * v for v in vxc]

    if deriv==2:
        v2rho2a_a = 2 * (gaa - 2*gab + gbb)
        v2rho2a_b = -2 * (gaa - 2*gab + gbb)
        v2rho2b_b = 2 * (gaa - 2*gab + gbb)
        v2rho2 = np.column_stack((v2rho2a_a, v2rho2a_b, v2rho2b_b))

        v2rhosigmaa_aa = 2 * (rhoa - rhob)
        v2rhosigmaa_ab = -4 * (rhoa - rhob)
        v2rhosigmaa_bb = 2 * (rhoa - rhob)
        v2rhosigmab_aa = -2 * (rhoa - rhob)
        v2rhosigmab_ab = 4 * (rhoa - rhob)
        v2rhosigmab_bb = -2 * (rhoa - rhob)
        v2rhosigma = np.column_stack((v2rhosigmaa_aa, v2rhosigmaa_ab, v2rhosigmaa_bb,
                                    v2rhosigmab_aa, v2rhosigmab_ab, v2rhosigmab_bb))
        
        v2sigma2aa_aa = np.zeros_like(rhoa)
        v2sigma2 = np.column_stack((v2sigma2aa_aa, v2sigma2aa_aa, v2sigma2aa_aa,
                                    v2sigma2aa_aa, v2sigma2aa_aa, v2sigma2aa_aa))

        fxc = [v2rho2, v2rhosigma, v2sigma2]
        fxc = [coef * f for f in fxc]
    else:
        fxc = None
    kxc = None
    return exc, vxc, fxc, kxc

def eval_fmat_g1(mf, dm):
    mol = mf.mol
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    s = dft.numint.eval_rho(mol, ao, dm[0]-dm[1], xctype='GGA')
    m, mx, my, mz = s
    fxc = np.zeros((4, 4, len(m)))
    fxc[0, 1:4] = 2 * m * s[1:4]
    fxc[1:4, 0] = 2 * m * s[1:4]
    
    wfxc = fxc * weights
    uv = lib.einsum('xiu,iv->xiuv', ao, ao[0])
    uv[1:4] += lib.einsum('iu,xiv->xiuv', ao[0], ao[1:4])
    wfuv = lib.einsum('xyi,xiuv->yiuv', wfxc, uv)
    k = lib.einsum('xiuv,xirs', wfuv, uv)

    return k

def eval_fmat_g2(mf, dm):
    mol = mf.mol
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    s = dft.numint.eval_rho(mol, ao, dm[0]-dm[1], xctype='GGA')
    m, mx, my, mz = s
    fxc = np.zeros((4, 4, len(m)))
    fxc[0, 0] = 2 * (mx**2 + my**2 + mz**2)
    fxc[1, 1] = fxc[2, 2] = fxc[3, 3] = 2 * m**2
    
    wfxc = fxc * weights
    uv = lib.einsum('xiu,iv->xiuv', ao, ao[0])
    uv[1:4] += lib.einsum('iu,xiv->xiuv', ao[0], ao[1:4])
    wfuv = lib.einsum('xyi,xiuv->yiuv', wfxc, uv)
    k = lib.einsum('xiuv,xirs', wfuv, uv)

    return k

def eval_fmat_g3(mf, dm):
    return 2/3 * eval_fmat_g1(mf, dm) + 1/3 * eval_fmat_g2(mf, dm)

def eval_fxc_gcol(mol, ao, dmz):

    s = dft.numint.eval_rho(mol, ao, dmz, xctype='GGA')
    m, mx, my, mz = s
    fxc = np.zeros((4, 4, len(m)))
    fxc[0, 0] = 2 * (mx**2 + my**2 + mz**2)
    fxc[0, 1:4] = 4 * m * s[1:4]
    fxc[1:4, 0] = 4 * m * s[1:4]
    fxc[1, 1] = fxc[2, 2] = fxc[3, 3] = 2 * m**2
    return fxc

def eval_fmat_mc(mf, dm, samples=100):
    mol = mf.mol
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=1)

    dmz = dm[0] - dm[1]
    x, w = roots_legendre(samples)
    w = 0.5 * w
    x = 0.5 * (x + 1)
    dmzs = np.expand_dims(x, axis=(1, 2)) * np.expand_dims(dmz, axis=0)
    fxcs = []
    for i in range(len(dmzs)):
        fxcs.append(eval_fxc_gcol(mol, ao, dmzs[i]))
    fxcs = np.array(fxcs)
    fxc = (fxcs * w[:, None, None, None]).sum(axis=0)

    wfxc = fxc * weights
    uv = lib.einsum('xiu,iv->xiuv', ao, ao[0])
    uv[1:4] += lib.einsum('iu,xiv->xiuv', ao[0], ao[1:4])
    wfuv = lib.einsum('xyi,xiuv->yiuv', wfxc, uv)
    k = lib.einsum('xiuv,xirs', wfuv, uv)

    return k

def eval_ab_sf(mf, dm, eval_fmat):
    '''
    return spin flip matrix A and B with subscripts ai,bj
    A_SFU, A_SFD, B_SFU, B_SFD
    '''
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    occidx_a = np.where(mo_occ[0]==1)[0]
    viridx_a = np.where(mo_occ[0]==0)[0]
    occidx_b = np.where(mo_occ[1]==1)[0]
    viridx_b = np.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    e_ai_ab = (mo_energy[0][viridx_a,None] - mo_energy[1][occidx_b]) # (nvir_a, nocc_b)
    e_ai_ba = (mo_energy[1][viridx_b,None] - mo_energy[0][occidx_a]) # (nvir_b, nocc_a)
    A_abab = np.diag(e_ai_ab.ravel()).reshape(nvir_a, nocc_b, nvir_a, nocc_b)
    A_baba = np.diag(e_ai_ba.ravel()).reshape(nvir_b, nocc_a, nvir_b, nocc_a)
    B_abba = np.zeros((nvir_a, nocc_b, nvir_b, nocc_a))
    B_baab = np.zeros((nvir_b, nocc_a, nvir_a, nocc_b))

    fmat = 2 * eval_fmat(mf, dm)
    A_abab += lib.einsum('ua,vi,rb,sj,uvrs->aibj', orbv_a, orbo_b, orbv_a, orbo_b, fmat)
    A_baba += lib.einsum('ua,vi,rb,sj,uvrs->aibj', orbv_b, orbo_a, orbv_b, orbo_a, fmat)
    B_abba += lib.einsum('ua,vi,rb,sj,uvrs->aibj', orbv_a, orbo_b, orbv_b, orbo_a, fmat)
    B_baab += lib.einsum('ua,vi,rb,sj,uvrs->aibj', orbv_b, orbo_a, orbv_a, orbo_b, fmat)

    def res(A):
        u, v, x, y = A.shape
        return A.reshape(u*v, x*y, order='C')
    
    return res(A_abab), res(A_baba), res(B_abba), res(B_baab)

def td(mf, s):
    dm = mf.make_rdm1()
    if s=='I':
        A_abab, A_baba, B_abba, B_baab = eval_ab_sf(mf, dm, eval_fmat_g1)
    elif s=='II':
        A_abab, A_baba, B_abba, B_baab = eval_ab_sf(mf, dm, eval_fmat_g2)
    elif s=='MC':
        A_abab, A_baba, B_abba, B_baab = eval_ab_sf(mf, dm, eval_fmat_mc)
    elif s=='MCA':
        A_abab, A_baba, B_abba, B_baab = eval_ab_sf(mf, dm, eval_fmat_g3)
    
    n_occ_a, n_occ_b = mf.mol.nelec
    n_virt_a, n_virt_b = mf.mol.nao_nr() - n_occ_a, mf.mol.nao_nr() - n_occ_b

    Cas = np.block([[A_baba, B_baab], [-B_abba.conj(), -A_abab.conj()]])
    eigenvals, eigenvecs = np.linalg.eig(Cas)

    idxt = eigenvals.real.argsort()
    eigenvals = eigenvals[idxt]
    eigenvecs = eigenvecs[:, idxt]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

    sfd_eigenvals = eigenvals[norms>0]
    sfd_eigenvecs = eigenvecs[:, norms>0].transpose(1,0)
    return sfd_eigenvals

def verify(mf, s):
    dm = mf.make_rdm1()
    if s=='I':
        fmat = eval_fmat_g1(mf, dm)
    elif s=='II':
        fmat = eval_fmat_g2(mf, dm)
    elif s=='MC':
        fmat = eval_fmat_mc(mf, dm)
    elif s=='MCA':
        fmat = eval_fmat_g3(mf, dm)
    fd = mf.get_veff()[0] - mf.get_veff()[1]
    kd = lib.einsum('uvrs,rs->uv', fmat, (dm[0] - dm[1]))
    return np.max(np.abs(fd - 2 * kd))


if __name__ == '__main__':
    # mol = gto.M(atom='Li 0.0 0.0 0.0', charge=0, spin=-1, basis='cc-pvdz')
    mol = gto.M(atom='C 0.0 0.0 0.0; H 0.0 0.0 1.0; H 0.0 1.0 0.0; H 1.0 0.0 0.0', charge=0, spin=-1, basis='sto-3g')
    mf = dft.UKS(mol)
    mf = mf.define_xc_(eval_xc, 'GGA')
    mf.kernel()
    
    print(verify(mf, 'I'))
    print(verify(mf, 'II'))
    print(verify(mf, 'MC'))
    print(verify(mf, 'MCA'))
    print(td(mf, 'I').real)
    print(td(mf, 'II').real)
    print(td(mf, 'MC').real)
    print(td(mf, 'MCA').real)
