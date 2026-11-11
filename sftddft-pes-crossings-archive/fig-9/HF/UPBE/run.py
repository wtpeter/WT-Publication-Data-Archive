import numpy as np
from pyscf import gto
from pyscf.lib import logger

r_list = np.concatenate((np.linspace(0.4, 3.0, 53), np.linspace(3.2, 6.0, 15)))[61:]
bas = '6-311g*'
e_list = []
s2_list = []

def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)
    cyc = 0
    converged = mf.converged
    while ((not stable) or (not converged)) and (cyc < 10):
        log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
        mf = mf.run(np.array(mo1[0]), mf.mo_occ)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc += 1
        converged = mf.converged
    if not stable:
        log.note('Stability Opt failed after %d attempts' % cyc)
    return mf

dm0=np.load('PBE.npy')
for r in r_list:
    print(f'R = {r:.2f} Angstrom')
    mol = gto.M(atom=f'H 0 0 0; F 0 0 {r}', basis=bas, symmetry='C2v')
    mf = mol.UKS(xc='PBE').newton()
    mf.kernel(dm0=dm0)
    mf = stable_opt_internal(mf)
    dm0 = (mf.make_rdm1() + np.load('PBE.npy') * 2) / 3
    if r>=5.79:
        dm0 = np.load('PBE.npy')
    e_list.append(mf.e_tot)
    s2_list.append(mf.spin_square()[0])

    np.save('E_PBE_1.npy', np.array(e_list))
    np.save('S2_PBE_1.npy', np.array(s2_list))