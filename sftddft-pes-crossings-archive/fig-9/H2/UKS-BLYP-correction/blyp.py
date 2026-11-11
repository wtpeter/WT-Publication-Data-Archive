import numpy as np
from pyscf import gto
from pyscf.lib import logger

r_list = np.concatenate((np.linspace(0.4, 3.0, 53), np.linspace(3.2, 6.0, 15)))
bas = '6-31G'
e_list = []
s2_list = []

for r in r_list:
    print(f'R = {r:.2f} Angstrom')
    mol = gto.M(atom=f'H; H 1 {r}', basis=bas, charge=0, spin=0)
    mf = mol.UKS(xc='BLYP')
    dm0 = np.zeros((2, mol.nao, mol.nao))
    dm0[0, 0, 0] = 1.0
    _, _, p1, _ = mol.aoslice_by_atom()[1]
    dm0[1, p1, p1] = 1.0
    mf.kernel(dm0=dm0)
    e_list.append(mf.e_tot)
    s2_list.append(mf.spin_square()[0])

    np.save('E_BLYP.npy', np.array(e_list))
    np.save('S2_BLYP.npy', np.array(s2_list))