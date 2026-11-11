import numpy as np
from pyscf import gto, fci

r_list = np.array([3.6, 4. , 4.4, 4.8, 5.2, 5.6, 6. ])
bas = '6-311g*'

hf_list = []
fci_list = []
for r in r_list:
    mol = gto.M(atom=f'H; F 1 {r}', basis=bas, charge=0, spin=0)
    mf = mol.RHF()
    mf.max_memory = 110000
    mf.kernel()
    hf_list.append(mf.e_tot)
    myci = fci.FCI(mf)
    myci.max_memory = 110000
    myci.kernel()
    fci_list.append(myci.e_tot)
    print(f'r = {r:.2f}, HF energy = {mf.e_tot:.6f}, FCI energy = {myci.e_tot:.6f}')
    np.save('FCI_r.npy', r_list)
    np.save('HF_e.npy', hf_list)
    np.save('FCI_e.npy', fci_list)