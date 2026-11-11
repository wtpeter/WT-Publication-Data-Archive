import numpy as np
from pyscf import gto, sftda
from pyscf.grad import tduks_sf
from pyscf.data.nist import BOHR
from pyscf.sftda.tools_td import extract_state
from time import time

basis = 'aug-cc-pvdz'
fun = 'BHandHLYP'

atom = '''
C          -0.01347941        0.03223283        0.03278602
C           1.37545297        0.00196587       -0.00563228
H           1.99197571       -0.90162307       -0.00018672
H           0.18071574       -0.61698880       -0.91621664
H          -0.54653218       -0.75574574        0.56677900
H           1.95055454        0.90227441       -0.19712514
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis=basis, max_memory=40000)

scf_times = []
tddftsf_times = []
grad_times = []

for i in range(5):
    mf = mol.UKS(xc=fun)
    t0 = time()
    mf.kernel()
    t1 = time()
    print(f"SCF time: {t1 - t0:.2f} s")
    td = mf.TDDFT_SF()
    td.extype = 1
    td.nstates = 5
    td.collinear_samples = -1
    td.positive_eig_threshold = -0.001
    t2 = time()
    td.kernel()
    t3 = time()
    print(f"TDDFT-SF time: {t3 - t2:.2f} s")
    S = extract_state(mf, td, Smin=0.0, Smax=0.7)
    T = extract_state(mf, td, Smin=1.7, Smax=2.9)
    tdg = td.Gradients()
    t4 = time()
    tdg.kernel(state=S[0]+1)
    t5 = time()
    print(f"TDDFT-SF Gradient time: {t5 - t4:.2f} s")
    scf_times.append(t1 - t0)
    tddftsf_times.append(t3 - t2)
    grad_times.append(t5 - t4)

np.save(f'{basis}_scf_times.npy', np.array(scf_times))
np.save(f'{basis}_tddftsf_times.npy', np.array(tddftsf_times))
np.save(f'{basis}_grad_times.npy', np.array(grad_times))