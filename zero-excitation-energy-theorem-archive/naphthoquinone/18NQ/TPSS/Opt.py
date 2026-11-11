import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.grad import tduks_sf
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state
from pyscf.sftda.tools_td import transition_analyze


bas = 'cc-pvtz'
fun = 'TPSS'

atom = '''
C 0.000000 1.300801 0.994575
C 0.000000 2.520134 0.186842
C 0.000000 2.479493 -1.204221
C 0.000000 1.239478 -1.871064
C 0.000000 -1.239478 -1.871064
C 0.000000 -2.479493 -1.204221
C 0.000000 -2.520134 0.186842
C 0.000000 -1.300801 0.994575
C 0.000000 0.000000 0.275641
C 0.000000 0.000000 -1.146275
O 0.000000 1.391977 2.243783
H 0.000000 3.460527 0.742496
H 0.000000 3.406668 -1.782598
H 0.000000 1.203823 -2.963616
H 0.000000 -1.203823 -2.963616
H 0.000000 -3.406668 -1.782598
H 0.000000 -3.460527 0.742496
O 0.000000 -1.391977 2.243783
'''
mol = gto.M(atom=atom, basis=bas, charge=0, spin=2)

# Singlet state optimization

geom1_list = []
e1_list = []

def f1(mol):
    mol.spin = 2
    mf = mol.UKS(xc=fun)
    mf.max_memory = 100000
    mf.kernel()
    e1 = mf.e_tot

    mftd = mf.TDDFT_SF()
    mftd.extype = 1
    mftd.collinear_samples = 50
    mftd.positive_eig_threshold = -0.1
    mftd.nstates = 3
    mftd.kernel()

    state = extract_state(mf, mftd, tdtype='TDDFT', Smin=0.0, Smax=0.8)
    target = state[0]

    mftdg1 = mftd.Gradients()
    g = mftdg1.kernel(state=target+1)

    e2 = e1 + mftd.e[target]

    print(transition_analyze(mf, mftd, mftd.e[target], mftd.xy[target], tdtype='TDDFT'))

    geom1_list.append(mol.atom_coords())
    e1_list.append(e2)

    print('energy', e2, '|grad|', np.linalg.norm(g))
    return e2, g

fake_method = as_pyscf_method(mol, f1)
new_mol = berny_solver.optimize(fake_method)

np.save('geom1.npy', np.array(geom1_list))
np.save('energy1.npy', np.array(e1_list))

# Triplet state optimization

geom3_list = []
e3_list = []

def f3(mol):
    mol.spin = 2
    mf = mol.UKS(xc=fun)
    mf.max_memory = 100000
    mf.kernel()
    e1 = mf.e_tot

    mftd = mf.TDDFT_SF()
    mftd.extype = 1
    mftd.collinear_samples = 50
    mftd.positive_eig_threshold = -0.1
    mftd.nstates = 3
    mftd.kernel()

    state = extract_state(mf, mftd, tdtype='TDDFT', Smin=1.9, Smax=2.8)
    target = state[0]

    mftdg1 = mftd.Gradients()
    g = mftdg1.kernel(state=target+1)

    e2 = e1 + mftd.e[target]

    print(transition_analyze(mf, mftd, mftd.e[target], mftd.xy[target], tdtype='TDDFT'))

    geom3_list.append(mol.atom_coords())
    e3_list.append(e2)

    print('energy', e2, '|grad|', np.linalg.norm(g))
    return e2, g

fake_method = as_pyscf_method(mol, f3)
new_mol = berny_solver.optimize(fake_method)

np.save('geom3.npy', np.array(geom3_list))
np.save('energy3.npy', np.array(e3_list))
