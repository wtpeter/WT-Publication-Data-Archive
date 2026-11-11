import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.sftda.tools_td import transition_analyze
from pyscf.data.nist import HARTREE2WAVENUMBER

fun_list = ['SVWN', 'BLYP', 'PBE', 'B3LYP', 'PBE0', 'TPSS']
bas = 'def2-tzvp'

atom = '''
Cu 0.000000 0.000000 1.695700
Cu 0.000000 0.000000 -1.695700
Cl 1.553900 0.000000 0.000000
Cl -1.553900 0.000000 0.000000
Cl 1.639300 0.000000 3.251400
Cl -1.639300 0.000000 3.251400
Cl 1.639300 0.000000 -3.251400
Cl -1.639300 0.000000 -3.251400
'''
mol_sym = gto.M(atom=atom, basis=bas, charge=0, spin=2, symmetry=True)
print(mol_sym.topgroup)
mol = gto.M(atom=atom, basis=bas, charge=-2, spin=2)

for fun in fun_list:
    print(f'Functional: {fun}')

    mf = mol.UKS(xc=fun)
    mf.max_memory = 50000
    mf.kernel()

    td = mf.TDDFT_SF()
    td.extype = 1
    td.max_space = 100
    td.collinear_samples = 50
    td.nstates = 5
    td.positive_eig_threshold = -0.01
    td.kernel()

    print(transition_analyze(mf, td, td.e[0], td.xy[0], tdtype='TDDFT'))
    print(transition_analyze(mf, td, td.e[1], td.xy[1], tdtype='TDDFT'))
    print(transition_analyze(mf, td, td.e[2], td.xy[2], tdtype='TDDFT'))
    print(transition_analyze(mf, td, td.e[3], td.xy[3], tdtype='TDDFT'))
    print(transition_analyze(mf, td, td.e[4], td.xy[4], tdtype='TDDFT'))

    print()
    print('-------------------------------------------')
    print()
