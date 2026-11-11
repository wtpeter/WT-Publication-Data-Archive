import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.sftda.tools_td import transition_analyze
from pyscf.data.nist import HARTREE2WAVENUMBER

fun_list = ['SVWN', 'BLYP', 'PBE', 'B3LYP', 'PBE0', 'TPSS']
bas = 'def2-tzvp'

atom = '''
V 0.933423136 1.16726372 0.372422819
O -0.0700049576 -0.33042 1.15800644
O 0.0700049576 0.33042 -1.15800644
O 5.90071107 1.569495 1.28536908
O 2.39817005 0.787501 1.58345186
O 2.45641842 1.778761 -0.79940382
O 0.180746881 2.5089892 0.765079136
C 4.89661546 0.473602 3.14612825
C 4.7420872 1.1025014 1.82282137
C 3.57782962 1.2236554 1.12277847
C 3.58154222 1.7754568 -0.175236544
C 4.81022296 2.2600728 -0.694623206
C 5.89627869 2.1400202 0.053293588
C -0.032677831 -0.7500534 2.51834785
H 4.78159995 0.958218 3.71248554
H 4.12440793 -0.088112 3.40536994
H 5.61133587 -0.088112 3.2879434
H 4.89763819 2.577276 -1.56267639
H 6.79481052 2.390038 -0.171623419
H -0.704359343 -1.189512 2.78210595
H 0.214483688 0.022028 3.24277934
H 0.667447647 -1.343708 2.62854815
V -0.933423136 -1.16726372 -0.372422819
O -5.90071107 -1.569495 -1.28536908
O -2.39817005 -0.787501 -1.58345186
O -2.45641842 -1.778761 0.79940382
O -0.180746881 -2.5089892 -0.765079136
C -4.89661546 -0.473602 -3.14612825
C -4.7420872 -1.1025014 -1.82282137
C -3.57782962 -1.2236554 -1.12277847
C -3.58154222 -1.7754568 0.175236544
C -4.81022296 -2.2600728 0.694623206
C -5.89627869 -2.1400202 -0.053293588
C 0.032677831 0.7500534 -2.51834785
H -4.78159995 -0.958218 -3.71248554
H -4.12440793 0.088112 -3.40536994
H -5.61133587 0.088112 -3.2879434
H -4.89763819 -2.577276 1.56267639
H -6.79481052 -2.390038 0.171623419
H 0.704359343 1.189512 -2.78210595
H -0.214483688 -0.022028 -3.24277934
H -0.667447647 1.343708 -2.62854815
'''
mol_sym = gto.M(atom=atom, basis=bas, charge=0, spin=2, symmetry=True)
print(mol_sym.topgroup)
mol = gto.M(atom=atom, basis=bas, charge=0, spin=2)

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
