import numpy as np
from pyscf import gto, sftda
from pyscf.sftda.tools_td import transition_analyze
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

basis = 'cc-pvtz'
fun_list = ['SVWN', 'BLYP', 'B3LYP', 'PBE', 'PBE0']

results = {}

atom = '''
C 0.00000000 -2.28707566 1.32044375
C 0.00000000 -4.25961422 -2.45928942
C 0.00000000 -0.00000000 -2.64088750
C 0.00000000 4.25961422 -2.45928942
C 0.00000000 2.28707566 1.32044375
C 0.00000000 -0.00000000 4.91857884
N 0.00000000 0.00000000 -0.00000000
N 0.00000000 -4.45372767 0.04447145
N 0.00000000 2.18835043 -3.87927703
N 0.00000000 2.26537724 3.83480558
N 0.00000000 -2.26537724 3.83480558
N 0.00000000 -2.18835043 -3.87927703
N 0.00000000 4.45372767 0.04447145
H 0.00000000 6.02612802 -3.47918663
H 0.00000000 0.00000000 6.95837327
H 0.00000000 -6.02612802 -3.47918663
'''

molT = gto.M(atom=atom, spin=2, charge=0, basis=basis, unit='B', max_memory=50000)
molS = gto.M(atom=atom, spin=0, charge=0, basis=basis, unit='B', max_memory=50000)
print(molT.tostring())

for fun in fun_list:
    mfS = molS.RKS(xc=fun)
    mfS.kernel()

    scTDS = mfS.TDDFT()
    scTDS.nstates = 5
    scTDS.kernel()

    scTDT = mfS.TDDFT()
    scTDT.nstates = 5
    scTDT.singlet = False
    scTDT.kernel()

    sctdeS1 = mfS.e_tot + scTDS.e[0]
    sctdeT1 = mfS.e_tot + scTDT.e[0]

    mfT = molT.UKS(xc=fun)
    mfT.kernel()

    sftdT = mfT.TDDFT_SF()
    sftdT.max_space = 30
    sftdT.extype = 1
    sftdT.positive_eig_threshold = -0.15
    sftdT.collinear_samples = 50
    sftdT.nstates = 5
    sftdT.kernel()

    S = extract_state(mfT, sftdT, tdtype='TDDFT')
    T = extract_state(mfT, sftdT, tdtype='TDDFT', Smin=1.9, Smax=2.7)
    sftdeS1 = mfT.e_tot + sftdT.e[S[1]]
    sftdeT1 = mfT.e_tot + sftdT.e[T[0]]

    results[fun] = {
        'S1_sc': sctdeS1,
        'T1_sc': sctdeT1,
        'S1_sf': sftdeS1,
        'T1_sf': sftdeT1,
        'dE_sc': sctdeS1 - sctdeT1,
        'dE_sf': sftdeS1 - sftdeT1
    }

    print(f'{fun}: SC-TDDFT gap={(sctdeS1-sctdeT1)*27.2114:.3f}, SF-TDDFT gap={(sftdeS1-sftdeT1)*27.2114:.3f}')

import pandas as pd

df = pd.DataFrame([
    {'Functional': fun, **res}
    for fun, res in results.items()
])

# 保存为csv
df.to_csv('S_T_gap.csv', index=False)

# 保存为Excel
df.to_excel('S_T_gap.xlsx', index=False)
