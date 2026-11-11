from pyscf import gto, dft
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/TwoGGAs')
from GGAnew import eval_xc, td, verify

atom = '''
6  0.000000    0.000000    0.103187
1 -0.000000    0.995526   -0.309561
1 -0.000000   -0.995526   -0.309561
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis='6-31g')
mf = dft.UKS(mol)
mf = mf.define_xc_(eval_xc, 'GGA')
mf.grids.level = 1
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.kernel()


print('-----------')
print('I')
print(verify(mf, 'I'))
print(td(mf, 'I') * 27.21138602)

print('-----------')
print('II')
print(verify(mf, 'II'))
print(td(mf, 'II') * 27.21138602)

print('-----------')
print('III')
print(verify(mf, 'MC'))
print(td(mf, 'MC') * 27.21138602)

print('-----------')
print('IV')
print(verify(mf, 'MCA'))
print(td(mf, 'MCA') * 27.21138602)