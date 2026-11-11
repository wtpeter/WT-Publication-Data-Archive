#!/usr/bin/env/python

'''
Spin flip TDA/ TDDFT. The kwarg 'extype' is used to control which
kind of excited energy to be calculated, 0 for spin flip up, 1 for
spin flip down.
'''

from pyscf import lib, gto, dft
from pyscf import sftda,tdscf
from pyscf.sftda import tools_td

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = '''
 N                  0.18099500    0.48678400   -0.00003600;
 O                 -1.04485200   -0.25546500   -0.00001600;
 H                 -1.71297000    0.44468900    0.00019300;
 O                  1.10060200   -0.22605700    0.00002300;
'''
mol.unit = 'A'
mol.basis = 'aug-cc-pvtz'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'b3lyP' # blyp, b3lyp, tpss
mf.kernel()



#
# 2. spin flip down TDA
#
mftd2 = sftda.uks_sf.TDA_SF(mf)
mftd2.nstates = 8 # the number of excited states
mftd2.extype = 1 # 1 for spin flip down excited energies
mftd2.collinear_samples=200 
mftd2.kernel()




mftd2.e # to get the excited energies
mftd2.xy # to get the transition vectors

tools_td.transition_analyze(mf, mftd2, mftd2.e[0], mftd2.xy[0], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[1], mftd2.xy[1], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[2], mftd2.xy[2], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[3], mftd2.xy[3], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[4], mftd2.xy[4], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[5], mftd2.xy[5], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[6], mftd2.xy[6], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[7], mftd2.xy[7], tdtype='TDA')

tools_td.full_diag_tddft(mf)
#
# 3. get_ab_sf()
#
a, b = sftda.TDDFT_SF(mf).get_ab_sf()
# List a has two items: (A_abab, A_baba) with A[i,a,j,b].
# List b has two items: (B_abba, B_baab) with B[i,a,j,b].

#
# 4. spin flip up TDDFT, which can not converged.
#    Just give an input example here.
#    Users can use get_ab_sf() to construct the whole TDDFT matrix
#    to get the excited energies, if the system is small.
# mftd3 = sftda.TDDFT_SF(mf) # equal to mftd3 = mf.TDDFT_SF()
# mftd3.nstates = 4
# mftd3.extype = 0
# mftd3.collinear_samples=200 
# mftd3.kernel()

# mftd3.e
# mftd3.xy

#
# 5. spin flip down TDDFT, which can not converged.
#    Just give an input example here.
# mftd4 = sftda.uks_sf.CasidaTDDFT(mf)
# mftd4.nstates = 4
# mftd4.extype = 1
# mftd4.collinear_samples=200 
# mftd4.kernel()

# mftd4.e
# mftd4.xy


