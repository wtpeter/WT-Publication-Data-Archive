#!/usr/bin/env/python

'''
Spin flip TDA/ TDDFT. The kwarg 'extype' is used to control which
kind of excited energy to be calculated, 0 for spin flip up, 1 for
spin flip down.
'''

from pyscf import lib, gto, dft
from pyscf import sftda,tdscf
from pyscf.sftda import tools_td
import numpy as np

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = '''
C -1.7849169944,0.0756037379,-0.4035511507;
C -0.4567500006,0.0947298614,-0.4388121878;
C -0.453558256,1.4256852423,0.4033397553;
C -1.7817251917,1.4065581283,0.4386024474;
H -2.5492202632,-0.579811979,-0.7910602342;
H 0.3044623657,-0.5387171752,-0.8668226544;
H 0.3107449427,2.0811010435,0.7908486157;
H -2.5429376028,2.040004913,0.866613405;
'''
mol.unit = 'A'
mol.basis = 'cc-pvtz'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'b3lyP' # blyp, b3lyp, tpss
mf.kernel()
nelec_alpha, nelec_beta = mol.nelec

HOMO_up_index = nelec_alpha - 1
LUMO_plus1_down_index = nelec_beta + 1

C_up = mf.mo_coeff[0][:, HOMO_up_index]
C_down = mf.mo_coeff[1][:, LUMO_plus1_down_index]

S = mf.get_ovlp()

overlap = np.dot(C_up.T, S @ C_down)

print("两个轨道的空间部分内积为：", overlap)
#





#
# 2. spin flip down TDA
#
mftd2 = sftda.uks_sf.TDA_SF(mf)
mftd2.nstates = 20 # the number of excited states
mftd2.extype = 1 # 1 for spin flip down excited energies
mftd2.collinear_samples=200 
mftd2.kernel()

tools_td.transition_analyze(mf, mftd2, mftd2.e[0], mftd2.xy[0], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[1], mftd2.xy[1], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[2], mftd2.xy[2], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[3], mftd2.xy[3], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[4], mftd2.xy[4], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[5], mftd2.xy[5], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[6], mftd2.xy[6], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[7], mftd2.xy[7], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[8], mftd2.xy[8], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[9], mftd2.xy[9], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[10], mftd2.xy[10], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[11], mftd2.xy[11], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[12], mftd2.xy[12], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[13], mftd2.xy[13], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[14], mftd2.xy[14], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[15], mftd2.xy[15], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[16], mftd2.xy[16], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[17], mftd2.xy[17], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[18], mftd2.xy[18], tdtype='TDA')
tools_td.transition_analyze(mf, mftd2, mftd2.e[19], mftd2.xy[19], tdtype='TDA')


mftd2.e # to get the excited energies
mftd2.xy # to get the transition vectors

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


