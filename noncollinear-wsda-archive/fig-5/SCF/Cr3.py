from pyscf import gto, dft, scf
import numpy as np
from pyscf.data.nist import BOHR
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/home/peter/Research/LegacyMaterials/HPC_PZC')
sys.path.append('/home/peter/Research/NCNLDFT/code/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT/src')
sys.path.append('/public1/home/scg0213/Jobs_WT/NCNLDFT/src/HPC_PZC')
from pySD import tools as tools_hss
from pySD import grids_util
from pySD import gksm_util
from pySD import mole_sym
from gks_nl_new import GKSMCNL
import numpy

bas = 'stuttgartrsc'

molcoords = """
Cr1          1.130432197164135     0.000000000000000     0.000000000000000    ;
Cr2          -0.565216098582067    0.978983000000000     0.000000000000000    ;
Cr3          -0.565216098582067    -0.978983000000000    0.000000000000000    ;
"""

molS = gto.Mole()
molS.atom = molcoords
molS.spin=18
molS.basis = bas
molS.ecp = bas
molS.build()
mfs = scf.UHF(molS)
mfs.kernel()
S = mfs.get_ovlp()


rm3d = R.from_rotvec(np.array([0.0, 0.0, np.pi*2/3])).as_matrix()

atom_grid = (15, 86)
mol0 = gto.M(atom='Cr 0.0 0.0 0.0', basis=bas, ecp=bas)
grids0 = dft.gen_grid.Grids(mol0)
grids0.prune = None
grids0.atom_grid = atom_grid
tab0 = grids0.gen_atomic_grids(mol0)['Cr']

tab1 = tab0
tab2 = ((rm3d @ tab0[0].T).T, tab0[1])
tab3 = ((rm3d @ rm3d @ tab0[0].T).T, tab0[1])

atom_grids_tab = {'Cr1': tab1, 'Cr2': tab2, 'Cr3': tab3}
grids = dft.gen_grid.Grids(molS)
grids.atom_grid = atom_grid
grids.prune = None

grids.coords, grids.weights = grids.get_partition(molS, atom_grids_tab)

molu = gto.Mole()
molu.atom = "Cr 0 0 0"
molu.spin=6
molu.basis = bas
molu.ecp = bas
molu.symmetry=False
molu.build()
mf = scf.UHF(molu)
mf.kernel()

molcoords = """
Cr          1.130432197164135     0.000000000000000     0.000000000000000    ;
Cr          -0.565216098582067    0.978983000000000     0.000000000000000    ;
Cr          -0.565216098582067    -0.978983000000000    0.000000000000000    ;
"""
mol = mole_sym.Mole_sym()
mol.Dsymmetry = True
mol.ovlp = S
mol.verbose = 6
mol.atom = molcoords
mol.basis = bas
mol.ecp = bas
mol.spin = 18
mol.build(singleatom=molu, dump_input=False)

mo_coeffu = mf.mo_coeff 
noccu = mf.mo_occ 
nao = mo_coeffu.shape[-1]
natom = mol.natm 
theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol, rotatez_negative = False,vortex = False) 

mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo(molu, mo_coeffu, natom
                                , numpy.array([theta_dict[i][0] for i in range(3)])                 , rotatem = True, rotatel =  True)
nocc = numpy.array((noccu[0].tolist()*natom+noccu[1].tolist()*natom))

mftot = gksm_util.GKSM_spha_ibp(mol)
mftot.xc = "PBE" # ^ functional
mftot.parallel = True
# ! NOTE: that the following part is Gauss-Legendre scheme.
# * By changing the phi angle, changing the symmetry.
dft.Grids.gen_atomic_grids = grids_util.Grids_hss.gen_atomic_grids_gauss_legendre
mftot.grids.atom_grid = {"Cr" : (140, 60, 48)} # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi
mftot.Ndirect=1800
dm_tot  = mftot.make_rdm1(mo_coeff = mo_coeffu , mo_occ  = nocc)

gwsda = GKSMCNL(molS, xc='', epsilon=1e-11)
gwsda.nproc = 64
gwsda.spin_samples = 110
gwsda.grids = grids
gwsda.max_memory = 500000

def save_dm(envs):
    cycle = envs['cycle']
    dm = envs['dm']
    np.save(f'dm_{cycle}', dm)

mf123 = gksm_util.gks_sym_general.GKS_symm(mol, xc='HF,')
mf123.chkfile = False
mf123.get_veff = gwsda.get_veff
mf123.energy_elec = gwsda.energy_elec
mf123.conv_tol = 1e-5
mf123.conv_tol_grad = 1e-2
mf123.callback = save_dm
mf123.kernel(np.load('dmi.npy'), dump_chk=False)

np.save('dm', mf123.make_rdm1())
np.save('mocoef', mf123.mo_coeff)
np.save('moenergy', mf123.mo_energy)
