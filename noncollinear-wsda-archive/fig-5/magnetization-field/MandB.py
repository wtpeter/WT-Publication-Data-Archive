from pyscf import gto, dft, scf
import numpy as np
from pyscf.data.nist import BOHR
from scipy.spatial.transform import Rotation as R
from nl_utils import eval_B

bas = 'stuttgartrsc'

molcoords = """
Cr1          1.130432197164135     0.000000000000000     0.000000000000000    ;
Cr2          -0.565216098582067    0.978983000000000     0.000000000000000    ;
Cr3          -0.565216098582067    -0.978983000000000    0.000000000000000    ;
"""

molS = gto.Mole()
molS.atom = molcoords
molS.spin=18 # ^ Spin
molS.basis = bas # ^ Basis
molS.ecp = bas
molS.build()

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

dm = np.load('dm.npy')

a = 0.5
x = np.linspace(0, 15, 16)[1:]
y = np.linspace(0, 15, 16)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
x1 = - x * a + y * 0.5 * a
y1 = y * a * np.sqrt(3) * 0.5
z1 = np.zeros_like(x1)
coords = np.vstack([x1, y1, z1]).T
rm3d = R.from_rotvec(np.array([0.0, 0.0, np.pi*2/3])).as_matrix()
coords1 = (rm3d @ coords.T).T
coords2 = (rm3d @ rm3d @ coords.T).T
coords = np.vstack([coords, coords1, coords2])

nelec, Ex, vmat, target_coords, B = eval_B(molS, coords, grids, dm, epsilon=1e-11,
                      max_memory=500000, nproc=64, spin_samples=110)

np.save('coords', target_coords)
np.save('B', B)
np.save('Ex', np.array([Ex]))
np.save('nelec', nelec)
np.save('vmat', vmat)
