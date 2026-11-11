from pyscf import gto, dft, lib
import numpy as np
from scipy.linalg import block_diag
import sys
sys.path.append('/home/peter/Research/Tools')
sys.path.append('/public1/home/scg0213/Jobs_WT/Tools')
from rotate import rotate_total_dm, rotate_spin_dm
from scipy.spatial.transform import Rotation

basis = '6-31g'
symmetry = False

rm3d = Rotation.from_rotvec(np.array([0, 0, 2*np.pi/5])).as_matrix()
x0 = np.array([0, 1.0 / 2 / np.sin(np.pi/5), 0])
x1 = (rm3d @ x0.T).T
x2 = (rm3d @ x1.T).T
x3 = (rm3d @ x2.T).T
x4 = (rm3d @ x3.T).T
atom = [['H', x0], ['H', x1], ['H', x2], ['H', x3], ['H', x4]]
mol0 = gto.M(atom=atom, charge=0, spin=5, basis=basis, symmetry=False)

molH = gto.M(atom='H 0,0 0.0 0.0', charge=0, spin=1, basis='6-31g')
mfH = molH.UHF()
mfH.kernel()
dmH = mfH.make_rdm1()
dmH = block_diag(dmH[0], dmH[1])

dmp1 = rotate_total_dm(molH, dmH, (1, 0, 0), -np.pi/2)
dmp2 = rotate_total_dm(molH, dmp1, (0, 0, 1), 2*np.pi/5)
dmp3 = rotate_total_dm(molH, dmp2, (0, 0, 1), 2*np.pi/5)
dmp4 = rotate_total_dm(molH, dmp3, (0, 0, 1), 2*np.pi/5)
dmp5 = rotate_total_dm(molH, dmp4, (0, 0, 1), 2*np.pi/5)
dmps = [dmp1, dmp2, dmp3, dmp4, dmp5]
dmaa, dmab, dmba, dmbb = np.zeros((4, mol0.nao_nr(), mol0.nao_nr()), dtype=complex)
for i in range(5):
    nao = dmps[i].shape[-1]//2
    dmaa[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, :nao]
    dmab[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][:nao, nao:]
    dmba[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, :nao]
    dmbb[i*nao:i*nao+nao, i*nao:i*nao+nao] = dmps[i][nao:, nao:]
dmi = np.block([[dmaa, dmab], [dmba, dmbb]])

mol = gto.M(atom=atom, charge=0, spin=5, basis=basis, verbose=4)

fun_list = ['HF', 'SVWN', 'BLYP', 'PBE', 'TPSS', 'SCAN', 'revSCAN', 'r2SCAN', 'B3LYP', 'PBE0', 'M062X', 'SCAN0']

for fun in fun_list:
    print('----------------------------------')
    print(fun)
    mf = mol.GKS(xc=fun)
    mf.collinear = 'mcol'
    mf._numint.collinear_thrd = None
    mf._numint.spin_samples = 3470
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.max_memory = 40000
    mf.kernel(dmi)
    mftd = mf.TDDFT()
    a, b = mftd.get_ab()
    o, v = a.shape[:2]
    a = a.reshape(o*v, o*v)
    b = b.reshape(o*v, o*v)
    Cas = np.block([[a, b], [-b.conj(), -a.conj()]])
    EigECas, EigVCas = np.linalg.eig(Cas)
    print(EigECas[np.abs(EigECas)<1e-2] * 27.211386245)
    np.save(f'{fun}.npy', EigECas*27.211386245)

