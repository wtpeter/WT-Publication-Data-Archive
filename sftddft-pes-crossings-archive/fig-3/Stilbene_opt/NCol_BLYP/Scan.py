import numpy as np
from pyscf import gto, sftda, scf, ao2mo
from pyscf.sftda.uhf_sf import get_ab_sf
from pyscf.sftda.tools_td import transition_analyze
import sys
sys.path.append('/home/peter/Research/ZeroExcitationEnergy/code/src')
sys.path.append('/data/home/db100401/db100401/Research/Tools/SFTD')
sys.path.append('/public1/home/scg0213/Jobs_WT/ZeroExcitationEnergy/src')
from tools import extract_state

fun = 'BLYP'
bas = '6-31g'

def extract_xyz_frames(filepath):
    frames = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    idx = 0
    n_atoms = None
    while idx < len(lines):
        # 跳过非数据行
        if lines[idx].strip() == "":
            idx += 1
            continue
        if n_atoms is None:
            try:
                n_atoms = int(lines[idx].strip())
            except ValueError:
                raise ValueError(f"文件第{idx+1}行不是原子数")
        atom_start = idx + 2
        atom_end = atom_start + n_atoms
        frame_lines = lines[atom_start:atom_end]
        # 组织为块字符串
        frame_str = "".join(frame_lines).rstrip("\n")
        frames.append(frame_str)
        idx = atom_end
        n_atoms = None  # 下一帧重新获取
    return frames

atoms = extract_xyz_frames("../traj.xyz")

eref_list = []
e1_list = []
e2_list = []
e3_list = []
for atom in atoms:
    mol = gto.M(atom=atom, basis=bas, charge=0, spin=2)
    mf = mol.UKS(xc=fun)
    mf.kernel()
    gse = mf.e_tot

    a, b = get_ab_sf(mf, collinear_samples=50)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b,n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b,n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a,n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a,n_occ_b*n_virt_a), order='C')

    Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                            [-B_baab_2d,-A_baba_2d]])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

    idxt = eigenvals.real.argsort()
    eigenvals = eigenvals[idxt].real
    eigenvecs = eigenvecs[:, idxt]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

    sfd_eigenvals = eigenvals[norms>0]
    sfd_eigenvecs = eigenvecs[:, norms>0].transpose(1,0)

    def norm_xy(z):
        x = z[:n_occ_a*n_virt_b].reshape(n_occ_a,n_virt_b)
        y = z[n_occ_a*n_virt_b:].reshape(n_occ_b,n_virt_a)
        norm = np.linalg.norm(x)**2 - np.linalg.norm(y)**2
        norm = np.sqrt(1./norm)
        return ((0,x*norm), (y*norm,0))

    mftd1 = sftda.TDA_SF(mf)
    mftd1.nstates = len(sfd_eigenvals)
    mftd1.extype = 1
    mftd1.collinear_samples = 50
    mftd1.e = sfd_eigenvals
    mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]

    S = extract_state(mf, mftd1, Smin=0.0, Smax=0.7)
    target1 = S[0]
    target2 = S[1]
    target3 = S[2]

    eref_list.append(gse)
    e1_list.append(gse + mftd1.e[target1])
    e2_list.append(gse + mftd1.e[target2])
    e3_list.append(gse + mftd1.e[target3])

    np.save('Ref', eref_list)
    np.save('S0', e1_list)
    np.save('S1', e2_list)
    np.save('S2', e3_list)