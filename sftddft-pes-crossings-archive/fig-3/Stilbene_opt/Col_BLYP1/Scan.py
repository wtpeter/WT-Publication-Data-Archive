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
e4_list = []
for atom in atoms:
    mol = gto.M(atom=atom, basis=bas, charge=0, spin=2)
    mf = mol.UKS(xc=fun)
    mf.kernel()
    gse = mf.e_tot

    na, nb = mf.mol.nelec
    oae1, oae2 = mf.mo_energy[0][na-2], mf.mo_energy[0][na-1]
    vbe1, vbe2 = mf.mo_energy[1][nb], mf.mo_energy[1][nb+1]
    e1, e2, e3, e4 = vbe1-oae1, vbe1-oae2, vbe2-oae1, vbe2-oae2
    e1, e2, e3, e4 = sorted([e1, e2, e3, e4])
    print(np.array([e1, e2, e3, e4]) * 27.2114)

    eref_list.append(gse)
    e1_list.append(gse + e1)
    e2_list.append(gse + e2)
    e3_list.append(gse + e3)
    e4_list.append(gse + e4)

    np.save('Ref.npy', eref_list)
    np.save('E1.npy', e1_list)
    np.save('E2.npy', e2_list)
    np.save('E3.npy', e3_list)
    np.save('E4.npy', e4_list)