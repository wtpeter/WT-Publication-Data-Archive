import numpy as np
from pyscf import gto, sftda
from pyscf.grad import tduks_sf
from pyscf.data.nist import BOHR
from pyscf.sftda.tools_td import extract_state

basis = '6-31+g**'
fun = 'TPSS'

atom = '''
C -0.012746 -0.024065 -0.012103
C 1.374611 0.024827 0.002314
C 2.178736 -1.105511 0.127972
C 3.635739 -1.041070 0.179075
C 4.432435 -1.162796 -0.983694
N 5.733934 -1.120471 -0.963068
H 4.146354 -0.916693 1.125658
H 1.729023 -2.079673 0.256391
H 3.963776 -1.294652 -1.945502
H 1.851305 0.988485 -0.087715
H 6.271011 -1.208554 -1.808123
H 6.248594 -1.000873 -0.106649
H -0.593470 0.872726 -0.113191
H -0.538669 -0.957439 0.079915
'''
mol = gto.M(atom=atom, charge=1, spin=2, basis=basis, max_memory=100000)

def tdgrad(mol):
    mf = mol.UKS(xc=fun)
    mf.kernel()
    gse = mf.e_tot

    td = mf.TDDFT_SF()
    td.extype = 1
    td.nstates = 5
    td.collinear_samples = 20
    td.positive_eig_threshold = -0.1
    td.kernel()

    T = extract_state(mf, td, Smin=1.7, Smax=2.9)
    S = extract_state(mf, td, Smin=0.0, Smax=0.7)
    target1 = S[0]
    target2 = S[1]

    tdg = td.Gradients()
    g1 = tdg.kernel(state=target1+1)
    g2 = tdg.kernel(state=target2+1)

    e1 = gse + td.e[target1]
    e2 = gse + td.e[target2]

    return e1, e2, g1, g2

from scipy.optimize import minimize, OptimizeResult

class MECIOptimizer:
    def __init__(self, mol, initial_sigma=10.0, sigma_factor=2.0, 
                 energy_gap_threshold=1e-4, max_outer_loops=10, 
                 alpha=0.02, tol_step=1e-6, tol_grad=1e-4):
        """
        初始化MECI优化器.

        Args:
            mol: PySCF mol 对象。优化将从这个对象当前的几何构型开始。
            initial_sigma: 初始惩罚因子.
            sigma_factor: 每次外循环 sigma 的增大倍数.
            energy_gap_threshold: 最终能量差收敛阈值 (in Hartree).
            max_outer_loops: 最大外循环次数，防止死循环.
            alpha: 惩罚函数中的平滑参数.
            tol_step: 内循环优化中函数值变化的收敛阈值.
            tol_grad: 内循环优化中梯度的收敛阈值.
        """
        self.mol = mol
        # 自动从 mol 对象获取初始坐标 (确保单位是 Bohr)
        self.R_initial = self.mol.atom_coords(unit='Bohr').flatten()
        self.natm = self.mol.natm
        
        self.initial_sigma = initial_sigma
        self.sigma_factor = sigma_factor
        self.energy_gap_threshold = energy_gap_threshold
        self.max_outer_loops = max_outer_loops
        self.alpha = alpha
        self.tol_step = tol_step
        self.tol_grad = tol_grad

        # 实例变量
        self.sigma = self.initial_sigma
        self.history = []
        self.tdgrad_cache = {}
        self.final_result = {}
        
    def _calculate_properties(self, R):
        """
        核心计算函数，使用鲁棒缓存和高效的in-place几何更新。
        """
        rounded_R = np.round(R, decimals=9)
        R_tuple = tuple(rounded_R)
        
        if R_tuple in self.tdgrad_cache:
            e1, e2, g1, g2 = self.tdgrad_cache[R_tuple]
        else:
            self.mol.set_geom_(R.reshape((self.natm, 3)), unit='Bohr')
            e1, e2, g1, g2 = tdgrad(self.mol)
            self.tdgrad_cache[R_tuple] = (e1, e2, g1, g2)

        g = (e1 - e2)**2 / (e2 - e1 + self.alpha)
        f = 0.5 * (e1 + e2) + self.sigma * g

        g1_flat = g1.flatten()
        g2_flat = g2.flatten()
        f_R = 0.5 * (g1_flat + g2_flat)
        u_penalty_term = ((e1 - e2)**2 + 2 * self.alpha * (e2 - e1)) / (e2 - e1 + self.alpha)**2
        f_R += self.sigma * u_penalty_term * (g2_flat - g1_flat)

        return f, f_R, e1, e2, g1_flat, g2_flat

    def F_and_grad(self, R):
        """提供给 scipy.minimize 的目标函数和梯度。"""
        f, f_R, e1, e2, g1, g2 = self._calculate_properties(R)
        self.current_f = f
        self.current_f_R = f_R
        self.current_e1 = e1
        self.current_e2 = e2
        self.current_g1 = g1
        self.current_g2 = g2
        return f, f_R

    def callback(self, intermediate_result: OptimizeResult):
        """内循环的回调函数，用于检查单次优化的收敛性。"""
        Rk = intermediate_result.x
        f = intermediate_result.fun
        assert self.current_f==f
        f_R = self.current_f_R
        e1 = self.current_e1
        e2 = self.current_e2
        g1 = self.current_g1
        g2 = self.current_g2
        print('-'*40)
        print(f"Iter {len(self.history)+1:>3d}")
        print(self.mol.tostring())
        print(f"F = {f:.8f}", end="")

        converged = False
        if self.history: # 检查历史记录是否为空
            f0 = self.history[-1]['f']
            delta_f = f - f0
            print(f" | Delta F = {delta_f:.2e}", end="")
            
            # 内循环收敛判据
            u = ((e1 - e2)**2 + 2 * self.alpha * (e2 - e1)) / (e2 - e1 + self.alpha)**2
            u *= (g2 - g1)
            tol1 = np.abs(delta_f) < self.tol_step
            tol2 = (f_R @ u / self.sigma) < self.tol_grad
            tol3 = np.linalg.norm(f_R - f_R @ u * u) < self.tol_grad
            if tol1 and tol2 and tol3:
                converged = True
        
        print() # 换行
        self.history.append({'R': Rk, 'f': f})

        if converged:
            raise StopIteration(f"Inner loop converged on energy change for sigma={self.sigma:.2f}.")

    def _run_single_optimization(self, R0):
        """执行单次BFGS优化。"""
        self.history = []
        self.final_result = {}

        result = minimize(fun=self.F_and_grad, x0=R0, method='BFGS', jac=True, callback=self.callback)
        if 'StopIteration' in result.message:
            print("Inner loop converged by custom criteria.")
            if self.history:
                self.final_result = {'R': self.history[-1]['R'], 'f': self.history[-1]['f'], 'success': True}
            else:
                self.final_result = {'R': result.x, 'f': result.fun, 'success': True}

        if not self.final_result:
            print(f"    Warning: Inner optimization loop did not converge. Message: {result.message}")
            return None
        return self.final_result

    def find_meci(self):
        """主执行函数，通过迭代调整 sigma 来寻找 MECI。"""
        print("="*50)
        print("Starting MECI search protocol.")
        print("="*50)

        current_R = self.R_initial
        self.sigma = self.initial_sigma

        for i in range(self.max_outer_loops):
            print(f"\n--- Outer Loop {i+1}/{self.max_outer_loops} | sigma = {self.sigma:.2f} ---")
            
            single_run_result = self._run_single_optimization(current_R)
            if single_run_result is None:
                print("Optimization failed. Aborting MECI search.")
                return None

            current_R = single_run_result['R']
            _, _, e1, e2, _, _ = self._calculate_properties(current_R)
            energy_gap = abs(e1 - e2)

            print(f"--- Result for sigma = {self.sigma:.2f} ---")
            print(f"  Converged penalty function value: {single_run_result['f']:.8f}")
            print(f"  Energy gap |e1 - e2|: {energy_gap:.6f} Hartree")

            if energy_gap < self.energy_gap_threshold:
                print("\n" + "="*50)
                print(f"SUCCESS: Energy gap ({energy_gap:.6f}) is below threshold ({self.energy_gap_threshold:.6f}).")
                print("MECI search converged!")
                print("="*50)
                return {'mol': self.mol, 'R': current_R}

            self.sigma *= self.sigma_factor

        print("\n" + "="*50)
        print(f"FAILURE: Reached max outer loops ({self.max_outer_loops}) but energy gap ({energy_gap:.6f}) is still larger than threshold.")
        print("="*50)
        return {'mol': self.mol, 'R': current_R}

params = {
    'mol': mol,
    'initial_sigma': 3.5,            # 初始惩罚因子
    'sigma_factor': 2.0,              # sigma 每次增大的倍数
    'energy_gap_threshold': 0.00036,     # 最终能量差收敛目标 (Hartree)
    'max_outer_loops': 8,             # 最多尝试8次
    'alpha': 0.02,                    # 平滑参数
    'tol_step': 1e-6,                 # 内循环能量变化收敛阈值
    'tol_grad': 0.005                  # 内循环梯度收敛阈值
}

# 2. 创建优化器实例
optimizer = MECIOptimizer(**params)

# 3. 运行全自动搜索
final_meci = optimizer.find_meci()

from time import time
mol = final_meci['mol']
mf = mol.UKS(xc=fun)
t0 = time()
mf.kernel()
t1 = time()
print(f"SCF time: {t1 - t0:.2f} s")
td = mf.TDDFT_SF()
td.extype = 1
td.nstates = 5
td.collinear_samples = 20
td.positive_eig_threshold = -0.1
t2 = time()
td.kernel()
t3 = time()
print(f"TDDFT-SF time: {t3 - t2:.2f} s")
S = extract_state(mf, td, Smin=0.0, Smax=0.7)
T = extract_state(mf, td, Smin=1.7, Smax=2.9)
mol.tofile(f'{basis}_meci.xyz', format='xyz')
np.save(f'{basis}_times.npy', np.array([t1 - t0, t3 - t2]))
