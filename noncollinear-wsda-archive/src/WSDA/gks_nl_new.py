#!/usr/bin/env python
'''
Module Name: gks_nl_new.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240706
Last Modified Date: 20240814

Description: generalized KS DFT for multicollinear nonlocal WSDA functional
'''

import numpy
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import gks
from nl_utils import numint2c_nl_gks, numint2c_nl_gks_nscf, numint2c_nl_gks_nc

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = isinstance(dm, numpy.ndarray) and dm.ndim == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n1, ex, vx = numint2c_nl_gks(mol, ks.grids, dm, epsilon=ks.epsilon,
                                     max_memory=max_memory, nproc=ks.nproc,
                                     spin_samples=ks.spin_samples,
                                     hermi=hermi)
        n2, ec, vc = ni.get_vxc(mol, ks.grids, ks.xc, dm,
                                hermi=hermi, max_memory=max_memory)
        n = tuple((np.array(n1) + np.array(n2)) * 0.5)
        exc = ex + ec
        vxc = vx + vc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    vk = None
    if (ks._eri is None and ks.direct_scf and
        getattr(vhf_last, 'vj', None) is not None):
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj = ks.get_j(mol, ddm, hermi)
        vj += vhf_last.vj
    else:
        vj = ks.get_j(mol, dm, hermi)
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

class GKSMCNL(gks.GKS):
    def __init__(self, mol, xc=',VWN', nproc=None, epsilon=1e-11,
                 spin_samples=266):
        if nproc is None:
            nproc = lib.num_threads()
        super().__init__(mol, xc)
        self._nproc = nproc
        self._epsilon = epsilon
        self._spin_samples = spin_samples
        self.collinear = 'mcol'
        self.grids.prune = None

    @property
    def nproc(self):
        return self._nproc
    @nproc.setter
    def nproc(self, value):
        self._nproc = value

    @property
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    get_veff = get_veff

def get_veff_nscf(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = isinstance(dm, numpy.ndarray) and dm.ndim == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n1, ex, vx = numint2c_nl_gks_nscf(mol, ks.grids, dm, epsilon=ks.epsilon,
                                     max_memory=max_memory, nproc=ks.nproc,
                                     spin_samples=ks.spin_samples,
                                     hermi=hermi)
        n2, ec, vc = ni.get_vxc(mol, ks.grids, ks.xc, dm,
                                hermi=hermi, max_memory=max_memory)
        n = (n1 + n2) * 0.5
        exc = ex + ec
        vxc = vx + vc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    vk = None
    if (ks._eri is None and ks.direct_scf and
        getattr(vhf_last, 'vj', None) is not None):
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj = ks.get_j(mol, ddm, hermi)
        vj += vhf_last.vj
    else:
        vj = ks.get_j(mol, dm, hermi)
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

class GKSMCNL_nscf(gks.GKS):
    def __init__(self, mol, xc=',VWN', nproc=None, epsilon=1e-11,
                 spin_samples=266):
        if nproc is None:
            nproc = lib.num_threads()
        super().__init__(mol, xc)
        self._nproc = nproc
        self._epsilon = epsilon
        self._spin_samples = spin_samples
        self.collinear = 'mcol'
        self.grids.prune = None

    @property
    def spin_samples(self):
        return self._spin_samples
    @spin_samples.setter
    def spin_samples(self, value):
        self._spin_samples = int(value) 

    @property
    def nproc(self):
        return self._nproc
    @nproc.setter
    def nproc(self, value):
        self._nproc = value

    @property
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    get_veff = get_veff_nscf


def get_veff_nc(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = isinstance(dm, numpy.ndarray) and dm.ndim == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n1, ex, vx = numint2c_nl_gks_nc(mol, ks.grids, dm, epsilon=ks.epsilon,
                                     max_memory=max_memory, nproc=ks.nproc,
                                     hermi=hermi)
        n2, ec, vc = ni.get_vxc(mol, ks.grids, ks.xc, dm,
                                hermi=hermi, max_memory=max_memory)
        n = (n1 + n2) * 0.5
        exc = ex + ec
        vxc = vx + vc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    vk = None
    if (ks._eri is None and ks.direct_scf and
        getattr(vhf_last, 'vj', None) is not None):
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj = ks.get_j(mol, ddm, hermi)
        vj += vhf_last.vj
    else:
        vj = ks.get_j(mol, dm, hermi)
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

class GKSNCNL(gks.GKS):
    def __init__(self, mol, xc=',VWN', nproc=None, epsilon=1e-11):
        if nproc is None:
            nproc = lib.num_threads()
        super().__init__(mol, xc)
        self._nproc = nproc
        self._epsilon = epsilon
        self.collinear = 'ncol'
        self.grids.prune = None

    @property
    def nproc(self):
        return self._nproc
    @nproc.setter
    def nproc(self, value):
        self._nproc = value

    @property
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    get_veff = get_veff_nc

if __name__=='__main__':
    from pyscf import gto
    mol = gto.M(atom='''
    B 0.0 0.0 0.0''',
    basis='6-31G', charge=0, spin=1)
    mf1 = mol.GKS(xc='HF,')
    mf1.init_guess = np.zeros((2*mol.nao_nr(), 2*mol.nao_nr()))
    mf1.kernel()
    print(mf1.get_veff().exc)
    # print(mf1.make_rdm1())

    mf = GKSMCNL(mol, xc='', nproc=32, epsilon=1e-11)
    mf.collinear = 'mcol'
    mf.spin_samples = 26
    mf.grids.level = 0
    mf.max_memory = 40000
    print(mf.get_veff(dm=mf1.make_rdm1()).exc)