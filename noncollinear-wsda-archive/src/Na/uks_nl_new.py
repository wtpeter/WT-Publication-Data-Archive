#!/usr/bin/env python
'''
Module Name: uks_nl_new.py
Author: Tai Wang <wtpeter@pku.edu.cn>
Create Date: 20240704
Last Modified Date: 20240704

Description: unrestriced KS DFT for nonlocal WSDA functional
'''

import numpy
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import uks
from nl_utils import numint_nl_uks

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    ni = ks._numint
    if hermi == 2:
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n1, ex, vx = numint_nl_uks(mol, ks.grids, dm, epsilon=ks.epsilon,
                                  max_memory=max_memory, nproc=ks.nproc)
        n2, ec, vc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        n = tuple((np.array(n1) + np.array(n2)) * 0.5)
        exc = ex + ec
        vxc = vx + vc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    vk = None
    if (ks._eri is None and ks.direct_scf and
        getattr(vhf_last, 'vj', None) is not None):
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
        vj += vhf_last.vj
    else:
        vj = ks.get_j(mol, dm[0]+dm[1], hermi)
    vxc += vj

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

class UKSNL(uks.UKS):
    def __init__(self, mol, xc=',VWN', nproc=None, epsilon=1e-12):
        if nproc is None:
            nproc = lib.num_threads()
        super().__init__(mol, xc)
        self._nproc = nproc
        self._epsilon = epsilon

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

    def dump_flags(self, verbose=None):
        uks.UKS.dump_flags(self, verbose)
        return self

    get_veff = get_veff

if __name__=='__main__':
    from pyscf import gto
    mol = gto.M(atom='''
    Li 0.0 0.0 0.0''',
    basis='6-31G', charge=0, spin=1)
    mf1 = mol.UKS(xc='HF,')
    mf1.kernel()
    # print(mf1.get_veff())
    mf = UKSNL(mol, xc='', nproc=32, epsilon=1)
    mf.grids.level = 3
    print(mf.get_veff(dm=mf1.make_rdm1()).exc)
    # # mf.print_nproc()
    # print(mf.nproc)
    # mf.grids.level = 0
    # mf.max_memory = 50000
    # mf.kernel()