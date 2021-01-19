import numpy as np
import time
from pyscf import gto, scf, fci

"""
MP2 implementaion
This module transforms AO integrals to MO basis
Obtains RHF energy and MO coefficients from pyscf
and calculates second order Moller Plesset energy
FAST implementation
"""

mol = gto.M(
    atom = [['H', (0, 0, 0)],
            ['H', (1, 0, 0)],
            ['H', (2, 0, 0)],
            ['H', (3, 0, 0)],
            ['H', (4, 0, 0)],
            ['H', (5, 0, 0)],
            ],
    basis = 'sto-3g',
    verbose = 0)

num_bf = mol.nao_nr()
print(num_bf)

# overlap, kinetic, nuclear attraction
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
v = mol.intor('cint1e_nuc_sph')

# The one-electron part of the H is the sum of the kinetic and nuclear integrals
h = t + v

# 2e integrals (electron repulsion integrals, "eri")
eri = mol.intor('cint2e_sph')

# ERI is stored as [pq, rs] 2D matrix
# print ("ERI shape=", eri.shape)

# Reshape it into a [p,q,r,s] 4D array
eri = eri.reshape([num_bf,num_bf,num_bf,num_bf])
# print ("ERI shape=", eri.shape)
g2e_ao = eri


# Perform HF calculation to obtain orbital energies and MO orbital coefficients.
conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
print((conv, e, mo_e, mo, mo_occ))
#mo_e and mo aka, mo coefficints are what is needed.
coeff_mat = mo
orb_e = mo_e
ehf = e

#######################################################
#
# FUNCTIONS
#
#######################################################

#Perform integral transformation for AO basis to MO basis
def transform_integrals_einsum(ge2_ao, coeff_mat):
    """AO to MO integral transformation using einsum
    """
    start_time = time.clock()
    g2e_mo = np.einsum('PQRS, Pp->pQRS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pQRS, Qq->pqRS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pqRS, Rr->pqrS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pqrS, Ss->pqrs', g2e_ao, coeff_mat)
    print(time.clock() - start_time, "seconds")
    return g2e_mo

# print(transform_integrals_einsum(eri, coeff_mat))

##################################################
#
# Compute MP2 energy
#
##################################################
nocc = 1
g2e_mo = transform_integrals_einsum(g2e_ao, coeff_mat)
def compute_mp2_energy(num_bf, nocc, g2e_mo, orb_e, ehf):
    """
    Compute the MP2 energy

    num_bf :: number of basis functions
    nocc  :: number of occupied orbitals
    g2e_mo :: two-electron integrals in MO basis
    orb_e ::: orbital energies as obtained from HF SCF procedure
    ehf :: HF energy
    """
    start_time = time.clock()
    E = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, num_bf):
                for b in range(nocc, num_bf):
                    E += g2e_mo[i, a, j, b]*(2*g2e_mo[i, a, j, b] - g2e_mo[i, b, j, a])/\
                         (orb_e[i] + orb_e[j] - orb_e[a] - orb_e[b])

    print('MP2 correlation energy: {:20.15f}\n'.format(E))
    print('Total MP2 energy: {:20.15f}\n'.format(E + ehf))
    print(time.clock() - start_time, "seconds")
    return E

print('MP2 energy = ', compute_mp2_energy(num_bf, nocc, g2e_mo, orb_e, ehf))
print('HF energy = ', ehf)
