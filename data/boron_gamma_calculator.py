"""
Quick estimate of 5^B^11 gamma from different injection energies 
"""
import sys
import numpy as np
sys.path.append('..')

from beam_gas_collisions import beam_gas_collisions

# Information for boron
Z_p = 5
q = 3
m_ion = 11.00930536 # in u
I_p = 0.2593715 #in keV, of B3+
n_0 = 1
MeV_per_u = np.array([4.2, 20.657, 10915.40]) # LEIR, PS, SPS injection energies, from Reyes notebook 

# Instantiate object
rest_gas = beam_gas_collisions() 

for e_kin in MeV_per_u:
    projectile_data = [Z_p, q, e_kin, I_p, n_0, m_ion]
    rest_gas.set_projectile_data(projectile_data, provided_beta=False)
    print(rest_gas.beta)