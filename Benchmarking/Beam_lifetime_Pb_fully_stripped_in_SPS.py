#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating how fully stripped Pb ions behave in the SPS 
"""
import numpy as np
import pandas as pd 
from beam_gas_collisions import BeamGasCollisions, Data

# Load data 
projectile = 'Pb54'
data = Data()
gas_fractions = data.gas_fractions  
pressure_data = data.pressure_data 
projectile_data = data.projectile_data



# Calculate lifetimes and cross sections for PS
projectile_data_SPS = np.array([projectile_data['Z'][projectile],
                                projectile_data['SPS_q'][projectile],
                                projectile_data['SPS_Kinj'][projectile],
                                projectile_data['I_p'][projectile], 
                                projectile_data['n_0'][projectile], 
                                projectile_data['SPS_beta'][projectile]])

# Initiate PS beam-gas interactions - first test simple case with 100% hydrogen
full_hydrogen_array = np.zeros(len(gas_fractions['SPS'].values))
full_hydrogen_array[0] = 1.0

SPS_rest_gas =  BeamGasCollisions(pressure_data['SPS'].values[0],
                                     full_hydrogen_array)
SPS_rest_gas.set_projectile_data(projectile_data_SPS)  

# Print out the estimated beam-gas lifetime of Pb in the SPS
tau_SPS = SPS_rest_gas.calculate_total_lifetime_full_gas()
sigmas = SPS_rest_gas.return_all_sigmas()

print("\n---- Pb82+ in the SPS with only hydrogen: ----\n")
print("Total lifetime: {}".format(tau_SPS))
print("sigma_EL = {:.3e} m^2, \nsigma_EC = {:.3e} m^2".format(sigmas[0][0], sigmas[1][0]))

sigma_EL = SPS_rest_gas.calculate_sigma_electron_loss(1.0, 
                                                      82.0, 
                                                      82.0, 
                                                      projectile_data['SPS_Kinj'][projectile], 
                                                      projectile_data['I_p'][projectile],
                                                      projectile_data['n_0'][projectile],
                                                      SI_units=True)
print("\nDirectly calculated EL loss of Pb82+ on H: {:.3e} m^2".format(sigma_EL))