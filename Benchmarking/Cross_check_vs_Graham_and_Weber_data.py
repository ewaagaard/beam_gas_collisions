#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking cross sections from semi-empirical formulae with previous data

G. Weber provided some data from 
Graham et al., PHYSICAL REVIEW A VOLUME 30, NUMBER 2 AUGUST 1984 
and PHYSICAL REVIEW SPECIAL TOPICS - ACCELERATORS AND BEAMS 18, 034403 (2015) for different charge states.
"""
import numpy as np
import sys
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

# Input data as arrays
Z_ref = np.array([82., 82., 82., 82., 82., 82.])  # projectile 
q_ref = np.array([52., 54., 59., 55., 54., 54.])
Ip_ref = np.array([3.2352, 5.414, 6.0252, 5.5778, 5.414, 5.414])  # ionization potential in keV
n0_ref = np.array([4., 3., 3., 3., 3., 3.])  # principal quantum number 
e_kin_ref = np.array([4.66, 4.66, 4.66, 5.9, 4.66, 4.66])  # in MeV/u
Z_t_ref = np.array([1., 1., 1., 7., 2., 7.]) 
sigma_measured_ref = np.array([1.75e-19, 1.25e-19, 0.7e-19, 4.3E-19, 2.8E-19, 1.6E-18])

# iterate over the different values
for i, Z in enumerate(Z_ref):
    
        projectile_data_Pb81 = np.array([Z,
                                         q_Pb81,  
                                         E_kin*1e3,  # in MeV/u
                                         Ip_Pb81, 
                                         n0_Pb81,
                                         beta_Pb81
                                         ])
    
    # First calculate on hydrogen - need to account for 2 hydrogen atoms per H2 molecule
    if Z_t_ref[i] == 1:

        sigmas_calculated_ref = 2*DuBois(Z_t_ref[i], Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], par) \
        *Shevelko(Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], n0_ref[i], par)
        print("Calculated vs measured sigma Pb{}+ on H2: {:.4e} cm^2   vs   {:.4e} cm^2".format(
                                                                                    int(q_ref[i]), \
                                                                                    sigmas_calculated_ref, \
                                                                                    sigma_measured_ref[i]))
    # Check helium - a single atom 
    elif Z_t_ref[i] == 2:
        sigma_calculated_ref = DuBois(Z_t_ref[i], Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], par) \
        *Shevelko(Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], n0_ref[i], par)
        print("Calculated vs measured sigma Pb{}+ on He: {:.4e} cm^2   vs   {:.4e} cm^2".format(
                                                                                    int(q_ref[i]), \
                                                                                    sigmas_calculated_ref, \
                                                                                    sigma_measured_ref[i]))    
    # If not hydrogen or helium, check nitrogen - also molecular composition 
    elif Z_t_ref[i] == 7:
        sigma_calculated_ref = 2*DuBois(Z_t_ref[i], Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], par) \
        *Shevelko(Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], n0_ref[i], par)
        print("Calculated vs measured sigma Pb{}+ on N2: {:.4e} cm^2   vs   {:.4e} cm^2".format(
                                                                                    int(q_ref[i]), \
                                                                                    sigmas_calculated_ref, \
                                                                                    sigma_measured_ref[i])) 