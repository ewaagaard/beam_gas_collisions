#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating beam lifetime as compared to Shevelko et al (2018) in FAIR: https://www.sciencedirect.com/science/article/pii/S0168583X1830096X
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

#### Rest gas composition of  ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'He', 'O2', 'Ar'] ####
gas_fractions = np.array([0.758, 0.049, 0.026, 0.119, 0.002, 0.034, 0.004, 0.008])
p = 1e-10
ion_beam_U28 = beam_gas_collisions(p, gas_fractions)

#### Projectile_data ####
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 