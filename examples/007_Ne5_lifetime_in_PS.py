"""
Script to compute Ne5+ ion lifetime in PS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from beam_gas_collisions import IonLifetimes, DataObject

"""
projectile_data : np.ndarray
    Z_p : atomic number Z of projectile
    A_p : mass number A of projectile
    q : charge of projectile.
    e_kin : collision energy in MeV/u.
    I_p : first ionization potential of projectile in keV
    n_0: principal quantum number
    m: mass of atom in Dalton (atomic unit)
"""

projectile_data_Ne5 = np.array([10., # Z_p : atomic number Z of projectile
                                20., # A_p : mass number A of projectile
                                5.,# q
                                0.0670880887510126*1e3, # in MeV/u, converted from GeV/u - for PS injection for Ne5+
                                1e-3 * 157.934,  # from NIST table
                                2,  # from NIST table
                                20.18 # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])

# Instantiate object and compute lifetime
PS_Ne5 = IonLifetimes(machine='PS')
PS_Ne5.set_projectile_data_manually(projectile_data_Ne5, gamma_is_provided=False)
tau_tot = PS_Ne5.calculate_total_lifetime_full_gas()    
print('\nExpected lifetime Ne5+: {:.5f} s'.format(tau_tot))

 