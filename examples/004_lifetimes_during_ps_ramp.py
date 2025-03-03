"""
Script to calculate lifetimes for Mg4+ and O4+ in PS for different energies
"""
from beam_gas_collisions import IonLifetimes
import numpy as np

# Instantiate objects
ps_O = IonLifetimes(projectile='O4', machine='PS')
ps_Mg = IonLifetimes(projectile='Mg7', machine='PS')

# Define kinetic energies where to calculate the lifetime
Ek_per_u_vals_O = np.array([0.067, 0.2456, 2.9635]) * 1e3 # from GeV/u to MeV/u
Ek_per_u_vals_Mg = np.array([0.090, 0.3226, 3.5782]) * 1e3 # from GeV/u to MeV/u
tau_values_O = np.zeros(len(Ek_per_u_vals_O))
tau_values_Mg = np.zeros(len(Ek_per_u_vals_Mg))

B_strings = ['\n----- Injection energy -----\n', '\n----- Intermediate plateau -----\n', 
             '\n----- Halfway through ramp ----------\n']

# Loop over energies
for i in range(len(Ek_per_u_vals_O)):
    # Update kinetic energies
    ps_O.e_kin = Ek_per_u_vals_O[i]
    ps_Mg.e_kin = Ek_per_u_vals_Mg[i]

    tau_values_O[i] = ps_O.calculate_total_lifetime_full_gas()
    tau_values_Mg[i] = ps_Mg.calculate_total_lifetime_full_gas()

    print(B_strings[i])

    print('--- O4+ --- ')
    print('Ek [MeV/u] = {:.5f}'.format(ps_O.e_kin))
    print('Beam lifetime = {:.3f}'.format(tau_values_O[i]))

    print('\n--- Mg7+ ---')
    print('Ek [MeV/u] = {:.5f}'.format(ps_Mg.e_kin))
    print('Beam lifetime = {:.3f}'.format(tau_values_Mg[i]))
