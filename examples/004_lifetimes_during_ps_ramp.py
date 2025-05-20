"""
Script to calculate lifetimes for Mg4+ and O4+ in PS for different energies
"""
import matplotlib.pyplot as plt
from beam_gas_collisions import IonLifetimes
import numpy as np

# Instantiate objects
ps_O = IonLifetimes(projectile='O4', machine='PS')
ps_Mg = IonLifetimes(projectile='Mg7', machine='PS')

O_mass_in_u_stripped = 15.9972
Mg_mass_in_u_stripped = 24.312
O_mass_in_eV = 15.022e9 
Mg_mass_in_eV = 22.586680e9 

# Define kinetic energies where to calculate the lifetime
Ek_per_u_vals_O = np.array([0.067, 0.2456, 2.9635, 5.723]) * 1e3 # from GeV/u to MeV/u
Ek_per_u_vals_Mg = np.array([0.090, 0.3226, 3.5782, 6.812]) * 1e3 # from GeV/u to MeV/u
tau_values_O = np.zeros(len(Ek_per_u_vals_O))
tau_values_Mg = np.zeros(len(Ek_per_u_vals_Mg))

B_strings = ['\n----- Injection energy -----\n', '\n----- Intermediate plateau -----\n', 
             '\n----- Halfway through ramp ----------\n', '\n----- Full ramp ----------\n']

# Loop over energies
for i in range(len(Ek_per_u_vals_O)):
    # Update kinetic energies but also gamma values
    
    # Update oxygen beta and gamma
    ps_O.e_kin = Ek_per_u_vals_O[i]
    ps_O.E_tot = O_mass_in_eV + 1e6*ps_O.e_kin * O_mass_in_u_stripped# total kinetic energy in eV per particle at injection
    ps_O.gamma = ps_O.E_tot/O_mass_in_eV
    ps_O.beta = np.sqrt(1 - 1/ps_O.gamma**2)
    
    ps_Mg.e_kin = Ek_per_u_vals_Mg[i]
    ps_Mg.E_tot = Mg_mass_in_eV + 1e6*ps_Mg.e_kin * Mg_mass_in_u_stripped# total kinetic energy in eV per particle at injection
    ps_Mg.gamma = ps_Mg.E_tot/Mg_mass_in_eV
    ps_Mg.beta = np.sqrt(1 - 1/ps_Mg.gamma**2)

    tau_values_O[i] = ps_O.calculate_total_lifetime_full_gas()
    tau_values_Mg[i] = ps_Mg.calculate_total_lifetime_full_gas()

    print(B_strings[i])

    print('--- O4+ --- ')
    print('Ek [MeV/u] = {:.5f}'.format(ps_O.e_kin))
    print('gamma = {:.5f}'.format(ps_O.gamma))
    print('Beam lifetime = {:.3f}'.format(tau_values_O[i]))

    print('\n--- Mg7+ ---')
    print('Ek [MeV/u] = {:.5f}'.format(ps_Mg.e_kin))
    print('gamma = {:.5f}'.format(ps_Mg.gamma))
    print('Beam lifetime = {:.3f}'.format(tau_values_Mg[i]))
    
# Small script to plot O lifetimes as a function of energy
eks = np.linspace(72., 6800)
taus_O = np.zeros(len(eks))
taus_Mg = np.zeros(len(eks))
for i, ek in enumerate(eks):
    ps_O.e_kin = ek
    ps_O.E_tot = O_mass_in_eV + 1e6*ps_O.e_kin * O_mass_in_u_stripped# total kinetic energy in eV per particle at injection
    ps_O.gamma = ps_O.E_tot/O_mass_in_eV
    ps_O.beta = np.sqrt(1 - 1/ps_O.gamma**2)
    taus_O[i] = ps_O.calculate_total_lifetime_full_gas()
    
    ps_Mg.e_kin = ek
    ps_Mg.E_tot = Mg_mass_in_eV + 1e6*ps_Mg.e_kin * Mg_mass_in_u_stripped# total kinetic energy in eV per particle at injection
    ps_Mg.gamma = ps_Mg.E_tot/Mg_mass_in_eV
    ps_Mg.beta = np.sqrt(1 - 1/ps_Mg.gamma**2)
    taus_Mg[i] = ps_Mg.calculate_total_lifetime_full_gas()
    
fig, ax = plt.subplots(1,1, figsize=(6, 4), constrained_layout=True)
ax.plot(eks, taus_O, marker='o', label='O')
ax.plot(eks, taus_Mg, marker='o', label='Mg')
ax.set_ylabel('Tau [s]')
ax.set_xlabel('Ekin [MeV/u]')
ax.grid(alpha=0.45)
ax.legend(fontsize=13)
ax.set_xscale('log')
plt.show()