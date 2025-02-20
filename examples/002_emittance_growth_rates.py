"""
Calculate emittance growth rates for different ions across accelerators
"""
from beam_gas_collisions import IonLifetimes
import matplotlib.pyplot as plt
import numpy as np

# Initialize for different ions
ions = ['Pb54', 'O8', 'Mg7']
machines = ['LEIR', 'PS', 'SPS']

growth_rates = np.zeros((len(ions), len(machines)))

for i, ion in enumerate(ions):
    for j, machine in enumerate(machines):
        beam = IonLifetimes(projectile=ion, machine=machine)
        growth_rates[i,j] = beam.calculate_emittance_growth_rate()

# Plot results
plt.figure(figsize=(10,6))
for i, ion in enumerate(ions):
    plt.semilogy(machines, growth_rates[i,:], 'o-', label=ion)
    
plt.xlabel('Accelerator')
plt.ylabel('Emittance growth rate [m rad/s]')
plt.legend()
plt.grid(True)
plt.title('Emittance Growth Rates Across CERN Complex')
plt.savefig('plots_and_output/emittance_growth_rates.png') 