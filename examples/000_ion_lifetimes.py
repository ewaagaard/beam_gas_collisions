"""
Calculate beam lifetimes and cross sections from beam-gas interactions
for different ions and machines
"""
import numpy as np
import matplotlib.pyplot as plt
from beam_gas_collisions import IonLifetimes, DataObject
import os

def calculate_single_gas_lifetime(ion, machine, gas_Z, pressure):
    """Calculate lifetime for a single ion on a single gas species"""
    beam = IonLifetimes(projectile=ion, machine=machine)
    tau, sigma_EL, sigma_EC = beam.calculate_lifetime_on_single_gas(
        p=pressure, Z_t=gas_Z)
    print(f'{machine}: {ion} lifetime on Z={gas_Z} at p={pressure} mbar -> {tau:.3f} seconds')
    print(f'Cross sections: σ_EL = {sigma_EL:.2e} m^2, σ_EC = {sigma_EC:.2e} m^2\n')
    return tau

def calculate_full_gas_lifetimes(ions, machines):
    """Calculate lifetimes for multiple ions across different machines"""
    results = {}
    for ion in ions:
        results[ion] = {}
        for machine in machines:
            beam = IonLifetimes(projectile=ion, machine=machine)
            tau = beam.calculate_total_lifetime_full_gas()
            results[ion][machine] = tau
            print(f'{machine}: {ion} lifetime on full gas -> {tau:.3f} seconds')
        print()
    return results

def main():
    os.makedirs('plots_and_output', exist_ok=True)
    
    # Single gas calculations
    print("=== Single Gas Calculations ===")
    calculate_single_gas_lifetime('Pb54', 'PS', gas_Z=18, pressure=5e-10)
    
    # Full gas calculations
    print("=== Full Gas Calculations ===")
    ions = ['Pb54', 'O8', 'Mg7']
    machines = ['LEIR', 'PS', 'SPS']
    results = calculate_full_gas_lifetimes(ions, machines)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for ion in ions:
        lifetimes = [results[ion][m] for m in machines]
        plt.semilogy(machines, lifetimes, 'o-', label=ion)
    
    plt.xlabel('Accelerator')
    plt.ylabel('Beam Lifetime [s]')
    plt.title('Ion Beam Lifetimes Across CERN Complex')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots_and_output/beam_lifetimes.png')

if __name__ == '__main__':
    main() 