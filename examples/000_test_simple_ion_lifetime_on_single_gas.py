"""
Simple example to calculate lifetime of simple ion projectile in PS, on single gas
"""
from beam_gas_collisions import IonLifetimes

# First calculate Pb lifetime on argon gas in PS
PS = IonLifetimes(projectile='Pb54', machine='PS')
tau_Ar_PS, sigma_EL_PS, sigma_EC_PS = PS.calculate_lifetime_on_single_gas(p=5e-10, Z_t=18)
print('PS: Pb54 lifetime on 100% Ar at p = {} mbar --> {:.3f} seconds'.format(5e-10, tau_Ar_PS))

# Calculate Pb54 full lifetime on standard residual gas in PS
tau_PS = PS.calculate_total_lifetime_full_gas()
print('PS: Pb54 lifetime on full rest gas composition --> {:.3f} seconds\n'.format(tau_PS))

# Check corresponding full lifetime in LEIR and SPS
LEIR = IonLifetimes(projectile='Pb54', machine='LEIR')
tau_LEIR = LEIR.calculate_total_lifetime_full_gas()

SPS = IonLifetimes(projectile='Pb54', machine='SPS')
tau_SPS = SPS.calculate_total_lifetime_full_gas()

print('Pb54 lifetime on full rest gas composition:\nLEIR = {:.3f} seconds\nSPS = {:.3e} seconds'.format(tau_LEIR, tau_SPS))