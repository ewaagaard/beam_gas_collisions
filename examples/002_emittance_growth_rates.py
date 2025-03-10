"""
Calculate emittance growth rates for different ions across accelerators
"""
from beam_gas_collisions import BeamGasCollisions
import matplotlib.pyplot as plt
import numpy as np
import re

beam_gas = BeamGasCollisions()
exyn_dict = beam_gas.calculate_full_elastic_emittance_growth()

# Function to convert to LaTeX-style charge states
def convert_to_charge_state(label):
    # Split the label into element and charge state using regex
    match = re.match(r"([A-Za-z]+)(\d+)", label)
    if match:
        element = match.group(1)
        charge = match.group(2)
        # Return the element and the LaTeX superscript for charge
        return f"{element}$^{{{charge}+}}$"

######## PLOT THE DATA ###########
x = np.arange(len(exyn_dict['Projectile']))
latex_labels = [convert_to_charge_state(label) for label in exyn_dict['Projectile'] ]

# Convert to numpy arrays, in emittance growth um rad per minute
deX_LEIR = 1e6*np.array(exyn_dict['X']['LEIR'])
deX_PS = 1e6*np.array(exyn_dict['X']['PS'])
deX_SPS = 1e6*np.array(exyn_dict['X']['SPS'])
deY_LEIR = 1e6*np.array(exyn_dict['Y']['LEIR'])
deY_PS = 1e6*np.array(exyn_dict['Y']['PS'])
deY_SPS = 1e6*np.array(exyn_dict['Y']['SPS'])

bar_width = 0.21
mini_fontsize = 8.5
fig, (ax, ax2) = plt.subplots(2, 1, figsize = (11, 6.8), sharex=True, constrained_layout=True)
bar1 = ax.bar(x - 1.15*bar_width, deX_LEIR, bar_width, color='cyan', label='LEIR') #
bar2 = ax.bar(x, deX_PS, bar_width, color='red', label='PS') #
bar3 = ax.bar(x + 1.15*bar_width, deX_SPS, bar_width, color='limegreen', label='SPS') #
ax.bar_label(bar1, labels=[f'{e:,.1e}'.replace('+0', '') for e in deX_LEIR], padding=3, color='black', fontsize=mini_fontsize) 
ax.bar_label(bar2, labels=[f'{e:,.1e}'.replace('+0', '') for e in deX_PS], padding=3, color='black', fontsize=mini_fontsize) 
ax.bar_label(bar3, labels=[f'{e:,.1e}'.replace('+0', '') for e in deX_SPS], padding=3, color='black', fontsize=mini_fontsize) 
bar12 = ax2.bar(x - 1.15*bar_width, deY_LEIR, bar_width, color='cyan', label='LEIR') #
bar22 = ax2.bar(x, deY_PS, bar_width, color='red', label='PS') #
bar32 = ax2.bar(x + 1.15*bar_width, deY_SPS, bar_width, color='limegreen', label='SPS') #
ax2.bar_label(bar12, labels=[f'{e:,.1e}'.replace('+0', '') for e in deY_LEIR], padding=3, color='black', fontsize=mini_fontsize) 
ax2.bar_label(bar22, labels=[f'{e:,.1e}'.replace('+0', '') for e in deY_PS], padding=3, color='black', fontsize=mini_fontsize) 
ax2.bar_label(bar32, labels=[f'{e:,.1e}'.replace('+0', '') for e in deY_SPS], padding=3, color='black', fontsize=mini_fontsize) 
for a in [ax, ax2]:
    a.set_yscale('log')
    a.set_xticks(x)
    a.grid(alpha=0.55)
ax.set_ylabel(r"$d\varepsilon^{n}_{x}/dt$ [urad/s]", fontsize=18)
ax2.set_ylabel(r"$d\varepsilon^{n}_{x}/dt$ [urad/s]", fontsize=18)
ax2.set_xticklabels(latex_labels)
ax.legend()
fig.savefig('plots_and_output/LEIR_PS_SPS_emittance_growth_plot.png', dpi=250)
plt.show()

'''
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
'''