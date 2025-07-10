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

ind0 = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12]
x0 = np.arange(len(x[ind0]))

# Convert to numpy arrays, in emittance growth um rad per minute
deX_LEIR = 1e6*np.array(exyn_dict['X']['LEIR'])[ind0]
deX_PS = 1e6*np.array(exyn_dict['X']['PS'])[ind0]
deX_SPS = 1e6*np.array(exyn_dict['X']['SPS'])[ind0]
deY_LEIR = 1e6*np.array(exyn_dict['Y']['LEIR'])[ind0]
deY_PS = 1e6*np.array(exyn_dict['Y']['PS'])[ind0]
deY_SPS = 1e6*np.array(exyn_dict['Y']['SPS'])[ind0]

bar_width = 0.21
mini_fontsize = 9.5
fig, (ax, ax2) = plt.subplots(2, 1, figsize = (9, 5.4), sharex=True, constrained_layout=True)
bar1 = ax.bar(x0 - 1.15*bar_width, deX_LEIR, bar_width, color='cyan', label='LEIR') #
bar2 = ax.bar(x0, deX_PS, bar_width, color='red', label='PS') #
bar3 = ax.bar(x0 + 1.15*bar_width, deX_SPS, bar_width, color='limegreen', label='SPS') #
ax.bar_label(bar1, labels=[f'{e:,.1e}'.replace('+0', '') for e in deX_LEIR], padding=3, color='black', fontsize=mini_fontsize) 
PS_labels_X=[f'{e:,.1e}'.replace('+0', '') for e in deX_PS]
SPS_labels_X = [f'{e:,.1e}'.replace('+0', '') for e in deX_SPS]
SPS_labels_X[5] = '' # no space otherwise
ax.bar_label(bar2, labels=PS_labels_X, padding=3, color='black', fontsize=mini_fontsize) 
ax.bar_label(bar3, labels=SPS_labels_X, padding=3, color='black', fontsize=mini_fontsize) 
bar12 = ax2.bar(x0 - 1.15*bar_width, deY_LEIR, bar_width, color='cyan', label='LEIR') #
bar22 = ax2.bar(x0, deY_PS, bar_width, color='red', label='PS') #
bar32 = ax2.bar(x0 + 1.15*bar_width, deY_SPS, bar_width, color='limegreen', label='SPS') #
ax2.bar_label(bar12, labels=[f'{e:,.1e}'.replace('+0', '') for e in deY_LEIR], padding=3, color='black', fontsize=mini_fontsize) 

PS_labels_Y=[f'{e:,.1e}'.replace('+0', '') for e in deY_PS]
PS_labels_Y[5] = '' # no space otherwise
ax2.bar_label(bar22, labels=PS_labels_Y, padding=2, color='black', fontsize=8.5) 
ax2.bar_label(bar32, labels=[f'{e:,.1e}'.replace('+0', '') for e in deY_SPS], padding=3, color='black', fontsize=mini_fontsize) 
for a in [ax, ax2]:
    a.set_yscale('log')
    a.set_xticks(x0)
    a.grid(alpha=0.45)
ax.set_ylabel(r"$d\varepsilon^{n}_{x}/dt$ [mm mrad/s]", fontsize=17)
ax2.set_ylabel(r"$d\varepsilon^{n}_{y}/dt$ [mm mrad/s]", fontsize=17)
ax.set_ylim(3e-4, 1e-2)
ax2.set_ylim(3e-4, 1e-2)
ax2.set_xticklabels(np.array(latex_labels)[ind0])
ax.legend(fontsize=11, ncol=2)
fig.savefig('plots_and_output/LEIR_PS_SPS_emittance_growth_plot.png', dpi=350)
plt.show()