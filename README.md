# Beam Gas Collisions

The present ion beams in the CERN accelerator complex suffer from heavy losses due to various effects, including interactions with residual gas molecules in the beam pipe. These beam-gas processes change the charge state of the ions, causing the altered ions to fall outside the accelerator acceptance. Assuming equivalent pressure in the accelerator and exponential decay, 
the beam intensity $I$ can be modelled with

$$
I(t) = I(t_0)\times\exp\bigg(- \frac{t}{\tau}\bigg),  \qquad \textrm{where} \quad \tau = \frac{1}{\sigma\,n\,\beta \, c }
$$

and $\sigma$ is the rest gas collision cross section, $\beta$ is the projectile relativistic beta factor, $c$ is the speed of light and $n$ is the molecular density in the beam pipe. The dominant contributions to the cross section $\sigma$ at PS and SPS ion beam energies are **electron capture (EC)** via pair production and **electron loss (EL)** via electron and nucleus impact ionization. Electron capture can be modelled with the [Schlachter formula](https://link.aps.org/doi/10.1103/PhysRevA.27.3372). We estimate the electron loss cross section with a semi-empirical formula combining studies from [Dubois](https://link.aps.org/doi/10.1103/PhysRevA.84.022702) and [Shevelko](https://www.sciencedirect.com/science/article/pii/S0168583X11003272).

The `beam_gas_collisions` class to represent the beam-gas collisions contains the structures and parameters to calculate the EC and EL cross sections. Assuming that no other charge-changing processes are relevant at these energies, also the beam lifetimes $\tau$ can be calculated. The class can be initiated without any input paramters, or providing pressure `p` in mbar and the molecular fractions of H2, H2O, CO, CH4, CO2, He, O2 and Ar to directly find the molecular density of each compound in the accelerator.  

### Instantiating the class

```python
import pandas as pd 
from beam_gas_collisions import beam_gas_collisions

# Load gas data
gas_fractions = pd.read_csv('Data/Gas_fractions.csv', index_col=0)
pressure_data = pd.read_csv('Data/Pressure_data.csv', index_col=0).T

# Instantiate PS class object
PS_rest_gas =  beam_gas_collisions(pressure_data['PS'].values[0], gas_fractions['PS'].values)
```
The class object can also be instantiated without providing pressure or gas to calculate single cross sections or lifetimes on a particular gas. The molecular densities and pressure (in mbar) can then be instantiated later:

```python
PS_rest_gas =  beam_gas_collisions()
PS_rest_gas.set_molecular_densities(gas_fractions['PS'].values, p = 5e-10)
```

### Calculating the cross sections

The respective EL and EC cross sections can be calculated from
```python
sigma_EL = PS_rest_gas.calculate_sigma_electron_loss(Z, Z_p, q, e_kin, I_p, n_0)
sigma_EC = PS_rest_gas.calculate_sigma_electron_captture(Z, q, e_kin_keV)
```
where `Z` is the $Z$ of the target atom, `Z_p` is the Z of the projectile, `q` is the charge of projectile, `e_kin` is the collision energy in MeV/u, ` e_kin_keV` is the collision energy in keV/u, `I_p` is the first ionization potential of projectile in keV and `n_0` is principle quantum number of outermost projectile electron. `n_0` and `I_0` for each ion can be found on the [NIST database](https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Pb&submit=Retrieve+Data&units=1&format=0&order=0&at_num_out=on&sp_name_out=on&ion_charge_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&ion_conf_out=on&e_out=0&unc_out=on&biblio=on).

### Calculating the full lifetime

Once the projectile data has been provided and the molecular composition have been set-up, the total lifetime $\tau$ of the projectile can be estimated. The projectile data is a vector, either providing `projectile_data = Z_p, q, e_kin, I_p, n_0, beta` (directly providing the relativistic $\beta$) 

```python
projectile_data = [Z_p, q, e_kin, I_p, n_0, beta]
PS_rest_gas.set_projectile_data(projectile_data)
tau = PS_rest_gas.calculate_total_lifetime_full_gas()
```

or providing the mass in Dalton and calculate the beta: `projectile_data = Z_p, q, e_kin, I_p, n_0, atomic_mass_in_u`:

```python
projectile_data = [Z_p, q, e_kin, I_p, n_0, atomic_mass_in_u]
PS_rest_gas.set_projectile_data(projectile_data, provided_beta=False)
tau = PS_rest_gas.calculate_total_lifetime_full_gas()
```

### Calculating the beam lifetime interacting with a single gas

Also the ion beam lifetime interacting with a single gas can be calculated, specifying the pressure and $Z$ of the target gas. 

```python
PS_rest_gas =  beam_gas_collisions()
PS_rest_gas.set_projectile_data([Z_p, q, e_kin, I_p, n_0, beta])  
taus_Ar = PS_rest_gas.calculate_lifetime_on_single_gas(p=5e-10, Z_t=19)
```

### Cross sections across the CERN accelerator complex 

In the module `LEIR_PS_SPS_lifetimes.py`, all the cross sections and estimated total lifetimes are calculated for various ions, as shown in these plots below. 

![Cross_sections_on_H2](https://github.com/ewaagaard/Beam-gas-collisions/assets/68541324/a6a61af0-a1d5-46cd-b74f-6bd805fdac8e)

![LEIR_PS_SPS_full_lifetime_plot_compact](https://github.com/ewaagaard/Beam-gas-collisions/assets/68541324/9820a9e1-3dd3-49d3-b28c-77bcde2decd3)

