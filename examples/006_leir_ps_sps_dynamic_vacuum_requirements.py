"""
Main script to compute dynamic vacuum requirements for all projectiles in LEIR, PS and SPS 
similar to E. Mahner (2007) 'The Vacuum System of the Low Energy Ion Ring at CERN' in http://cds.cern.ch/record/902810
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
import scipy.constants as constants
from beam_gas_collisions import IonLifetimes, DataObject

# --- Configuration ---
OUTPUT_DIR = 'plots_and_output_partial_pressure'
TARGET_LIFETIME_S = 30.0  # Target lifetime for the table (Part a)
LIFETIME_RANGE_S = np.logspace(0, 2, 20) # 1s to 100s range for plots (Part b)
TARGET_GASES = ['H2', 'CH4', 'CO', 'CO2']
TEMPERATURE_K = 298 # Default temperature from IonLifetimes, ensure consistency

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function (calculates pressure from sigma) ---
def calculate_allowed_partial_pressure_mbar(tau_required_s, sigma_tot_gas_m2, beta, c_light, K, T):
    """Calculates the allowed partial pressure (mbar) for a single gas."""
    if sigma_tot_gas_m2 <= 0 or beta <= 0 or tau_required_s <= 0:
        return np.inf
    # Calculate required density n = 1 / (tau * sigma * beta * c)
    n_required = 1.0 / (tau_required_s * sigma_tot_gas_m2 * beta * c_light)
    # Calculate pressure p = n * K * T (in Pascal)
    p_pascal = n_required * K * T
    # Convert Pascal to mbar (1 Pa = 0.01 mbar)
    p_mbar = p_pascal * 1e-2
    return p_mbar

# --- Main Logic ---
data = DataObject() # Load base data once
all_projectiles = data.projectile_data.index.values
machines = ['LEIR', 'PS', 'SPS']

results_table_a = [] # To store data for the CSV file (Part a)

# Use a single IonLifetimes object and update it
# Initialize with arbitrary valid values, will be overwritten in loop
lifetime_calculator = IonLifetimes(projectile='Pb54', machine='LEIR', T=TEMPERATURE_K)

# Physical constants
K_boltzmann = constants.Boltzmann
c_light_speed = constants.c

print("Starting calculations...")

for machine in machines:
    print(f"--- Processing Machine: {machine} ---")

    for projectile in all_projectiles:
        print(f"  Processing Projectile: {projectile}")

        # --- Update IonLifetimes object for current machine/projectile ---
        # This sets up self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, self.beta, etc.
        try:
             # Use the existing method to load machine/projectile specific data
            lifetime_calculator.set_projectile_and_machine(projectile=projectile, machine=machine, p=None, molecular_fraction_array=None)
            lifetime_calculator.T = TEMPERATURE_K # Ensure temperature is correct
        except KeyError:
             print(f"    WARNING: Data likely missing for {projectile} in {machine}. Skipping.")
             continue
        except Exception as e:
            print(f"    Skipping {projectile} in {machine} due to error during setup: {e}")
            continue # Skip if data is missing or causes error

        # --- Calculate EL/EC Cross Sections using the new class method ---
        try:
            molecular_sigmas = lifetime_calculator.get_molecular_cross_sections_for_gases(TARGET_GASES)
            # Verify that all target gases were returned
            if not all(gas in molecular_sigmas for gas in TARGET_GASES):
                 missing_gases = [gas for gas in TARGET_GASES if gas not in molecular_sigmas]
                 print(f"    WARNING: Cross sections missing for gases {missing_gases} for {projectile} in {machine}. Skipping this projectile.")
                 continue

        except AttributeError:
             print("ERROR: The 'get_molecular_cross_sections_for_gases' method was not found.")
             print("Please ensure your IonLifetimes class in beam_gas_collisions.py is updated.")
             exit() # Stop the script if the method is missing
        except Exception as e:
            print(f"    Error calculating cross sections for {projectile} in {machine}: {e}. Skipping.")
            continue


        # --- a) Calculate allowed pressure for fixed lifetime & store ---
        partial_pressures_a = {}
        for gas in TARGET_GASES:
            sigma_tot = molecular_sigmas[gas]['Total']
            p_allowed = calculate_allowed_partial_pressure_mbar(
                TARGET_LIFETIME_S,
                sigma_tot,
                lifetime_calculator.beta,
                c_light_speed,
                K_boltzmann,
                lifetime_calculator.T
            )
            partial_pressures_a[gas] = p_allowed
            results_table_a.append({
                'Machine': machine,
                'Projectile': projectile,
                'Charge': int(lifetime_calculator.q),
                'Target Gas': gas,
                f'Allowed Pressure (mbar) for {TARGET_LIFETIME_S}s': p_allowed
            })

        # --- b) Calculate pressure for range of lifetimes & Plot ---
        plt.figure(figsize=(12, 7))
        for gas in TARGET_GASES:
            sigma_tot = molecular_sigmas[gas]['Total']
            pressures_b = [calculate_allowed_partial_pressure_mbar(
                                tau,
                                sigma_tot,
                                lifetime_calculator.beta,
                                c_light_speed,
                                K_boltzmann,
                                lifetime_calculator.T
                            ) for tau in LIFETIME_RANGE_S]
            # Filter out potential inf values for plotting if sigma_tot was zero
            valid_indices = np.isfinite(pressures_b)
            if np.any(valid_indices): # Only plot if there are valid points
                 plt.plot(LIFETIME_RANGE_S[valid_indices], np.array(pressures_b)[valid_indices], label=gas)
            elif sigma_tot <=0: # Indicate if pressure is always infinite
                 print(f"    Note: Sigma_tot for {gas} is <= 0. Allowed pressure is infinite.")


        plt.xlabel('Required Beam Lifetime (s)')
        plt.ylabel('Allowed Partial Pressure (mbar)')
        plt.title(f'Allowed Pressure vs Lifetime for {projectile} in {machine}')
        #plt.xscale('log')
        plt.yscale('log')
        # Add legend only if plots were actually made
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
             plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        filename_b = os.path.join(OUTPUT_DIR, f'{machine}_{projectile}_pressure_vs_lifetime.png')
        plt.savefig(filename_b)
        plt.close()
        #print(f"    Saved plot: {filename_b}")

        # --- c) Plot cross-sections ---
        el_values = [molecular_sigmas[gas]['EL'] for gas in TARGET_GASES]
        ec_values = [molecular_sigmas[gas]['EC'] for gas in TARGET_GASES]
        x = np.arange(len(TARGET_GASES)) # the label locations
        width = 0.35 # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        # Check for zero values before plotting log scale
        valid_el = np.array(el_values) > 0
        valid_ec = np.array(ec_values) > 0

        # Plot bars only for non-zero values
        if np.any(valid_el):
             rects1 = ax.bar(x[valid_el] - width/2, np.array(el_values)[valid_el], width, label='Electron Loss (EL)')
             ax.bar_label(rects1, padding=3, fmt='%.2e')
        if np.any(valid_ec):
             rects2 = ax.bar(x[valid_ec] + width/2, np.array(ec_values)[valid_ec], width, label='Electron Capture (EC)')
             ax.bar_label(rects2, padding=3, fmt='%.2e')

        # Handle cases where EL or EC might be zero entirely for legend
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = list(handles) # Copy existing handles
        legend_labels = list(labels) # Copy existing labels
        if not np.any(valid_el) and np.any(np.array(el_values) == 0): # Add dummy entry if EL was always zero
             legend_handles.append(plt.Rectangle((0,0),1,1, color='tab:blue', alpha=0))
             legend_labels.append('Electron Loss (EL=0)')
        if not np.any(valid_ec) and np.any(np.array(ec_values) == 0): # Add dummy entry if EC was always zero
             legend_handles.append(plt.Rectangle((0,0),1,1, color='tab:orange', alpha=0))
             legend_labels.append('Electron Capture (EC=0)')

        ax.set_ylabel('Cross Section (m$^2$)')
        ax.set_title(f'EL/EC Cross Sections for {projectile} in {machine}')
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_GASES)
        if legend_handles: # Only show legend if there's something to label
             ax.legend(legend_handles, legend_labels)

        # Use log scale only if there are positive values, otherwise linear
        all_vals = np.array(el_values + ec_values)
        positive_vals = all_vals[all_vals > 0]

        if len(positive_vals) > 0:
             ax.set_yscale('log')
             min_val = positive_vals.min()
             max_val = positive_vals.max()
             ax.set_ylim(min_val * 0.5, max_val * 2) # Adjust y-limits for log scale
        else:
            ax.set_yscale('linear') # Fallback to linear if all are zero or negative
            ax.set_ylim(bottom=0) # Ensure linear scale starts at 0 if non-negative

        plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        filename_c = os.path.join(OUTPUT_DIR, f'{machine}_{projectile}_cross_sections.png')
        plt.savefig(filename_c)
        plt.close()
        #print(f"    Saved plot: {filename_c}")

# --- Save Table a) ---
df_results_a = pd.DataFrame(results_table_a)
filename_a = os.path.join(OUTPUT_DIR, f'allowed_partial_pressures_{TARGET_LIFETIME_S}s_lifetime.csv')
df_results_a.to_csv(filename_a, index=False)
print(f"\nSaved table: {filename_a}")

print(f"\nProcessing complete. Results saved in '{OUTPUT_DIR}'.")


# --- Generate LaTeX-style CSV Tables ---
print("\nGenerating LaTeX-style CSV tables...")

# Ensure the results were collected in results_table_a
if 'results_table_a' not in locals() or not results_table_a:
    print("ERROR: No results found in results_table_a to generate tables.")
    # Consider exiting or handling the error appropriately
    exit()

# Convert results to DataFrame for easier manipulation
try:
    df_results = pd.DataFrame(results_table_a)
    # Ensure the 'Charge' column exists (needs to be added during result collection)
    if 'Charge' not in df_results.columns:
         print("ERROR: 'Charge' column missing in results. Please modify the loop that creates 'results_table_a' to include it.")
         exit()
except Exception as e:
    print(f"ERROR creating DataFrame from results: {e}")
    exit()

# --- Formatting Functions ---
def format_projectile_latex_simplified(projectile_str, charge_val):
    """Formats projectile string like He1 to He$^{1+}$ for LaTeX."""
    if pd.isna(charge_val): return projectile_str # Handle missing charge
    charge_int = int(charge_val)
    # Use regex to extract only the element letters at the beginning
    match = re.match(r"([A-Za-z]+)", projectile_str) # Only capture letters
    if match:
        element = match.group(1)
        # Format as Element$^{charge+}$
        return f"{element}$^{{{charge_int}+}}$"
    else:
        # Fallback if no letters found (unlikely for element symbols)
        return f"{projectile_str}$^{{{charge_int}+}}$" # Keep original name + charge

def format_pressure_scientific(p_val):
    """Formats pressure value into scientific notation (e.g., 9.7e-13)."""
    if pd.isna(p_val) or not np.isfinite(p_val):
        return "-" # Use hyphen for missing/invalid data
    # Format to one decimal place in scientific notation
    return f"{p_val:.1e}"

# --- Create one CSV per machine ---
for machine in machines:
    print(f"  Processing table for: {machine}")
    df_machine = df_results[df_results['Machine'] == machine].copy()

    if df_machine.empty:
        print(f"    No results found for machine {machine}, skipping CSV generation.")
        continue

    # Check for charge column again before proceeding
    if 'Charge' not in df_machine.columns:
         print(f"    ERROR: Charge column missing for machine {machine} data. Cannot format projectile names.")
         continue

    # Apply *simplified* projectile formatting (create a new column)
    try:
        df_machine['Projectile_LaTeX'] = df_machine.apply(
            lambda row: format_projectile_latex_simplified(row['Projectile'], row['Charge']), axis=1
        )
    except Exception as e:
         print(f"    ERROR formatting projectile names for {machine}: {e}")
         continue # Skip this machine if formatting fails

    # Pivot the table
    pressure_col_name = f'Allowed Pressure (mbar) for {TARGET_LIFETIME_S}s'
    try:
        # Group by the formatted projectile name and get the first entry for each gas
        # This handles potential duplicate formatted names if inputs were like 'He1', 'He2' mapping to He^1+, He^2+

        # Get the ordered unique values from the 'Projectile_LaTeX' column
        ordered_projectile_latex = df_machine['Projectile_LaTeX'].unique().tolist()
        
        df_pivot = df_machine.pivot_table(
            index= 'Projectile_LaTeX', # Use the formatted name as index
            columns='Target Gas',
            values=pressure_col_name,
            aggfunc='first' # Use 'first' aggfunc
        )
        
        df_pivot = df_pivot.reindex(ordered_projectile_latex)
    except Exception as e:
         print(f"    ERROR pivoting data for machine {machine}: {e}")
         continue # Skip this machine if pivoting fails

    # Ensure columns are in the desired order H2, CH4, CO, CO2
    try:
        # Use reindex and fill missing values with NaN which format_pressure handles
        df_pivot = df_pivot.reindex(columns=TARGET_GASES, fill_value=np.nan)
    except Exception as e:
        print(f"    ERROR reindexing columns for {machine}: {e}")
        continue

    # Format the pressure values in the pivoted table
    df_formatted = df_pivot.applymap(format_pressure_scientific)

    # --- Prepare CSV content ---
    csv_filename = os.path.join(OUTPUT_DIR, f'{machine}_partial_pressure_table_{TARGET_LIFETIME_S}s.csv')

    # UPDATED Header line 1: Simplified format
    header_line1 = f"\"{machine} projectile\""

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            # Write the custom header line 1
            f.write(header_line1 + '\n')

            # Write the second header line - "Projectile" and Gas names
            # Ensure gas names are quoted if they contain commas, spaces etc. (unlikely here)
            gas_headers_str = ",".join([f"\"{gas}\"" for gas in df_formatted.columns])
            projectile_header = f"\"Projectile\",{gas_headers_str}"
            f.write(projectile_header + '\n')

            # Write the DataFrame to CSV
            # index=True writes the 'Projectile_LaTeX' index column
            # header=False prevents pandas from writing its own header row again
            df_formatted.to_csv(f, index=True, header=False)

        print(f"    Saved table: {csv_filename}")
    except Exception as e:
        print(f"    ERROR writing CSV file {csv_filename}: {e}")


print("\nCSV table generation complete.")