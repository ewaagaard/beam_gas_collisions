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
LIFETIME_RANGE_S = np.linspace(1, 40, num=40) #np.logspace(0, 2, 20) # 1s to 100s range for plots (Part b)
TARGET_GASES = ['H2', 'H2O', 'CH4', 'CO', 'CO2']
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


# Function to convert to LaTeX-style charge states
def convert_to_molecule_state(label):
    # Split the label into element and charge state using regex
    match = re.match(r"([A-Za-z]+)(\d+)", label)
    if label == 'H2O':
        return 'H$_{{2}}$O'
    else:
        if match:
            element = match.group(1)
            number = match.group(2)
            # Return the element and the LaTeX superscript for charge
            return f"{element}$_{{{number}}}$"
        else:
            return label

def format_cross_section_scientific(cs_val):
    """Formats cross section value into scientific notation (e.g., 1.23e-20)."""
    if pd.isna(cs_val) or not np.isfinite(cs_val) or cs_val < 0:
        return "-"
    return f"{cs_val:.1e}"

def format_energy(e_val):
    """Formats energy value (e.g., 4.2)."""
    if pd.isna(e_val) or not np.isfinite(e_val):
        return "-"
    return f"{e_val:.1f}" # Format to one decimal place

# --- Main Logic ---
data = DataObject() # Load base data once
all_projectiles = data.projectile_data.index.values
machines = ['LEIR', 'PS', 'SPS']

results_table_a = [] # To store data for the CSV file (Part a)
results_cross_sections = []

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

            # --- Store Cross Sections ---
            for gas in TARGET_GASES:
                # Ensure charge is correctly retrieved/available
                current_charge = int(lifetime_calculator.q) if hasattr(lifetime_calculator, 'q') else None
                current_energy = lifetime_calculator.e_kin if hasattr(lifetime_calculator, 'e_kin') else None
                if current_charge is None or current_energy is None:
                    print(f"    WARNING: Could not retrieve charge for {projectile} in {machine}. Skipping CS data storage.")
                    continue # Skip if charge is missing

                results_cross_sections.append({
                    'Machine': machine,
                    'Projectile': projectile,
                    'Charge': current_charge,
                    'Energy': current_energy,
                    'Target Gas': gas,
                    'EL Cross Section': molecular_sigmas[gas]['EL'], # m^2
                    'EC Cross Section': molecular_sigmas[gas]['EC']  # m^2
                })

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
        fig0, ax0 = plt.subplots(1, 1, figsize=(4.6, 4.1), constrained_layout=True)
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
                 ax0.plot(LIFETIME_RANGE_S[valid_indices], np.array(pressures_b)[valid_indices], lw=2.6, label=convert_to_molecule_state(gas))
                 ax0.text(0.05, 0.925, '{:.1f} MeV/u: {}'.format(lifetime_calculator.e_kin, format_projectile_latex_simplified(projectile, lifetime_calculator.q)),
                          fontsize=16.2, transform=ax0.transAxes)
            elif sigma_tot <=0: # Indicate if pressure is always infinite
                 print(f"    Note: Sigma_tot for {gas} is <= 0. Allowed pressure is infinite.")


        ax0.set_xlabel('Beam lifetime [s]', fontsize=17)
        ax0.set_ylabel('Target pressure [mbar]', fontsize=17)
        ax0.set_ylim(1e-12, 1e-6)
        #plt.title(f'Allowed Pressure vs Lifetime for {projectile} in {machine}')
        #plt.xscale('log')
        ax0.grid(alpha=0.45)
        ax0.set_yscale('log')
        # Add legend only if plots were actually made
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
             plt.legend(fontsize=12, loc='upper right')
        #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        filename_b = os.path.join(OUTPUT_DIR, f'{machine}_{projectile}_pressure_vs_lifetime.png')
        plt.savefig(filename_b)
        plt.close()
        #print(f"    Saved plot: {filename_b}")

        # --- c) Plot cross-sections ---
        el_values = [molecular_sigmas[gas]['EL'] for gas in TARGET_GASES]
        ec_values = [molecular_sigmas[gas]['EC'] for gas in TARGET_GASES]
        x = np.arange(len(TARGET_GASES)) # the label locations
        width = 0.25 # the width of the bars

        fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
        # Check for zero values before plotting log scale
        valid_el = np.array(el_values) > 0
        valid_ec = np.array(ec_values) > 0

        # Plot bars only for non-zero values
        if np.any(valid_el):
             rects1 = ax.bar(x[valid_el] - width/2, np.array(el_values)[valid_el], width, label='EL')
             #ax.bar_label(rects1, padding=3, fmt='%.2e', fontsize=8.2)
        if np.any(valid_ec):
             rects2 = ax.bar(x[valid_ec] + width/2, np.array(ec_values)[valid_ec], width, label='EC')
             #ax.bar_label(rects2, padding=3, fmt='%.2e', fontsize=8.2)

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

        ax.set_ylabel('$\\sigma$ [m$^2$]', fontsize=17)
        ax.text(0.013, 0.94, '{:.1f} MeV/u: {}'.format(lifetime_calculator.e_kin, format_projectile_latex_simplified(projectile, lifetime_calculator.q)),
                fontsize=14.5, transform=ax.transAxes)
        ax.set_xticks(x)
        ax.grid(alpha=0.45)
        # Apply the function to each label in the index
        latex_labels = [convert_to_molecule_state(label) for i, label in enumerate(TARGET_GASES)]
                
        ax.set_xticklabels(latex_labels, fontsize=13)
        if legend_handles: # Only show legend if there's something to label
             ax.legend(legend_handles, legend_labels, fontsize=12.5, loc='lower right')

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
        ax.set_ylim(1e-29, 1e-19)

        #plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5)
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


def format_pressure_scientific(p_val):
    """Formats pressure value into scientific notation (e.g., 9.7e-13)."""
    if pd.isna(p_val) or not np.isfinite(p_val):
        return "-" # Use hyphen for missing/invalid data
    # Format to one decimal place in scientific notation
    return f"{p_val:.1e}"


# ... (Check results_table_a, create df_results - same as before) ...
if 'df_results' in locals() and df_results is not None: # Check if df_results was created

    # --- Define Formatting Functions (if not already defined globally) ---
    # format_projectile_latex_simplified(...)
    # format_pressure_scientific(...)

    # --- Create one CSV per machine ---
    for machine in machines:
        print(f"  Processing Pressure table for: {machine}")
        df_machine = df_results[df_results['Machine'] == machine].copy()

        if df_machine.empty:
            print(f"    No results found for machine {machine}, skipping CSV generation.")
            continue

        # Apply projectile formatting
        # ... (Try/except block as before) ...
        try:
            df_machine['Projectile_LaTeX'] = df_machine.apply(
                lambda row: format_projectile_latex_simplified(row['Projectile'], row['Charge']), axis=1
            )
        except NameError: # Ensure formatter function exists
             print("    ERROR: 'format_projectile_latex_simplified' function not defined.")
             break
        except Exception as e:
             print(f"    ERROR formatting projectile names for Pressure table ({machine}): {e}")
             continue

        # Pivot the table
        pressure_col_name = f'Allowed Pressure (mbar) for {TARGET_LIFETIME_S}s'
        try:
            df_pivot = df_machine.pivot_table(
                index='Projectile_LaTeX', columns='Target Gas', values=pressure_col_name, aggfunc='first'
            )
        except Exception as e:
             print(f"    ERROR pivoting data for Pressure table ({machine}): {e}")
             continue

        # Reorder columns
        # ... (Try/except block for reindex as before) ...
        try:
            df_pivot = df_pivot.reindex(columns=TARGET_GASES, fill_value=np.nan)
        except Exception as e:
            print(f"    ERROR reindexing columns for Pressure table ({machine}): {e}")
            continue

        # Format the pressure values
        df_formatted = df_pivot.applymap(format_pressure_scientific)

        # --- Sort by Charge State ---
        df_formatted_sorted = df_formatted # Default to unsorted
        if 'Charge' not in df_machine.columns:
            print(f"   WARNING: Charge column missing for machine {machine}. Cannot sort table by charge.")
        else:
            try:
                # Create map from LaTeX name back to original Charge
                # Use groupby().first() to handle potential duplicate Projectile_LaTeX index entries safely
                charge_map = df_machine.groupby('Projectile_LaTeX')['Charge'].first()
                # Get the index sorted by the charge values from the map
                sorted_index = charge_map.sort_values().index
                # Reindex the formatted table according to the sorted index
                # Ensure all indices in df_formatted are present in sorted_index
                df_formatted_sorted = df_formatted.reindex(sorted_index.intersection(df_formatted.index))
            except Exception as e:
                print(f"    WARNING: Failed to sort pressure table by charge for {machine}: {e}")
        # --- End Sort ---

        # Prepare CSV content
        csv_filename = os.path.join(OUTPUT_DIR, f'{machine}_partial_pressure_table_{TARGET_LIFETIME_S}s.csv')
        header_line1 = f"\"{machine} projectile\""

        try:
             with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                f.write(header_line1 + '\n')
                # Use df_formatted_sorted here for column headers if needed (though format is same)
                gas_headers_str = ",".join([f"\"{gas}\"" for gas in df_formatted_sorted.columns])
                projectile_header = f"\"Projectile\",{gas_headers_str}"
                f.write(projectile_header + '\n')
                # Write the SORTED DataFrame to CSV
                df_formatted_sorted.to_csv(f, index=True, header=False)

             print(f"    Saved sorted Pressure table: {csv_filename}")
        except Exception as e:
             print(f"    ERROR writing sorted Pressure CSV file {csv_filename}: {e}")

# else: # Handle case where df_results was not created
#    print("Skipping Pressure table generation as initial DataFrame failed.")

print("\nPressure CSV table generation complete.")

# --- Generate Cross Section LaTeX-style CSV Tables ---
print("\nGenerating Cross Section LaTeX-style CSV tables...")

df_cs_results = None # Initialize DataFrame variable to None

# First, check if the source list exists and has data
if 'results_cross_sections' not in locals() or not results_cross_sections:
    print("WARNING: No cross section results found in 'results_cross_sections' list. Skipping CS table generation.")
else:
    # Attempt to create the DataFrame and validate columns
    try:
        df_cs_results = pd.DataFrame(results_cross_sections)
        # Ensure required columns exist after DataFrame creation
        required_cs_cols = ['Machine', 'Projectile', 'Charge', 'Energy', 'Target Gas', 'EL Cross Section', 'EC Cross Section']
        if not all(col in df_cs_results.columns for col in required_cs_cols):
             missing_cols = [col for col in required_cs_cols if col not in df_cs_results.columns]
             print(f"ERROR: Required columns missing in cross section results DataFrame: {missing_cols}. Skipping CS table generation.")
             df_cs_results = None # Invalidate DataFrame if columns missing
        elif df_cs_results.empty:
             print("WARNING: Cross section results DataFrame is empty. Skipping CS table generation.")
             df_cs_results = None # Invalidate DataFrame if empty

    except Exception as e:
        print(f"ERROR creating DataFrame from cross section results: {e}. Skipping CS table generation.")
        df_cs_results = None # Invalidate DataFrame on creation error


# Proceed only if DataFrame was successfully created, validated, and is not empty
if df_cs_results is not None:

    # --- Define Formatting Functions (ensure these are defined) ---
    try:
        # Example check, replace with actual function names if needed
        assert callable(format_projectile_latex_simplified)
        assert callable(format_cross_section_scientific)
        assert callable(format_energy)
    except (NameError, AssertionError):
        print("ERROR: Formatting functions (e.g., format_projectile_latex_simplified) are not defined. Skipping CS table generation.")
        df_cs_results = None # Prevent proceeding

# Re-check df_cs_results before entering the loop
if df_cs_results is not None:
    # --- Create one EL and one EC CSV per machine ---
    for machine in machines:
        print(f"  Processing CS tables for: {machine}")
        # Filter data for the current machine
        df_cs_machine = df_cs_results[df_cs_results['Machine'] == machine].copy()

        if df_cs_machine.empty:
            # This check might be redundant given the earlier checks, but safe
            print(f"    No cross section results found for machine {machine}, skipping CSV generation.")
            continue

        # Apply projectile formatting
        try:
            df_cs_machine['Projectile_LaTeX'] = df_cs_machine.apply(
                lambda row: format_projectile_latex_simplified(row['Projectile'], row['Charge']), axis=1
            )
        except Exception as e:
             print(f"    ERROR formatting projectile names for CS table ({machine}): {e}")
             continue # Skip this machine

        # --- Create Charge map for sorting ---
        charge_map_cs = None
        sorted_index_cs = None
        # No need to check 'Charge' column existence here again, checked at df_cs_results creation
        try:
            charge_map_cs = df_cs_machine.groupby('Projectile_LaTeX')['Charge'].first()
            sorted_index_cs = charge_map_cs.sort_values().index # Get the sorted index
        except Exception as e:
            print(f"    WARNING: Failed to create charge map/sorted index for CS tables ({machine}): {e}")
            # Sorting will be skipped if sorted_index_cs remains None

        # --- Create Energy map ---
        energy_map = None
        # No need to check 'Energy' column existence here again
        try:
             energy_map = df_cs_machine.groupby('Projectile_LaTeX')['Energy'].first()
        except Exception as e:
             print(f"    WARNING: Failed to create energy map for {machine}: {e}")
             # Energy column addition will be skipped if energy_map remains None

        # --- Generate EL Table ---
        try:
            df_el_pivot = df_cs_machine.pivot_table(
                index='Projectile_LaTeX', columns='Target Gas', values='EL Cross Section', aggfunc='first'
            )
            df_el_pivot = df_el_pivot.reindex(columns=TARGET_GASES, fill_value=np.nan)
            df_el_formatted = df_el_pivot.applymap(format_cross_section_scientific)

            # Add Energy column if map exists
            if energy_map is not None:
                 try:
                    energy_col_data = df_el_formatted.index.map(energy_map)
                    formatted_energy_col = energy_col_data.map(format_energy)
                    df_el_formatted.insert(0, 'Energy (MeV/u)', formatted_energy_col)
                 except Exception as e:
                    print(f"    WARNING: Failed to add energy column to EL table for {machine}: {e}")
            else:
                 print(f"    Skipping energy column addition for EL table ({machine}) due to map error.")


            # Sort EL Table if sorted index exists
            df_el_formatted_sorted = df_el_formatted # Default to unsorted
            if sorted_index_cs is not None:
                 try:
                    df_el_formatted_sorted = df_el_formatted.reindex(sorted_index_cs.intersection(df_el_formatted.index))
                 except Exception as e:
                    print(f"    WARNING: Failed to sort EL table by charge for {machine}: {e}")
            else:
                 print(f"    Skipping charge sort for EL table ({machine}).")


            # Write EL CSV
            csv_el_filename = os.path.join(OUTPUT_DIR, f'{machine}_EL_cross_section_table.csv')
            header_line1_el = f"\"{machine} EL cross section (m^2)\""
            with open(csv_el_filename, 'w', newline='', encoding='utf-8') as f:
                f.write(header_line1_el + '\n')
                # Adjust header dynamically based on whether Energy column was added
                gas_headers_str = ",".join([f"\"{gas}\"" for gas in df_el_pivot.columns])
                energy_header = ""
                if 'Energy (MeV/u)' in df_el_formatted_sorted.columns:
                     energy_header = ",\"Energy (MeV/u)\""
                projectile_header = f"\"Projectile\"{energy_header},{gas_headers_str}"
                f.write(projectile_header + '\n')
                df_el_formatted_sorted.to_csv(f, index=True, header=False)
            print(f"    Saved EL table: {csv_el_filename}")

        except Exception as e:
            print(f"    ERROR generating EL table for {machine}: {e}")


        # --- Generate EC Table ---
        try:
            df_ec_pivot = df_cs_machine.pivot_table(
                index='Projectile_LaTeX', columns='Target Gas', values='EC Cross Section', aggfunc='first'
            )
            df_ec_pivot = df_ec_pivot.reindex(columns=TARGET_GASES, fill_value=np.nan)
            df_ec_formatted = df_ec_pivot.applymap(format_cross_section_scientific)

            # Add Energy column if map exists
            if energy_map is not None:
                 try:
                    energy_col_data = df_ec_formatted.index.map(energy_map)
                    formatted_energy_col = energy_col_data.map(format_energy)
                    df_ec_formatted.insert(0, 'Energy (MeV/u)', formatted_energy_col)
                 except Exception as e:
                     print(f"    WARNING: Failed to add energy column to EC table for {machine}: {e}")

            else:
                 print(f"    Skipping energy column addition for EC table ({machine}) due to map error.")


            # Sort EC Table if sorted index exists
            df_ec_formatted_sorted = df_ec_formatted # Default to unsorted
            if sorted_index_cs is not None:
                 try:
                    df_ec_formatted_sorted = df_ec_formatted.reindex(sorted_index_cs.intersection(df_ec_formatted.index))
                 except Exception as e:
                    print(f"    WARNING: Failed to sort EC table by charge for {machine}: {e}")
            else:
                 print(f"    Skipping charge sort for EC table ({machine}).")


            # Write EC CSV
            csv_ec_filename = os.path.join(OUTPUT_DIR, f'{machine}_EC_cross_section_table.csv')
            header_line1_ec = f"\"{machine} EC cross section (m^2)\""
            with open(csv_ec_filename, 'w', newline='', encoding='utf-8') as f:
                f.write(header_line1_ec + '\n')
                # Adjust header dynamically
                gas_headers_str = ",".join([f"\"{gas}\"" for gas in df_ec_pivot.columns])
                energy_header = ""
                if 'Energy (MeV/u)' in df_ec_formatted_sorted.columns:
                     energy_header = ",\"Energy (MeV/u)\""
                projectile_header = f"\"Projectile\"{energy_header},{gas_headers_str}"
                f.write(projectile_header + '\n')
                df_ec_formatted_sorted.to_csv(f, index=True, header=False)
            print(f"    Saved EC table: {csv_ec_filename}")

        except Exception as e:
            print(f"    ERROR generating EC table for {machine}: {e}")

# else: # Handle case where df_cs_results is None or initial checks failed
#    print("Skipping Cross Section table generation due to initialization issues.")

print("\nCross Section CSV table generation complete.")