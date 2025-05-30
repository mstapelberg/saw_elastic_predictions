import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import scipy.signal as sig
import scipy.optimize as opt
from defdap import ebsd
import pandas as pd
from scipy.stats import ks_2samp, kurtosis

# Assuming these are in the same parent directory or installed
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator

# --- Constants ---
N_PEAKS_TO_EXTRACT = 3  # Number of peaks to extract parameters for from experimental data
HIST_XMIN_MHZ = 250 # Set to a float value (e.g., 200.0) or None for auto
HIST_XMAX_MHZ = 350 # Set to a float value (e.g., 500.0) or None for auto
N_EXP_PEAKS_FOR_HISTOGRAM = 2 # Number of top experimental peaks (by amplitude) to include in analysis (e.g., 1 or 2)
FILTER_EXP_MIN_MHZ = 200.0 # Min frequency in MHz for filtering experimental data, set to None to disable
FILTER_EXP_MAX_MHZ = 400.0 # Max frequency in MHz for filtering experimental data, set to None to disable

# --- Helper Functions ---
def _gauss(x, A, mu, sig_val):
    return A * np.exp(-(x - mu)**2 / (2 * sig_val**2))

def refine_exp_peaks(freq, amp, N_candidates=5, prom=0.04, min_peak_height_rel=0.1):
    """
    Finds multiple peaks in the spectrum, fits Gaussians, and returns parameters
    for N_PEAKS_TO_EXTRACT peaks, sorted by frequency (mu), padded with NaNs.
    Returns an array of shape (N_PEAKS_TO_EXTRACT, 3) for [A, mu, sigma].
    """
    default_peak_params = np.full((N_PEAKS_TO_EXTRACT, 3), np.nan)
    if amp.size == 0 or amp.max() <= 1e-9:
        return default_peak_params

    min_abs_height = amp.max() * min_peak_height_rel
    initial_idx, prop = sig.find_peaks(amp, prominence=prom, height=min_abs_height)

    if len(initial_idx) == 0:
        return default_peak_params

    sorted_candidate_indices = initial_idx[np.argsort(prop["prominences"])[::-1]][:N_candidates]
    
    fitted_peaks = []
    for p_idx in sorted_candidate_indices:
        sl = slice(max(0, p_idx - 5), min(len(freq), p_idx + 6))
        try:
            A0, mu0 = amp[p_idx], freq[p_idx]
            sigma_0_guess = max(5e4, (freq[sl][-1] - freq[sl][0]) / 6) 
            popt, pcov = opt.curve_fit(_gauss, freq[sl], amp[sl], p0=(A0, mu0, sigma_0_guess), maxfev=8000)
            if popt[0] > 0 and popt[2] > 0 and freq.min() <= popt[1] <= freq.max():
                 fitted_peaks.append({'A': popt[0], 'mu': popt[1], 'sigma': popt[2]})
        except (RuntimeError, ValueError):
            pass

    if not fitted_peaks:
        return default_peak_params

    fitted_peaks.sort(key=lambda p: p['A'], reverse=True)
    selected_peaks = fitted_peaks[:N_PEAKS_TO_EXTRACT]
    selected_peaks.sort(key=lambda p: p['mu'])

    output_params = np.full((N_PEAKS_TO_EXTRACT, 3), np.nan)
    for i, peak in enumerate(selected_peaks):
        if i < N_PEAKS_TO_EXTRACT:
            output_params[i, 0] = peak['A']
            output_params[i, 1] = peak['mu']
            output_params[i, 2] = peak['sigma']
            
    return output_params

def calculate_predicted_saw_frequency(euler_angles, material):
    try:
        calculator = SAWCalculator(material, euler_angles)
        v, _, _ = calculator.get_saw_speed(0.0, sampling=400, psaw=0)
        wavelength = 8.8e-6  # Assuming same wavelength as before
        return v[0] / wavelength if len(v) > 0 else np.nan
    except Exception:
        return np.nan

# --- Main Script Logic ---
if __name__ == "__main__":
    # 1. Load Experimental FFT Data and Get Fits
    print("--- Loading Experimental FFT Data ---")
    exp_hdf5_path = '/home/myless/Documents/saw_freq_analysis/fftData.h5' # USER: Please verify path
    dominant_exp_freq_list = []
    multi_peak_spectra_count = 0  # Diagnostic counter
    spectra_with_multiple_printed = 0 # Diagnostic print limit

    try:
        with h5py.File(exp_hdf5_path, 'r') as h5:
            exp_freq_axis = h5["/freq"][:]
            exp_x_coords_raw = h5["/X"][:]
            exp_y_coords_raw = h5["/Ycoord"][:]
            exp_amplitude_raw = h5["/amplitude"]
            
            freq_axis_idx_exp = next((i for i, dim_size in enumerate(exp_amplitude_raw.shape) if dim_size == len(exp_freq_axis)), -1)
            if freq_axis_idx_exp == -1: raise ValueError("Freq axis not found in exp data.")
            exp_amplitude_data = np.moveaxis(exp_amplitude_raw[()], freq_axis_idx_exp, 0) if freq_axis_idx_exp != 0 else exp_amplitude_raw[()]

        Ny_exp, Nx_exp = exp_amplitude_data.shape[1], exp_amplitude_data.shape[2] # Assuming freq is first axis now
        print(f"Experimental data grid: {Ny_exp} (Y) x {Nx_exp} (X)")

        print("Fitting peaks to experimental SAW FFT data...")
        for iy in tqdm(range(Ny_exp), desc="Processing exp data Y rows"):
            for ix in range(Nx_exp):
                amp_trace = exp_amplitude_data[:, iy, ix]
                peak_params_all = refine_exp_peaks(exp_freq_axis, amp_trace) # Shape (N_PEAKS_TO_EXTRACT, 3)
                
                # Extract frequencies of the top N_EXP_PEAKS_FOR_HISTOGRAM peaks by amplitude
                valid_peaks_mask = ~np.isnan(peak_params_all[:, 0]) # Mask for peaks with valid amplitude
                if np.any(valid_peaks_mask):
                    actual_valid_params = peak_params_all[valid_peaks_mask, :]
                    
                    if actual_valid_params.shape[0] > 1:
                        multi_peak_spectra_count += 1
                        if spectra_with_multiple_printed < 5:
                            print(f"\nSpectrum (iy={iy}, ix={ix}) found multiple valid peaks ({actual_valid_params.shape[0]}):\n{actual_valid_params}")
                            spectra_with_multiple_printed += 1

                    # Sort these valid peaks by amplitude in descending order
                    sorted_amp_indices = np.argsort(actual_valid_params[:, 0])[::-1]
                    
                    num_peaks_to_add = min(N_EXP_PEAKS_FOR_HISTOGRAM, len(sorted_amp_indices))
                    
                    for i in range(num_peaks_to_add):
                        mu_to_add = actual_valid_params[sorted_amp_indices[i], 1] # Get mu (frequency)
                        if pd.notna(mu_to_add):
                            dominant_exp_freq_list.append(mu_to_add)
        
        dominant_exp_freq_array = np.array(dominant_exp_freq_list)
        print(f"\nTotal spectra where refine_exp_peaks found >1 valid peak: {multi_peak_spectra_count}") # Print diagnostic count
        print(f"Processed {len(dominant_exp_freq_array)} experimental peak frequencies (using top {N_EXP_PEAKS_FOR_HISTOGRAM} per spectrum).")
        if len(dominant_exp_freq_array) > 0:
            print(f"Min/Max dominant exp freq (raw): {np.nanmin(dominant_exp_freq_array):.2e} / {np.nanmax(dominant_exp_freq_array):.2e} Hz")
        else:
            print("No dominant experimental frequencies extracted (raw).")

        # Filter experimental frequencies
        if len(dominant_exp_freq_array) > 0 and FILTER_EXP_MIN_MHZ is not None and FILTER_EXP_MAX_MHZ is not None:
            min_freq_hz = FILTER_EXP_MIN_MHZ * 1e6
            max_freq_hz = FILTER_EXP_MAX_MHZ * 1e6
            original_count = len(dominant_exp_freq_array)
            dominant_exp_freq_array = dominant_exp_freq_array[
                (dominant_exp_freq_array >= min_freq_hz) & (dominant_exp_freq_array <= max_freq_hz)
            ]
            print(f"Filtered experimental frequencies to range [{FILTER_EXP_MIN_MHZ:.1f} MHz - {FILTER_EXP_MAX_MHZ:.1f} MHz].")
            print(f"Retained {len(dominant_exp_freq_array)} peaks out of {original_count}.")
            if len(dominant_exp_freq_array) > 0:
                 print(f"Min/Max dominant exp freq (filtered): {np.nanmin(dominant_exp_freq_array):.2e} / {np.nanmax(dominant_exp_freq_array):.2e} Hz")
            else:
                print("No dominant experimental frequencies remaining after filtering.")

        # Add a diagnostic print here:
        if len(dominant_exp_freq_array) > 0:
            print(f"DIAGNOSTIC: After filtering, dominant_exp_freq_array has len: {len(dominant_exp_freq_array)}, sum: {np.sum(dominant_exp_freq_array):.2e}")
        else:
            print("DIAGNOSTIC: After filtering, dominant_exp_freq_array is empty.")

    except FileNotFoundError:
        print(f"ERROR: Experimental HDF5 data file not found at {exp_hdf5_path}")
        dominant_exp_freq_array = np.array([]) # Ensure it exists for plotting
    except Exception as e:
        print(f"Error processing experimental data: {e}")
        dominant_exp_freq_array = np.array([]) # Ensure it exists for plotting

    # 2. Load EBSD Data
    print("\n--- Loading EBSD Data ---")
    ebsd_data_path = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected" # USER: Please verify path
    predicted_ebsd_freq_list = []

    try:
        ebsd_map = ebsd.Map(ebsd_data_path, dataType="OxfordText")
        ebsd_map.buildQuatArray()
        print(f"EBSD Phases: {[phase.name for phase in ebsd_map.phases]}")
        ebsd_map.findBoundaries(boundDef=5)
        ebsd_map.findGrains(minGrainSize=10)
        print(f"Identified {len(ebsd_map.grainList)} EBSD grains.")

        # 3. Calculate Predicted SAW Frequencies from EBSD
        print("\n--- Calculating Predicted SAW Frequencies from EBSD ---")
        vanadium = Material(formula='V', C11=229e9, C12=119e9, C44=43e9, density=6110, crystal_class='cubic')
        
        for grain in tqdm(ebsd_map, desc="Processing EBSD grains", unit="grain"):
            grain.calcAverageOri()
            euler = grain.refOri.eulerAngles()
            peak_saw_freq = calculate_predicted_saw_frequency(euler, vanadium)
            if pd.notna(peak_saw_freq):
                predicted_ebsd_freq_list.append(peak_saw_freq)
        
        predicted_ebsd_freq_array = np.array(predicted_ebsd_freq_list)
        print(f"Calculated {len(predicted_ebsd_freq_array)} predicted EBSD frequencies.")
        if len(predicted_ebsd_freq_array) > 0:
            print(f"Min/Max predicted EBSD freq: {np.nanmin(predicted_ebsd_freq_array):.2e} / {np.nanmax(predicted_ebsd_freq_array):.2e} Hz")
        else:
            print("No EBSD frequencies calculated.")

    except FileNotFoundError:
        print(f"ERROR: EBSD data file not found at {ebsd_data_path}")
        predicted_ebsd_freq_array = np.array([]) # Ensure it exists
    except Exception as e:
        print(f"Error processing EBSD data: {e}")
        predicted_ebsd_freq_array = np.array([]) # Ensure it exists

    # Perform KS test if both datasets have data
    if len(dominant_exp_freq_array) > 0 and len(predicted_ebsd_freq_array) > 0:
        print("\n--- Kolmogorov-Smirnov Test for Distribution Similarity (using potentially filtered experimental data) ---")
        ks_statistic, p_value = ks_2samp(dominant_exp_freq_array, predicted_ebsd_freq_array)
        print(f"KS Statistic: {ks_statistic:.4f}")
        print(f"P-value: {p_value:.4g}")
        if p_value < 0.05:
            print("The p-value is less than 0.05, suggesting the distributions are statistically different.")
        else:
            print("The p-value is greater than or equal to 0.05, suggesting no statistically significant difference between the distributions.")
    else:
        print("\nSkipping KS test as one or both datasets are empty.")

    # Calculate and print summary statistics
    print("\n--- Summary Statistics (MHz) (using potentially filtered experimental data) ---")
    if len(dominant_exp_freq_array) > 0:
        exp_freq_mhz = dominant_exp_freq_array / 1e6
        print("Experimental Frequencies:")
        print(f"  Mean: {np.nanmean(exp_freq_mhz):.2f} MHz")
        print(f"  Median: {np.nanmedian(exp_freq_mhz):.2f} MHz")
        print(f"  Std Dev: {np.nanstd(exp_freq_mhz):.2f} MHz")
        print(f"  Kurtosis: {kurtosis(exp_freq_mhz, nan_policy='omit'):.2f}")
    else:
        print("Experimental Frequencies: No data")

    if len(predicted_ebsd_freq_array) > 0:
        pred_freq_mhz = predicted_ebsd_freq_array / 1e6
        print("Predicted EBSD Frequencies:")
        print(f"  Mean: {np.nanmean(pred_freq_mhz):.2f} MHz")
        print(f"  Median: {np.nanmedian(pred_freq_mhz):.2f} MHz")
        print(f"  Std Dev: {np.nanstd(pred_freq_mhz):.2f} MHz")
        print(f"  Kurtosis: {kurtosis(pred_freq_mhz, nan_policy='omit'):.2f}")
    else:
        print("Predicted EBSD Frequencies: No data")

    # 4. Plot Histograms and CDFs
    print("\n--- Plotting Histograms and CDFs (using potentially filtered experimental data) ---")
    fig, axs = plt.subplots(2, 2, figsize=(18, 12)) # Changed to 2x2 subplots
    axs_flat = axs.flatten() # Flatten for easier indexing if needed

    # Determine common x-limits if set
    common_xlim = []
    if HIST_XMIN_MHZ is not None:
        common_xlim.append(HIST_XMIN_MHZ)
    if HIST_XMAX_MHZ is not None:
        common_xlim.append(HIST_XMAX_MHZ)

    # Histogram of Experimental Frequencies (axs[0])
    if len(dominant_exp_freq_array) > 0:
        axs_flat[0].hist(dominant_exp_freq_array / 1e6, bins=50, alpha=0.7, color='blue', edgecolor='black') # Consider adjusting bins
        axs_flat[0].set_title('Histogram of Dominant Experimental SAW Frequencies')
        axs_flat[0].set_xlabel('Frequency (MHz)')
        axs_flat[0].set_ylabel('Counts')
        axs_flat[0].grid(True, alpha=0.3)
        if len(common_xlim) == 2:
            axs_flat[0].set_xlim(common_xlim)
        elif HIST_XMIN_MHZ is not None:
            axs_flat[0].set_xlim(left=HIST_XMIN_MHZ)
        elif HIST_XMAX_MHZ is not None:
            axs_flat[0].set_xlim(right=HIST_XMAX_MHZ)
    else:
        axs_flat[0].text(0.5, 0.5, "No experimental data to plot", ha='center', va='center', transform=axs_flat[0].transAxes)
        axs_flat[0].set_title('Experimental SAW Frequencies')

    # Histogram of Predicted EBSD Frequencies (axs[1])
    if len(predicted_ebsd_freq_array) > 0:
        axs_flat[1].hist(predicted_ebsd_freq_array / 1e6, bins=10, alpha=0.7, color='green', edgecolor='black')
        axs_flat[1].set_title('Histogram of Predicted EBSD SAW Frequencies')
        axs_flat[1].set_xlabel('Frequency (MHz)')
        axs_flat[1].set_ylabel('Counts')
        axs_flat[1].grid(True, alpha=0.3)
        if len(common_xlim) == 2:
            axs_flat[1].set_xlim(common_xlim)
        elif HIST_XMIN_MHZ is not None:
            axs_flat[1].set_xlim(left=HIST_XMIN_MHZ)
        elif HIST_XMAX_MHZ is not None:
            axs_flat[1].set_xlim(right=HIST_XMAX_MHZ)
    else:
        axs_flat[1].text(0.5, 0.5, "No predicted EBSD data to plot", ha='center', va='center', transform=axs_flat[1].transAxes)
        axs_flat[1].set_title('Predicted EBSD SAW Frequencies')

    # CDF Plot (axs[2])
    exp_freq_sorted_mhz, pred_freq_sorted_mhz = None, None # Define for later use
    if len(dominant_exp_freq_array) > 0:
        exp_freq_sorted_mhz = np.sort(dominant_exp_freq_array / 1e6)
        exp_cdf = np.arange(1, len(exp_freq_sorted_mhz) + 1) / len(exp_freq_sorted_mhz)
        axs_flat[2].plot(exp_freq_sorted_mhz, exp_cdf, color='blue', label='Experimental Dominant')

    if len(predicted_ebsd_freq_array) > 0:
        pred_freq_sorted_mhz = np.sort(predicted_ebsd_freq_array / 1e6)
        pred_cdf = np.arange(1, len(pred_freq_sorted_mhz) + 1) / len(pred_freq_sorted_mhz)
        axs_flat[2].plot(pred_freq_sorted_mhz, pred_cdf, color='green', label='Predicted EBSD')
    
    axs_flat[2].set_title('Cumulative Distribution Functions (CDFs)')
    axs_flat[2].set_xlabel('Frequency (MHz)')
    axs_flat[2].set_ylabel('Cumulative Probability')
    axs_flat[2].grid(True, alpha=0.3)
    if len(dominant_exp_freq_array) > 0 or len(predicted_ebsd_freq_array) > 0:
        axs_flat[2].legend(loc='best')
    if len(common_xlim) == 2:
        axs_flat[2].set_xlim(common_xlim)
    elif HIST_XMIN_MHZ is not None:
        axs_flat[2].set_xlim(left=HIST_XMIN_MHZ)
    elif HIST_XMAX_MHZ is not None:
        axs_flat[2].set_xlim(right=HIST_XMAX_MHZ)

    if not (len(dominant_exp_freq_array) > 0 or len(predicted_ebsd_freq_array) > 0) :
        axs_flat[2].text(0.5, 0.5, "No data for CDF plot", ha='center', va='center', transform=axs_flat[2].transAxes)

    # CDF Difference Plot (axs[3])
    if exp_freq_sorted_mhz is not None and pred_freq_sorted_mhz is not None:
        # Create a common frequency axis for interpolation
        # Ensure we use the MHz versions which are sorted
        all_freqs_mhz = np.sort(np.unique(np.concatenate((exp_freq_sorted_mhz, pred_freq_sorted_mhz))))
        
        # Interpolate CDFs onto the common axis
        # Need original CDF values corresponding to exp_freq_sorted_mhz and pred_freq_sorted_mhz
        exp_cdf_full = np.arange(1, len(exp_freq_sorted_mhz) + 1) / len(exp_freq_sorted_mhz)
        pred_cdf_full = np.arange(1, len(pred_freq_sorted_mhz) + 1) / len(pred_freq_sorted_mhz)

        # To correctly interpolate step CDFs, we need to add points just before each step
        # and ensure the first point starts at 0 probability.
        
        # Experimental CDF
        interp_exp_freqs = np.concatenate(([all_freqs_mhz.min()], exp_freq_sorted_mhz))
        interp_exp_cdf_vals = np.concatenate(([0], exp_cdf_full))
        # Ensure unique points for interpolation, keeping last for duplicates (steps)
        unique_exp_freqs, unique_exp_indices = np.unique(interp_exp_freqs, return_index=True)
        interp_exp_cdf_on_common = np.interp(all_freqs_mhz, unique_exp_freqs, interp_exp_cdf_vals[unique_exp_indices], left=0, right=1)

        # Predicted CDF
        interp_pred_freqs = np.concatenate(([all_freqs_mhz.min()], pred_freq_sorted_mhz))
        interp_pred_cdf_vals = np.concatenate(([0], pred_cdf_full))
        # Ensure unique points for interpolation
        unique_pred_freqs, unique_pred_indices = np.unique(interp_pred_freqs, return_index=True)
        interp_pred_cdf_on_common = np.interp(all_freqs_mhz, unique_pred_freqs, interp_pred_cdf_vals[unique_pred_indices], left=0, right=1)

        cdf_difference = interp_exp_cdf_on_common - interp_pred_cdf_on_common
        
        axs_flat[3].plot(all_freqs_mhz, cdf_difference, color='purple', label='CDF Diff (Exp - Pred)')
        axs_flat[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axs_flat[3].set_title('CDF Difference (Experimental - Predicted)')
        axs_flat[3].set_xlabel('Frequency (MHz)')
        axs_flat[3].set_ylabel('Difference in Cumulative Probability')
        axs_flat[3].grid(True, alpha=0.3)
        axs_flat[3].legend(loc='best')
        if len(common_xlim) == 2:
            axs_flat[3].set_xlim(common_xlim)
        elif HIST_XMIN_MHZ is not None:
            axs_flat[3].set_xlim(left=HIST_XMIN_MHZ)
        elif HIST_XMAX_MHZ is not None:
            axs_flat[3].set_xlim(right=HIST_XMAX_MHZ)
    else:
        axs_flat[3].text(0.5, 0.5, "Not enough data for CDF difference plot", ha='center', va='center', transform=axs_flat[3].transAxes)
        axs_flat[3].set_title('CDF Difference')

    plt.tight_layout()
    plt.show()

    print("\nBasic analysis script finished.") 