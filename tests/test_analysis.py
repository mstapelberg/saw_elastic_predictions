from defdap import ebsd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import scipy.signal as sig
import scipy.optimize as opt
from scipy.interpolate import griddata
from skimage.transform import estimate_transform, SimilarityTransform # Using SimilarityTransform for simplicity first
import skimage.draw
import skimage.transform as sktransform # For rotation

# Import your SAW calculator
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator

# --- Constants ---
N_PEAKS_TO_EXTRACT = 3 # Number of peaks to extract parameters for
EBSD_FIDUCIAL_SIDE_LENGTH_MM = 5.0
EBSD_FIDUCIAL_LINE_THICKNESS_MM = 0.3 # Estimated thickness of the mark
BC_THRESHOLD_PERCENTILE = 10 # Pixels below this percentile of BC are 'no signal'
FIT_SEARCH_WINDOW_PX = 20 # Search window around clicked center (+/- pixels)
FIT_ANGLE_RANGE_DEG = 10  # Search angle range (+/- degrees)
FIT_ANGLE_STEP_DEG = 1    # Angle step for search

# --- New Helper Function for Fiducial Fitting ---
def get_square_corners(center_px, side_length_px, angle_deg):
    """Calculates the four corners of a square given its center, side length, and rotation angle."""
    half_side = side_length_px / 2
    # Corners relative to center (0,0) before rotation
    corners = np.array([
        [-half_side, -half_side], # Top-left
        [ half_side, -half_side], # Top-right
        [ half_side,  half_side], # Bottom-right
        [-half_side,  half_side]  # Bottom-left
    ])
    tf_rotate = sktransform.SimilarityTransform(rotation=np.deg2rad(angle_deg))
    rotated_corners = tf_rotate(corners) + center_px # Add center coordinates
    return rotated_corners

def get_rotated_square_band_pixels(center_px, side_length_px, line_thickness_px, angle_deg, map_shape):
    """
    Generates pixel coordinates for a band around a rotated square.
    Returns a tuple of (row_coords, col_coords) for pixels within the band.
    Coordinates are for numpy array indexing (row, col).
    """
    # Outer square
    outer_corners = get_square_corners(center_px, side_length_px, angle_deg)
    
    # Inner square (for the hollow part)
    # Ensure inner side length is not negative if line thickness is large
    inner_side_length_px = max(0, side_length_px - 2 * line_thickness_px) 
    
    # Create a mask for the band
    # skimage.draw.polygon expects (rows_coords, cols_coords)
    # Our corners are (x,y) which correspond to (col, row) for indexing if origin is top-left
    # So, polygon(corners[:,1], corners[:,0]) for (rows, cols)
    
    rr_outer, cc_outer = skimage.draw.polygon(outer_corners[:, 1], outer_corners[:, 0], shape=map_shape)
    
    band_mask_canvas = np.zeros(map_shape, dtype=bool)
    # Clip coordinates to be within map_shape bounds before indexing
    rr_outer_clipped = np.clip(rr_outer, 0, map_shape[0] - 1)
    cc_outer_clipped = np.clip(cc_outer, 0, map_shape[1] - 1)
    band_mask_canvas[rr_outer_clipped, cc_outer_clipped] = True
    
    if inner_side_length_px > 1e-3: # Only subtract inner if it has a meaningful area (avoid issues with tiny/zero inner)
        inner_corners = get_square_corners(center_px, inner_side_length_px, angle_deg)
        rr_inner, cc_inner = skimage.draw.polygon(inner_corners[:, 1], inner_corners[:, 0], shape=map_shape)
        
        # Create a temporary canvas for the inner polygon to subtract it
        temp_mask_inner = np.zeros(map_shape, dtype=bool)
        rr_inner_clipped = np.clip(rr_inner, 0, map_shape[0] - 1)
        cc_inner_clipped = np.clip(cc_inner, 0, map_shape[1] - 1)
        temp_mask_inner[rr_inner_clipped, cc_inner_clipped] = True
        band_mask_canvas[temp_mask_inner] = False # Subtract inner area
            
    return np.where(band_mask_canvas)

# --- Functions adapted from saw_upsample_rot.py ---
def _gauss(x, A, mu, sig_val):
    return A * np.exp(-(x-mu)**2 / (2*sig_val**2))

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
    # Find more initial candidates than N_PEAKS_TO_EXTRACT to choose from
    initial_idx, prop = sig.find_peaks(amp, prominence=prom, height=min_abs_height)

    if len(initial_idx) == 0:
        return default_peak_params

    # Sort candidates by prominence to prioritize stronger ones for fitting
    sorted_candidate_indices = initial_idx[np.argsort(prop["prominences"])[::-1]][:N_candidates]
    
    fitted_peaks = []
    for p_idx in sorted_candidate_indices:
        sl = slice(max(0, p_idx - 5), min(len(freq), p_idx + 6))
        try:
            A0, mu0 = amp[p_idx], freq[p_idx]
            # Ensure sigma_0 is not zero and positive
            sigma_0_guess = max(5e4, (freq[sl][-1] - freq[sl][0]) / 6) # Heuristic for initial sigma
            popt, pcov = opt.curve_fit(_gauss, freq[sl], amp[sl], p0=(A0, mu0, sigma_0_guess), maxfev=8000)
            # Check for reasonable fit: positive A, positive sigma, mu within freq range
            if popt[0] > 0 and popt[2] > 0 and freq.min() <= popt[1] <= freq.max():
                 fitted_peaks.append({'A': popt[0], 'mu': popt[1], 'sigma': popt[2], 'prominence': prop["prominences"][initial_idx == p_idx][0]})
        except (RuntimeError, ValueError):
            # Fallback: use peak properties if fit fails, but mark with lower confidence (e.g. A=amp[p_idx])
            # For simplicity now, we only take successfully fitted peaks with positive A and sigma
            pass # Or append with raw A0, mu0 if desired as fallback

    if not fitted_peaks:
        return default_peak_params

    # Sort fitted peaks by frequency (mu)
    fitted_peaks.sort(key=lambda p: p['mu'])
    
    # Select up to N_PEAKS_TO_EXTRACT peaks (could also sort by 'A' or 'prominence' before slicing)
    # For now, taking the first N_PEAKS_TO_EXTRACT after sorting by mu.
    # Or, more robustly, sort by amplitude AFTER fitting, then take top N_PEAKS_TO_EXTRACT, then sort THOSE by mu.
    # Let's sort by Amplitude (desc) to pick the strongest N_PEAKS_TO_EXTRACT ones first
    fitted_peaks.sort(key=lambda p: p['A'], reverse=True)
    selected_peaks = fitted_peaks[:N_PEAKS_TO_EXTRACT]
    
    # Now sort these selected peaks by frequency for consistent output order
    selected_peaks.sort(key=lambda p: p['mu'])

    output_params = np.full((N_PEAKS_TO_EXTRACT, 3), np.nan)
    for i, peak in enumerate(selected_peaks):
        if i < N_PEAKS_TO_EXTRACT:
            output_params[i, 0] = peak['A']
            output_params[i, 1] = peak['mu']
            output_params[i, 2] = peak['sigma']
            
    return output_params

def load_exp_fiducials_csv(path):
    df = pd.read_csv(path)
    if not ({'X_coord', 'Y_coord', 'Fiducial'}.issubset(df.columns)):
        raise ValueError("CSV must have 'X_coord', 'Y_coord', 'Fiducial'.")
    if sum(df["Fiducial"] == 1) != 1:
        raise ValueError("CSV must have exactly one 'Fiducial' == 1 (marked corner).")
    poly_coords_mm = df[["X_coord", "Y_coord"]].to_numpy(float)
    marked_corner_idx = df[df["Fiducial"] == 1].index[0]
    return np.roll(poly_coords_mm, -marked_corner_idx, axis=0)
# --- End of adapted functions ---

# --- EBSD Processing ---
data_path = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected"
ebsd_map = ebsd.Map(data_path, dataType="OxfordText")
ebsd_map.buildQuatArray()
print(f"Phases: {[phase.name for phase in ebsd_map.phases]}")
ebsd_map.findBoundaries(boundDef=5)
ebsd_map.findGrains(minGrainSize=10)
print(f"Identified {len(ebsd_map.grainList)} EBSD grains")
ebsd_step_size_mm = ebsd_map.stepSize / 1000.0
print(f"EBSD: {ebsd_map.shape}, xDim: {ebsd_map.xDim}, yDim: {ebsd_map.yDim}, Step: {ebsd_step_size_mm:.4f} mm")

vanadium = Material(formula='V', C11=229e9, C12=119e9, C44=43e9, density=6110, crystal_class='cubic')

def calculate_predicted_saw_frequency(euler_angles, material):
    try:
        calculator = SAWCalculator(material, euler_angles)
        v, _, _ = calculator.get_saw_speed(0.0, sampling=400, psaw=0)
        wavelength = 8.8e-6
        return v[0] / wavelength if len(v) > 0 else np.nan
    except Exception: return np.nan

print("Calculating predicted SAW frequencies for EBSD grains...")
ebsd_grains_data = []
for grain in tqdm(ebsd_map, desc="Processing EBSD grains", unit="grain"):
    grain.calcAverageOri()
    euler = grain.refOri.eulerAngles()
    peak_saw_freq = calculate_predicted_saw_frequency(euler, vanadium)
    ebsd_grains_data.append({
        "Grain ID": grain.grainID, "Euler1": euler[0], "Euler2": euler[1], "Euler3": euler[2],
        "Predicted SAW Frequency": peak_saw_freq,
        "Centroid_X_px": grain.centreCoords()[0], "Centroid_Y_px": grain.centreCoords()[1],
        "Centroid_X_mm_ebsd": grain.centreCoords()[0] * ebsd_step_size_mm,
        "Centroid_Y_mm_ebsd": grain.centreCoords()[1] * ebsd_step_size_mm
    })
df_ebsd = pd.DataFrame(ebsd_grains_data)
df_ebsd.dropna(subset=['Predicted SAW Frequency'], inplace=True)
print(f"EBSD DataFrame head:\n{df_ebsd.head()}")

# --- Programmatic EBSD Fiducial Definition ---
print("\n--- Programmatic EBSD Fiducial Definition ---")
ebsd_fiducial_mm = None
best_fit_params = None # To store (center_px, angle_deg, score)
no_signal_mask = None # Initialize for later potential plotting

if hasattr(ebsd_map, 'bandContrastArray') and ebsd_map.bandContrastArray is not None:
    bc_image = ebsd_map.bandContrastArray
    map_shape_ebsd = bc_image.shape
    
    # 1. User clicks approximate center
    fig_bc, ax_bc = plt.subplots(figsize=(10,8))
    ax_bc.imshow(bc_image, cmap='gray', origin='lower')
    ax_bc.set_title('Click the APPROXIMATE center of the 5x5mm fiducial mark')
    ax_bc.set_aspect('equal', adjustable='box')
    print("Close the plot window after clicking the approximate center.")
    clicked_center_px_list = plt.ginput(1, timeout=-1)
    plt.close(fig_bc)

    if not clicked_center_px_list:
        print("No approximate center clicked. Skipping programmatic fiducial fitting.")
    else:
        approx_center_px = np.array(clicked_center_px_list[0])
        print(f"Approximate EBSD fiducial center (pixels): {approx_center_px}")

        # 2. Create "No Signal" Mask
        bc_threshold_val = np.percentile(bc_image[np.isfinite(bc_image)], BC_THRESHOLD_PERCENTILE)
        no_signal_mask = bc_image < bc_threshold_val
        print(f"Using BC threshold: < {bc_threshold_val:.2f} ({BC_THRESHOLD_PERCENTILE}th percentile) for 'no signal' mask.")
        print(f"'No signal' pixels count: {np.sum(no_signal_mask)}")

        # 3. & 4. Search for Best Fit (Center and Small Rotation)
        fiducial_side_px = EBSD_FIDUCIAL_SIDE_LENGTH_MM / ebsd_step_size_mm
        fiducial_line_thick_px = EBSD_FIDUCIAL_LINE_THICKNESS_MM / ebsd_step_size_mm
        
        best_score = -1
        best_center_px = None
        best_angle_deg = None

        angles_to_search = np.arange(-FIT_ANGLE_RANGE_DEG, FIT_ANGLE_RANGE_DEG + FIT_ANGLE_STEP_DEG, FIT_ANGLE_STEP_DEG)
        
        print(f"Starting fiducial fit search. Angles: {angles_to_search} deg, Search window: +/-{FIT_SEARCH_WINDOW_PX} px")

        for angle_deg in tqdm(angles_to_search, desc="Searching angles"):
            # Search around the clicked approximate center
            min_x = int(max(0, approx_center_px[0] - FIT_SEARCH_WINDOW_PX))
            max_x = int(min(map_shape_ebsd[1] -1, approx_center_px[0] + FIT_SEARCH_WINDOW_PX))
            min_y = int(max(0, approx_center_px[1] - FIT_SEARCH_WINDOW_PX))
            max_y = int(min(map_shape_ebsd[0] -1, approx_center_px[1] + FIT_SEARCH_WINDOW_PX))

            for cx_px in range(min_x, max_x + 1):
                for cy_px in range(min_y, max_y + 1):
                    current_center_px = np.array([cx_px, cy_px])
                    band_rows, band_cols = get_rotated_square_band_pixels(
                        current_center_px, 
                        fiducial_side_px, 
                        fiducial_line_thick_px, 
                        angle_deg, 
                        map_shape_ebsd
                    )
                    
                    if band_rows.size > 0:
                        score = np.sum(no_signal_mask[band_rows, band_cols])
                        if score > best_score:
                            best_score = score
                            best_center_px = current_center_px
                            best_angle_deg = angle_deg
        
        if best_center_px is not None:
            print(f"Best fit found: Center={best_center_px}, Angle={best_angle_deg:.1f} deg, Score={best_score}")
            # 5. Calculate EBSD Fiducial Corners
            fitted_ebsd_corners_px = get_square_corners(best_center_px, fiducial_side_px, best_angle_deg)
            ebsd_fiducial_mm = fitted_ebsd_corners_px * ebsd_step_size_mm
            # Ensure order: Top-left, Top-right, Bottom-right, Bottom-left (relative to image axes if angle=0)
            # This order should be consistent with how exp_fiducial_corners_mm_stage is prepared (marked first, then sequential)
            # The get_square_corners returns them in a TL, TR, BR, BL sequence if angle is 0.
            # We need to ensure the 'marked' corner from experimental data corresponds to one of these.
            # For now, we assume the order from get_square_corners is used directly.
            # This might need adjustment if a specific corner needs to be 'first'.
            # Typically, the experimental marked corner is top-left of the physical stage movement.
            # Let's assume the default order is fine for now and matches the experimental CSV's intent after load_exp_fiducials_csv.
            
            print(f"Calculated EBSD fiducial corners (local EBSD mm) from fit:\n{ebsd_fiducial_mm}")
            best_fit_params = {'center_px': best_center_px, 'angle_deg': best_angle_deg, 'score': best_score, 'corners_px': fitted_ebsd_corners_px}
        else:
            print("Programmatic fiducial fitting failed to find a suitable fit.")

else:
    print("Band contrast array not available. Skipping programmatic fiducial fitting.")
    print("Falling back to manual 4-point EBSD fiducial selection.")
    # Fallback to manual clicking if programmatic fails or BC not available
    # ... (The previous manual 4-point ginput logic would go here) ...
    # This part needs to be re-added carefully if we want a fallback.
    # For now, if programmatic fails, ebsd_fiducial_mm will be None and registration won't happen.
    print("\n--- MANUAL EBSD Fiducial Definition as Fallback ---")
    print("Please click the 4 corners of the ~5x5 mm fiducial square on the EBSD map.")
    # ... (rest of manual click logic from before) ...
    fig_ebsd_select, ax_ebsd_select = plt.subplots(figsize=(10,8))
    try:
        if hasattr(ebsd_map, 'bandContrastArray') and ebsd_map.bandContrastArray is not None:
            ax_ebsd_select.imshow(ebsd_map.bandContrastArray, cmap='gray', origin='lower')
            ax_ebsd_select.set_title('MANUAL FALLBACK: Click 4 fiducial corners (marked first, then sequential)')
        else: 
            grain_map_display = np.zeros(ebsd_map.shape, dtype=int)
            for grain in ebsd_map:
                for coord_px in grain.coordList: 
                     if 0 <= coord_px[1] < grain_map_display.shape[0] and 0 <= coord_px[0] < grain_map_display.shape[1]:
                        grain_map_display[coord_px[1], coord_px[0]] = grain.grainID
            ax_ebsd_select.imshow(grain_map_display, cmap='viridis', origin='lower')
            ax_ebsd_select.set_title('MANUAL FALLBACK: Click 4 fiducial corners on Grain ID map (marked first, then sequential)')
    except Exception as e_plot:
        print(f"Error preparing EBSD map for manual fallback: {e_plot}. Using scatter.")
        ax_ebsd_select.scatter(df_ebsd['Centroid_X_px'], df_ebsd['Centroid_Y_px'], s=1)
        ax_ebsd_select.set_xlim(0, ebsd_map.xDim); ax_ebsd_select.set_ylim(0, ebsd_map.yDim)
        if ebsd_map.shape[0] > ebsd_map.shape[1]: ax_ebsd_select.invert_yaxis() 

    ax_ebsd_select.set_aspect('equal', adjustable='box')
    print("Close the plot window after clicking the 4 corners.")
    clicked_points_px_manual = plt.ginput(4, timeout=-1) 
    plt.close(fig_ebsd_select)

    if not clicked_points_px_manual or len(clicked_points_px_manual) != 4:
        print("Manual fallback: Exactly 4 fiducial corners must be clicked. Registration will likely fail.")
        ebsd_fiducial_mm = None # Ensure it's None
    else:
        ebsd_fiducial_pixels_manual = np.array(clicked_points_px_manual)
        ebsd_fiducial_mm = ebsd_fiducial_pixels_manual * ebsd_step_size_mm
        print(f"Manual EBSD fiducial corners (pixels):\n{ebsd_fiducial_pixels_manual}")
        print(f"Manual EBSD fiducial corners (local EBSD mm): \n{ebsd_fiducial_mm}")


# --- Load Experimental SAW Data and Fiducials ---
exp_hdf5_path = '/home/myless/Documents/saw_freq_analysis/fftData.h5'
exp_fiducial_csv_path = '/home/myless/Documents/saw_freq_analysis/ref_corners.csv'

print("\n--- Experimental Data Loading ---")
measured_saw_freq_map_exp, exp_x_coords, exp_y_coords, exp_fiducial_corners_mm_stage = [None]*4
try:
    exp_fiducial_corners_mm_stage = load_exp_fiducials_csv(exp_fiducial_csv_path)
    print(f"Experimental fiducial corners (stage mm, marked corner first from CSV): \n{exp_fiducial_corners_mm_stage}")

    with h5py.File(exp_hdf5_path, 'r') as h5:
        exp_freq_axis = h5["/freq"][:]
        exp_x_coords_raw = h5["/X"][:] 
        exp_y_coords_raw = h5["/Ycoord"][:]
        exp_amplitude_raw = h5["/amplitude"]
        freq_axis_idx_exp = next((i for i, dim_size in enumerate(exp_amplitude_raw.shape) if dim_size == len(exp_freq_axis)), -1)
        if freq_axis_idx_exp == -1: raise ValueError("Freq axis not found in exp data.")
        exp_amplitude_data = np.moveaxis(exp_amplitude_raw[()], freq_axis_idx_exp, 0) if freq_axis_idx_exp != 0 else exp_amplitude_raw[()]
    
    # Ensure exp_x_coords and exp_y_coords are 1D
    exp_x_coords = exp_x_coords_raw[0,:] if exp_x_coords_raw.ndim > 1 else exp_x_coords_raw
    exp_y_coords = exp_y_coords_raw[:,0] if exp_y_coords_raw.ndim > 1 else exp_y_coords_raw

    print(f"Exp data: X {exp_x_coords.shape}, Y {exp_y_coords.shape}, Freq {len(exp_freq_axis)}, Amp {exp_amplitude_data.shape}")

    print("Fitting peaks to experimental SAW FFT data...")
    Ny_exp, Nx_exp = len(exp_y_coords), len(exp_x_coords)
    # Correct initialization for multi-peak data
    measured_saw_freq_map_exp = np.full((Ny_exp, Nx_exp, N_PEAKS_TO_EXTRACT, 3), np.nan, dtype=float)
    for iy in tqdm(range(Ny_exp), desc="Processing exp data Y rows"):
        for ix in range(Nx_exp):
            amp_trace = exp_amplitude_data[:, iy, ix]
            measured_saw_freq_map_exp[iy, ix, :, :] = refine_exp_peaks(exp_freq_axis, amp_trace)
    print(f"Exp peak fitting done. Map shape: {measured_saw_freq_map_exp.shape}")
    # To get a representative min/max, let's look at the mu of the first found peak (if any)
    # This is just for a summary print, the full data is preserved.
    first_peak_mus = measured_saw_freq_map_exp[:, :, 0, 1] # All y, All x, 0th peak, 1st param (mu)
    print(f"Min/Max exp freq (of 1st peak where available): {np.nanmin(first_peak_mus):.2e} / {np.nanmax(first_peak_mus):.2e} Hz")

    # The following processing (transformation and dominant peak extraction for interpolation)
    # should only happen if the transformation matrix was successfully estimated earlier.
    # This logic is moved to the "Registration and Transformation" and "Correlate Data" sections.

except FileNotFoundError:
    print(f"ERROR: Exp data files not found. HDF5: {exp_hdf5_path}, CSV: {exp_fiducial_csv_path}")
except Exception as e:
    print(f"Error loading/processing experimental data: {e}")

# --- Registration and Transformation ---
trans_matrix = None
exp_coords_transformed_mm = None
if ebsd_fiducial_mm is not None and exp_fiducial_corners_mm_stage is not None:
    print("\n--- Data Registration ---")
    print(f"EBSD fiducial (mm):\n{ebsd_fiducial_mm}")
    print(f"Exp fiducial (mm from CSV):\n{exp_fiducial_corners_mm_stage}")

    # Ensure both have 4 points
    if ebsd_fiducial_mm.shape[0] == 4 and exp_fiducial_corners_mm_stage.shape[0] == 4:
        # Use SimilarityTransform (allows scale, rotation, translation)
        transform_obj = SimilarityTransform() # Renamed to avoid conflict if it's None
        if transform_obj.estimate(exp_fiducial_corners_mm_stage, ebsd_fiducial_mm): # src, dst
            trans_matrix = transform_obj.params
            print(f"Transformation matrix (Exp -> EBSD space):\n{trans_matrix}")

            # Transform all experimental grid coordinates
            exp_X_mesh, exp_Y_mesh = np.meshgrid(exp_x_coords, exp_y_coords)
            exp_coords_flat = np.vstack([exp_X_mesh.ravel(), exp_Y_mesh.ravel()]).T
            exp_coords_transformed_flat = transform_obj(exp_coords_flat) # Apply transform
            
            exp_X_transformed_mm = exp_coords_transformed_flat[:,0].reshape(exp_X_mesh.shape)
            exp_Y_transformed_mm = exp_coords_transformed_flat[:,1].reshape(exp_Y_mesh.shape)
            print("Experimental coordinates transformed.")
        else:
            print("ERROR: Could not estimate transformation. Check fiducial points.")
            # Ensure exp_X_transformed_mm remains None if transform fails
            exp_X_transformed_mm = None 
            exp_Y_transformed_mm = None
    else:
        print("ERROR: Fiducial point sets do not both have 4 points. Cannot estimate transform.")
else:
    print("Skipping registration: Missing EBSD or Experimental fiducial data.")

# --- Correlate Data: Interpolate measured SAW onto EBSD grain centroids ---
# This section now relies on exp_X_transformed_mm being successfully computed.
if exp_X_transformed_mm is not None and exp_Y_transformed_mm is not None and measured_saw_freq_map_exp is not None and not df_ebsd.empty:
    print("\n--- Correlating Data ---")
    points_to_interpolate_at = df_ebsd[['Centroid_X_mm_ebsd', 'Centroid_Y_mm_ebsd']].values
    
    # Prepare source points for griddata using the transformed coordinates
    # exp_X_mesh and exp_Y_mesh are defined in the transformation block if successful
    known_points_exp_transformed = np.array([exp_Y_transformed_mm.ravel(), exp_X_transformed_mm.ravel()]).T
    
    print("\nWARNING: Temporarily using dominant frequency (highest amplitude peak) from experimental data for interpolation.")
    # Reshape measured_saw_freq_map_exp correctly for iterating over spatial points
    num_exp_points = exp_X_transformed_mm.size
    temp_measured_map_flat = measured_saw_freq_map_exp.reshape(num_exp_points, N_PEAKS_TO_EXTRACT, 3)

    dominant_mu_values_flat = np.full(num_exp_points, np.nan) # Renamed from dominant_mu_values_exp for clarity
    for i in range(temp_measured_map_flat.shape[0]):
        peaks_at_point = temp_measured_map_flat[i, :, :] # Shape (N_PEAKS_TO_EXTRACT, 3) for A, mu, sigma
        valid_peaks_mask = ~np.isnan(peaks_at_point[:, 0]) # Check for NaNs in Amplitude
        if np.any(valid_peaks_mask):
            amps = peaks_at_point[valid_peaks_mask, 0]
            mus = peaks_at_point[valid_peaks_mask, 1]
            if len(amps) > 0:
                dominant_mu_values_flat[i] = mus[np.argmax(amps)]

    known_values_exp = dominant_mu_values_flat # Use this for griddata
    
    # Reshape for pcolormesh plotting later
    dominant_mu_map_exp_2d = dominant_mu_values_flat.reshape(exp_X_transformed_mm.shape)

    # Filter out NaNs from known experimental data to avoid issues with some interpolators
    nan_mask_exp = np.isnan(known_values_exp)
    known_points_exp_valid = known_points_exp_transformed[~nan_mask_exp] # Use transformed points for interpolation
    known_values_exp_valid = known_values_exp[~nan_mask_exp]

    if len(known_points_exp_valid) > 0:
        interpolated_measured_freq = griddata(
            known_points_exp_valid,          # Points where experimental data is known (transformed)
            known_values_exp_valid,          # The experimental SAW frequencies at these points
            points_to_interpolate_at,        # EBSD grain centroids (in EBSD mm space)
            method='nearest'                 # Use 'linear' or 'cubic' for smoother, 'nearest' for speed/robustness
        )
        df_ebsd['Measured SAW Frequency (Interp)'] = interpolated_measured_freq
        df_ebsd['Prediction Error (Hz)'] = df_ebsd['Predicted SAW Frequency'] - df_ebsd['Measured SAW Frequency (Interp)']
        print("Measured SAW frequencies interpolated onto EBSD grain centroids.")
        print(df_ebsd[['Predicted SAW Frequency', 'Measured SAW Frequency (Interp)', 'Prediction Error (Hz)']].head())
    else:
        print("No valid (non-NaN) experimental data points to interpolate from.")
        df_ebsd['Measured SAW Frequency (Interp)'] = np.nan
        df_ebsd['Prediction Error (Hz)'] = np.nan

else:
    print("Skipping data correlation due to missing transformed coordinates or data.")
    df_ebsd['Measured SAW Frequency (Interp)'] = np.nan
    df_ebsd['Prediction Error (Hz)'] = np.nan

# --- Visualization ---
print("\n--- Plotting Results ---")
fig_final, axs = plt.subplots(2, 2, figsize=(18, 14))
axs = axs.ravel() # Flatten array for easier indexing

# Plot 1: Predicted SAW from EBSD (using defdap's grain map plotting)
ax = axs[0]
if not df_ebsd.empty and 'Predicted SAW Frequency' in df_ebsd.columns:
    # Create a data array for plotGrainDataMap: value for each grain ID
    max_grain_id = df_ebsd['Grain ID'].max()
    predicted_freq_for_plot = np.full(max_grain_id + 1, np.nan)
    for _, row in df_ebsd.iterrows():
        if pd.notna(row['Predicted SAW Frequency']):
            predicted_freq_for_plot[int(row['Grain ID'])] = row['Predicted SAW Frequency']
    
    try:
        ebsd_map.plotGrainDataMap(data=predicted_freq_for_plot, ax=ax, caxlabel='Predicted SAW Freq (Hz)', cmap='viridis')
        ax.set_title('Predicted SAW Frequencies (EBSD Grains)')
    except Exception as e_defdap_plot:
        print(f"Error using defdap plot: {e_defdap_plot}. Falling back to scatter.")
        sc = ax.scatter(df_ebsd['Centroid_X_mm_ebsd'], df_ebsd['Centroid_Y_mm_ebsd'], c=df_ebsd['Predicted SAW Frequency'], cmap='viridis', s=5)
        plt.colorbar(sc, ax=ax, label='Predicted SAW Freq (Hz)')
        ax.set_title('Predicted SAW (EBSD - Scatter)')
    # Overlay EBSD fiducials
    if ebsd_fiducial_mm is not None:
        ax.add_patch(Polygon(ebsd_fiducial_mm, closed=True, fill=False, ec='red', lw=1.5, label='EBSD Fiducial'))
    ax.legend()
    ax.set_aspect('equal', adjustable='box')


# Plot 2: Measured SAW (transformed experimental data)
ax = axs[1]
if exp_X_transformed_mm is not None and measured_saw_freq_map_exp is not None:
    # Use the 2D dominant mu map for pcolormesh
    im = ax.pcolormesh(exp_X_transformed_mm, exp_Y_transformed_mm, dominant_mu_map_exp_2d, 
                        cmap='magma', shading='auto', 
                        vmin=np.nanpercentile(dominant_mu_map_exp_2d, 5), 
                        vmax=np.nanpercentile(dominant_mu_map_exp_2d, 95))
    plt.colorbar(im, ax=ax, label='Measured SAW Freq (Dominant Peak, Hz)')
    ax.set_title('Measured SAW Frequencies (Transformed Experimental - Dominant Peak)')
    if ebsd_fiducial_mm is not None: # Overlay EBSD fiducial for alignment check
        ax.add_patch(Polygon(ebsd_fiducial_mm, closed=True, fill=False, ec='cyan', lw=1.5, label='EBSD Fiducial (Target)'))
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (EBSD local mm)')
    ax.set_ylabel('Y (EBSD local mm)')
else:
    ax.text(0.5, 0.5, "Experimental data not processed/transformed", ha='center', va='center')
ax.set_title('Measured SAW (Transformed Experimental)')


# Plot 3: Prediction Error Map (on EBSD grains)
ax = axs[2]
if not df_ebsd.empty and 'Prediction Error (Hz)' in df_ebsd.columns and df_ebsd['Prediction Error (Hz)'].notna().any():
    max_grain_id_err = df_ebsd['Grain ID'].max()
    error_for_plot = np.full(max_grain_id_err + 1, np.nan)
    for _, row in df_ebsd.iterrows():
        if pd.notna(row['Prediction Error (Hz)']):
            error_for_plot[int(row['Grain ID'])] = row['Prediction Error (Hz)']
    
    # Determine symmetric color limits for error
    err_abs_max = np.nanpercentile(np.abs(df_ebsd['Prediction Error (Hz)']), 98) if df_ebsd['Prediction Error (Hz)'].notna().any() else 1e6

    try:
        ebsd_map.plotGrainDataMap(data=error_for_plot, ax=ax, caxlabel='Prediction Error (Hz)', cmap='coolwarm', vmin=-err_abs_max, vmax=err_abs_max)
        ax.set_title('Prediction Error (Predicted - Measured)')
    except Exception as e_defdap_plot_err:
        print(f"Error using defdap plot for error: {e_defdap_plot_err}. Falling back to scatter.")
        sc_err = ax.scatter(df_ebsd['Centroid_X_mm_ebsd'], df_ebsd['Centroid_Y_mm_ebsd'], c=df_ebsd['Prediction Error (Hz)'], cmap='coolwarm', s=5, vmin=-err_abs_max, vmax=err_abs_max)
        plt.colorbar(sc_err, ax=ax, label='Prediction Error (Hz)')
        ax.set_title('Prediction Error (EBSD - Scatter)')
    if ebsd_fiducial_mm is not None:
        ax.add_patch(Polygon(ebsd_fiducial_mm, closed=True, fill=False, ec='black', lw=1, ls='--'))
    ax.set_aspect('equal', adjustable='box')

else:
    ax.text(0.5, 0.5, "Error data not available", ha='center', va='center')
ax.set_title('Prediction Error Map')


# Plot 4: Histogram of Prediction Errors
ax = axs[3]
if not df_ebsd.empty and 'Prediction Error (Hz)' in df_ebsd.columns and df_ebsd['Prediction Error (Hz)'].notna().any():
    error_values = df_ebsd['Prediction Error (Hz)'].dropna()
    ax.hist(error_values, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Prediction Error (Hz)')
    ax.set_ylabel('Number of Grains')
    ax.set_title('Histogram of Prediction Errors')
    ax.grid(True, alpha=0.3)
    mean_err = error_values.mean()
    std_err = error_values.std()
    ax.text(0.05, 0.95, f"Mean: {mean_err:.2e} Hz\nStd: {std_err:.2e} Hz", transform=ax.transAxes, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

else:
    ax.text(0.5, 0.5, "Error data not available for histogram", ha='center', va='center')
ax.set_title('Error Histogram')


# Add a new debug plot for fiducial fitting
if best_fit_params is not None and no_signal_mask is not None and hasattr(ebsd_map, 'bandContrastArray'):
    fig_debug_fit, ax_debug_fit = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Band contrast with fitted fiducial
    ax_debug_fit[0].imshow(ebsd_map.bandContrastArray, cmap='gray', origin='lower')
    fitted_corners_plot_px = best_fit_params['corners_px']
    # Create a polygon patch (corners are x,y)
    poly = Polygon(fitted_corners_plot_px, closed=True, fill=False, ec='red', lw=1.5, label='Fitted EBSD Fiducial')
    ax_debug_fit[0].add_patch(poly)
    ax_debug_fit[0].scatter(best_fit_params['center_px'][0], best_fit_params['center_px'][1], c='cyan', marker='+', s=100, label='Fitted Center')
    ax_debug_fit[0].set_title(f"EBSD BC Map with Fitted Fiducial\nCenter: {best_fit_params['center_px']}, Angle: {best_fit_params['angle_deg']:.1f} deg, Score: {best_fit_params['score']}")
    ax_debug_fit[0].legend()
    ax_debug_fit[0].set_aspect('equal', adjustable='box')

    # Plot 2: "No signal" mask with fitted fiducial band
    ax_debug_fit[1].imshow(no_signal_mask, cmap='gray', origin='lower')
    # Draw the band used for scoring for visualization
    band_rows_viz, band_cols_viz = get_rotated_square_band_pixels(
        best_fit_params['center_px'],
        EBSD_FIDUCIAL_SIDE_LENGTH_MM / ebsd_step_size_mm,
        EBSD_FIDUCIAL_LINE_THICKNESS_MM / ebsd_step_size_mm,
        best_fit_params['angle_deg'],
        ebsd_map.bandContrastArray.shape
    )
    # Create an overlay for the band
    band_overlay_viz = np.zeros((*ebsd_map.bandContrastArray.shape, 3), dtype=np.uint8) # RGB
    band_overlay_viz[band_rows_viz, band_cols_viz, 0] = 255 # Red channel for band
    ax_debug_fit[1].imshow(band_overlay_viz, alpha=0.3) # Overlay with transparency
    ax_debug_fit[1].set_title("'No Signal' Mask & Fitted Fiducial Band Outline")
    ax_debug_fit[1].set_aspect('equal', adjustable='box')

    plt.suptitle("Debug Info: Programmatic EBSD Fiducial Fitting")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle


# plt.tight_layout(pad=2.0) # This was for the main 2x2 plot
# Ensure all plots are shown
plt.show()

df_ebsd.to_csv('ebsd_grains_with_saw_comparison.csv', index=False)
print(f"\nSaved final EBSD grain data with comparisons to 'ebsd_grains_with_saw_comparison.csv'")
print("Script complete.")
