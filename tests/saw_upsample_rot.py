#!/usr/bin/env python3
"""
SAW-FFT up-sampler  • 2025-05-23
--------------------------------
* Refits Gaussian peaks (returns ⟨μ, A, σ⟩ in that order – fixed)
* 10× bilinear up-sampling with amplitude weighting
* Optional raw-FFT inspection (--inspect-mm X Y)
* Overlay of corner polygon + fiducial, with line thickness scaled to data units.
* User colour limits via --vmin / --vmax  (MHz, default 250-350)
* Added minimum relative peak height for peak detection.
* Added option to rotate map based on fiducial position.
"""

import argparse, sys, time
import numpy as np, h5py, scipy.signal as sig, scipy.optimize as opt, scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
import pandas as pd
try:
    import cupy as cp
except ImportError:
    cp = None


# ---------- helper: choose NumPy / CuPy ----------
def xp(use_gpu: bool):
    """
    Selects NumPy or CuPy backend.
    Args:
        use_gpu (bool): If True, attempts to use CuPy.
    Returns:
        module: np or cp.
    """
    if use_gpu and cp is not None:
        print("[GPU] CuPy backend"); return cp
    if use_gpu: print("[GPU] CuPy not found – using NumPy")
    return np


# ---------- peak refit (A, μ, σ  →  μ, A, σ) ----------
def _gauss(x, A, mu, sig_val):
    """
    Gaussian function definition.
    Args:
        x (array-like): Input x values.
        A (float): Amplitude.
        mu (float): Mean.
        sig_val (float): Standard deviation.
    Returns:
        array-like: Gaussian values.
    """
    return A * np.exp(-(x-mu)**2 / (2*sig_val**2))

def refine(freq, amp, N=3, prom=0.04, min_peak_height_rel=0.1):
    """
    Refits Gaussian peaks to FFT data.
    Args:
        freq (array-like): Frequency array.
        amp (array-like): Amplitude array.
        N (int): Maximum number of peaks to find.
        prom (float): Prominence for peak finding.
        min_peak_height_rel (float): Minimum peak height relative to max amplitude of the trace.
    Returns:
        list: List of tuples (mu, A, sigma) for each peak, up to N.
    """
    if amp.size == 0 or amp.max() <= 1e-9: # Handle empty or effectively all-zero amplitude arrays
        out = []
        while len(out) < N:
            out.append((np.nan, np.nan, np.nan))
        return out

    # Find peaks based on prominence and relative height
    min_abs_height = amp.max() * min_peak_height_rel
    idx, prop = sig.find_peaks(amp, prominence=prom, height=min_abs_height)
    
    # Sort peaks by prominence (descending) and take at most N peaks
    sorted_indices = np.argsort(prop["prominences"])[::-1]
    idx = idx[sorted_indices[:N]]
    
    out = []
    for p_idx in idx: # Iterate through the filtered and sorted peak indices
        # Define a slice around the peak for fitting
        sl = slice(max(0, p_idx - 3), min(len(freq), p_idx + 4))
        try:
            A0, mu0 = amp[p_idx], freq[p_idx] # Initial guesses for amplitude and mean
            # Perform curve fitting
            popt, _ = opt.curve_fit(_gauss, freq[sl], amp[sl],
                                    p0=(A0, mu0, 5e5), maxfev=5000) # p0 is (A, mu, sigma), added maxfev
            # Store fitted parameters (mu, A, sigma)
            out.append((popt[1], popt[0], popt[2])) # Note: storing as mu, A, sigma
        except RuntimeError:
            # If fitting fails, use initial peak values with a default sigma
            out.append((freq[p_idx], amp[p_idx], 6e5)) # mu, A, default_sigma
    
    # Fill with NaNs if fewer than N peaks are found/fitted
    while len(out) < N:
        out.append((np.nan, np.nan, np.nan))
    return out


# ---------- peak matching & cell up-sample ----------
def same(p1, p2, k):
    """
    Checks if two peaks are "the same" based on their means and sigmas.
    Args:
        p1 (tuple): First peak (mu, A, sigma).
        p2 (tuple): Second peak (mu, A, sigma).
        k (float): Tolerance factor.
    Returns:
        bool: True if peaks are considered the same.
    """
    mu1, _, s1 = p1
    mu2, _, s2 = p2
    if not (np.isfinite(mu1) and np.isfinite(mu2) and np.isfinite(s1) and np.isfinite(s2)):
        return False
    # Avoid issues with zero sigma if peaks are perfectly sharp (should not happen with fitting)
    s1 = max(s1, 1e-9) 
    s2 = max(s2, 1e-9)
    return abs(mu1 - mu2) < k * np.hypot(s1, s2)

def groups(corners, k):
    """
    Groups similar peaks from corner data.
    Args:
        corners (list): List of peak lists for each of the 4 corners.
                        Each peak is (mu, A, sigma).
        k (float): Tolerance factor for `same` function.
    Returns:
        list: List of groups, where each group has a representative peak
              and a list of member peaks with their corner index.
    """
    g = [] 
    for ci, plist in enumerate(corners): 
        for p_item in plist: 
            if not np.isfinite(p_item[0]): 
                continue
            added_to_group = False
            for G_group in g:
                if G_group and G_group[0].get("p") and same(p_item, G_group[0]["p"], k): # Check G_group[0] exists
                    G_group.append({"ci": ci, "p": p_item})
                    added_to_group = True
                    break
            if not added_to_group:
                g.append([{"ci": ci, "p": p_item}])
    return [{"rep": group_members[0]["p"], "members": group_members} 
            for group_members in g if group_members and group_members[0].get("p")]


def upsample(corners, m=10, k=2.):
    """
    Upsamples a single cell based on its corner peaks.
    Args:
        corners (list): List of peak lists for the 4 corners of the cell.
        m (int): Upsampling factor (e.g., 10 means 10x10 sub-pixels).
        k (float): Tolerance factor for grouping peaks.
    Returns:
        tuple: (f_upsampled, a_upsampled)
               f_upsampled: Upsampled frequency map (m x m).
               a_upsampled: Upsampled amplitude map (m x m).
    """
    yy, xx = np.mgrid[0:m, 0:m] / (m - 1)
    W = [(1-yy)*(1-xx), (1-yy)*xx, yy*(1-xx), yy*xx] 

    f_upsampled = np.full((m, m), np.nan, np.float32) 
    a_upsampled = np.zeros_like(f_upsampled)          

    peak_groups = groups(corners, k)
    for G in peak_groups:
        num = np.zeros_like(f_upsampled, dtype=np.float64) 
        den = np.zeros_like(f_upsampled, dtype=np.float64) 
        for d_member in G["members"]:
            mu_peak, A_peak, _ = d_member["p"] 
            if not (np.isfinite(mu_peak) and np.isfinite(A_peak) and A_peak > 1e-9): 
                continue
            num += W[d_member["ci"]] * A_peak * mu_peak 
            den += W[d_member["ci"]] * A_peak          
        
        mu_group_avg = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9) 
        msk_valid = den > 1e-9 
        sel_dominant = (den > a_upsampled) & msk_valid
        
        f_upsampled[sel_dominant] = mu_group_avg[sel_dominant]
        a_upsampled[sel_dominant] = den[sel_dominant]

    if np.isnan(f_upsampled).any():
        dom_freqs_at_corners = []
        for corner_peaks_list in corners: 
            valid_peaks_in_corner = [p for p in corner_peaks_list if np.isfinite(p[0]) and np.isfinite(p[1]) and p[1] > 1e-9]
            if valid_peaks_in_corner:
                dominant_peak = max(valid_peaks_in_corner, key=lambda t: t[1])
                dom_freqs_at_corners.append(dominant_peak[0]) 
            else:
                dom_freqs_at_corners.append(np.nan)
        
        # Bilinear interpolation for NaNs using dominant corner frequencies
        bil_interpolated_f = np.full((m, m), np.nan, dtype=np.float64)
        total_weights = np.zeros((m,m), dtype=np.float64)
        temp_sum_freq = np.zeros((m,m), dtype=np.float64) # Temporary sum for frequencies

        all_corners_nan = True
        for i_w, w_corner_factor in enumerate(W):
            if np.isfinite(dom_freqs_at_corners[i_w]):
                all_corners_nan = False
                temp_sum_freq += w_corner_factor * dom_freqs_at_corners[i_w]
                total_weights += w_corner_factor
        
        if not all_corners_nan:
            valid_bil_mask = total_weights > 1e-9
            bil_interpolated_f[valid_bil_mask] = temp_sum_freq[valid_bil_mask] / total_weights[valid_bil_mask]

        nan_mask_in_f = np.isnan(f_upsampled)
        f_upsampled[nan_mask_in_f] = bil_interpolated_f[nan_mask_in_f]
            
    return f_upsampled, a_upsampled


# ---------- overlay helpers ----------
def load_csv(path):
    """
    Loads corner coordinates and fiducial marker from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        tuple: (poly_coords, fid_coords)
               poly_coords: NumPy array of (X, Y) for polygon corners.
               fid_coords: NumPy array of (X, Y) for fiducial.
    """
    df = pd.read_csv(path)
    # Ensure "Fiducial" column exists and has at least one 1
    if "Fiducial" not in df.columns or not (df["Fiducial"] == 1).any():
        raise ValueError("CSV file must contain a 'Fiducial' column with at least one entry marked as 1.")
    fid_coords = df.loc[df["Fiducial"] == 1, ["X_coord", "Y_coord"]].to_numpy(float)[0]
    poly_coords = df[["X_coord", "Y_coord"]].to_numpy(float)
    return poly_coords, fid_coords

def overlay(ax, poly, fid, line_thickness_data_mm, fid_radius_data_mm):
    """
    Overlays a polygon and a fiducial marker on a matplotlib Axes.
    The line thickness and fiducial size are scaled to the data coordinates.
    """
    fig = ax.get_figure()
    x_data_min, x_data_max = ax.get_xlim()
    
    lw_points, fid_dot_points = 1.0, 1.0 # Defaults

    if abs(x_data_max - x_data_min) < 1e-9: 
        print_warning = getattr(overlay, 'print_warning', True) 
        if print_warning:
            print("[Warning] X-axis data range is effectively zero. Overlay line thickness may not scale as expected.")
            overlay.print_warning = False 
        lw_points, fid_dot_points = 0.5, 1.0
    else:
        data_width_mm = x_data_max - x_data_min
        bbox_inches = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axis_width_inches = bbox_inches.width
        
        if axis_width_inches <= 1e-6: 
            print_warning_axis = getattr(overlay, 'print_warning_axis', True)
            if print_warning_axis:
                print("[Warning] Axis width in inches is effectively zero. Overlay may not scale as expected.")
                overlay.print_warning_axis = False
            lw_points, fid_dot_points = 0.5, 1.0
        else:
            mm_per_inch_on_axis = data_width_mm / axis_width_inches
            if mm_per_inch_on_axis <= 1e-6:
                lw_points, fid_dot_points = 0.5, 1.0
            else:
                line_thickness_inches_on_axis = line_thickness_data_mm / mm_per_inch_on_axis
                lw_points = line_thickness_inches_on_axis * 72.0
                fid_dot_diameter_data_mm = line_thickness_data_mm 
                fid_dot_inches_on_axis = fid_dot_diameter_data_mm / mm_per_inch_on_axis
                fid_dot_points = fid_dot_inches_on_axis * 72.0

    lw_points = max(lw_points, 0.1) 
    fid_dot_points = max(fid_dot_points, 0.5)

    ax.add_patch(Polygon(poly, closed=True, fill=False, ec='k', lw=lw_points))
    
    cx, cy = fid
    mx, my = poly.mean(0) 
    ang = (np.degrees(np.arctan2(my-cy, mx-cx)) + 180) % 360
    
    arc_diameter_data_mm = fid_radius_data_mm * 2.0
    ax.add_patch(Arc((cx, cy), 
                     width=arc_diameter_data_mm, 
                     height=arc_diameter_data_mm, 
                     theta1=ang-135, theta2=ang+135,
                     ec='k', lw=lw_points)) 

    ax.plot(cx, cy, 'ko', ms=fid_dot_points, mec='k', mew=max(lw_points*0.1, 0.05) )

def rotate_point(px, py, angle_rad, cx, cy):
    """Rotates a point (px,py) around a center (cx,cy) by angle_rad."""
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    px_c = px - cx
    py_c = py - cy
    px_rot = px_c * c - py_c * s + cx
    py_rot = px_c * s + py_c * c + cy
    return px_rot, py_rot

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="SAW-FFT Up-sampler. Overlay line/fiducial sizes are in data units and scale with plot.")
    p.add_argument("--fft", required=True, help="Path to HDF5 file with FFT data.")
    p.add_argument("--n-peaks", type=int, default=3, help="Maximum number of peaks to fit per FFT trace.")
    p.add_argument("--k", type=float, default=2.0, help="Tolerance for matching peaks (factor of sigma sum).")
    p.add_argument("--upsamp", type=int, default=10, help="Upsampling factor for each dimension (e.g., 10 for 10x10).")
    p.add_argument("--min-peak-height-rel", type=float, default=0.1, 
                     help="Minimum peak height relative to the max amplitude of the current FFT trace (0.0 to 1.0). Default: 0.1 (10%%).")
    p.add_argument("--prominence", type=float, default=0.04, help="Required prominence of peaks. Default: 0.04.")
    p.add_argument("--gpu", action="store_true", help="Use CuPy for GPU acceleration if available.")
    p.add_argument("--corners", help="Path to CSV file with corner and fiducial coordinates for overlay.")
    p.add_argument("--line-mm", type=float, default=0.15, 
                     help="Target linewidth for overlay in data units (mm), scales with plot. E.g., 0.15 for 150 microns.")
    p.add_argument("--fid-mm", type=float, default=0.25, 
                     help="Target radius for fiducial arc in data units (mm), scales with plot.")
    p.add_argument("--rotate-fiducial", type=str, default="NE", 
                     choices=['none', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                     help="Rotate map so fiducial is in specified direction (e.g., NE for North-East/Upper-Right). Default: NE.")
    p.add_argument("--vmin", type=float, default=250, help="Min frequency for colormap (MHz).")
    p.add_argument("--vmax", type=float, default=350, help="Max frequency for colormap (MHz).")
    p.add_argument("--inspect-mm", nargs=2, type=float, metavar=("X", "Y"),
                     help="Inspect raw FFT at specified X Y (mm) coords and exit.")
    p.add_argument("--show", action="store_true", help="Show the final upsampled plot.")
    args = p.parse_args()
    
    # Load HDF5 data
    print(f"Loading FFT data from: {args.fft}")
    with h5py.File(args.fft, 'r') as h5:
        freq_coords = h5["/freq"][:]
        x_coords_orig = h5["/X"][:] 
        y_coords_orig = h5["/Ycoord"][:] 
        raw_amplitude_data = h5["/amplitude"]
        freq_axis_index = -1
        for i, s_dim in enumerate(raw_amplitude_data.shape):
            if s_dim == len(freq_coords):
                freq_axis_index = i
                break
        if freq_axis_index == -1:
            if len(raw_amplitude_data.shape) == 3 and raw_amplitude_data.shape[0] == len(freq_coords): 
                 freq_axis_index = 0
            elif len(raw_amplitude_data.shape) == 3 and raw_amplitude_data.shape[2] == len(freq_coords): 
                 freq_axis_index = 2
            else: 
                raise ValueError(f"Frequency dimension not found or length mismatch in amplitude data. Amplitude shape: {raw_amplitude_data.shape}, Freq length: {len(freq_coords)}")

        A_all_ffts = raw_amplitude_data[()]
        if freq_axis_index != 0: 
            A_all_ffts = np.moveaxis(A_all_ffts, freq_axis_index, 0)
            
    if A_all_ffts.shape[0] != len(freq_coords):
        raise ValueError(f"Mismatch after axis move. Amp shape: {A_all_ffts.shape}, Freq len: {len(freq_coords)}")

    if np.nanmax(freq_coords) < 20: 
        print("Frequencies appear to be in GHz (max < 20), converting to Hz.")
        freq_coords *= 1e9
        
    _Flen, Ny_orig, Nx_orig = A_all_ffts.shape

    # For extent and rotation, use copies that might be modified
    current_x_coords = np.copy(x_coords_orig)
    current_y_coords = np.copy(y_coords_orig)


    if args.inspect_mm:
        inspect_x, inspect_y = args.inspect_mm
        ix_inspect = np.abs(current_x_coords - inspect_x).argmin()
        iy_inspect = np.abs(current_y_coords - inspect_y).argmin()
        plt.figure(figsize=(8, 6))
        plt.plot(freq_coords / 1e6, A_all_ffts[:, iy_inspect, ix_inspect])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(f"Raw FFT @ X={current_x_coords[ix_inspect]:.2f} mm, Y={current_y_coords[iy_inspect]:.2f} mm")
        plt.grid(True)
        plt.show()
        sys.exit()

    print(f"Performing peak refitting for {Ny_orig}x{Nx_orig} grid points using N_max={args.n_peaks}, prom={args.prominence}, min_height_rel={args.min_peak_height_rel}...")
    peaks_data = np.empty((Ny_orig, Nx_orig, args.n_peaks, 3), np.float32)
    t0 = time.time()
    for iy_idx in range(Ny_orig):
        for ix_idx in range(Nx_orig):
            current_amp_trace = A_all_ffts[:, iy_idx, ix_idx]
            peaks_data[iy_idx, ix_idx] = refine(freq_coords, current_amp_trace, args.n_peaks, prom=args.prominence, min_peak_height_rel=args.min_peak_height_rel)
        if (iy_idx + 1) % 10 == 0 or (iy_idx + 1) == Ny_orig:
            elapsed_time = time.time() - t0
            print(f" Processed row {iy_idx+1}/{Ny_orig}   {elapsed_time:.1f}s")
    print("Peak refitting complete.")

    print(f"Upsampling grid by a factor of {args.upsamp}...")
    NY_upsampled, NX_upsampled = Ny_orig * args.upsamp, Nx_orig * args.upsamp
    hi_res_f = np.full((NY_upsampled, NX_upsampled), np.nan, np.float32)
    hi_res_a = np.zeros_like(hi_res_f) # hi_res_a is not globally rotated currently

    for iy_orig_cell in range(Ny_orig - 1):
        for ix_orig_cell in range(Nx_orig - 1):
            corner_peaks_list = [
                peaks_data[iy_orig_cell,     ix_orig_cell].tolist(),
                peaks_data[iy_orig_cell,     ix_orig_cell + 1].tolist(),
                peaks_data[iy_orig_cell + 1, ix_orig_cell].tolist(),
                peaks_data[iy_orig_cell + 1, ix_orig_cell + 1].tolist()
            ]
            f_block_upsampled, a_block_upsampled = upsample(corner_peaks_list, args.upsamp, args.k)
            y_slice_target = slice(iy_orig_cell * args.upsamp, (iy_orig_cell + 1) * args.upsamp)
            x_slice_target = slice(ix_orig_cell * args.upsamp, (ix_orig_cell + 1) * args.upsamp)
            hi_res_f[y_slice_target, x_slice_target] = f_block_upsampled
            hi_res_a[y_slice_target, x_slice_target] = a_block_upsampled # hi_res_a filled here
        if (iy_orig_cell + 1) % 10 == 0 or (iy_orig_cell + 1) == (Ny_orig -1) :
             print(f" Upsampled cell row {iy_orig_cell+1}/{Ny_orig-1}")
    print("Main upsampling loop complete.")

    print("Filling edges of the upsampled map...")
    if Ny_orig > 1 and args.upsamp > 0: 
        last_processed_block_start_row = (Ny_orig - 2) * args.upsamp
        target_fill_rows_slice = slice((Ny_orig - 1) * args.upsamp, NY_upsampled)
        if last_processed_block_start_row + args.upsamp <= NY_upsampled and last_processed_block_start_row >=0 and target_fill_rows_slice.stop > target_fill_rows_slice.start:
            source_row_f_data = hi_res_f[last_processed_block_start_row + args.upsamp - 1, :]
            hi_res_f[target_fill_rows_slice, :] = source_row_f_data[None, :] 
    if Nx_orig > 1 and args.upsamp > 0: 
        last_processed_block_start_col = (Nx_orig - 2) * args.upsamp
        target_fill_cols_slice = slice((Nx_orig - 1) * args.upsamp, NX_upsampled)
        if last_processed_block_start_col + args.upsamp <= NX_upsampled and last_processed_block_start_col >=0 and target_fill_cols_slice.stop > target_fill_cols_slice.start:
            source_col_f_data = hi_res_f[:, last_processed_block_start_col + args.upsamp - 1]
            hi_res_f[:, target_fill_cols_slice] = source_col_f_data[:, None]
    
    # Prepare for plotting (extent and overlay coords might be rotated)
    plot_extent = [current_x_coords[0], current_x_coords[-1], current_y_coords[0], current_y_coords[-1]]
    poly_coords_to_plot, fid_coords_to_plot = None, None

    if args.corners:
        try:
            poly_coords_to_plot, fid_coords_to_plot = load_csv(args.corners)
        except Exception as e:
            print(f"Warning: Could not load corners/fiducial CSV: {e}. Overlay and rotation might be affected.")
            args.corners = None # Disable further corner-dependent operations if loading failed

    # Rotation logic
    if args.rotate_fiducial != 'none' and args.corners and fid_coords_to_plot is not None:
        print(f"Rotating map to place fiducial towards: {args.rotate_fiducial}")
        center_of_extent_x = (plot_extent[0] + plot_extent[1]) / 2.0
        center_of_extent_y = (plot_extent[2] + plot_extent[3]) / 2.0

        vector_to_fid_x = fid_coords_to_plot[0] - center_of_extent_x
        vector_to_fid_y = fid_coords_to_plot[1] - center_of_extent_y
        current_angle_rad = np.arctan2(vector_to_fid_y, vector_to_fid_x)

        target_angles_map = {
            'N':  np.pi / 2, 'NE': np.pi / 4, 'E':  0, 'SE': -np.pi / 4,
            'S': -np.pi / 2, 'SW': -3 * np.pi / 4, 'W': np.pi, 'NW': 3 * np.pi / 4
        }
        target_angle_rad = target_angles_map[args.rotate_fiducial]
        
        # scipy.ndimage.rotate rotates counter-clockwise for positive angle.
        # We want the fiducial vector (currently at current_angle_rad) to align with target_angle_rad *after* data rotation.
        # If data rotates by rot_data, new_fid_angle = current_angle_rad + rot_data.
        # So, rot_data = target_angle_rad - current_angle_rad.
        rotation_rad_for_data = target_angle_rad - current_angle_rad
        rotation_deg_for_data = np.degrees(rotation_rad_for_data)

        print(f" Current fiducial angle: {np.degrees(current_angle_rad):.1f} deg. Target: {np.degrees(target_angle_rad):.1f} deg. Data rotation: {rotation_deg_for_data:.1f} deg.")

        # Rotate the hi_res_f data array
        # Note: hi_res_a is not rotated here as it's not directly plotted.
        # If it were, it should be rotated consistently.
        hi_res_f = scipy.ndimage.rotate(hi_res_f, rotation_deg_for_data, reshape=True, cval=np.nan, order=1, mode='constant')

        # Calculate new extent for the rotated image
        # Original data width and height
        W_data_orig = plot_extent[1] - plot_extent[0]
        H_data_orig = plot_extent[3] - plot_extent[2]
        
        # New bounding box dimensions after rotating the original extent box
        cos_abs = abs(np.cos(rotation_rad_for_data))
        sin_abs = abs(np.sin(rotation_rad_for_data))
        W_data_rot = W_data_orig * cos_abs + H_data_orig * sin_abs
        H_data_rot = W_data_orig * sin_abs + H_data_orig * cos_abs
        
        plot_extent = [
            center_of_extent_x - W_data_rot / 2.0,
            center_of_extent_x + W_data_rot / 2.0,
            center_of_extent_y - H_data_rot / 2.0,
            center_of_extent_y + H_data_rot / 2.0
        ]

        # Rotate overlay coordinates
        poly_coords_to_plot = np.array([
            rotate_point(px, py, rotation_rad_for_data, center_of_extent_x, center_of_extent_y) 
            for px, py in poly_coords_to_plot
        ])
        fid_coords_to_plot = np.array(
            rotate_point(fid_coords_to_plot[0], fid_coords_to_plot[1], rotation_rad_for_data, center_of_extent_x, center_of_extent_y)
        )
    elif args.rotate_fiducial != 'none':
        print(f"Warning: Rotation requested ({args.rotate_fiducial}) but --corners not specified or failed to load. Skipping rotation.")

    print(f"[debug] Final plot_extent: [{plot_extent[0]:.2f}, {plot_extent[1]:.2f}, {plot_extent[2]:.2f}, {plot_extent[3]:.2f}]")
    if np.isnan(hi_res_f).all():
        print("[Warning] hi_res_f is all NaN before plotting. Check upsampling and peak finding.")
    else:
        print(f"[debug] hi_res_f range before plot: {np.nanmin(hi_res_f)/1e6:.2f} – {np.nanmax(hi_res_f)/1e6:.2f} MHz")


    if args.show:
        print("Generating plot...")
        fig, ax = plt.subplots(figsize=(9,7))
        im = ax.imshow(
            hi_res_f / 1e6, 
            origin='lower', 
            extent=plot_extent, 
            aspect='equal',
            cmap='viridis', 
            vmin=args.vmin, 
            vmax=args.vmax
        )
        cbar = fig.colorbar(im, ax=ax, label="Peak Frequency (MHz)", shrink=0.8)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("High-Resolution SAW Peak-Frequency Map")
        
        if args.corners and poly_coords_to_plot is not None and fid_coords_to_plot is not None:
            try:
                overlay(ax, poly_coords_to_plot, fid_coords_to_plot, args.line_mm, args.fid_mm)
                print(f"Overlayed polygon and fiducial from {args.corners}")
            except Exception as e:
                print(f"Error drawing overlay: {e}")
        
        plt.tight_layout() 
        plt.show()
    else:
        print("Plotting skipped as --show was not specified.")
    
    print("Script finished.")

if __name__ == "__main__":
    main()

