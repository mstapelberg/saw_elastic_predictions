# saw_elastic_predictions/src/saw_calculator.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, det
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from euler_transformations import C_modifi, EulerAngles # Assuming euler_transformations.py is in the same directory or in PYTHONPATH

class SAWCalculator:
    def __init__(self, material, euler_angles):
        """
        Initializes the SAWCalculator with material properties and Euler angles.

        Args:
            material: An object representing the material. Must have methods:
                      get_cijkl() returning the (3,3,3,3) elastic tensor C
                      get_density() returning the density rho
            euler_angles: A list or numpy array of three Euler angles in radians.

        Raises:
            ValueError: If material or euler_angles are invalid
            TypeError: If material doesn't have required methods
        """
        # Validate material
        if not hasattr(material, 'get_cijkl') or not hasattr(material, 'get_density'):
            raise TypeError("Material must have get_cijkl() and get_density() methods")
        
        # Validate euler angles
        euler_angles = np.asarray(euler_angles)
        if euler_angles.shape != (3,):
            raise ValueError("euler_angles must be a 3-element array")
        
        self.material = material
        self.euler_angles = euler_angles

    def get_saw_speed(self, deg, sampling=4000, psaw=0, draw_plot=False):
        """
        Calculates the SAW velocity and optionally plots the displacement-slowness profile.

        Args:
            deg: float, the angle rotated on surface plane referred to y direction, 0<=deg<180 in degree.
            sampling: int, resolution in k space (e.g., 400, 4000, 40000).
            psaw: int, flag for PSAW calculation (0 for SAW only, 1 for SAW and PSAW).
            draw_plot: bool, if True, generates a plot of displacement vs slowness.

        Returns:
            v: numpy array, SAW velocity(ies).
            index: numpy array, direction of SAW in crystalline coordinates.
            intensity: numpy array, intensity of SAW mode(s).

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If calculation fails
        """
        # Validate inputs
        if not isinstance(deg, (int, float)):
            raise TypeError("deg must be a number")
        if not isinstance(sampling, int):
            raise TypeError("sampling must be an integer")
        if not isinstance(psaw, int) or psaw not in [0, 1]:
            raise ValueError("psaw must be 0 or 1")
        
        if deg < 0 or deg >= 180:
            raise ValueError("deg must be in range [0, 180)")
        if sampling <= 0:
            raise ValueError("sampling must be positive")

        # Get material properties
        try:
            C = self.material.get_cijkl()
            rho = self.material.get_density()
        except Exception as e:
            raise RuntimeError(f"Failed to get material properties: {str(e)}")

        # Validate material properties
        if not isinstance(C, np.ndarray) or C.shape != (3, 3, 3, 3):
            raise ValueError("Invalid elastic tensor shape")
        if not isinstance(rho, (int, float)) or rho <= 0:
            raise ValueError("Invalid density value")

        # --- Parameter Checks ---
        self._check_euler_range(self.euler_angles)
        self._check_deg_range(deg)

        # --- Sampling Parameter ---
        T = self._get_sampling_parameter(sampling)

        # --- Initialization ---
        lambda_val = 7 * 10 ** -6  # m
        k0 = 2 * np.pi / lambda_val
        w0 = 2 * np.pi / T
        w = w0 + complex(0, 0.000001 * w0)

        # --- Alignment and Transformation ---
        try:
            MM = self._get_alignment_matrix(deg)
            rotation_crystal_sample = EulerAngles.to_matrix(self.euler_angles) @ MM.T
            C_transformed = C_modifi(C, rotation_crystal_sample)
            index = rotation_crystal_sample @ np.array([0, 1, 0])
        except Exception as e:
            raise RuntimeError(f"Failed during transformation: {str(e)}")

        # --- Core SAW Calculation ---
        try:
            G33, ynew, slownessnew = self._calculate_g33(C_transformed, rho, w, k0, sampling)
        except Exception as e:
            raise RuntimeError(f"Failed during G33 calculation: {str(e)}")

        if len(slownessnew) == 0:
            raise RuntimeError("No valid SAW solutions found")

        # --- Calculate velocities ---
        v = 1.0 / slownessnew

        # --- Intensity Calculation ---
        try:
            intensity = self._calculate_intensity(ynew, slownessnew)
        except Exception as e:
            raise RuntimeError(f"Failed during intensity calculation: {str(e)}")

        # --- Optional Plotting ---
        if draw_plot:
            try:
                self._plot_saw_profile(G33, k0, w, slownessnew, psaw, ynew)
            except Exception as e:
                print(f"Warning: Failed to generate plot: {str(e)}")

        return v, index, intensity

    def _check_euler_range(self, Euler):
        """Check if Euler angles are in reasonable range."""
        if np.any(np.abs(Euler) > 2 * np.pi):
            raise ValueError("Euler angles must be in radians and within [-2π, 2π]")

    def _check_deg_range(self, deg):
        """Check if degree is in valid range."""
        if not 0 <= deg < 180:
            raise ValueError("deg must be in range [0, 180)")

    # --- Helper Functions ---
    def _get_sampling_parameter(self, sampling):
        if sampling == 40000:
            T = 20 * 10 ** -14
        elif sampling == 4000:
            T = 20 * 10 ** -13
        elif sampling == 400:
            T = 20 * 10 ** -12
        else:
            T = 20 * 10 ** -12
            print("Warning: Sampling is not a value often used.")
        return T

    def _get_alignment_matrix(self, deg):
        MM = np.array([[np.cos(np.deg2rad(deg)), np.cos(np.deg2rad(90 - deg)), 0],
                       [np.cos(np.deg2rad(90 + deg)), np.cos(np.deg2rad(deg)), 0],
                       [0, 0, 1]]) # Corrected to be 3x3
        return MM

    def _calculate_g33(self, C_transformed, rho, w, k0, sampling):
        """
        Calculate G33 and related values.
        
        Args:
            C_transformed: Transformed elastic tensor
            rho: Material density
            w: Angular frequency
            k0: Wave vector
            sampling: Number of sampling points
            
        Returns:
            G33: Complex displacement array
            ynew: Interpolated displacement values
            slownessnew: Calculated slowness values
        """
        G33 = np.zeros((1, sampling), dtype=complex)
        POL = [[0] * 3 for _ in range(3)]

        for ny_idx in range(sampling):
            ny_val = ny_idx + 1
            k_vec = np.array([0, ny_val * k0]) # nx is fixed to 0

            # --- Calculate F, M, and N coefficients ---
            F = np.zeros((3, 3), dtype=complex)
            M = np.zeros((3, 3), dtype=complex)
            N = np.zeros((3, 3), dtype=complex)
            B = np.zeros((3, 3), dtype=complex)

            for i in range(3):
                for j in range(3):
                    F[i, j] = C_transformed[i, 2, 2, j]

            for i in range(3):
                for l in range(3):
                    for u_idx in range(2):
                        B[i, l] -= k_vec[u_idx] * C_transformed[i, u_idx, 2, l]
                    for v_idx in range(2):
                        B[i, l] -= k_vec[v_idx] * C_transformed[i, 2, v_idx, l]
                    M[i, l] = 1j * B[i, l]

            for i in range(3):
                for l in range(3):
                    N[i, l] += rho * (w**2) * deltaij(i+1, l+1)
                    for u_idx in range(2):
                        for v_idx in range(2):
                            N[i, l] -= C_transformed[i, u_idx, v_idx, l] * k_vec[u_idx] * k_vec[v_idx]

            # --- Calculate polynomial determinant coefficients (corrected determinant calculation) ---
            for i in range(3):
                for j in range(3):
                    POL[i][j] = [F[i, j], M[i, j], N[i, j]]

            Poly_coeffs = self._calculate_polynomial_determinant(POL)
            ppC = np.roots(Poly_coeffs)
            pp = ppC[np.real(ppC) > 0] # Select positive real roots

            # --- Solve for A (eigenvectors) ---
            A_matrices = []
            for i in range(len(pp)): # Iterate over the valid roots pp
                S_matrix = F * (pp[i]**2) + M * pp[i] + N
                U, s_val, V = svd(S_matrix)
                Sol = V[-1, :] # Eigenvector corresponding to smallest singular value
                A_matrices.append(Sol)
            A = np.array(A_matrices) # A is now (len(pp), 3)

            if A.size == 0: # Handle cases where no valid root and eigenvector is found
                G33[0, ny_idx] = 0 # Or handle as needed, maybe raise a warning
                continue # Skip rest of loop and proceed to next ny_idx

            # --- Apply boundary conditions to determine a ---
            R = np.zeros((3, len(pp)), dtype=complex) # R and I are now matrices, (3, len(pp))
            I = np.zeros((3, len(pp)), dtype=complex)

            for i in range(3):
                for r_idx in range(len(pp)): # Loop over valid roots and corresponding A vectors
                    for l in range(3):
                        R[i, r_idx] += C_transformed[i, 2, 2, l] * pp[r_idx] * A[r_idx, l]
                        for u_idx in range(2):
                            I[i, r_idx] += C_transformed[i, 2, u_idx, l] * k_vec[u_idx] * A[r_idx, l]

            Comb = -R + 1j * I # Comb is also a matrix (3, len(pp))
            delta_vec = np.array([0, 0, 1])

            a_coeffs = np.zeros((3, len(pp)), dtype=complex) # a is now a matrix (3, len(pp))

            for r_idx in range(len(pp)): # For each valid root/eigenvector set
                Comb_current = Comb[:, r_idx].reshape(3, 1) # Get Comb vector for current root
                Aug = np.copy(Comb_current)

                current_a_set = np.zeros(3, dtype=complex)
                for i in range(3): # Cramer's rule for each component of 'a'
                    Aug_i = np.copy(Comb_current)
                    Aug_i[:, 0] = delta_vec # Replace i-th column with delta_vec
                    if i == 0:
                        Aug_i_matrix = np.column_stack((delta_vec, Comb_current[:,0].reshape(3,1), Comb_current[:,0].reshape(3,1))) # need to reconstruct matrix for det
                    elif i == 1:
                        Aug_i_matrix = np.column_stack((Comb_current[:,0].reshape(3,1), delta_vec, Comb_current[:,0].reshape(3,1)))
                    elif i == 2:
                        Aug_i_matrix = np.column_stack((Comb_current[:,0].reshape(3,1), Comb_current[:,0].reshape(3,1), delta_vec))

                    current_a_set[i] = det_3x3_poly_coeffs_to_complex(Aug_i_matrix) / det_3x3_poly_coeffs_to_complex(Comb_current) # Determinant of 3x3 complex matrix

                a_coeffs[:,r_idx] = current_a_set # Store the 'a' coefficients for this root

            # --- Calculate G33 value ---
            g33_values = np.zeros(len(pp), dtype=complex)
            for r_idx in range(len(pp)):
                g33_values[r_idx] = np.sum(a_coeffs[:,r_idx] * A[r_idx, :].conj()) # Using conj to match potential MATLAB behavior if A is normalized.

            # For now, take the max abs G33 from all valid root sets. Can refine this selection if needed.
            if g33_values.size > 0: # Handle case where g33_values might be empty if no valid roots.
                 G33[0, ny_idx] = np.max(np.abs(g33_values))  # Take maximum absolute value for G33


        # --- Interpolation and Peak Finding ---
        inc = 1
        xx = np.arange(1, sampling + 1)
        yy = np.real(G33[0, :])
        xnew = np.arange(1, sampling + 1, inc)
        cs = CubicSpline(xx, yy)
        ynew = cs(xnew)

        # Get peak indices using _h_l_peak method with psaw=0 for SAW only
        YYnew_indices = self._h_l_peak(ynew, psaw=0)  # Default to SAW only
        Num = 1 + inc * (YYnew_indices + 1)
        slownessnew = Num * k0 / np.real(w)

        return G33, ynew, slownessnew


    def _calculate_polynomial_determinant(self, POL):
        """
        Calculates the determinant of a 3x3 matrix of polynomials (represented by coefficient lists).
        This needs to be expanded explicitly as numpy's poly determinant doesn't directly handle polynomial matrices.
        """
        # Directly expand the determinant for a 3x3 matrix
        p11, p12, p13 = POL[0]
        p21, p22, p23 = POL[1]
        p31, p32, p33 = POL[2]

        term1 = np.convolve(np.convolve(p11, p22), p33)
        term2 = np.convolve(np.convolve(p12, p23), p31)
        term3 = np.convolve(np.convolve(p13, p21), p32)
        term4 = np.convolve(np.convolve(p11, p23), p32)
        term5 = np.convolve(np.convolve(p12, p21), p33)
        term6 = np.convolve(np.convolve(p13, p22), p31)

        Poly_coeffs = term1 + term2 + term3 - term4 - term5 - term6
        return Poly_coeffs


    def _h_l_peak(self, y_new, psaw_flag):
        """
        Find peaks in the y_new array using a more robust method matching MATLAB implementation.
        
        Args:
            y_new: numpy array of displacement values
            psaw_flag: boolean, whether to look for pseudo-SAW peaks
            
        Returns:
            numpy array of peak indices
        """
        # Find positions with negative slope (matching MATLAB implementation)
        peak_candidates = np.where(np.diff(y_new) < 0)[0]
        
        if len(peak_candidates) == 0:
            return np.array([])
            
        if psaw_flag and len(peak_candidates) >= 2:
            # Return last two peaks for PSAW
            return peak_candidates[-2:]
        else:
            # Return last peak for SAW
            return np.array([peak_candidates[-1]])


    def _calculate_intensity(self, ynew, slownessnew):
        """
        Calculates the relative intensity of each SAW mode.
        Simplified intensity calculation focusing on peak height difference.
        """
        YYnew_indices = self._h_l_peak(ynew, False) # Find peaks again, adjust psaw if needed
        intensity = np.zeros(len(YYnew_indices))

        for jj_idx, peak_index in enumerate(YYnew_indices):
            pos_ind = peak_index # Directly use index from find_peaks
            go = True
            loop_idx = pos_ind

            neg_ind = pos_ind # Initialize in case loop doesn't run
            if loop_idx > 1: # Avoid index out of bounds
                loop_idx_inner = loop_idx
                while go and loop_idx_inner > 1: # loop_idx_inner instead of loop_idx
                    if ynew[loop_idx_inner - 2] < ynew[loop_idx_inner - 1]: # Check within bounds
                        go = False
                        neg_ind = loop_idx_inner - 1 # Store neg_ind when condition met
                    else:
                        loop_idx_inner -= 1 # Decrement loop_idx_inner
            if pos_ind < len(ynew) - 1: # Check bounds for pos_ind
                 intensity[jj_idx] = ynew[neg_ind] - ynew[pos_ind] # Intensity as difference

        return intensity


    def _plot_saw_profile(self, G33, k0, w, slownessnew, psaw, ynew):
        """Generates and displays the displacement-slowness profile plot."""
        plt.figure()
        hax = plt.gca()
        ny = np.arange(1, len(G33[0,:]) + 1) # Corrected range based on G33 shape
        Nsx = 0 # Python 0-indexed
        plt.plot(ny * k0 / np.real(w), np.real(G33[Nsx, :]), 'b', linewidth=2)
        plt.hold(True) # plt.hold(True) is deprecated, use plt.show() and separate plotting commands or plt.plot(...) multiple times on the same axes

        if slownessnew.size > 0: # Check if slownessnew has values before plotting lines
            plt.axvline(x=slownessnew[0], color='r')
            if psaw and len(slownessnew) > 1: # Check for PSAW and enough values in slownessnew
                plt.axvline(x=slownessnew[1], color='r')

        plt.xlabel('Slowness (s/m)', fontsize=16)
        plt.ylabel('Displacement (arb. unit)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()


def deltaij(i, j):
    """Kronecker delta function."""
    return 1 if i == j else 0

def det_3x3_poly_coeffs_to_complex(matrix_of_complex_vec):
    """Determinant of 3x3 complex matrix directly without polynomial representation."""
    m = matrix_of_complex_vec
    return m[0,0] * (m[1,1] * m[2,2] - m[1,2] * m[2,1]) - m[0,1] * (m[1,0] * m[2,2] - m[1,2] * m[2,0]) + m[0,2] * (m[1,0] * m[2,1] - m[1,1] * m[2,0])


if __name__ == '__main__':
    # --- Example Material Class (replace with your actual Material class) ---
    class ExampleMaterial:
        def get_cijkl(self):
            # Example C tensor (replace with your actual elastic constants)
            C_example = np.zeros((3, 3, 3, 3))
            C_example = np.eye(3*3).reshape(3,3,3,3) # Example Identity-like C tensor - replace!
            return C_example

        def get_density(self):
            return 5000  # Example density

    # --- Example Usage ---
    material_example = ExampleMaterial()
    euler_example = np.array([0.1, 0.2, 0.3])
    deg_example = 30.0
    sampling_example = 400
    psaw_example = 1

    saw_calculator = SAWCalculator(material_example, euler_example)
    v_saw, index_saw, intensity_saw = saw_calculator.get_saw_speed(deg_example, sampling_example, psaw_example, draw_plot=True)

    print("SAW Velocity (v):", v_saw)
    print("SAW Direction (index):", index_saw)
    print("SAW Intensity (intensity):", intensity_saw)