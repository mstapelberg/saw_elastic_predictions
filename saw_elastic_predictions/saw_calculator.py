# saw_elastic_predictions/saw_calculator.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, det
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from .euler_transformations import C_modifi, EulerAngles  # Using relative import

class SAWCalculator:
    def __init__(self, material, euler_angles):
        """
        Initialize SAWCalculator with material properties and Euler angles.
        
        Args:
            material: Material object with get_cijkl() and get_density() methods
            euler_angles: Euler angles in radians, must be within [-2π, 2π]
        
        Raises:
            TypeError: If material doesn't have required methods
            ValueError: If euler_angles are invalid
        """
        # Validate material
        if not hasattr(material, 'get_cijkl') or not hasattr(material, 'get_density'):
            raise TypeError("Material must have get_cijkl() and get_density() methods")
        
        # Convert euler_angles to numpy array
        euler_angles = np.asarray(euler_angles)
        
        # Check Euler angle shape
        if euler_angles.shape != (3,):
            raise ValueError("euler_angles must be a 3-element array")
            
        # Check Euler angle range - strict validation for Python
        if np.any(np.abs(euler_angles) > 2 * np.pi):
            raise ValueError("Euler angles must be in radians and within [-2π, 2π]")
        elif np.any(np.abs(euler_angles) > 9):
            print('Warning: Euler angles are expressed in radian not degree. Please check!')
            
        self.material = material
        self.euler_angles = euler_angles

    def get_saw_speed(self, deg, sampling=4000, psaw=0, draw_plot=False, debug=False):
        """
        Calculate SAW velocity and related parameters.
        
        Args:
            deg: Angle rotated on surface plane referred to y direction (0<=deg<180 in degree)
            sampling: Resolution in k space (400, 4000, or 40000)
            psaw: Flag for PSAW calculation (0 for SAW only, 1 for SAW and PSAW)
            draw_plot: If True, generates displacement-slowness plot
            debug: If True, returns intermediate values for testing
            
        Returns:
            v: SAW velocity(ies)
            index: Direction of SAW in crystalline coordinates
            intensity: Intensity of SAW mode(s)
        """
        # Strict validation for deg
        if not (0 <= deg < 180):
            raise ValueError("deg must be in range [0, 180)")
        elif deg > 180:  # This won't be reached due to above check, but kept for MATLAB compatibility
            print('Warning: Check convention, deg should be between 0 and 180 in degree')
            
        # Validate sampling parameter
        if sampling <= 0:
            raise ValueError("sampling must be positive")
        
        # Validate psaw parameter
        if psaw not in [0, 1]:
            raise ValueError("psaw must be 0 or 1")
        
        # Get material properties - EXACTLY as in MATLAB
        C = self.material.get_cijkl()  # This should now match MATLAB's getCijkl
        rho = self.material.get_density()  # Should be 7.57e3
        
        if debug:
            print("\nMaterial properties:")
            print(f"C11 = {C[0,0,0,0]}")
            print(f"C12 = {C[0,0,1,1]}")
            print(f"C44 = {C[0,1,0,1]}")
            print(f"density = {rho}")
        
        # Get alignment matrix - EXACTLY as in MATLAB
        MM = self._get_alignment_matrix(deg)
        
        # Transform elastic constants - EXACTLY match MATLAB order
        #print(f"Euler angles: {self.euler_angles}")
        euler_obj = EulerAngles(self.euler_angles[0], self.euler_angles[1], self.euler_angles[2])
        rotation_matrix = euler_obj.to_matrix()
        
        # MATLAB: C = C_modifi(C,(Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)')
        transform_matrix = (rotation_matrix @ MM).T  # Exactly as MATLAB: (A*B)'
        C_transformed = C_modifi(C, transform_matrix)
        
        # Get sampling parameter and initialize variables
        T = self._get_sampling_parameter(sampling)
        lambda_val = 7e-6  # m
        k0 = 2 * np.pi / lambda_val
        w0 = 2 * np.pi / T
        w = w0 + complex(0, 0.000001 * w0)
        index = transform_matrix @ np.array([0, 1, 0]) # What is this for? 
        
        if debug:
            initial_debug_values = {
                'C_transformed': C_transformed,
                'euler_rotation': rotation_matrix,
                'MM': MM,
                'combined_rotation': transform_matrix
            }
            G33, ynew, slownessnew, g33_debug_values = self._calculate_g33(
                C_transformed, rho, w, k0, sampling, psaw, debug=True)
            # Calculate final values
            v = 1.0 / slownessnew
            intensity = self._calculate_intensity(ynew, slownessnew)
            # Merge debug dictionaries
            debug_values = {**initial_debug_values, **g33_debug_values}
            return v, index, intensity, debug_values
        else:
            G33, ynew, slownessnew = self._calculate_g33(
                C_transformed, rho, w, k0, sampling, psaw)
            v = 1.0 / slownessnew
            intensity = self._calculate_intensity(ynew, slownessnew)
            return v, index, intensity

    def _get_alignment_matrix(self, deg):
        """Get alignment matrix (exactly matching MATLAB's MM matrix).
        
        Args:
            deg: Angle in degrees
        
        Returns:
            3x3 rotation matrix that aligns y-axis with direction of interest
        """
        # Use cosd directly like MATLAB (convert to radians internally)
        def cosd(angle_deg):
            return np.cos(np.deg2rad(angle_deg))
        
        # Exactly match MATLAB's MM matrix construction
        MM = np.array([
            [cosd(deg), cosd(90-deg), 0],
            [cosd(90+deg), cosd(deg), 0],
            [0, 0, 1]
        ])
        
        return MM

    def _get_sampling_parameter(self, sampling):
        """Get sampling parameter T (exactly matching MATLAB)."""
        if sampling == 40000:
            T = 20e-14
        elif sampling == 4000:
            T = 20e-13
        elif sampling == 400:
            T = 20e-12
        else:
            T = 20e-12
            print('Warning: sampling is not a value often used')
        return T

    def _stable_det(self, matrix):
        """Calculate determinant with improved numerical stability."""
        # Scale the matrix to improve condition number
        scale = np.max(np.abs(matrix))
        if scale > 0:
            scaled_matrix = matrix / scale
            return np.linalg.det(scaled_matrix) * (scale**3)
        return 0.0

    def _select_roots(self, roots, tol=1e-10):
        """Select roots with positive real parts, handling numerical noise."""
        # Remove roots with tiny real parts (numerical noise)
        cleaned_roots = np.where(np.abs(np.real(roots)) < tol,
                               1j * np.imag(roots),
                               roots)
        
        # Select roots with positive real parts or positive imaginary parts if real part is zero
        selected = []
        for root in cleaned_roots:
            if np.real(root) > tol or (abs(np.real(root)) < tol and np.imag(root) > 0):
                selected.append(root)
        return np.array(selected)

    def _calculate_g33(self, C_transformed, rho, w, k0, sampling, psaw, debug=False):
        """
        Calculate G33 following the MATLAB implementation exactly.
        
        Args:
            C_transformed: Transformed elastic tensor
            rho: Material density
            w: Angular frequency
            k0: Wave vector
            sampling: Number of sampling points
            psaw: Flag for PSAW calculation
            debug: If True, returns intermediate values for testing
        """
        if debug:
            print("\nNumerical stability diagnostics:")
            print(f"C_transformed condition number: {np.linalg.cond(C_transformed.reshape(9,9))}")
        
        # Pre-allocate arrays (matching MATLAB)
        G33 = np.zeros((1, sampling), dtype=complex)
        F = np.zeros((3, 3), dtype=np.complex128)  # Use complex128 for better precision
        B = np.zeros((3, 3), dtype=np.complex128)
        M = np.zeros((3, 3), dtype=np.complex128)
        N = np.zeros((3, 3), dtype=complex)
        POL = [[[] for _ in range(3)] for _ in range(3)]  # 3x3 list of lists for polynomials
        pp = np.zeros(3, dtype=complex)  # Pre-allocate for up to 3 roots
        A = np.zeros((3, 3), dtype=complex)
        R = np.zeros((3, 3), dtype=complex)
        I = np.zeros((3, 3), dtype=complex)
        a = np.zeros(3, dtype=complex)

        # Fixed nx=1 as in MATLAB
        nx = 1
        
        # Store debug values for first iteration
        debug_values = {}
        
        # Main loop over ny (sampling points)
        for ny_idx in range(sampling):
            ny = ny_idx + 1
            
            # Set up wave vector k (2D as in MATLAB)
            k = np.array([nx*k0, ny*k0])
            
            # Calculate F matrix - EXACTLY as MATLAB
            F.fill(0)
            for i in range(3):
                for j in range(3):
                    # Print the value we're about to assign
                    if debug and ny_idx == 0:
                        print(f"C_transformed[{i},2,2,{j}] = {C_transformed[i,2,2,j]}")
                    # Convert to float64, taking real part if complex
                    F[i,j] = np.float64(np.real(C_transformed[i,2,2,j]))
                    
                    # Add extra debug info
                    if debug and ny_idx == 0 and abs(F[i,j]) > 1e5:
                        print(f"F[{i},{j}] = {F[i,j]}")
            
            # Clean up numerical noise in F matrix
            # Separate handling for diagonal and off-diagonal elements
            max_diagonal = np.max(np.abs(np.diag(F)))
            for i in range(3):
                for j in range(3):
                    if i != j:  # Off-diagonal elements
                        if np.abs(F[i,j]) < 1e-4 * max_diagonal:  # More strict threshold for off-diagonal
                            F[i,j] = 0
                    else:  # Diagonal elements
                        if np.abs(F[i,j]) < 1e-8 * max_diagonal:  # Original threshold for diagonal
                            F[i,j] = 0
            
            # Calculate B and M matrices - EXACTLY as MATLAB
            B.fill(0)
            for i in range(3):
                for l in range(3):
                    for u in range(2):
                        B[i,l] -= k[u] * C_transformed[i,u,2,l]
                    for v in range(2):
                        B[i,l] -= k[v] * C_transformed[i,2,v,l]
                    M[i,l] = B[i,l] * 1j
            
            # Calculate N matrix - EXACTLY as MATLAB
            N.fill(0)
            for i in range(3):
                for l in range(3):
                    N[i,l] = rho * (w**2) * (1 if i==l else 0)
                    for u in range(2):
                        for v in range(2):
                            N[i,l] -= C_transformed[i,u,v,l] * k[u] * k[v]
            
            # Set up polynomial coefficients
            for i in range(3):
                for j in range(3):
                    POL[i][j] = [F[i,j], M[i,j], N[i,j]]
            
            # Calculate determinant polynomial using convolution (as in MATLAB)
            Poly = np.convolve(np.convolve(POL[0][0], POL[1][1]), POL[2][2])
            Poly += np.convolve(np.convolve(POL[0][1], POL[1][2]), POL[2][0])
            Poly += np.convolve(np.convolve(POL[0][2], POL[1][0]), POL[2][1])
            Poly -= np.convolve(np.convolve(POL[0][0], POL[1][2]), POL[2][1])
            Poly -= np.convolve(np.convolve(POL[0][1], POL[1][0]), POL[2][2])
            Poly -= np.convolve(np.convolve(POL[0][2], POL[1][1]), POL[2][0])
            
            # Find roots and select positive real parts - EXACTLY as MATLAB
            ppC = np.roots(Poly)
            pp = np.array([root for root in ppC if np.real(root) > 0])
            
            if len(pp) == 0:
                G33[0,ny_idx] = 0
                continue
            
            # Calculate eigenvectors using SVD
            for i in range(len(pp)):
                S = F * (pp[i]**2) + M * pp[i] + N
                U, s, Vh = np.linalg.svd(S)
                Sol = Vh[-1,:].conj()  # Last row of V^H (equivalent to last column of V in MATLAB)
                A[i,:] = Sol
            
            # Calculate R and I matrices
            R.fill(0)
            I.fill(0)
            for i in range(3):
                for r in range(len(pp)):
                    for l in range(3):
                        R[i,r] += C_transformed[i,2,2,l] * pp[r] * A[r,l]
                        for u in range(2):
                            I[i,r] += C_transformed[i,2,u,l] * k[u] * A[r,l]
            
            # Form combined matrix and solve for a using Cramer's rule
            Comb = -R + 1j * I
            del_vec = np.array([0, 0, 1])
            
            # Add numerical stability checks for determinant
            for r in range(3):
                Aug = Comb.copy()
                Aug[:,r] = del_vec
                det_comb = self._stable_det(Comb)
                det_aug = self._stable_det(Aug)
                
                if abs(det_comb) < 1e-10:
                    # Use SVD-based solution instead
                    U, s, Vh = np.linalg.svd(Comb)
                    tol = np.max(s) * 1e-10
                    s_inv = np.array([1/x if x > tol else 0 for x in s])
                    a[r] = (Vh.T @ np.diag(s_inv) @ U.T @ del_vec)[r]
                else:
                    a[r] = det_aug / det_comb
            
            # Calculate G33
            G33[0,ny_idx] = sum(a[r] * A[r,2] for r in range(len(pp)))
            
            # Add debug prints
            #print(f"F matrix at ny={ny}:")
            #print(F)
            #print(f"M matrix at ny={ny}:")
            #print(M)
            #print(f"N matrix at ny={ny}:")
            #print(N)
            
            if ny_idx == 0 and debug:  # Store values for first iteration
                debug_values = {
                    'F': F.copy(),
                    'M': M.copy(),
                    'N': N.copy(),
                    'roots': ppC.copy(),
                    'selected_roots': pp.copy(),
                    'first_G33': G33[0,0]
                }
            
            # Add condition number checks in the loop
            if debug and ny_idx == 0:
                print(f"F matrix condition number: {np.linalg.cond(F)}")
                print(f"Comb matrix condition number: {np.linalg.cond(Comb)}")
        
        # Post-processing
        inc = 1
        xx = np.arange(1, sampling + 1)
        yy = np.real(G33[0,:])
        xnew = np.arange(1, sampling + 1, inc)
        cs = CubicSpline(xx, yy)
        ynew = cs(xnew)
        
        # Debug prints for post-processing
        if debug:
            print("\nPost-processing debug:")
            print(f"G33 shape: {G33.shape}")
            print(f"First few values of real(G33): {yy[:5]}")
            print(f"Last few values of real(G33): {yy[-5:]}")
            
            # Find peaks before converting to slowness
            YYnew_indices = self._h_l_peak(ynew, psaw)
            print(f"\nPeak indices: {YYnew_indices}")
            
            # Debug peak finding
            print(f"Number of peaks found: {len(YYnew_indices)}")
            if len(YYnew_indices) > 0:
                print(f"Peak positions: {YYnew_indices}")
                print(f"Peak values: {ynew[YYnew_indices]}")
            
            # Debug slowness calculation
            Num = 1 + inc * YYnew_indices
            print(f"\nNum values: {Num}")
            print(f"k0: {k0}")
            print(f"w: {w}")
            slownessnew = Num * k0 / np.real(w)
            print(f"Calculated slowness values: {slownessnew}")
            print(f"Final velocities: {1.0/slownessnew}")

        # Calculate final values
        YYnew_indices = self._h_l_peak(ynew, psaw)
        Num = 1 + inc * YYnew_indices
        slownessnew = Num * k0 / np.real(w)
        
        if debug:
            return G33, ynew, slownessnew, debug_values
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
        """Find peaks in the y_new array using a more robust method."""
        # Find local maxima by comparing with neighbors
        peak_candidates = []
        for i in range(1, len(y_new)-1):
            if y_new[i] > y_new[i-1] and y_new[i] > y_new[i+1]:
                # Only consider significant peaks (adjust threshold as needed)
                if abs(y_new[i]) > 1e-22:  # Threshold based on observed values
                    peak_candidates.append(i)
        
        peak_candidates = np.array(peak_candidates)
        
        if len(peak_candidates) == 0:
            return np.array([])
            
        # Remove debug prints - commented out
        # print(f"\nPeak finding debug:")
        # print(f"Number of significant peaks found: {len(peak_candidates)}")
        # print(f"Peak positions: {peak_candidates}")
        # print(f"Peak values: {y_new[peak_candidates]}")
        
        # Sort peaks by magnitude
        peak_magnitudes = abs(y_new[peak_candidates])
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # Sort in descending order
        peak_candidates = peak_candidates[sorted_indices]

        if psaw_flag and len(peak_candidates) >= 2:
            # Return two largest peaks for PSAW
            return peak_candidates[:2]
        else:
            # Return largest peak for SAW
            return np.array([peak_candidates[0]])

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
        ny = np.arange(1, len(G33[0,:]) + 1)
        Nsx = 0  # Python 0-indexed
        
        # Plot displacement profile
        plt.plot(ny * k0 / np.real(w), np.real(G33[Nsx, :]), 'b', linewidth=2)
        
        # Plot SAW lines
        if slownessnew.size > 0:
            plt.axvline(x=slownessnew[0], color='r')
            if psaw and len(slownessnew) > 1:
                plt.axvline(x=slownessnew[1], color='r')

        plt.xlabel('Slowness (s/m)', fontsize=16)
        plt.ylabel('Displacement (arb. unit)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.show()

def deltaij(i, j):
    """Kronecker delta function."""
    return 1 if i == j else 0

def det_3x3_poly_coeffs_to_complex(matrix_of_complex_vec):
    """Determinant of 3x3 complex matrix directly without polynomial representation."""
    m = matrix_of_complex_vec
    det_val = m[0,0] * (m[1,1] * m[2,2] - m[1,2] * m[2,1]) - m[0,1] * (m[1,0] * m[2,2] - m[1,2] * m[2,0]) + m[0,2] * (m[1,0] * m[2,1] - m[1,1] * m[2,0])
    print(f"Determinant value: {det_val}") # DEBUG: Print determinant value

    return det_val

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