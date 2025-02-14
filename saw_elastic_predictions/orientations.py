"""Database of common crystallographic orientations and their Euler angles."""

import numpy as np
from typing import Tuple, Dict

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def cross_product_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Create orientation matrix from two vectors.
    First column is v1 × v2 (normalized)
    Second column is v1 (normalized)
    Third column is v2 (normalized)
    """
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    v3 = np.cross(v1_norm, v2_norm)
    v3_norm = normalize(v3)
    
    return np.column_stack([v3_norm, v1_norm, v2_norm])

def matrix_to_euler(M: np.ndarray) -> Tuple[float, float, float]:
    """Convert orientation matrix to Euler angles (in radians).
    
    Following the Bunge convention (z-x'-z''):
    phi1: rotation around z
    Phi: rotation around x'
    phi2: rotation around z''
    """
    if abs(abs(M[2,2]) - 1.0) < 1e-8:
        # Special case: Phi = 0 or pi
        if M[2,2] > 0:
            # Phi = 0
            phi1 = np.arctan2(M[0,1], M[0,0])
            Phi = 0
            phi2 = 0
        else:
            # Phi = pi
            phi1 = np.arctan2(-M[0,1], M[0,0])
            Phi = np.pi
            phi2 = 0
    else:
        phi1 = np.arctan2(M[2,0], -M[2,1])
        Phi = np.arccos(M[2,2])
        phi2 = np.arctan2(M[0,2], M[1,2])
    
    return phi1, Phi, phi2

class Orientations:
    """Standard crystallographic orientations and their Euler angles."""
    
    @staticmethod
    def get_euler_angles(name: str) -> np.ndarray:
        """Get Euler angles for a standard orientation.
        
        Args:
            name: Orientation name (e.g., '100_001', '110_111')
            
        Returns:
            numpy array of [phi1, Phi, phi2] in radians
        
        Raises:
            ValueError: If orientation name is not recognized
        """
        # Dictionary of orientation matrices
        matrices = {
            # {100}<001>
            '100_001': np.eye(3),
            
            # {110}<111>
            '110_111': cross_product_matrix(
                np.array([1, -1, 0]) / np.sqrt(2),  # direction on surface
                np.array([1, 1, 1]) / np.sqrt(3)    # surface normal
            ),
            
            # {112}<111>
            '112_111': cross_product_matrix(
                np.array([1, 1, -2]) / np.sqrt(6),
                np.array([1, 1, 1]) / np.sqrt(3)
            ),
            
            # {100}<011>
            '100_011': cross_product_matrix(
                np.array([1, 0, 0]),
                np.array([0, 1, 1]) / np.sqrt(2)
            ),
            
            # Add more orientations as needed...
        }
        
        if name not in matrices:
            valid_names = list(matrices.keys())
            raise ValueError(f"Unknown orientation '{name}'. Valid options are: {valid_names}")
            
        # Convert matrix to Euler angles
        phi1, Phi, phi2 = matrix_to_euler(matrices[name])
        return np.array([phi1, Phi, phi2])
    
    @staticmethod
    def list_orientations() -> Dict[str, str]:
        """List available orientations with descriptions."""
        return {
            '100_001': '{100}<001> Cube orientation',
            '110_111': '{110}<111> Common orientation in FCC metals',
            '112_111': '{112}<111> Common orientation in BCC metals',
            '100_011': '{100}<011> Common orientation in cubic materials',
            # Add more as needed...
        } 