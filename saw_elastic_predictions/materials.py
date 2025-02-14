# saw_elastic_predictions/src/materials.py
import numpy as np
import matplotlib.pyplot as plt

class Material:
    def __init__(self, formula, C11, C12, C44, density, crystal_class, **kwargs):
        """
        Initialize material with elastic constants and properties.
        
        Args:
            formula: Chemical formula of the material
            C11, C12, C44: Primary elastic constants
            density: Material density
            crystal_class: Crystal system ('cubic', 'hexagonal', etc.)
            **kwargs: Additional elastic constants for non-cubic systems
        """
        self.formula = formula
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44
        self.density = density
        self.crystal_class = crystal_class
        
        # Additional constants for non-cubic systems
        self.C13 = kwargs.get('C13', None)
        self.C33 = kwargs.get('C33', None)
        self.C66 = kwargs.get('C66', None)

    def get_cijkl(self):
        """Convert from Voigt notation to full tensor for cubic crystal."""
        C11 = self.C11  # 150.4e9
        C12 = self.C12  # 81.7e9
        C44 = self.C44  # 107.8e9
        
        C = np.zeros((3, 3, 3, 3))
        
        # Fill in the tensor components
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if i == j and k == l:  # Diagonal terms
                            C[i,j,k,l] = C12
                            if i == k:
                                C[i,j,k,l] = C11
                        elif (i == k and j == l) or (i == l and j == k):  # Off-diagonal terms
                            C[i,j,k,l] = C44
        
        """
        # Debug print in MATLAB-like format
        print("\nPython C tensor:")
        for i in range(3):
            for j in range(3):
                print(f"\n(:,:,{i+1},{j+1}) =")
                print(C[:,:,i,j])
        """
        
        return C

    def get_density(self):
        """Get material density in kg/m^3"""
        return self.density