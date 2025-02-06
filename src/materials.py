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
        """
        Get the full elastic constant tensor.
        Matches MATLAB implementation from getCijkl.m
        """
        C = np.zeros((3, 3, 3, 3))
        
        if self.crystal_class == 'cubic':
            # Cubic crystal system
            C11, C12, C44 = self.C11, self.C12, self.C44
            for i in range(3):
                C[i,i,i,i] = C11
            pairs = [(0,1), (0,2), (1,2)]
            for i,j in pairs:
                C[i,i,j,j] = C[i,j,i,j] = C[j,j,i,i] = C12
            C[1,2,1,2] = C[2,1,2,1] = C[1,2,2,1] = C[2,1,1,2] = C44
            C[0,2,0,2] = C[2,0,2,0] = C[0,2,2,0] = C[2,0,0,2] = C44
            C[0,1,0,1] = C[1,0,1,0] = C[0,1,1,0] = C[1,0,0,1] = C44
            
        elif self.crystal_class == 'hexagonal':
            if None in (self.C13, self.C33):
                raise ValueError("C13 and C33 required for hexagonal crystals")
            C11, C12, C13 = self.C11, self.C12, self.C13
            C33, C44 = self.C33, self.C44
            C66 = (C11 - C12) / 2
            
            # Set primary constants
            for i in range(2):
                C[i,i,i,i] = C11
            C[2,2,2,2] = C33
            
            # Set C12, C13 components
            C[0,0,1,1] = C[1,1,0,0] = C12
            C[0,0,2,2] = C[1,1,2,2] = C[2,2,0,0] = C[2,2,1,1] = C13
            
            # Set C44, C66 components
            C[1,2,1,2] = C[2,1,2,1] = C[1,2,2,1] = C[2,1,1,2] = C44
            C[0,2,0,2] = C[2,0,2,0] = C[0,2,2,0] = C[2,0,0,2] = C44
            C[0,1,0,1] = C[1,0,1,0] = C[0,1,1,0] = C[1,0,0,1] = C66
            
        else:
            raise ValueError(f"Crystal class {self.crystal_class} not implemented")
            
        # Ensure tensor symmetry
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if C[i,j,k,l] != 0:
                            C[j,i,k,l] = C[i,j,l,k] = C[j,i,l,k] = C[k,l,i,j] = C[i,j,k,l]
        
        return C

    def get_density(self):
        """Get material density in kg/m^3"""
        return self.density