import numpy as np
from materials import Material
from euler_transformations import EulerAngles
from saw_calculator import SAWCalculator

# Define material
ni3al = Material(
    formula="Ni3Al",
    C11=150.4e9,
    C12=81.7e9,
    C44=107.8e9,
    density=7.57e3,
    crystal_class="cubic",
)

# Define Euler angles
euler_angles = EulerAngles(a=0, b=0, r=0)  # Replace with actual values

# Create SAW calculator
saw_calculator = SAWCalculator(material=ni3al, euler_angles=euler_angles)

# Calculate and plot
angles = np.arange(0, 61)
saw_calculator.plot_saw_speeds(angles)