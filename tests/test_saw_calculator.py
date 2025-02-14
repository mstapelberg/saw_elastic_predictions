import pytest
import numpy as np
import matplotlib.pyplot as plt
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator
import os
import contextlib
from saw_elastic_predictions.euler_transformations import EulerAngles

# Test fixtures
@pytest.fixture
def ni3al_material():
    """Create Ni3Al material with properties at 500C"""
    return Material(
        formula='Ni3Al',
        C11=150.4e9,  # Pa
        C12=81.7e9,   # Pa
        C44=107.8e9,  # Pa
        density=7.57e3,  # kg/m^3
        crystal_class='cubic'
    )

@pytest.fixture
def euler_110_111():
    """Euler angles for {110}<111> orientation (in radians)"""
    return np.array([np.pi/2, 0.9553, np.pi/4])  # Match MATLAB's values exactly

@pytest.fixture
def matlab_reference_data():
    """Load MATLAB reference data"""
    data = np.loadtxt('matlab_test/Ni3Al_SAW_111_data.txt', delimiter='\t')
    return data[:, 0], data[:, 1]  # angles, speeds

# Test initialization
def test_saw_calculator_initialization(ni3al_material, euler_110_111):
    """Test basic initialization of SAWCalculator"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    assert calculator.material == ni3al_material
    assert np.allclose(calculator.euler_angles, euler_110_111)

# Test input validation
def test_invalid_material():
    """Test initialization with invalid material"""
    class InvalidMaterial:
        pass
    
    with pytest.raises(TypeError, match=r"Material must have get_cijkl\(\) and get_density\(\) methods"):
        SAWCalculator(InvalidMaterial(), np.array([0, 0, 0]))

def test_invalid_euler_angles(ni3al_material):
    """Test initialization with invalid Euler angles"""
    # Wrong shape
    with pytest.raises(ValueError, match=r"euler_angles must be a 3-element array"):
        SAWCalculator(ni3al_material, np.array([0, 0]))
    
    # Out of range
    with pytest.raises(ValueError, match=r"Euler angles must be in radians and within \[-2π, 2π\]"):
        SAWCalculator(ni3al_material, np.array([10, 10, 10]))

# Test SAW speed calculation
def test_saw_speed_basic(ni3al_material, euler_110_111):
    """Test basic SAW speed calculation"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    v, index, intensity = calculator.get_saw_speed(30.0, sampling=400, psaw=0)
    
    # Basic checks
    assert isinstance(v, np.ndarray)
    assert isinstance(index, np.ndarray)
    assert isinstance(intensity, np.ndarray)
    assert v.size > 0
    assert index.size == 3
    assert intensity.size > 0

def test_saw_speed_invalid_inputs(ni3al_material, euler_110_111):
    """Test SAW speed calculation with invalid inputs"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    
    # Invalid degree
    with pytest.raises(ValueError, match="deg must be in range"):
        calculator.get_saw_speed(-1)
    with pytest.raises(ValueError, match="deg must be in range"):
        calculator.get_saw_speed(180)
    
    # Invalid sampling
    with pytest.raises(ValueError, match="sampling must be positive"):
        calculator.get_saw_speed(30, sampling=0)
    
    # Invalid psaw
    with pytest.raises(ValueError, match="psaw must be 0 or 1"):
        calculator.get_saw_speed(30, psaw=2)

def test_saw_speed_sampling_values(ni3al_material, euler_110_111):
    """Test different sampling values"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    
    # Test standard sampling values
    samplings = [400, 4000, 40000]
    for sampling in samplings:
        v, _, _ = calculator.get_saw_speed(30.0, sampling=sampling)
        assert v.size > 0

def test_saw_speed_with_psaw(ni3al_material, euler_110_111):
    """Test SAW speed calculation with PSAW enabled"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    
    # Without PSAW
    v1, _, _ = calculator.get_saw_speed(30.0, sampling=4000, psaw=0)
    
    # With PSAW
    v2, _, _ = calculator.get_saw_speed(30.0, sampling=4000, psaw=1)
    
    # PSAW should potentially give more solutions
    assert v2.size >= v1.size

#def test_saw_speed_angle_sweep(ni3al_material, euler_110_111):
    #"""Test SAW speed calculation over a range of angles (matching Ni3Al_SAW.m)"""
    #calculator = SAWCalculator(ni3al_material, euler_110_111)
    #angles = np.arange(0, 61)
    #saw_speeds = np.zeros((2, len(angles)))
    
    #for i, angle in enumerate(angles):
        #v, _, _ = calculator.get_saw_speed(angle, sampling=4000, psaw=1)
        #saw_speeds[:len(v), i] = v
    
    ## Basic checks on the sweep results
    #assert not np.any(np.isnan(saw_speeds))  # No NaN values
    #assert np.all(saw_speeds >= 0)  # All speeds should be positive
    #assert saw_speeds.shape == (2, 61)  # Expected shape for SAW and PSAW

def test_comparison_with_matlab(ni3al_material, euler_110_111, matlab_reference_data):
    """Compare Python implementation results with MATLAB reference data"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    matlab_angles, matlab_speeds = matlab_reference_data
    
    # Calculate speeds using Python implementation
    python_speeds = np.zeros_like(matlab_speeds)
    for i, angle in enumerate(matlab_angles):
        v, _, _ = calculator.get_saw_speed(angle, sampling=4000, psaw=1)
        python_speeds[i] = v[0]  # Take first (SAW) speed
    
    # Calculate relative differences
    relative_diff = np.abs(python_speeds - matlab_speeds) / matlab_speeds
    max_allowed_diff = 0.01  # 1% maximum allowed difference
    
    # Create diagnostic plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Speed comparison
    plt.subplot(2, 1, 1)
    plt.plot(matlab_angles, matlab_speeds, 'b-', label='MATLAB')
    plt.plot(matlab_angles, python_speeds, 'r--', label='Python')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Speed (m/s)')
    plt.title('SAW Speed Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Relative difference
    plt.subplot(2, 1, 2)
    plt.plot(matlab_angles, relative_diff * 100, 'k-')
    plt.axhline(y=max_allowed_diff * 100, color='r', linestyle='--', 
                label=f'{max_allowed_diff*100}% threshold')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Difference between MATLAB and Python')
    plt.legend()
    plt.grid(True)
    
    # Print some statistics
    print("\nDiagnostic Statistics:")
    print(f"Mean relative difference: {np.mean(relative_diff)*100:.2f}%")
    print(f"Max relative difference: {np.max(relative_diff)*100:.2f}%")
    print(f"Min relative difference: {np.min(relative_diff)*100:.2f}%")
    print("\nAngles with largest differences:")
    worst_indices = np.argsort(relative_diff)[-5:]  # Get indices of 5 worst matches
    for idx in worst_indices:
        print(f"Angle: {matlab_angles[idx]:.1f}°, "
              f"MATLAB: {matlab_speeds[idx]:.2f}, "
              f"Python: {python_speeds[idx]:.2f}, "
              f"Diff: {relative_diff[idx]*100:.2f}%")
    
    plt.tight_layout()
    plt.show()
    
    # Original assertion
    assert np.all(relative_diff < max_allowed_diff), \
        f"Maximum relative difference ({np.max(relative_diff)*100:.2f}%) exceeds allowed threshold ({max_allowed_diff*100:.2f}%)"

# Test plotting functionality
def test_saw_speed_with_plotting(ni3al_material, euler_110_111):
    """Test SAW speed calculation with plotting enabled"""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    v, _, _ = calculator.get_saw_speed(30.0, sampling=400, psaw=0, draw_plot=True)
    assert v.size > 0  # Basic check that calculation succeeded

def test_saw_speed_at_30_degrees(ni3al_material, euler_110_111):
    """Test SAW speed calculation specifically at 30 degrees."""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    v, _, _ = calculator.get_saw_speed(30.0, sampling=4000, psaw=1)
    
    expected_speed = 2517.98561151079  # From MATLAB reference
    calculated_speed = v[0]  # Take first (SAW) speed
    
    print("\nDiagnostic output for 30 degrees:")
    print(f"Expected speed (MATLAB): {expected_speed:.8f} m/s")
    print(f"Calculated speed (Python): {calculated_speed:.8f} m/s")
    print(f"Absolute difference: {abs(calculated_speed - expected_speed):.8f} m/s")
    print(f"Relative difference: {abs(calculated_speed - expected_speed)/expected_speed*100:.4f}%")
    
    # Assert with a slightly larger tolerance for this specific test
    np.testing.assert_allclose(
        calculated_speed, 
        expected_speed, 
        rtol=0.01,  # 1% relative tolerance
        err_msg="Speed at 30 degrees doesn't match MATLAB reference"
    )

def test_saw_calculation_at_30_degrees(ni3al_material, euler_110_111):
    """Test SAW calculation specifically at 30 degrees against MATLAB values."""
    calculator = SAWCalculator(ni3al_material, euler_110_111)
    
    # First, verify initial elastic tensor matches MATLAB exactly
    C = ni3al_material.get_cijkl()
    
    # MATLAB reference values for initial C tensor
    matlab_C = {
        (1,1,1,1): 1.5040e11,
        (1,1,2,2): 0.8170e11,
        (1,1,3,3): 0.8170e11,
        (1,2,1,2): 1.0780e11,
        (2,2,2,2): 1.5040e11,
        (3,3,3,3): 1.5040e11
    }
    
    print("\nComparing initial C tensor values:")
    print("Index\t\tMATLAB\t\t\tPython\t\t\tRel. Diff")
    print("-" * 80)
    for (i,j,k,l), matlab_val in matlab_C.items():
        python_val = C[i-1,j-1,k-1,l-1]
        rel_diff = abs(matlab_val - python_val) / abs(matlab_val)
        print(f"C[{i},{j},{k},{l}]\t{matlab_val:.6e}\t{python_val:.6e}\t{rel_diff:.6f}")
    
    # MATLAB reference values for transformed C tensor
    matlab_C_transformed = {
        (1,1,1,1): 2.238500e11,
        (1,1,2,2): 5.721667e10,
        (1,1,2,3): -3.462466e10,
        (1,1,3,2): -3.462466e10,
        (1,1,3,3): 3.273333e10,
        (2,2,2,2): 2.238500e11,
        (2,2,3,3): 3.273333e10,
        (3,3,3,3): 2.483333e11
    }
    
    # MATLAB reference values for matrices
    matlab_F = np.array([
        [0.5883e11, 0, 0],
        [0, 0.5883e11, 0],
        [0, 0, 2.4833e11]
    ])
    
    matlab_M = np.array([
        [6.2158e16j, 6.2158e16j, -8.2190e16j],
        [6.2158e16j, -6.2158e16j, -8.2190e16j],
        [-8.2190e16j, -8.2190e16j, 0j]
    ])
    
    matlab_N = np.array([
        [7.4713e28, 0, 0],
        [0, 7.4713e28, 0],
        [0, 0, 7.4713e28]
    ], dtype=complex)
    
    matlab_G33 = -7.3415e-27 - 7.3415e-21j
    
    # Add before the get_saw_speed call:
    print("\nRotation matrix construction:")
    euler_obj = EulerAngles(euler_110_111[0], euler_110_111[1], euler_110_111[2])
    rotation_matrix = euler_obj.to_matrix()
    MM = calculator._get_alignment_matrix(30.0)
    transform_matrix = (rotation_matrix @ MM).T
    
    print("\nEuler rotation matrix:")
    print(rotation_matrix)
    print("\nMM matrix:")
    print(MM)
    print("\nTransform matrix:")
    print(transform_matrix)
    
    # MATLAB reference values for these matrices (from getSAW.m output)
    matlab_rotation = np.array([
        [-0.05198636, 0.74343119, 0.66678893],
        [-0.74343119, -0.47462252, 0.47121494],
        [0.66678893, -0.47121494, 0.57736384]
    ])
    
    print("\nMatrix differences:")
    print("Rotation matrix relative difference:")
    print(np.abs(rotation_matrix - matlab_rotation) / np.abs(matlab_rotation))
    
    # Run calculation with debug mode but suppress F,M,N prints
    #with open(os.devnull, 'w') as f:
        #with contextlib.redirect_stdout(f):
    v, _, _, debug_values = calculator.get_saw_speed(30.0, sampling=4000, psaw=1, debug=True)
    
    # Extract Python's transformed C tensor
    C_transformed = debug_values['C_transformed']
    
    print("\nComparing transformed C tensor values:")
    print("Index\t\tMATLAB\t\t\tPython\t\t\tRel. Diff")
    print("-" * 80)
    for (i,j,k,l), matlab_val in matlab_C_transformed.items():
        python_val = C_transformed[i-1,j-1,k-1,l-1]
        rel_diff = abs(matlab_val - python_val) / abs(matlab_val)
        print(f"C[{i},{j},{k},{l}]\t{matlab_val:.6e}\t{python_val:.6e}\t{rel_diff:.6f}")
    
    # Print only the final comparison values
    print("\nF matrix comparison:")
    print("MATLAB F:")
    print(matlab_F)
    print("Python F:")
    print(debug_values['F'])
    
    print("\nG33 comparison:")
    print(f"MATLAB G33: {matlab_G33}")
    print(f"Python G33: {debug_values['first_G33']}")
    print(f"Relative difference: {abs(matlab_G33 - debug_values['first_G33']) / abs(matlab_G33)}")
    
    # Test with reasonable tolerances
    rtol = 1e-5  # relative tolerance
    np.testing.assert_allclose(debug_values['F'], matlab_F, rtol=rtol*10, 
                              err_msg="F matrix doesn't match MATLAB")
    np.testing.assert_allclose(debug_values['first_G33'], matlab_G33, rtol=rtol,
                              err_msg="G33 doesn't match MATLAB")

if __name__ == '__main__':
    pytest.main([__file__]) 