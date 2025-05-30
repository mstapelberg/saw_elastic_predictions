from defdap import ebsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm

# Import your SAW calculator
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator

data_path = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected"
# Load the EBSD map from a .ctf file
ebsd_map = ebsd.Map(data_path, dataType="OxfordText")  # Provide the .ctf file path
ebsd_map.buildQuatArray()           # Convert orientations to quaternions for calculations

print(f"Phases: {[phase.name for phase in ebsd_map.phases]}")

ebsd_map.findBoundaries(boundDef=5)      # mark grain boundaries at >5° misorientation
ebsd_map.findGrains(minGrainSize=10)     # group pixels into grains, remove grains <10 pixels
print(f"Identified {len(ebsd_map.grainList)} grains")

# Check the actual dimensions and grain map attributes
print(f"Map shape: {ebsd_map.shape}")
print(f"xDim: {ebsd_map.xDim}, yDim: {ebsd_map.yDim}")
print(f"Step size: {ebsd_map.stepSize}")

# Create Vanadium material object
# Note: You'll need to find/define the actual elastic constants for V
# These are placeholder values - replace with actual V properties
vanadium = Material(
    formula='V',
    C11=229e9,  # Pa - replace with actual values for Vanadium
    C12=119e9,  # Pa 
    C44=43e9,   # Pa
    density=6110,  # kg/m^3 - replace with actual V density
    crystal_class='cubic'
)

# Function to calculate peak SAW frequency based on Euler angles using your real calculator
def calculate_peak_saw_frequency(euler_angles, material):
    """
    Calculate peak SAW frequency based on grain orientation using the real SAW calculator
    """
    try:
        # Create SAW calculator for this grain's orientation
        calculator = SAWCalculator(material, euler_angles)
        
        # Calculate SAW speed at a reference angle (e.g., 0 degrees)
        # You can experiment with different angles or use multiple angles
        v, index, intensity = calculator.get_saw_speed(0.0, sampling=400, psaw=0)
        
        # Convert SAW speed to frequency (simplified relationship)
        # Peak SAW frequency is related to speed and wavelength: f = v/λ
        # For a typical grating wavelength of ~4 μm:
        wavelength = 8.8e-6  # meters
        peak_saw_freq = v[0] / wavelength  # Hz
        
        return peak_saw_freq
        
    except Exception as e:
        print(f"Error calculating SAW frequency for Euler angles {euler_angles}: {e}")
        # Return a default value if calculation fails
        return np.nan

# Extract grains data
print("Calculating SAW frequencies for grains...")
grains_data = []
for grain in tqdm(ebsd_map, desc="Processing grains", unit="grain"): 
    grain.calcAverageOri()
    euler = grain.refOri.eulerAngles()  # (phi1, PHI, phi2)
    
    # Calculate peak SAW frequency for this grain
    peak_saw_freq = calculate_peak_saw_frequency(euler, vanadium)
    
    grains_data.append({
        "Grain ID": grain.grainID,
        "Euler1": euler[0], 
        "Euler2": euler[1], 
        "Euler3": euler[2],
        "Size (pixels)": len(grain),  # Use len(grain) instead of grain.nPixels
        "Size (um^2)": len(grain) * (ebsd_map.stepSize**2),
        "Mean Misorientation": getattr(grain, "averageMisOri", None),
        "Peak SAW Frequency": peak_saw_freq
    })

df = pd.DataFrame(grains_data)
print(df.head())

# Remove grains with NaN frequencies for plotting
df_valid = df.dropna(subset=['Peak SAW Frequency'])
print(f"Valid grains for plotting: {len(df_valid)} out of {len(df)}")

# Create a mapping from grain ID to SAW frequency for plotting
grain_id_to_saw_freq = dict(zip(df_valid["Grain ID"], df_valid["Peak SAW Frequency"]))

# Try to find the correct way to access grain map data
try:
    # Check if there's a grains attribute that contains the map
    if hasattr(ebsd_map, 'grains') and hasattr(ebsd_map.grains, 'shape'):
        grain_map = ebsd_map.grains
        print(f"Using ebsd_map.grains with shape: {grain_map.shape}")
    else:
        # Alternative: create grain map manually
        print("Creating grain map manually...")
        grain_map = np.zeros(ebsd_map.shape, dtype=int)
        
        # Fill grain map by iterating through grains
        for grain in tqdm(ebsd_map, desc="Building grain map", unit="grain"):
            for coord in grain.coordList:
                # coord is likely in (x, y) format
                if coord[1] < grain_map.shape[0] and coord[0] < grain_map.shape[1]:
                    grain_map[coord[1], coord[0]] = grain.grainID
        
        print(f"Created grain map with shape: {grain_map.shape}")

    # Create SAW frequency map
    saw_freq_map = np.full(grain_map.shape, np.nan)
    
    # Fill in the SAW frequency values for each pixel based on its grain ID
    for row in range(grain_map.shape[0]):
        for col in range(grain_map.shape[1]):
            grain_id = grain_map[row, col]
            if grain_id > 0 and grain_id in grain_id_to_saw_freq:  # Valid grain ID
                saw_freq_map[row, col] = grain_id_to_saw_freq[grain_id]

    # Plot the EBSD map colored by SAW frequency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: SAW frequency map
    im1 = ax1.imshow(saw_freq_map, cmap='viridis', origin='lower')
    ax1.set_title('EBSD Map Colored by Peak SAW Frequency')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Peak SAW Frequency (Hz)')

    # Plot 2: Histogram of SAW frequencies with improved binning
    freq_values = df_valid["Peak SAW Frequency"].values
    
    # Calculate better bin edges to avoid spacing issues
    freq_min = freq_values.min()
    freq_max = freq_values.max()
    freq_range = freq_max - freq_min
    
    # Use automatic binning but ensure reasonable number of bins
    n_bins = min(50, max(10, int(np.sqrt(len(freq_values)))))  # Between 10-50 bins
    
    # Create evenly spaced bins
    bin_edges = np.linspace(freq_min, freq_max, n_bins + 1)
    
    ax2.hist(freq_values, bins=bin_edges, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Peak SAW Frequency (Hz)')
    ax2.set_ylabel('Number of Grains')
    ax2.set_title('Distribution of Peak SAW Frequencies')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis to use scientific notation if values are large
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nSAW Frequency Statistics:")
    print(f"Mean: {df_valid['Peak SAW Frequency'].mean():.2e} Hz")
    print(f"Std:  {df_valid['Peak SAW Frequency'].std():.2e} Hz")
    print(f"Min:  {df_valid['Peak SAW Frequency'].min():.2e} Hz")
    print(f"Max:  {df_valid['Peak SAW Frequency'].max():.2e} Hz")

except Exception as e:
    print(f"Error creating grain map: {e}")
    print("Let's try a different approach...")

# Save the dataframe for further analysis
df.to_csv('grains_data_with_saw_frequency.csv', index=False)
print(f"\nSaved grain data with SAW frequencies to 'grains_data_with_saw_frequency.csv'")