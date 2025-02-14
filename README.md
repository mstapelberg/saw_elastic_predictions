# SAW Elastic Predictions

A Python package for calculating Surface Acoustic Wave (SAW) velocities in elastic materials.

## Installation

You can install the package in development mode:

```bash
pip install -e .
```

For development with test dependencies:

```bash
pip install -e ".[test]"
```

## Usage

Basic usage example:

```python
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator
import numpy as np

# Create a material (Ni3Al at 500C)
ni3al = Material(
    formula='Ni3Al',
    C11=150.4e9,  # Pa
    C12=81.7e9,   # Pa
    C44=107.8e9,  # Pa
    density=7.57e3,  # kg/m^3
    crystal_class='cubic'
)

# Define Euler angles for {110}<111> orientation
euler_angles = np.array([2.186, 0.9553, 2.186])

# Create calculator
calculator = SAWCalculator(ni3al, euler_angles)

# Calculate SAW speed for a specific angle
v, index, intensity = calculator.get_saw_speed(30.0, sampling=4000, psaw=1)
```

## Testing

Run tests with:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=saw_elastic_predictions
```
