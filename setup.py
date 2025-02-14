from setuptools import setup, find_packages

setup(
    name="saw_elastic_predictions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    author="Myles",
    description="Surface Acoustic Wave (SAW) velocity calculator for elastic materials",
    python_requires=">=3.8",
) 