#!/bin/bash
# Compiles the Cython extensions in place

# Change to the entities directory
cd ./core/entities

# Run the setup script to build the extensions
# The --inplace option puts the compiled files next to the .pyx files
python3 _cython_setup.py build_ext --inplace

echo "Cython compilation complete."