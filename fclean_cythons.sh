#!/bin/bash
# Cleans up the Cython build artifacts

# Change to the entities directory
cd ./core/entities

# Remove the compiled .so files
find . -name "*.so" -delete

# Remove the generated .c and .cpp files
find . -name "*.c" -delete
find . -name "*.cpp" -delete

# Remove the build directory
rm -rf build

echo "Cython build artifacts cleaned up."