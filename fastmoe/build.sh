#!/bin/bash

# Set paths for include and libraries
export CPLUS_INCLUDE_PATH=/home/wjbang/anaconda3/envs/pMoE/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/wjbang/anaconda3/envs/pMoE/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/wjbang/anaconda3/envs/pMoE/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/home/wjbang/anaconda3/envs/pMoE/lib/python3.11/site-packages/nvidia/nccl/include:/home/wjbang/anaconda3/envs/pMoE/lib/python3.11/site-packages/torch/include/torch/csrc/cuda:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/wjbang/anaconda3/envs/pMoE/lib/python3.11/site-packages/nvidia/nccl/include:/home/wjbang/anaconda3/envs/pMoE/lib/python3.11/site-packages/torch/include/torch/csrc/cuda:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/wjbang/anaconda3/envs/pMoE/lib/python3.11/site-packages/nvidia/nccl/lib/:/home/wjbang/anaconda3/envs/pMoE/lib:$LD_LIBRARY_PATH



# Clean Python build artifacts and cache
echo "Deleting build artifacts and cache..."
rm -rf build/ dist/ *.egg-info
find . -name "__pycache__" -exec rm -rf {} +  # Remove Python cache directories
find . -name "*.pyc" -exec rm -f {} +         # Remove Python bytecode files
find . -name "*.pyo" -exec rm -f {} +         # Remove optimized Python bytecode

# Clean C++ and CUDA artifacts
echo "Cleaning C++/CUDA artifacts..."
rm -rf *.so                               # Remove compiled shared objects
find . -name "*.o" -exec rm -f {} +       # Remove object files
find . -name "*.a" -exec rm -f {} +       # Remove static libraries
find . -name "*.d" -exec rm -f {} +       # Remove dependency files
find . -name "*.log" -exec rm -f {} +     # Remove log files

# Force clean Python installation
echo "Running Python clean and install..."
python setup.py clean --all
python setup.py build
python setup.py install --force

echo "Build and installation completed."
