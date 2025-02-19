#!/bin/bash

rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
TORCH_CUDA_ARCH_LIST=7.0 python3 setup.py install # This is for h100. change this if needed