#!/bin/bash

export FLAGS_call_stack_level=2
# Add this to reduce cpu memory!
export CUDA_MODULE_LOADING=LAZY

if [ ! -f "diff_case.txt" ]; then
    echo "no need to rerun"
    exit 0
fi

cat diff_case.txt

python run_diff_case.py
