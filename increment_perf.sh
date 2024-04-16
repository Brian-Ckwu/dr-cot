#!/bin/bash

# Define arrays for models, methods, and clinical conditions
models=("gemini-1.0-pro" "gpt-3.5-turbo-0125")
methods=("naivezs" "zsdrcot")
ccs=("cough" "fever" "nasal congestion" "shortness of breath")

# Nested loops to iterate over each combination of model, method, and clinical condition
for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        for cc in "${ccs[@]}"
        do
            echo "Running experiment for model: $model, method: $method, clinical condition: $cc"
            python increment_perf.py \
                --exp_dir "/home/brianckwu/dr-cot/experiments/$model/$method/$cc/single_prompt_w_dxs-allddx" \
                --pipeline "calc_acc"
        done
    done
done
