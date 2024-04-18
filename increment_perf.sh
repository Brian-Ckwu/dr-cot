#!/bin/bash

# Define arrays for models, methods, and clinical conditions
ddxs=("" "-5ddx" "-20ddx")
models=("gemini-1.0-pro" "gpt-3.5-turbo-0125")
ccs=("cough" "fever" "nasal congestion" "shortness of breath")
methods=("naivezs" "zsdrcot")

# Nested loops to iterate over each combination of ddxs, model, method, and clinical condition
for ddx in "${ddxs[@]}"
do
    for model in "${models[@]}"
    do
        for cc in "${ccs[@]}"
        do
            for method in "${methods[@]}"
            do
                echo "Running experiment for num_ddxs: $ddx, model: $model, clinical condition: $cc, method: $method"
                python increment_perf.py \
                    --exp_dir "/home/brianckwu/dr-cot/experiments/$model/$method/$cc/single_prompt_w_dxs$ddx" \
                    --pipeline "calc_acc"
            done
        done
    done
done
