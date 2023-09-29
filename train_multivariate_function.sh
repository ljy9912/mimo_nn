#!/bin/bash
# Define an array of learning rates
work_dir="./work_dirs"

# Loop to run main.py 5 times with different seeds
for dataset in "LeakyReLU" "soc_2dim"; do
    for seed in {1..3}; do
        command="python data_generation.py --fun \"${dataset}\" --seed ${seed}"
        echo "Executing: ${command}"
        eval ${command}
        for act_fun in "soc_2dim" "LeakyReLU" "ReLU" "PReLU"; do
            command="python main_multivariate_function.py --work_dir \"${work_dir}\" --seed ${seed} --act_fun \"${act_fun}\" --Dataset \"${dataset}\""
            echo "Executing: ${command}"
            eval ${command}
        done
    done
done