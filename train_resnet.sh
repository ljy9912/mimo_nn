#!/bin/bash

# Loop to run main.py 5 times with different seeds
for act_fun in "ReLU" "PReLU" "LeakyReLU" "soc_2dim_leaky" "soc" "soc_2dim"; do
    for angle_tan in "0.84"; do
        for seed in {1..3}; do
            work_dir="./work_dirs/resnet18_${act_fun}_angle_tan_${angle_tan}_seed_${seed}"
            command="python main_resnet.py --work_dir \"${work_dir}\" --seed ${seed} --act_fun \"${act_fun}\""
            echo "Executing: ${command}"
            eval ${command}
        done
    done
done
