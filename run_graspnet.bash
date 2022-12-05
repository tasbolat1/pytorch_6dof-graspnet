#!/bin/bash

# NUM_GRASP_SAMPLES=10
# REFINE_STEPS=50
# SAVE_DIR="../experiments/generated_grasps_experiment2"
# BATCH_SIZE=64

# RUN AS: source run_graspnet.bas 10 50 ../experiments/generated_grasps_experiment4 128

NUM_GRASP_SAMPLES="$1"
REFINE_STEPS="$2"
SAVE_DIR="$3"
BATCH_SIZE="$4"

run_graspnet_sampler() {
    python generate_data_from_isaac_pcs.py --batch_size ${BATCH_SIZE} --cat $1 --idx $2 --num_grasp_samples ${NUM_GRASP_SAMPLES}   --refinement_method gradient --refine_steps ${REFINE_STEPS} --save_dir ${SAVE_DIR}
}

cat="scissor"
for idx in 4 7
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="mug"
for idx in 2 8 14
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="bottle"
for idx in 3 12 19
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="box"
for idx in 14 17
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="bowl"
for idx in 1 16
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="cylinder"
for idx in 2 11
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="pan"
for idx in 3 6
    do
    run_graspnet_sampler ${cat} ${idx}
    done


cat="fork"
for idx in 1 11
    do
    run_graspnet_sampler ${cat} ${idx}
    done

cat="hammer"
for idx in 15
    do
    run_graspnet_sampler ${cat} ${idx}
    done


cat="spatula"
for idx in 1 14
    do
    run_graspnet_sampler ${cat} ${idx}
    done