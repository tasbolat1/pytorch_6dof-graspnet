#!/bin/bash

experiment_type='shelf008'

run_graspnet_sampler() {
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat pan --idx 12 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat bottle --idx 14 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat bowl --idx 8 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat bowl --idx 10 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat fork --idx 6 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    # python generate_data_from_isaac_pcs.py --batch_size 128 --cat mug --idx 8 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat pan --idx 6 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
    python generate_data_from_isaac_pcs.py --batch_size 128 --cat scissor --idx 7 --num_grasp_samples $2 --refinement_method gradient --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$1 --experiment_type $experiment_type --seed $3
}

# run_graspnet_sampler 301 10 1
# run_graspnet_sampler 302 50 1
# run_graspnet_sampler 303 500 1
# run_graspnet_sampler 304 1000 1
# run_graspnet_sampler 305 5000 1

run_graspnet_sampler 306 10 2
run_graspnet_sampler 307 50 2
run_graspnet_sampler 308 500 2
run_graspnet_sampler 309 1000 2
run_graspnet_sampler 310 5000 2

run_graspnet_sampler 311 10 3
run_graspnet_sampler 312 50 3
run_graspnet_sampler 313 500 3
run_graspnet_sampler 314 1000 3
run_graspnet_sampler 315 5000 3

# run_graspnet_sampler 101 10 1
# run_graspnet_sampler 102 50 1
# run_graspnet_sampler 103 500 1
# run_graspnet_sampler 104 1000 1
# run_graspnet_sampler 105 5000 1

# run_graspnet_sampler 106 10 2
# run_graspnet_sampler 107 50 2
# run_graspnet_sampler 108 500 2
# run_graspnet_sampler 109 1000 2
# run_graspnet_sampler 110 5000 2

# run_graspnet_sampler 111 10 3
# run_graspnet_sampler 112 50 3
# run_graspnet_sampler 113 500 3
# run_graspnet_sampler 114 1000 3
# run_graspnet_sampler 115 5000 3

# run_graspnet_sampler 116 10 4
# run_graspnet_sampler 117 50 4
# run_graspnet_sampler 118 500 4
# run_graspnet_sampler 119 1000 4
# run_graspnet_sampler 120 5000 4

# run_graspnet_sampler 121 10 5
# run_graspnet_sampler 122 50 5
# run_graspnet_sampler 123 500 5
# run_graspnet_sampler 124 1000 5
# run_graspnet_sampler 125 5000 5

# run_graspnet_sampler 201 10 1
# run_graspnet_sampler 202 50 1
# run_graspnet_sampler 203 500 1
# run_graspnet_sampler 204 1000 1
# run_graspnet_sampler 205 5000 1

# run_graspnet_sampler 206 10 2
# run_graspnet_sampler 207 50 2
# run_graspnet_sampler 208 500 2
# run_graspnet_sampler 209 1000 2
# run_graspnet_sampler 210 5000 2


# run_graspnet_sampler 211 10 3
# run_graspnet_sampler 212 50 3
# run_graspnet_sampler 213 500 3
# run_graspnet_sampler 214 1000 3
# run_graspnet_sampler 215 5000 3

# run_graspnet_sampler 216 10 4
# run_graspnet_sampler 217 50 4
# run_graspnet_sampler 218 500 4
# run_graspnet_sampler 219 1000 4
# run_graspnet_sampler 220 5000 4

# run_graspnet_sampler 221 10 5
# run_graspnet_sampler 222 50 5
# run_graspnet_sampler 223 500 5
# run_graspnet_sampler 224 1000 5
# run_graspnet_sampler 225 5000 5