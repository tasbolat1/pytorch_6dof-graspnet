#!/bin/bash

# RUN AS: bash gpd_postprocess.bash 10 ../experiments/generated_grasps_experiment2 ../experiments/gpd_raw_grasps_experiment2

# NUM_GRASP_SAMPLES=10
# GRASP_FOLDER="../experiments/generated_grasps_experiment2"
# GPD_GRASP_FOLDER="../experiments/gpd_raw_grasps_experiment2"

NUM_GRASP_SAMPLES="$1"
GRASP_FOLDER="$2"
GPD_GRASP_FOLDER="$3"

python gpd_postprocess.py --cat mug --idx 2  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat mug --idx 8  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat mug --idx 14  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat bottle --idx 3  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat bottle --idx 12  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat bottle --idx 19  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat box --idx 14  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat box --idx 17  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat bowl --idx 1  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat bowl --idx 16  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat cylinder --idx 2  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat cylinder --idx 11 --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat pan --idx 3  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat pan --idx 6  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat scissor --idx 4  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat scissor --idx 7  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat fork --idx 1  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat fork --idx 11  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat hammer --idx 2  --num_grasp_samples ${NUM_GRASP_SAMPLES}  --grasp_folder ${GRASP_FOLDER}
python gpd_postprocess.py --cat hammer --idx 15  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}

python gpd_postprocess.py --cat spatula --idx 1  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}
python gpd_postprocess.py --cat spatula --idx 14  --num_grasp_samples ${NUM_GRASP_SAMPLES} --grasp_folder ${GRASP_FOLDER} --gpd_raw_grasp_folder ${GPD_GRASP_FOLDER}