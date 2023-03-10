cat=1
##### WORKSTATION RUNS SHELF ONLY #####
declare -a obj=('pan 12' 'spatula 14' 'bottle 0' 'bowl 8' 'fork 6')
#declare -a obj=('bottle 14' 'bowl 8' 'bowl 10' 'pan 6' 'pan 12' 'fork 6' 'scissor 7')
declare -a classifier=('SC')
export CUDA_VISIBLE_DEVICES=1
scene='diner001'
f1=1051
f2=1052
f3=1053
o1=1060
upper=1063
seed=44
for j in $f1 $f2 $f3;
    do
    let "seed++"
    for i in "${obj[@]}"
        do
            arrIN=(${i// / })
            cat=${arrIN[0]} 
            idx=${arrIN[1]}
            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 1000 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 0 )) \
            --experiment_type $scene --seed $seed

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 50 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 10 )) \
            --experiment_type $scene --seed $seed
        done
    done