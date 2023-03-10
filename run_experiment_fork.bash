cat=1 ## RUN THIS ONCE ONLY PER EXPERIMENT SET
declare -a obj=('pan 12' 'spatula 14' 'bottle 0' 'bowl 8' 'fork 6')
declare -a classifier=('SECN')
num_samples=10
scene='diner001'
export CUDA_VISIBLE_DEVICES=1;
f1=701
f2=706
f3=711
o1=716
for j in $f1 $f2 $f3;
    do
    for i in "${obj[@]}"
        do
            arrIN=(${i// / })
            cat=${arrIN[0]} 
            idx=${arrIN[1]}
            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 10 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 0 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 50 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 1 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 100 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 2 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 1000 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 3 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 5000 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 4 )) \
            --experiment_type $scene --seed $j
        done
    done

for ((j = $o1 ; j < $o1 + 3 ; j++ ));
    do
    for i in "${obj[@]}"
        do
            arrIN=(${i// / })
            cat=${arrIN[0]} 
            idx=${arrIN[1]}

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 10 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 0 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 50 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 3 )) \
            --experiment_type $scene --seed $j

            python generate_data_from_isaac_pcs.py\
            --cat $cat --idx $idx --num_grasp_samples 100 --refinement_method gradient \
            --refine_steps 2 --save_dir ../experiments/generated_grasps_experiment$(( $j + 6 )) \
            --experiment_type $scene --seed $j
        done
    done
