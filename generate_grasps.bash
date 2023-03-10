cat=1
declare -a obj=('bottle 0' 'mug 8' 'bowl 8' 'bowl 9' 'pan 12' 'spatula 14' 'fork 6' 'fork 15')

for i in "${obj[@]}"
    do
        arrIN=(${i// / })
        cat=${arrIN[0]} 
        idx=${arrIN[1]} 
        python generate_data_from_isaac_pcs.py --batch_size 10 \
        --cat $cat --idx $idx --num_grasp_samples 30 --refinement_method gradient \
        --refine_steps 10 --save_dir ../experiments/generated_grasps_experiment86 \
        --experiment_type diner001
    done    



