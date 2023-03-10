cat=1
declare -a obj=('pan 12' 'spatula 14' 'bottle 0' 'mug 8' 'bowl 8' 'bowl 9' 'fork 6' 'fork 15')
declare -a obj=('mug 8')
classifier="SECN"

for i in "${obj[@]}"
    do
        arrIN=(${i// / })
        cat=${arrIN[0]} 
        idx=${arrIN[1]} 
        python visualize_grasps.py --cat $cat --idx $idx\
         --data_dir ../experiments/generated_grasps_experiment86 --experiment_type diner001\
          --show_body 0 --method GraspOptES --classifier $classifier --grasp_space SO3 \
          --bad_grasp_threshold 0.5

    done    



