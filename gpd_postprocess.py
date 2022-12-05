import numpy as np
import argparse

np.random.seed(42)

from auxilary import *
from pathlib import Path
import trimesh
from numpy import genfromtxt
from robot_ik_model import RobotModel
import tqdm.auto as tqdm

'''
RUN as:
python gpd_postprocess.py --cat box --idx 14 --num_grasp_samples 20
'''


# GPD TIME EXTRACTOR
def get_time(cat, idx, raw_file):
    '''
    GPD time extractor from log files
    '''
    log_files = Path(raw_file).glob(f'{cat}/log_{cat}{idx}_*.txt')
    word1 = '1. Candidate generation:'
    word2 = '2. Descriptor extraction:'
    # word3 = '3. Classification:'

    total_time=0
    for log_file in log_files:
        with open(log_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find(word1) != -1:
                    cand_gen_time = float(line.split(' ')[-1][:-2])
                    total_time+=cand_gen_time
                if line.find(word2) != -1:
                    desc_extr_time = float(line.split(' ')[-1][:-2])
                    total_time+=desc_extr_time
                # if line.find(word3) != -1:
                #     class_time = float(line.split(' ')[-1][:-2])
    return total_time


objs = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser(
    description='Visualize grasps',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cat", type=str, help="Choose simulation obj name.", choices=objs, default='box' )
parser.add_argument("--idx", type=int, help="Choose obj id.", default=14)
parser.add_argument("--num_grasp_samples", type=int, help="Number of grasp samples.", default=20)
parser.add_argument("--grasp_folder", type=str, help="Npz grasp_folder.", default="../experiments/generated_grasps_experiment4")
parser.add_argument("--gpd_raw_grasp_folder", type=str, help="Raw gpd grasps.", default="../experiments/gpd_raw_grasps_experiment4")


args = parser.parse_args()

cat = args.cat
idx = args.idx

print(f'Working with {cat}{idx:003} ... ')

def move_backward_grasp(transform, standoff = 0.2):
    standoff_mat = np.eye(4)
    standoff_mat[2] = -standoff
    new = np.matmul(transform,standoff_mat)
    return new[:3,3]

def normalize(x):
    low = np.min(x)-1
    high = np.max(x)+1
    return (x-low) / (high - low)

def gpd2grasps(rotations, translations):
    '''
    rotations: [B,3,3]
    translations: [B,3]
    scores: [B]
    '''
    # create grasps
    grasps = np.eye(4)
    grasps = np.repeat(grasps, translations.shape[0], axis=1).reshape(4,4,-1)
    grasps = np.transpose(grasps, (2,0,1))
    # apply rotations
    r = R.from_matrix(rotations)
    r2 = R.from_quat([0,0.707,0,0.707])
    r3 = R.from_quat([0,0,0.707,0.707])
    grasps[:, :3,:3] = (r*r2*r3).as_matrix()
    # apply translations
    grasps[:,:3,3] = translations
    
    for i in range(grasps.shape[0]):
        grasps[i,:3,3] = move_backward_grasp(grasps[i], standoff=0.1075) # original 0.0725, then I set 0.0825 also too much

    return grasps

# grasps_file = f'{args.grasp_folder}/{cat}{idx:003}.npz'
# data = np.load(grasps_file)

all_pc1, all_pc_world, obj_stable_t, obj_stable_q, obj_t, obj_q, view1, view_rotmat_pre, isaac_seed = parse_isaac_data(args.cat, args.idx, data_dir='/home/tasbolat/some_python_examples/graspflow_models/experiments/pointclouds')
grasps_dir = f'{args.gpd_raw_grasp_folder}/{cat}'

all_grasps_translations = []
all_grasps_quaternions = []
all_gpd_scores = []
all_theta = []
all_theta_pre = []

# TODO
all_time = []

panda_robot = RobotModel(30)

for i in range(all_pc_world.shape[0]):

    grasps_data = genfromtxt(f'{grasps_dir}/{cat}{idx:003}_{i}_gpd_grasps.csv', delimiter=',')

    gpd_scores = grasps_data[:,0]
    gpd_scores = normalize(gpd_scores) # normalize scores

    ts = grasps_data[:,1:4]
    rs = grasps_data[:,4:].reshape([-1, 3,3])

    # do grasp transformations from gpd world to isaac world and compensate gripper stuff
    grasps = gpd2grasps(rotations=rs, translations=ts)

    # filter out grasps that not reachable  
    theta, theta_pre = panda_robot.solve_ik_batch2(grasps)
    reachable_flag = ~np.isnan(theta).any(axis=1)
    reachable_idx = np.squeeze(np.argwhere(reachable_flag), axis=1)

    if np.sum(reachable_flag) == 0:
        raise ValueError(f'There is no reachable grasps available from gpd for: {cat}{idx:003}!')

    if np.sum(reachable_flag) < args.num_grasp_samples:
        left_over = args.num_grasp_samples - np.sum(reachable_flag)
        new_idx = np.random.choice(reachable_idx, size=left_over)
        reachable_idx = np.concatenate([reachable_idx, new_idx], axis=0)        


    theta, theta_pre = theta[reachable_idx], theta_pre[reachable_idx]
    grasps = grasps[reachable_idx]
    gpd_scores = gpd_scores[reachable_idx]

    if grasps.shape[0] > args.num_grasp_samples:
        theta, theta_pre = theta[:args.num_grasp_samples], theta_pre[:args.num_grasp_samples]
        grasps = grasps[:args.num_grasp_samples]
        gpd_scores = gpd_scores[:args.num_grasp_samples]

    
    all_grasps_translations.append(grasps[:,:3,3])
    all_grasps_quaternions.append(R.from_matrix(grasps[:,:3,:3]).as_quat())

    all_theta.append(theta)
    all_theta_pre.append(theta_pre)
    
    all_gpd_scores.append(gpd_scores)

    all_time.append(get_time(cat, idx, args.gpd_raw_grasp_folder))


all_grasps_translations = np.stack(all_grasps_translations)
all_grasps_quaternions = np.stack(all_grasps_quaternions)
all_gpd_scores = np.stack(all_gpd_scores)
all_theta = np.stack(all_theta)
all_theta_pre = np.stack(all_theta_pre)
all_time = np.stack(all_time)

print(f'Total sampled grasps shape = {all_grasps_translations.shape}')

#grasps_file = f'{args.grasp_folder}/{cat}{idx:003}_gpd'
Path(args.grasp_folder).mkdir(parents=True, exist_ok=True)
np.savez(f'{args.grasp_folder}/{args.cat}{args.idx:003}_gpd',
        gpd_N_N_N_grasps_translations = all_grasps_translations,
        gpd_N_N_N_grasps_quaternions = all_grasps_quaternions,
        gpd_N_N_N_original_scores = all_gpd_scores,
        gpd_N_N_N_theta = all_theta,
        gpd_N_N_N_theta_pre = all_theta_pre,
        gpd_N_N_N_time = all_time,

        pc = all_pc_world,
        obj_translations =  obj_t,
        obj_quaternions = obj_q,
        obj_stable_translations =  obj_stable_t,
        obj_stable_quaternions = obj_stable_q,
        seed = isaac_seed
)

# # append on npz
# data_dict = dict(data)
# data_dict["gpd_N_N_N_grasps_translations"] = all_grasps_translations
# data_dict["gpd_N_N_N_grasps_quaternions"] = all_grasps_quaternions
# data_dict["gpd_N_N_N_scores"] = all_scores
# data_dict["gpd_N_N_N_theta"] = all_theta
# data_dict["gpd_N_N_N_theta_pre"] = all_theta_pre


# np.savez(grasps_file, **data_dict)

print('Success')