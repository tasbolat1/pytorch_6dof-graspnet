import numpy as np

from auxilary import *

save_dir = '../experiments/generated_grasps_experiment28'

def get_pc(cat, idx):
    all_pc1, all_pc_world, all_pc_world_raw, obj_stable_t, obj_stable_q, obj_t, obj_q, view1, view_rotmat_pre, isaac_seed = parse_isaac_data(cat, idx, data_dir='/home/tasbolat/some_python_examples/graspflow_models/experiments/pointclouds')
    fname = f'{save_dir}/{cat}{idx:003}_pc'
    save_raw_pc(fname, all_pc_world_raw)


test_info = {
    'mug': [2, 8, 14],
    'bottle': [3, 12, 19],
    'box':[14, 17],
    'bowl':[1, 16],
    'cylinder': [2, 11],
    'pan': [3,6],
    'scissor': [4,7],
    'fork': [1, 11],
    'hammer': [2, 15],
    'spatula': [1, 14]
}

for cat in test_info.keys():
    for idx in test_info[cat]:
        get_pc(cat, idx)


# data = np.load(f'{save_dir}/box014_pc.npz')
# print(len(data.keys()))
# print(data['0'].shape)