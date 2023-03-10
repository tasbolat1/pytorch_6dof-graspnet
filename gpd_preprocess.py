import numpy as np
import open3d as o3d
import argparse
import complex_environment_utils

np.random.seed(42)

from auxilary import *
from pathlib import Path

objs = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser(
    description='Visualize grasps',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cat", type=str, help="Choose simulation obj name.", choices=objs, default='box' )
parser.add_argument("--idx", type=int, help="Choose obj id.", default=14)
parser.add_argument('--experiment_type', type=str, help='define experiment type for isaac. Ex: complex, single', default='single', choices=['single', 'complex'])
args = parser.parse_args()

cat = args.cat
idx = args.idx

print(f'Processing for {cat}{idx:003} ... ')

data_dir = '../experiments/pointclouds'

if args.experiment_type == 'single':
    _, all_pcs, _, _, _, _, _, _, _ = parse_isaac_data(args.cat, args.idx, data_dir=data_dir)
elif args.experiment_type == 'complex':
    pc, pc_env, obj_stable_t, obj_stable_q, pc1, pc1_view, isaac_seed = complex_environment_utils.parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz',
                                                            cat=args.cat, idx=args.idx, env_num=0,
                                                            filter_epsion=1.0)
    all_pcs = np.expand_dims(pc, axis=0)

for i in range( all_pcs.shape[0] ) :

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pcs[i])

    save_dir = f'{data_dir}/pcd/{cat}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(f'{save_dir}/{cat}{idx:003}_{i}.pcd', pcd)

print(f'Number of unique pcs are {i+1} .')
print(f'Done.')
