from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
# import os
# import glob
# import mayavi.mlab as mlab
# from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
import pickle
import trimesh
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import time
from auxilary import *

def make_parser():

    objs = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder', type=str, default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder', type=str, default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method', choices={"gradient", "sampling"}, default='sampling')
    parser.add_argument('--refine_steps', type=int, default=1) # set it one for just sampling
    parser.add_argument("--cat", type=str, help="Choose simulation obj name.", choices=objs, default='box' )
    parser.add_argument("--idx", type=int, help="Choose obj id.", default=1)
    parser.add_argument('--visualize', action='store_true', help='If set, visualizes the grasps', default=False)
    parser.add_argument('--threshold', type=float, default=0.8,
    help= "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument( '--choose_fn',
        choices={ "all", "better_than_threshold", "better_than_threshold_in_sequence"},
        default='better_than_threshold',
        help= "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )
    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=60,
        help="Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    # parser.add_argument('--train_data', action='store_true')
    return parser



def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    grasp_evaluator_args.generate_dense_grasps = False
    args.choose_fn = 'all'
    args.generate_dense_grasps = False
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args, grasp_evaluator_args, args)

    # load the pointclouds
    pc, obj_pose_relative, camera_pose = read_pc(args.cat, args.idx)


    # the following functions returns all sequence (incl. refined traj.), so use last grasps.
    generated_grasps, generated_scores = estimator.generate_and_refine_grasps(pc)
    generated_grasps = generated_grasps[-args.num_grasp_samples:]
    generated_scores = generated_scores[-args.num_grasp_samples:]
    
    # check enough samples are given
    if not (len(generated_grasps) == args.num_grasp_samples):
        print('Not enough samples, generating more ...')
        left_over_samples = args.num_grasp_samples - len(generated_grasps)
        _generated_grasps, _generated_scores = estimator.generate_and_refine_grasps(pc)
        generated_grasps += _generated_grasps[-left_over_samples:]
        generated_scores += _generated_scores[-left_over_samples:]

    all_grasps = np.array(generated_grasps)
    all_scores = np.array(generated_scores)
    
    for i in range(len(all_grasps)):
        grasp = all_grasps[i]
        grasp_tran = np.matmul(np.linalg.inv(camera_pose.T), grasp)
        all_grasps[i] = grasp_tran
    
    translations = all_grasps[:,:3,3]
    quaternions = R.from_matrix(all_grasps[:,:3,:3]).as_quat()

    if args.visualize:
        pc_mesh = trimesh.points.PointCloud(pc)
        scene = trimesh.scene.Scene()
        scene.add_geometry(pc_mesh)
        for i, (grasp, score) in enumerate(zip(all_grasps, all_scores)):
            print(score)
            print(grasp)
            scene.add_geometry( gripper_bd(score), transform = grasp)
        scene.show()
        
    # save_path = f'../grasps_generated_graspnet_5second/'
    # Path(save_path).mkdir(parents=True, exist_ok=True)

    # # save results
    # _f_dir = f'{save_path}/{args.cat}{args.idx:03}'
    # # if 'heuristics' in args:
    # #     _f_dir = f'{save_path}/grasps_heuristics'

    # print(translations.shape)
    # np.savez(_f_dir,
    #              obj_pose_relative = obj_pose_relative,
    #              translations = translations,
    #              quaternions = quaternions,
    #              graspnet_scores = all_scores)

    # pickle.dump(my_pcs, open(f'{save_path}/pcs.pkl', 'wb'))

    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])


