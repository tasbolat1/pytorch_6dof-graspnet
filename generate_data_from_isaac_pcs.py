from __future__ import print_function
from cmath import pi

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
import json
import torch
import tqdm.auto as tqdm

np.random.seed(42)
torch.manual_seed(42)

from robot_ik_model import RobotModel


'''
Run code as
python generate_data_from_isaac_pcs.py --cat box --idx 14 --n 20   --refinement_method gradient --refine_steps 50 --visualize
'''

# PARAMETERS TO TUNE
TOP_DOWN_PHI = 60 # angle between grasp vector and plane on xy. [degrees]
TABLE_HEIGHT = 0.20 # table height to specify z threshold on grasps. [cm]
IK_Q7_ITERATIONS = 30 # q7 angle iterations for checking IK solver [unitless], higher is better


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
    parser.add_argument('--batch_size', type=int, default=128,
        help="Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--save_dir', type=str, help='directory to save the generated grasps.', default='../experiments/generated_grasps')
    return parser



# def load_mesh(cat, idx):
#     # load an object
#     if cat in ['box', 'cylinder']:
#         mesh = trimesh.load(f'/home/tasbolat/some_python_examples/graspflow_models/grasper/grasp_data/meshes/{cat}/{cat}{idx:003}.stl')
#     else:
#         mesh = trimesh.load(f'/home/tasbolat/some_python_examples/graspflow_models/grasper/grasp_data/meshes/{cat}/{cat}{idx:003}.obj', force='mesh')

#     info = json.load(open(f'/home/tasbolat/some_python_examples/graspflow_models/grasper/grasp_data/info/{cat}/{cat}{idx:003}.json'))

#     mesh.apply_scale(info['scale'])

#     return mesh

def map2world(grasps, camera_view, view_rotmat_pre):

        grasps_transforms = np.einsum('ai,Nib -> Nab', view_rotmat_pre, grasps)
        # grasp_tran = np.matmul(view_rotmat_pre, grasp)
        grasps_transforms = np.einsum('ai,Nib -> Nab', camera_view, grasps_transforms)
        # grasp_tran = np.matmul(view1[0], grasp_tran)

        return grasps_transforms

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

    panda_robot = RobotModel(angle_iterations=IK_Q7_ITERATIONS)
    
    # parse isaac data
    print(f'Working with {args.cat}{args.idx:003} to generate {args.num_grasp_samples} samples ...')
    all_pc1, all_pc_world, obj_stable_t, obj_stable_q, obj_t, obj_q, view1, view_rotmat_pre, isaac_seed = parse_isaac_data(args.cat, args.idx, data_dir='/home/tasbolat/some_python_examples/graspflow_models/experiments/pointclouds')
    
    num_unique_pcs = all_pc1.shape[0] # corresponds to num_env in isaac
    print(f'num_unique_pcs shape = {num_unique_pcs}')

    obj_mesh = load_mesh(args.cat, args.idx)

    # prepare to sample
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args, grasp_evaluator_args, args)

#     

# # #     # # load the pointclouds
#     pc, obj_pose_relative, camera_pose = read_pc(args.cat, args.idx)
#     print(pc.shape)
#     print(obj_pose_relative)

# #     # the following functions returns all sequence (incl. refined traj.), so use last grasps.
# #     start_time = time.time()
#     print(args.batch_size)
#     generated_grasps, generated_scores = estimator.generate_and_refine_grasps(pc)
#     print(len(generated_grasps))
#     # print(generated_grasps[0])
#     # print(generated_grasps[300])
#     # generated_grasps = generated_grasps[-args.num_grasp_samples:]
#     # generated_scores = generated_scores[-args.num_grasp_samples:]

#     # print 

    # print('-----------------------------------------------------------------------------')

    
    total_refined_grasps_translations = np.zeros([num_unique_pcs, args.num_grasp_samples, 3])
    total_refined_grasps_quaternions =  np.zeros([num_unique_pcs, args.num_grasp_samples, 4])
    total_refined_grasps = np.zeros([num_unique_pcs, args.num_grasp_samples, 4,4])
    total_refined_scores = np.zeros([num_unique_pcs, args.num_grasp_samples])
    total_refined_time = np.zeros([num_unique_pcs, args.num_grasp_samples])
    total_refined_theta = np.zeros([num_unique_pcs, args.num_grasp_samples, 7])
    total_refined_theta_pre = np.zeros([num_unique_pcs, args.num_grasp_samples, 7])

    total_sampled_grasps_translations = np.zeros([num_unique_pcs, args.num_grasp_samples, 3])
    total_sampled_grasps_quaternions =  np.zeros([num_unique_pcs, args.num_grasp_samples, 4])
    total_sampled_grasps = np.zeros([num_unique_pcs, args.num_grasp_samples, 4,4])
    total_sampled_scores = np.zeros([num_unique_pcs, args.num_grasp_samples])
    total_sampled_time = np.zeros([num_unique_pcs, args.num_grasp_samples])
    total_sampled_theta = np.zeros([num_unique_pcs, args.num_grasp_samples, 7])
    total_sampled_theta_pre = np.zeros([num_unique_pcs, args.num_grasp_samples, 7])


    for i in tqdm.tqdm(range(num_unique_pcs), disable=False):

        left_over_size = args.num_grasp_samples
        all_sampled_grasps = []
        all_sampled_scores = []
        all_sampled_theta = []
        all_sampled_theta_pre = []
        
        all_refined_grasps = []
        all_refined_scores = []
        all_refined_theta = []
        all_refined_theta_pre = []
        
        current_size = 0
        all_sampling_time = 0
        all_refined_time = 0
        while left_over_size > 0:

            # do this, otherwise it modifies underline pc
            pc_to_model = all_pc1[i].copy()

            # sample grasps
            tic = time.time()
            generated_grasps, generated_scores, sampling_time = estimator.generate_and_refine_grasps(pc_to_model)

            toc = time.time()
            all_refined_time += toc-tic - sampling_time
            all_sampling_time += sampling_time

            sampled_grasps_size = int(len(generated_grasps)/(args.refine_steps+1)) #

            if sampled_grasps_size == 0:
                continue

            generated_grasps = np.array(generated_grasps)
            generated_scores = np.array(generated_scores)

            refined_grasps = generated_grasps[-sampled_grasps_size:]
            refined_scores = generated_scores[-sampled_grasps_size:]
            sampled_grasps = generated_grasps[:sampled_grasps_size]
            sampled_scores = generated_scores[:sampled_grasps_size]

            # Transfer grasps to the world frame
            refined_grasps = map2world(refined_grasps, view1[i], view_rotmat_pre)
            sampled_grasps = map2world(sampled_grasps, view1[i], view_rotmat_pre)

            # filter out graps that are not top-down
            sampled_topdown_flags = is_topdown_batch(sampled_grasps, z_threshold=TABLE_HEIGHT, alpha_threshold=TOP_DOWN_PHI*np.pi/180)
            refined_topdown_flags = is_topdown_batch(refined_grasps, z_threshold=TABLE_HEIGHT, alpha_threshold=TOP_DOWN_PHI*np.pi/180)
            topdown_flags = np.zeros_like(sampled_topdown_flags)
            topdown_flags[(refined_topdown_flags == 1) & (sampled_topdown_flags == 1)] = 1


            refined_grasps = refined_grasps[topdown_flags == 1]
            refined_scores = refined_scores[topdown_flags == 1]
            sampled_grasps = sampled_grasps[topdown_flags == 1]
            sampled_scores = sampled_scores[topdown_flags == 1]



            # Filter out grasps that sampled far: issue with GraspnetVAE - some grasps are super far
            far_grasps_flag = is_far_grasp(refined_grasps, all_pc_world[i])
            refined_grasps = refined_grasps[far_grasps_flag]
            refined_scores = refined_scores[far_grasps_flag]
            sampled_grasps = sampled_grasps[far_grasps_flag]
            sampled_scores = sampled_scores[far_grasps_flag]

            if refined_grasps.shape[0] == 0:
                continue

            # add transform to the grasps
            sampled_grasps = compensate_camera_frame(sampled_grasps, standoff=0.03)
            refined_grasps = compensate_camera_frame(refined_grasps, standoff=0.015)
            
            # filter out grasps that not reachable
            refined_theta, refined_theta_pre = panda_robot.solve_ik_batch2(refined_grasps)
            sampled_theta, sampled_theta_pre = panda_robot.solve_ik_batch2(sampled_grasps)

            reachable_idx_refined = ~np.isnan(refined_theta).any(axis=1)
            reachable_idx_sampled = ~np.isnan(sampled_theta).any(axis=1)
            reachable_idx = reachable_idx_refined & reachable_idx_sampled

            refined_theta, refined_theta_pre = refined_theta[reachable_idx], refined_theta_pre[reachable_idx]
            refined_grasps = refined_grasps[reachable_idx]
            refined_scores = refined_scores[reachable_idx]
            sampled_theta, sampled_theta_pre = sampled_theta[reachable_idx], sampled_theta_pre[reachable_idx]
            sampled_grasps = sampled_grasps[reachable_idx]
            sampled_scores = sampled_scores[reachable_idx]

            # add to all grasps
            selected_grasps_size = min(refined_grasps.shape[0], left_over_size)

            all_refined_grasps += list(refined_grasps)[-selected_grasps_size:]
            all_refined_scores += list(refined_scores)[-selected_grasps_size:]
            all_refined_theta += list(refined_theta)[-selected_grasps_size:]
            all_refined_theta_pre += list(refined_theta_pre)[-selected_grasps_size:]

            all_sampled_grasps += list(sampled_grasps)[-selected_grasps_size:]
            all_sampled_scores += list(sampled_scores)[-selected_grasps_size:]
            all_sampled_theta += list(sampled_theta)[-selected_grasps_size:]
            all_sampled_theta_pre += list(sampled_theta_pre)[-selected_grasps_size:]

            # count what is left of
            left_over_size = args.num_grasp_samples - selected_grasps_size - current_size
            current_size += selected_grasps_size

            #print(current_size)

        all_refined_grasps = np.array(all_refined_grasps)
        all_refined_theta = np.array(all_refined_theta)
        all_refined_theta_pre = np.array(all_refined_theta_pre)

        all_sampled_grasps = np.array(all_sampled_grasps)
        all_sampled_theta = np.array(all_sampled_theta)
        all_sampled_theta_pre = np.array(all_sampled_theta_pre)

        total_refined_grasps_translations[i, :, :] = all_refined_grasps[:,:3, 3]
        total_refined_grasps_quaternions[i, :, :] = R.from_matrix(all_refined_grasps[:, :3, :3]).as_quat()
        total_refined_grasps[i, :, :, :] = all_refined_grasps
        total_refined_scores[i, :] = all_refined_scores
        total_refined_time[i, :] = all_refined_time
        total_refined_theta[i,:,:] = all_refined_theta
        total_refined_theta_pre[i,:,:] = all_refined_theta_pre


        total_sampled_grasps_translations[i, :, :] = all_sampled_grasps[:,:3, 3]
        total_sampled_grasps_quaternions[i, :, :] = R.from_matrix(all_sampled_grasps[:, :3, :3]).as_quat()
        total_sampled_grasps[i, :, :, :] = all_sampled_grasps
        total_sampled_scores[i, :] = all_sampled_scores
        total_sampled_time[i, :] = all_sampling_time
        total_sampled_theta[i,:,:] = all_sampled_theta
        total_sampled_theta_pre[i,:,:] = all_sampled_theta_pre

        
    print(f'Total sampled grasps shape = {total_sampled_grasps.shape}')
    print(f'Total refined grasps shape = {total_refined_grasps.shape}')

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    np.savez(f'{args.save_dir}/{args.cat}{args.idx:003}_graspnet',
            graspnet_N_N_N_grasps_translations = total_sampled_grasps_translations,
            graspnet_N_N_N_grasps_quaternions = total_sampled_grasps_quaternions,
            graspnet_N_N_N_original_scores = total_sampled_scores,
            graspnet_N_N_N_theta = total_sampled_theta,
            graspnet_N_N_N_theta_pre = total_sampled_theta_pre,
            graspnet_N_N_N_time = total_sampled_time,


            graspnet_Sminus_graspnet_Euler_grasps_translations = total_refined_grasps_translations,
            graspnet_Sminus_graspnet_Euler_grasps_quaternions = total_refined_grasps_quaternions,
            graspnet_Sminus_graspnet_Euler_scores = total_refined_scores,
            graspnet_Sminus_graspnet_Euler_theta = total_refined_theta,
            graspnet_Sminus_graspnet_Euler_theta_pre = total_refined_theta_pre,
            graspnet_Sminus_graspnet_Euler_time = total_refined_time,

            pc = all_pc_world,
            obj_translations =  obj_t,
            obj_quaternions = obj_q,
            obj_stable_translations =  obj_stable_t,
            obj_stable_quaternions = obj_stable_q,
            seed = isaac_seed
    )



    ################## VISUALIZE ####################################
     
    which_env = 0
    # which_trial = 0
    pc_mesh = trimesh.points.PointCloud(all_pc_world[which_env])
    obj_mesh.visual.face_colors = [128,128,128,128]
    obj_transform = get_rot_matrix(obj_stable_t[which_env], obj_stable_q[which_env])

    _box_mesh = trimesh.primitives.Box(extents=[0.03,0.03,0.03])

    # visualize the pointclouds inorld frame
    print('VISUALIZING SAMPLED GRASPS ... ')
    if args.visualize:


        scene = trimesh.Scene()
        
        scene.add_geometry(pc_mesh)
        
        scene.add_geometry(obj_mesh, transform=obj_transform)
        scene.add_geometry( _box_mesh )
        for i, (grasp, score) in enumerate(zip(total_sampled_grasps[which_env], total_sampled_scores[which_env])):

            scene.add_geometry( gripper_bd(score), transform = grasp)
            # panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
            # panda_gripper.apply_transformation(transform=grasp)
            # for _mesh in panda_gripper.get_meshes():
            #     _mesh.visual.face_colors = [125,125,125,80]
            #     scene.add_geometry(_mesh)
            
        scene.show()

    print('VISUALIZING REFINED GRASPS ... ')
    if args.visualize:


        scene = trimesh.Scene()
        scene.add_geometry(pc_mesh)
        scene.add_geometry(obj_mesh, transform=obj_transform)
        scene.add_geometry( _box_mesh )
        for i, (grasp, score) in enumerate(zip(total_refined_grasps[which_env], total_refined_scores[which_env])):
            scene.add_geometry( gripper_bd(score), transform = grasp)
            # panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
            # panda_gripper.apply_transformation(transform=grasp)
            # for _mesh in panda_gripper.get_meshes():
            #     _mesh.visual.face_colors = [125,125,125,80]
            #     scene.add_geometry(_mesh)
            
        scene.show()


if __name__ == '__main__':
    main(sys.argv[1:])


