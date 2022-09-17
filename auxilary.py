'''
Author: Tasbolat Taunyazov
Some utility functions for graspnet. These functions are collected from different github repos
'''

import pickle
import numpy as np
import trimesh


def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def read_pc(cat, id):
    '''
    Return:
    - pcs_in_camera_frame
    - obj_pose_relative: object pose in world frame
    - camera_pose: camera pose in world frame
    '''

    with open(f'/home/tasbolat/some_python_examples/GRASP/grasp_network/data/pcs/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
        data = pickle.load(fp)
        obj_pose_relative = data['obj_pose_relative'] # obj pose in world frame
        pcs = data['pcs'] # pcs in world frame, size 1000
        camera_poses = data["camera_poses"] # Camera pose in world frame

    pcs_camera = []
    for pc, camera_pose in zip(pcs, camera_poses):
        trimesh_pc = trimesh.points.PointCloud(pc)
        trimesh_pc.apply_transform(camera_pose.T)
        pcs_camera.append(trimesh_pc.vertices.view(np.ndarray))

    return pcs_camera[0], obj_pose_relative, camera_poses[0]

def normalize_pc_and_translation(pcs, trans):
    '''
    Shifts pointcloud and grasp translation
    Input:
    - pcs: [B,N,3]
    - trans: [B,3]
    Return:
    - pcs: [B,N,3]
    - trans: [B,3]
    - pc_mean: [B,3]
    '''
    pc_mean = pcs.mean()
    trans = trans-pc_mean

    return pcs, trans, pc_mean

def gripper_bd(quality=None):
    gripper_line_points_main_part = np.array([
        [0.0401874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0401874312758446, -0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0401874312758446, 0.0000599553131906, 0.1055731475353241],
    ])


    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    

    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
    else:
        color = None
    small_gripper_main_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0,1,2,3,4], color=color)],
                                                vertices = gripper_line_points_main_part)
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                                vertices = gripper_line_points_handle_part)
    small_gripper = trimesh.path.util.concatenate([small_gripper_main_part,
                                    small_gripper_handle_part])
    return small_gripper


def compensate_camera_frame(transforms, standoff = 0.2):
    # this is ad-hoc
    # TODO: check

    new_transforms = transforms.copy()
    for i, transform in enumerate(transforms):
        standoff_mat = np.eye(4)
        standoff_mat[2] = -standoff
        
        new_transforms[i, :3, 3] = np.matmul(transform,standoff_mat)[:3,3]
    return new_transforms