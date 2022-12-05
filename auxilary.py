'''
Author: Tasbolat Taunyazov
Some utility functions for graspnet. These functions are collected from different github repos
'''

import pickle
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import json

def load_mesh(cat, idx, grasp_data_dir = '/home/tasbolat/some_python_examples/graspflow_models/grasper/grasp_data'):
    # load an object
    if cat in ['box', 'cylinder']:
        mesh = trimesh.load(f'{grasp_data_dir}/meshes/{cat}/{cat}{idx:003}.stl')
    else:
        mesh = trimesh.load(f'{grasp_data_dir}/meshes/{cat}/{cat}{idx:003}.obj', force='mesh')

    info = json.load(open(f'{grasp_data_dir}/info/{cat}/{cat}{idx:003}.json'))

    mesh.apply_scale(info['scale'])

    return mesh

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


def get_gripper_pc(t,q):
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

    gripper_pc = np.concatenate([gripper_line_points_main_part, gripper_line_points_handle_part], axis=0)
    
    _gripper_pc = np.hstack([gripper_pc, np.expand_dims(np.ones(gripper_pc.shape[0]), axis=1)])
    H = get_rot_matrix(t,q)

    gripper_pc = np.matmul(H, _gripper_pc.T).T[:,:3]
    
    return gripper_pc

def is_topdown(t,q, z_threshold, alpha_threshold):
    '''
    Checks weather grasps are top down based on angle between an orientation of grasp and xy plane
    '''

    # check table distance
    gripper_pc = get_gripper_pc(t,q)
    gripper_pc_z_mean = np.mean(gripper_pc, axis=1)[3]

    if gripper_pc_z_mean < z_threshold:
        return 0
    
    rot = R.from_quat(q)
    new_dir = rot.apply([0,0,1])
    # get angle
    angle = np.arccos((-1.0 * new_dir[2]))
    # print(f'angle = {angle*180/np.pi}')

    if alpha_threshold > angle:
        return 1

    return 0

def is_far_grasp(grasps, pc):
    _pc_mean = np.mean(pc, axis=0)
    _pc_trimesh = trimesh.points.PointCloud(pc)
    bounds = _pc_trimesh.bounds
    threshold_coeff = 2
    l2_threshold = threshold_coeff*np.linalg.norm(bounds[0] - bounds[1])
    _t = grasps[:,:3,3]

    l2_distance = np.linalg.norm(_t-np.expand_dims(_pc_mean, axis=0),axis=1)
    far_grasps_flag = (l2_distance < l2_threshold)

    return far_grasps_flag


def is_topdown_batch(grasps, z_threshold, alpha_threshold):

    res = np.zeros(grasps.shape[0])

    t = grasps[:,:3,3]
    q = R.from_matrix(grasps[:,:3,:3]).as_quat()

    for i in range(t.shape[0]):
        res[i] = is_topdown(t[i], q[i], z_threshold=z_threshold, alpha_threshold=alpha_threshold)

    return res


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

def parse_isaac_data(cat, idx, data_dir):

    NUM_POINTS = 1024

    # load data
    data_dir = f'{data_dir}/{cat}{idx:003}.npz'
    data = np.load(data_dir)
    # print(list(data.keys()))

    view_q0 = data['view_rot_0']
    view_t0 = data['view_pos_0']
    view_q1 = data['view_rot_1']
    view_t1 = data['view_pos_1']
    view_q2 = data['view_rot_2']
    view_t2 = data['view_pos_2']
    view_q3 = data['view_rot_3']
    view_t3 = data['view_pos_3']

    view0 = get_rot_matrix_batch(view_t0, view_q0) # 4x4 w.r.t WORLD
    view1 = get_rot_matrix_batch(view_t1, view_q1)
    view2 = get_rot_matrix_batch(view_t2, view_q2)
    view3 = get_rot_matrix_batch(view_t3, view_q3)

    obj_stable_t = data['obj_stable_translations']
    obj_stable_q = data['obj_stable_quaternions']

    obj_t = data['obj_translations']
    obj_q = data['obj_quaternions']

    isaac_seed = data['seed']

    num_of_env = view_q0.shape[0]

    i = 0

    all_pc_world = np.zeros([num_of_env, NUM_POINTS, 3])
    all_pc1 = np.zeros([num_of_env, NUM_POINTS, 3])

    view_rotmat_pre = get_rot_matrix(np.array([0,0,0]), R.from_euler('zyx', [0, np.pi/2, -np.pi/2]).as_quat())

    for i in range(num_of_env):    
        pc0 = data[f'pc_0_{i}']
        pc1 = data[f'pc_1_{i}']
        pc2 = data[f'pc_2_{i}']
        pc3 = data[f'pc_3_{i}']

        _pc0 = np.hstack([pc0, np.expand_dims(np.ones(pc0.shape[0]), axis=1)]) # Nx4 w.r.t CAMERA
        pc0_world = np.matmul(view_rotmat_pre, _pc0.T).T 
        pc0_world = np.matmul(view0[i], pc0_world.T).T 

        _pc1 = np.hstack([pc1, np.expand_dims(np.ones(pc1.shape[0]), axis=1)])
        pc1_world = np.matmul(view_rotmat_pre, _pc1.T).T 
        pc1_world = np.matmul(view1[i], pc1_world.T).T 

        _pc2 = np.hstack([pc2, np.expand_dims(np.ones(pc2.shape[0]), axis=1)])
        pc2_world = np.matmul(view_rotmat_pre, _pc2.T).T 
        pc2_world = np.matmul(view2[i], pc2_world.T).T 

        _pc3 = np.hstack([pc3, np.expand_dims(np.ones(pc3.shape[0]), axis=1)])
        pc3_world = np.matmul(view_rotmat_pre, _pc3.T).T 
        pc3_world = np.matmul(view3[i], pc3_world.T).T 

        pc_world = np.concatenate([pc0_world, pc1_world, pc2_world, pc3_world], axis=0)
        pc_world = regularize_pc_point_count(pc=pc_world[:,:3], npoints=NUM_POINTS)

        all_pc_world[i, :] = pc_world

        all_pc1[i, :] = regularize_pc_point_count(pc=pc1, npoints=NUM_POINTS)


    return all_pc1, all_pc_world, obj_stable_t, obj_stable_q, obj_t, obj_q, view1, view_rotmat_pre, isaac_seed

def compensate_camera_frame(transforms, standoff = 0.2):
    # this is ad-hoc
    # TODO: check

    new_transforms = transforms.copy()
    for i, transform in enumerate(transforms):
        standoff_mat = np.eye(4)
        standoff_mat[2] = -standoff
        
        new_transforms[i, :3, 3] = np.matmul(transform,standoff_mat)[:3,3]
    return new_transforms


def get_rot_matrix(t1,q1):
    view_rot_matrix = np.eye(4)
    view_rot_matrix[:3,:3] = R.from_quat(q1).as_matrix()
    view_rot_matrix[:3,3] = t1
    return view_rot_matrix

def get_rot_matrix_batch(t,q):

    all_view_matrix = []
    for t1, q1 in zip(t,q):
        view_rot_matrix = np.eye(4)
        view_rot_matrix[:3,:3] = R.from_quat(q1).as_matrix()
        view_rot_matrix[:3,3] = t1
        all_view_matrix.append(view_rot_matrix)
    
    all_view_matrix = np.stack(all_view_matrix)
    return all_view_matrix

import trimesh.transformations as tra
from pathlib import Path

# define gripper
class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder='', num_get_distance_rays=20):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
            face_color {list of 4 int} (optional) -- RGBA, make A less than 255 to have transparent mehs visualisation
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04
        self.num_contact_points_per_finger = num_contact_points_per_finger
        self.num_get_distance_rays = num_get_distance_rays

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q

        self.base = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/hand.obj')
        self.base.metadata['name'] = 'base'
        self.finger_left = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/finger.obj')
        self.finger_left.metadata['name'] = 'finger_left'
        self.finger_right = self.finger_left.copy()
        self.finger_right.metadata['name'] = 'finger_right'

        # transform fingers relative to the base
        self.finger_left.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_left.apply_translation([0, -q, 0.0584])  # moves relative to y
        self.finger_right.apply_translation([0, +q, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_left, self.finger_right])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])


        # this makes to rotate the gripper to match with real world
        self.apply_transformation(tra.euler_matrix(0, 0, -np.pi/2))



    def apply_transformation(self, transform):
        #transform = transform.dot(tra.euler_matrix(0, 0, -np.pi/2))
        # applies relative to the latest transform
        self.finger_left.apply_transform(transform)
        self.finger_right.apply_transform(transform)
        self.base.apply_transform(transform)
        self.fingers.apply_transform(transform)
        self.hand.apply_transform(transform)


    def get_obbs(self):
        """Get list of obstacle meshes.
        Returns:
            list of trimesh -- bounding boxes used for collision checking
        """
        return [self.finger_left.bounding_box, self.finger_right.bounding_box, self.base.bounding_box]

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_left, self.finger_right, self.base]

    def get_bb(self, all=False):
        if all:
            return trimesh.util.concatenate(self.get_meshes()).bounding_box
        return trimesh.util.concatenate(self.get_meshes())
