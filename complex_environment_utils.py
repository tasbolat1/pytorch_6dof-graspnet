import numpy as np
import trimesh
# from utils.points import regularize_pc_point_count
from scipy.spatial.transform import Rotation as R


def get_transform(quat, trans):
    '''
    Converts translation and quaternion into transform
    '''
    transform = np.eye(4)
    transform[:3,:3] = R.from_quat(quat).as_matrix()
    transform[:3,3] = trans
    return transform

def depth_to_pointcloud(depth):
        '''
        Converts depth to pointcloud.
        Arguments:
            * ``depth`` ((W,H) ``float``): Depth data.
        Returns:
            * pc ((3,N) ``float``): Pointcloud.
        W        '''
        fov = 75.0 * np.pi/180
        width = height = 512
        fy = fx = 0.5 / np.tan(fov * 0.5) # aspectRatio is one.
        # print(self.fov)        
        # height = depth.shape[0]
        # width = depth.shape[1]
        
        # depth = np.nan_to_num(depth, posinf=0, neginf=0, nan=0)
                 
        mask = np.where(depth > 0)
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        
        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        pc = np.vstack((world_x, world_y, world_z)).T

        pc = pc[~np.isinf(pc).any(axis=1)] # ad-hoc?
        return pc

def isaac_cam2world(pc, view):
    '''
    Converts pointcloud from camera view to world frame
    '''
    view_rotmat_pre = get_transform(R.from_euler('zyx', [0, np.pi/2, -np.pi/2]).as_quat(), np.array([0,0,0]))
    pc = np.hstack([pc, np.expand_dims(np.ones(pc.shape[0]), axis=1)]) # Nx4 w.r.t CAMERA
    pc_world = np.matmul(view_rotmat_pre, pc.T).T 
    pc_world = np.matmul(view, pc_world.T).T 
    return pc_world[:,:3]

def filter_pc_by_distance(pc, axis=0, lim=[-np.inf, np.inf]):
    '''
    Filters pointclouds by distance
    '''
    mask = (pc[:,int(axis)] >= lim[0]) & (pc[:,int(axis)] <= lim[1])
    return pc[mask]


def parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/complex.npz',
                             cat='scissor', idx=7,
                             env_num = 0, filter_epsion=0.075):

    # read data
    data = np.load(path_to_npz)

    isaac_seed = data['seed']
    
    total_view = 12
    seg_labels = {
    'bottle000': [1,11],
    'pan012': [2,11],
    'shelf001': [3,11],
    'bottle014': [4,11],
    'bowl008': [5,11],
    'bowl010': [6,11],
    'bowl016': [7,11],
    'fork006': [8,10],
    'fork015': [9,11],
    'mug002': [10,11],
    'mug008': [11,10],
    'pan006': [12,11],
    'scissor007': [13,10],
    'table000': [100, 0]
    }

    pcs = []
    pcs_env = []
    obj = f'{cat}{idx:003}'

    if obj not in seg_labels:
        raise f"Object {obj} is not in environment!"

    for view_id in range(total_view):

        # extract imgs
        depth = data[f'depth_{view_id}_{env_num}'] # [512,512]
        segment = data[f'segment_{view_id}_{env_num}'] # [512, 512]
        view_trans = data[f'view_pos_{view_id}'] # (4,)
        view_rots = data[f'view_rot_{view_id}'] # (3,)

        # ### use the following code to find view_id with max point
        # idxs, counts=np.unique(segment, return_counts=True)
        # if seg_labels[obj][0] in idxs:
        #     cur_idx = np.argwhere(idxs==seg_labels[obj][0])[0][0]
        #     # print(cur_idx)
        #     if seg_labels[obj][1] < counts[cur_idx]:
        #         seg_labels[obj][1] = view_id

        # continue

        # get segments
        depth_segmented_obj = np.where(segment == seg_labels[obj][0], depth, np.inf)
        depth_segmented_env = np.where(( (segment > 0)  ) & (segment != seg_labels[obj][0]), depth, np.inf)

        # convert to pc
        pc = depth_to_pointcloud(depth_segmented_obj)
        pc_env = depth_to_pointcloud(depth_segmented_env)

        # get pc r.t camera frame
        if seg_labels[obj][1] == view_id:
            pc1 = np.copy(pc)
            pc1_view = get_transform(quat=view_rots, trans=view_trans)

        # map to world frame
        pc = isaac_cam2world(pc, get_transform(quat=view_rots, trans=view_trans))
        pc_env = isaac_cam2world(pc_env, view=get_transform(quat=view_rots, trans=view_trans))

        # can filter by distance
        pc_mean = pc.mean(axis=0)
        for i in range(3):
            pc_env = filter_pc_by_distance(pc_env, axis=i, lim=[pc_mean[i]-filter_epsion,pc_mean[i]+filter_epsion])

        # exclude empty ones
        if pc.shape[0] > 0:
            pcs.append(pc)
        if pc_env.shape[0] > 0:
            pcs_env.append(pc_env)

    pc = np.concatenate(pcs, axis=0)
    pc_env = np.concatenate(pcs_env, axis=0)

    # print(seg_labels[obj])

    return pc, pc_env, data['obj_stable_translations'][seg_labels[obj][0]-1], data['obj_stable_quaternions'][seg_labels[obj][0]-1], pc1, pc1_view, isaac_seed

def load_shape_for_complex(cat, idx, path='../experiments/composites/shelf'):
    '''
    Loads mesh from complex environment
    '''
    f=f'{path}/{cat}{idx:003}.obj'
    mesh = trimesh.load(f, force='mesh')
    return mesh


if __name__ == "__main__":

    seg_labels = {
        'bottle000': 1,
        'pan012': 2,
        'shelf001': 3,
        'bottle014': 4,
        'bowl008': 5,
        'bowl010': 6,
        'bowl016': 7,
        'fork006': 8,
        'fork015': 9,
        'mug002': 10,
        'mug008': 11,
        'pan006': 12,
        'scissor007': 13
        }

    cat = 'bottle'
    idx = 0
    pc, pc_env, obj_trans, obj_quat, pc1, pc1_view, isaac_seed = parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz',
                                                               cat=cat, idx=idx, env_num=0,
                                                               filter_epsion=1.0)
    obj_mesh = load_shape_for_complex(cat=cat, idx=idx)
    obj_pose = get_transform(obj_quat, obj_trans)

    # pc_env = regularize_pc_point_count(pc_env, npoints=10000)
    # pc = regularize_pc_point_count(pc, npoints=1024)

    pc_mesh = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 50])
    pc_env = trimesh.points.PointCloud(pc_env, colors=[0, 255, 0, 50])
    scene = trimesh.Scene()

    scene.add_geometry(obj_mesh, transform=obj_pose)
    scene.add_geometry(pc_mesh)
    scene.add_geometry(pc_env)

    scene.show()

    # for cat in seg_labels.keys():
    #     pc, pc_env, obj_trans, obj_quat,  pc1, pc1_view, isaac_seed = parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz', cat=cat[:-3], idx=int(cat[-3:]), env_num=0)
        

    #     pc_mesh = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 50])
        
    #     # pc = regularize_pc_point_count(pc, npoints=100)

    #     obj_mesh = load_shape_for_complex(cat=cat[:-3], idx=int(cat[-3:]))
    #     obj_pose = get_transform(obj_quat, obj_trans)

    #     scene.add_geometry(obj_mesh, transform=obj_pose)
    #     # scene.add_geometry(obj_mesh)

    #     scene.add_geometry(pc_mesh)
    # # scene.add_geometry(pc_env)

    # scene.show()



    
