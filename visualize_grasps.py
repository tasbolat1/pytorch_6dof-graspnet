import numpy as np
import trimesh
from auxilary import *
import argparse
import complex_environment_utils

'''
RUN AS:
python visualize_grasps.py --classifier Sminus --method graspnet --cat scissor --idx 7

'''

objs = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser(
    description='Visualize grasps',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cat", type=str, help="Choose simulation obj name.", choices=objs, default='box' )
parser.add_argument("--idx", type=int, help="Choose obj id.", default=1)
parser.add_argument("--sampler", type=str, help="Choose sampler.", choices=['graspnet', 'gpd'], default='graspnet')
parser.add_argument("--classifier", type=str, help="Choose classifier.", default='N')
parser.add_argument("--method", type=str, help="Choose refine method.", choices=['N', 'graspnet', 'GraspFlow', 'metropolis', 'GraspOpt'], default='N')
parser.add_argument("--data_dir", type=str, help="data directory to npz files.", default='../experiments/generated_grasps')
parser.add_argument("--grasp_space", type=str, help=".", choices=['Euler', 'SO3', 'Theta', 'N'], default='N', )
parser.add_argument("--which_env", type=int, help="Env number", default=0)
parser.add_argument('--experiment_type', type=str, help='define experiment type for isaac. Ex: complex, single', default='single', choices=['single', 'complex'])
args = parser.parse_args()

cat = args.cat
idx = args.idx

sampler=args.sampler # graspnet, gpd
classifier = args.classifier # N, Sminus, S, SE, E
refineMethod = args.method # N, graspnet, GraspFlow, MH
which_env = args.which_env
which_trial = 7


grasps_file = f'{args.data_dir}/{cat}{idx:003}_{sampler}.npz'

data = np.load(grasps_file)

print(dict(data).keys())

prefix = f"{sampler}_{classifier}_{refineMethod}_{args.grasp_space}"

ts = data[f'{prefix}_grasps_translations']
qs = data[f'{prefix}_grasps_quaternions']
try:
    scores = 1
    for classifier in args.classifier:
        _scores = data[f'{prefix}_{classifier}_final_scores']
        scores *= _scores
except:
    scores = data[f'{prefix}_original_scores']

grasps = get_rot_matrix_batch(ts[which_env], qs[which_env])
scores = scores[which_env]

print(grasps.shape)

if args.experiment_type == 'single':
    all_pc_world = data['pc']
    pc_env = None
    obj_stable_t = data['obj_stable_translations']
    obj_stable_q = data['obj_stable_quaternions']
    obj_mesh = load_mesh(cat, idx)

elif args.experiment_type == 'complex':
    pc, pc_env, obj_stable_t, obj_stable_q, pc1, pc1_view, isaac_seed = complex_environment_utils.parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz',
                                                            cat=cat, idx=idx, env_num=0,
                                                            filter_epsion=1.0)

    all_pc_world = np.expand_dims( regularize_pc_point_count(pc, npoints=1024), axis=0)
    obj_stable_t = np.expand_dims(obj_stable_t, axis=0)
    obj_stable_q = np.expand_dims(obj_stable_q, axis=0)
    view_rotmat_pre = None
    pc_env = regularize_pc_point_count(pc_env, npoints=int(pc_env.shape[0]*0.2))
    pc_env = np.expand_dims(pc_env, axis=0)
    obj_mesh = complex_environment_utils.load_shape_for_complex(cat=cat, idx=idx)

pc_mesh = trimesh.points.PointCloud(all_pc_world[which_env])

obj_mesh.visual.face_colors = [128,128,128,128]
obj_transform = get_rot_matrix(obj_stable_t[which_env], obj_stable_q[which_env])

print(f'obj_transform = {obj_transform}')

# visualize the pointclouds inorld frame
print('VISUALIZING SAMPLED GRASPS ... ')

scene = trimesh.Scene()
scene.add_geometry(pc_mesh)
scene.add_geometry(obj_mesh, transform=obj_transform)

center_box_mesh = trimesh.primitives.Box(extents=[0.03,0.03,0.03])
scene.add_geometry( center_box_mesh )

if pc_env is not None:
    pc_env = trimesh.points.PointCloud(pc_env[which_env], colors=[0,255,0,50])
    scene.add_geometry(pc_env)


for i, (g, s) in enumerate(zip(grasps, scores)):
    # if i != which_trial:
    #     continue
    # print(ts[which_env][i])
    # print(qs[which_env][i])
    # print(s)
    scene.add_geometry( gripper_bd(s), transform = g)
    panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
    panda_gripper.apply_transformation(transform=g)
    for _mesh in panda_gripper.get_meshes():
        _mesh.visual.face_colors = [125,125,125,80]
        scene.add_geometry(_mesh)

    # if i == which_trial:
    #     break
        
scene.show()

