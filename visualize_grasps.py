import numpy as np
import trimesh
from auxilary import *
import argparse

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
parser.add_argument("--classifier", type=str, help="Choose classifier.", choices=['N', 'Sminus', 'S', 'SE', 'E'], default='N')
parser.add_argument("--method", type=str, help="Choose refine method.", choices=['N', 'graspnet', 'GraspFlow', 'metropolis'], default='N')
parser.add_argument("--data_dir", type=str, help="data directory to npz files.", default='../experiments/generated_grasps')
parser.add_argument("--grasp_space", type=str, help=".", choices=['Euler', 'SO3', 'Theta', 'N'], default='N', )
args = parser.parse_args()

cat = args.cat
idx = args.idx

sampler=args.sampler # graspnet, gpd
classifier = args.classifier # N, Sminus, S, SE, E
refineMethod = args.method # N, graspnet, GraspFlow, MH
which_env = 0
which_trial = 0

# def get_rot_matrix(t1,q1):
#     view_rot_matrix = np.eye(4)
#     view_rot_matrix[:3,:3] = R.from_quat(q1).as_matrix()
#     view_rot_matrix[:3,3] = t1
#     return view_rot_matrix

grasps_file = f'{args.data_dir}/{cat}{idx:003}_{sampler}.npz'

data = np.load(grasps_file)

ts = data[f'{sampler}_{classifier}_{refineMethod}_{args.grasp_space}_grasps_translations']
qs = data[f'{sampler}_{classifier}_{refineMethod}_{args.grasp_space}_grasps_quaternions']
try:
    scores = data[f'{sampler}_{classifier}_{refineMethod}_{args.grasp_space}_scores']
except:
    scores = data[f'{sampler}_{classifier}_{refineMethod}_{args.grasp_space}_original_scores']

grasps = get_rot_matrix_batch(ts[which_env], qs[which_env])
scores = scores[which_env]

print(grasps.shape)

all_pc_world = data['pc']
pc_mesh = trimesh.points.PointCloud(all_pc_world[which_env])
obj_stable_t = data['obj_stable_translations']
obj_stable_q = data['obj_stable_quaternions']


obj_mesh = load_mesh(cat, idx)
obj_mesh.visual.face_colors = [128,128,128,128]
obj_transform = get_rot_matrix(obj_stable_t[which_env], obj_stable_q[which_env])


# visualize the pointclouds inorld frame
print('VISUALIZING SAMPLED GRASPS ... ')

scene = trimesh.Scene()
scene.add_geometry(pc_mesh)
scene.add_geometry(obj_mesh, transform=obj_transform)

center_box_mesh = trimesh.primitives.Box(extents=[0.03,0.03,0.03])
scene.add_geometry( center_box_mesh )


for i, (g, s) in enumerate(zip(grasps, scores)):
    if i != which_trial:
        continue
    print(ts[which_env][i])
    print(qs[which_env][i])
    scene.add_geometry( gripper_bd(s), transform = g)
    panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
    panda_gripper.apply_transformation(transform=g)
    for _mesh in panda_gripper.get_meshes():
        _mesh.visual.face_colors = [125,125,125,80]
        scene.add_geometry(_mesh)

    if i == which_trial:
        break
        
scene.show()

