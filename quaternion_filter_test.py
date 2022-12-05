from math import degrees
from re import S
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import trimesh
import json
from auxilary import *

cat = 'box'
idx = 14
# load mesh
mesh = trimesh.load(f'/home/tasbolat/some_python_examples/graspflow_models/grasper/grasp_data/meshes/{cat}/{cat}{idx:003}.stl')
mesh.visual.face_colors = [125,125,125,200]


    




# q = R.from_euler('xyz', [0, 90, 0], degrees=True).as_quat()
# q = R.from_quat([-0.22708954,  0.13176556, -0.53958311, -0.7999489 ]).as_quat() #R.random().as_quat()
q = R.random().as_quat()

# print(q)
t = np.array([0,0,0])

# print(75*(np.pi)/180)
is_it_good = is_topdown(t=t, q=q, z_threshold=0, alpha_threshold=45*(np.pi)/180)
print( is_it_good )

scene = trimesh.Scene()
scene.add_geometry(mesh)
panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
panda_gripper.apply_transformation(transform=get_rot_matrix(q1=q,t1=t))
for _mesh in panda_gripper.get_meshes():
    _mesh.visual.face_colors = [125,125,125,120]
    scene.add_geometry(_mesh)
scene.show()