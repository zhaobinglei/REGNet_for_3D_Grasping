import open3d
import numpy as np
import copy
from configs import config

create_box = open3d.geometry.TriangleMesh.create_box
#create_box = open3d.geometry.create_mesh_box

HALF_BOTTOM_WIDTH = 0.06/2 + 0.01
FINGER_LENGTH = 0.06
FINGER_WIDTH = 0.01
HALF_HAND_THICKNESS = 0.005
BOTTOM_LENGTH = 0.06
BACK_COLLISION_MARGIN = 0
def get_hand_geometry(T_global_to_local, color=(0.1, 0.6, 0.3)):
    back_hand = create_box(height=2 * HALF_BOTTOM_WIDTH,
                           depth=HALF_HAND_THICKNESS * 2,
                           width=BOTTOM_LENGTH - BACK_COLLISION_MARGIN)
    # back_hand = open3d.geometry.TriangleMesh.create_cylinder(height=0.1, radius=0.02)

    temp_trans = np.eye(4)
    temp_trans[0, 3] = -BOTTOM_LENGTH
    temp_trans[1, 3] = -HALF_BOTTOM_WIDTH
    temp_trans[2, 3] = -HALF_HAND_THICKNESS
    back_hand.transform(temp_trans)

    finger = create_box((FINGER_LENGTH + BACK_COLLISION_MARGIN),
                        FINGER_WIDTH,
                        HALF_HAND_THICKNESS * 2)
    finger.paint_uniform_color(color)
    back_hand.paint_uniform_color(color)
    left_finger = copy.deepcopy(finger)

    temp_trans = np.eye(4)
    temp_trans[1, 3] = HALF_BOTTOM_SPACE
    temp_trans[2, 3] = -HALF_HAND_THICKNESS
    temp_trans[0, 3] = -BACK_COLLISION_MARGIN
    left_finger.transform(temp_trans)
    temp_trans[1, 3] = -HALF_BOTTOM_WIDTH
    finger.transform(temp_trans)

    # Global transformation
    T_local_to_global = np.linalg.inv(T_global_to_local)

    back_hand.transform(T_local_to_global)
    finger.transform(T_local_to_global)
    left_finger.transform(T_local_to_global)

    vis_list = [back_hand, left_finger, finger]
    for vis in vis_list:
        vis.compute_vertex_normals()
    return vis_list
