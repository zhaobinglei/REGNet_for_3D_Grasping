from .eval_utils.torch_scene_point_cloud import TorchScenePointCloud
from .eval_utils.evaluation_data_generator import EvalDataValidate, EvalDataTest

def eval_test(points, predicted_grasp, view_num, table_height, depth, width, gpu: int):
    '''
    points         : [N, 3]
    predicted_grasp: [B, 8]
    '''
    print(depth, width)
    view_cloud = EvalDataTest(points, predicted_grasp, view_num, table_height, depth, width, gpu)
    grasp_nocoll_view = view_cloud.run_collision_view()
    return grasp_nocoll_view

def eval_validate(formal_dict, predicted_grasp, view_num: int, table_height, depth, width, gpu: int):
    '''
    formal_dict:  {}
    predicted_grasp: [B, 8]
    '''
    print(depth, width)
    view_cloud = EvalDataValidate(formal_dict, predicted_grasp, view_num, table_height, depth, width, gpu)
    vgr, score, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = view_cloud.run_collision()
    return vgr, score, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene
    
if __name__ == "__main__":
    pass