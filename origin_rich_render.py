import os
import sys
import bpy 
import bmesh
import numpy as np
import os.path as osp
# from read_data import process_data
import argparse
from mathutils import Matrix, Vector
import math

context = bpy.context
scene = context.scene
render = scene.render

g_depth_clip_start = 0
g_depth_clip_end = 4

g_segm_clip_start = 0
g_segm_clip_end = 2

import os.path as osp
import os
from xml.dom.minidom import parse
import numpy as np

def clear_mesh():
    """ clear all meshes in the scene
    """
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select = True
    bpy.ops.object.delete()

def str_list_to_matrix(str_list):
    int_list = []
    for str_index in str_list:
        if str_index is not '':
            int_list.append(float(str_index))
    int_list = np.array(int_list)
    return int_list

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def render_settings(action_name, view_index):
    view_index = str(view_index)
    #change render engine to cycles
    bpy.data.scenes['Scene'].render.engine = 'CYCLES'
    scene = bpy.context.scene
    view_layer = scene.view_layers['View Layer']
    bpy.data.scenes['Scene'].render.image_settings.color_depth = '8'
    view_layer.use_pass_vector = True
    view_layer.use_pass_material_index = True
    view_layer.use_pass_emit = True
    view_layer.use_pass_z = True
    view_layer.use_pass_material_index = True
    base_path = 'D:\\Work\\Datasets\\Render_Rich\\'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    segm_output_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_map_value_node = tree.nodes.new('CompositorNodeMapValue')
    segm_output_node.format.file_format = 'PNG'
    if not osp.exists(osp.join(base_path, action_name)):
        os.mkdir(osp.join(base_path, action_name))
    if not osp.exists(osp.join(base_path, action_name, 'segm')):
        os.mkdir(osp.join(base_path, action_name, 'segm'))
    if not osp.exists(osp.join(base_path, action_name, 'segm', view_index)):
        os.mkdir(osp.join(base_path, action_name, 'segm', view_index))
    segm_output_node.base_path = osp.join(base_path, action_name, 'segm', view_index)
    segm_output_node.file_slots[0].path = 'segm-#####.png'
    depth_output_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_output_node.format.file_format = 'PNG' 
    if not osp.exists(osp.join(base_path, action_name, 'depth')):
        os.mkdir(osp.join(base_path, action_name, 'depth'))
    if not osp.exists(osp.join(base_path, action_name, 'depth', view_index)):
        os.mkdir(osp.join(base_path, action_name, 'depth', view_index))
    depth_output_node.base_path = osp.join(base_path, action_name, 'depth', view_index)
    depth_output_node.file_slots[0].path = 'depth-#####.png'
    #define map values 
    depth_map_value_node.offset[0] = - g_depth_clip_start
    depth_map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
    depth_map_value_node.use_min = True
    depth_map_value_node.use_max = True
    depth_map_value_node.min[0] = 0.0
    depth_map_value_node.max[0] = 1.0
    links.new(render_layer_node.outputs['Depth'], depth_map_value_node.inputs[0])
    links.new(depth_map_value_node.outputs[0], depth_output_node.inputs[0])
    links.new(render_layer_node.outputs['IndexMA'], segm_output_node.inputs[0])
        
def set_camera_intrinsic(camera_ob_data, K):
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camera_ob_data.sensor_fit, camera_ob_data.sensor_width, camera_ob_data.sensor_height)
    sensor_fit = get_sensor_fit(
        camera_ob_data.sensor_fit, 
        scene.render.pixel_aspect_x * resolution_x_in_px, 
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
        camera_ob_data.lens = K[0, 0] * sensor_size_in_mm / view_fac_in_px
        camera_ob_data.shift_x = (resolution_x_in_px / 2 - K[0, 2]) / view_fac_in_px
        camera_ob_data.shift_y = ((K[1, 2] - resolution_y_in_px / 2) * pixel_aspect_ratio) / view_fac_in_px

def process_data(end_dir='', ):
    ##### process calibration data #####
    scan_calibration_dir = osp.join(end_dir, 'scan_calibration')
    train_body_dir = osp.join(end_dir, 'train_body')
    data_dict = {}
    for action_name in os.listdir(train_body_dir):
        action_path = osp.join(train_body_dir, action_name)
        scene_name = action_name.split('_')[0]
        scene_path = osp.join(scan_calibration_dir, scene_name)
        scene_calibration_path = osp.join(scene_path, 'calibration')
        total_camera_matrix = []
        total_intrinsic = []
        total_distortion = []
        action_dict = {}
        for camera_dict_name in os.listdir(scene_calibration_path):
            camera_dict_path = osp.join(scene_calibration_path, camera_dict_name)
            camera_dom = parse(camera_dict_path).documentElement
            camera_matrix_stus = camera_dom.getElementsByTagName('CameraMatrix')
            intrinsic_stus = camera_dom.getElementsByTagName('Intrinsics')
            distortion_stus = camera_dom.getElementsByTagName('Distortion')
            str_camera_matrix = camera_matrix_stus[0].getElementsByTagName('data')[0].childNodes[0].nodeValue
            str_intrinsic = intrinsic_stus[0].getElementsByTagName('data')[0].childNodes[0].nodeValue
            str_distortion = distortion_stus[0].getElementsByTagName('data')[0].childNodes[0].nodeValue
            str_camera_matrix_list = str_camera_matrix.replace('\n', '').split(' ')
            str_intrinsic_list = str_intrinsic.replace('\n', '').split(' ')
            str_distortion_list = str_distortion.replace('\n', '').split(' ')
            camera_matrix = str_list_to_matrix(str_camera_matrix_list).reshape(3, 4)
            intrinsic = str_list_to_matrix(str_intrinsic_list).reshape(3, 3)
            distortion = str_list_to_matrix(str_distortion_list)
            total_camera_matrix.append(camera_matrix)
            total_intrinsic.append(intrinsic)
            total_distortion.append(distortion)
        total_camera_matrix = np.array(total_camera_matrix)
        total_intrinsic = np.array(total_intrinsic)
        total_distortion = np.array(total_distortion)
        action_dict['camera_matrix'] = total_camera_matrix
        action_dict['intrinsic'] = total_intrinsic
        action_dict['distortion'] = total_distortion
        ##### load body and scene meshes #####
        scene_mesh_path = []
        for scan_coord in os.listdir(scene_path):
            if scan_coord.endswith('.ply'):
                scene_mesh_path.append(osp.join(scene_path, scan_coord))
        action_sequence_body_path = []
        for frame_idx in os.listdir(action_path):
            frame_total_body_path = []
            action_frame_path = osp.join(action_path, frame_idx)
            for body_name in os.listdir(action_frame_path):
                if body_name.endswith('.ply'):
                    frame_total_body_path.append(osp.join(action_frame_path, body_name))
            action_sequence_body_path.append(frame_total_body_path)
        action_dict['scene_mesh_path'] = scene_mesh_path # scene list[]
        action_dict['body_mesh_list'] = action_sequence_body_path # frame_list[body_list[]]
        data_dict[action_name] = action_dict
    return data_dict

def generate_argparse():
    parser = argparse.ArgumentParser(description='generate depth image render based on given camera position.')
    parser.add_argument('--views', type=int, default=6)
    parser.add_argument('--output_folder', type=str, default='C:\\Users\\shizhelun\\Desktop\\rendered_rich')
    parser.add_argument('--color_depth', type=int, default=8)
    parser.add_argument('--format', type=str, default='PNG')
    parser.add_argument('--resolution_x', type=int, default=600)
    parser.add_argument('--resolution_y', type=int, default=600)
    parser.add_argument('--engine', type=str, default='BLENDER_EEVEE')
    return parser.parse_args()

def create_segmentation(object_name_list=['001']):
    scene_name = 'scan_camcoord'
    if 'Material' not in bpy.data.materials.keys():
        bpy.data.materials.new('Material')
    bpy.ops.object.select_all(action='DESELECT')
    for object_name in object_name_list:
        smplx_mesh = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)
        smplx_mesh.active_material = bpy.data.materials['Material']
        smplx_mesh.active_material.pass_index = 2
        bpy.ops.object.select_all(action='DESELECT')
    """
    bpy.data.objects[scene_name].select_set(True)
    scene_mesh = bpy.data.objects[scene_name]
    scene_mesh.active_material = bpy.data.materials['Material']
    scene_mesh.active_material.pass_index = 1
    """
    bpy.ops.object.select_all(action='DESELECT')
 
def camera_rot_to_XYZEuler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def matrix_to_quaternion(R):
    w = math.sqrt(R[0][0]+R[1][1]+R[2][2]+1)/2
    x = math.sqrt(R[0][0]-R[1][1]-R[2][2]+1)/2
    y = math.sqrt(-R[0][0]+R[1][1]-R[2][2]+1)/2
    z = math.sqrt(-R[0][0]-R[1][1]+R[2][2]+1)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (R[2][1]-R[1][2])/(4*w)
        y = (R[0][2]-R[2][0])/(4*w)
        z = (R[1][0]-R[0][1])/(4*w)
    if m == 1:
        w = (R[2][1]-R[1][2])/(4*x)
        y = (R[0][1]+R[1][0])/(4*x)
        z = (R[2][0]+R[0][2])/(4*x)
    if m == 2:
        w = (R[0][2]-R[2][0])/(4*y)
        x = (R[0][1]+R[1][0])/(4*y)
        z = (R[1][2]+R[2][1])/(4*y)
    if m == 3:
        w = (R[1][0]-R[0][1])/(4*z)
        x = (R[2][0]+R[0][2])/(4*z)
        y = (R[1][2]+R[2][1])/(4*z)
    return np.array([w,x,y,z])

def process_single_frame(action_name, scene_mesh_dir_list, frame_body_mesh_list, camera_matrix, intrinsic, distortion, frame_index):
    #load cameras
    cam_ob = bpy.data.objects['Camera']
    num_views = camera_matrix.shape[0]
    object_name_list = []
    for frame_body_mesh_path in frame_body_mesh_list:
        object_name_list.append(osp.basename(frame_body_mesh_path).replace('.ply', ''))
        bpy.ops.import_mesh.ply(filepath=frame_body_mesh_path)
    bpy.ops.import_mesh.ply(filepath=scene_mesh_dir_list[0])
    create_segmentation(object_name_list)
    for index in range(num_views):
        single_camera_matrix = camera_matrix[index]
        rot = np.array([[-1,0,0],[0,1,0],[0,0,1]])
        single_intrinsic = intrinsic[index]
        single_distortion = distortion[index]
        single_camera_rotation = single_camera_matrix[:3, :3]
        single_camera_translation = single_camera_matrix[:3, 3]
        single_camera_rotation = np.matmul(rot , single_camera_rotation)
        rot2 = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
        single_camera_rotation = -single_camera_rotation
        cam_ob.location[0] = -single_camera_translation[0]
        cam_ob.location[1] = single_camera_translation[1]
        cam_ob.location[2] = single_camera_translation[2]
        single_camera_euler = camera_rot_to_XYZEuler(single_camera_rotation)
        cam_ob.rotation_euler[0] = single_camera_euler[0]
        cam_ob.rotation_euler[1] = single_camera_euler[1]
        cam_ob.rotation_euler[2] = single_camera_euler[2]
        cam_ob.data.lens = 20
        render_settings(action_name=action_name, view_index=index)
        bpy.ops.render.render(write_still=True)
    clear_mesh()
        
# bpy.data.objects['Cube'].select_set(True)
# bpy.ops.object.delete()
total_data = process_data('D:\\Work\\Datasets\\Rich\\')
# load scene, body and camera
for action_name in total_data.keys():
    action_data = total_data[action_name]
    action_scene_mesh_path = action_data['scene_mesh_path']
    action_body_mesh_list = action_data['body_mesh_list']
    #load camera parameters
    camera_matrix = action_data['camera_matrix']
    intrinsic_matrix = action_data['intrinsic']
    distortion = action_data['distortion']
    num_frames = len(action_body_mesh_list)
    #setup output_path
    for index in range(num_frames):
        single_frame_body_path = action_body_mesh_list[index]
        process_single_frame(action_name,action_scene_mesh_path, single_frame_body_path, camera_matrix, intrinsic_matrix, distortion, index)
