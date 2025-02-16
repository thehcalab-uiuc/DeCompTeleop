# ROS
import rospy
from sensor_msgs.msg import Image
import tf
import message_filters
import ros_numpy

# Standard
import csv
import os

# Third Party
import torch
from pytorch3d.transforms import quaternion_to_matrix
import cv2
import numpy as np


# Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save dataset
SAVE_DIR = '/path/to/data/save/folder/' # directory path to save data into (will be created if not exists)
FILENAME = 'sub_folder_name'            # sub-directory to create under save dir for current data collection run

RGB_TOPIC = '/zed2/zed_node/rgb/image_rect_color'       # RGB image topic
DEPTH_TOPIC = '/zed2/zed_node/depth/depth_registered'   # Depth map topic
CAM_FRAME = 'zed2_left_camera_optical_frame'            # Camera frame
ODOM_FRAME = 'odom'                                     # World frame
CAM_FRAME_TORCH3D = 'pytorch3d_cam'                     # Camera frame in PyTorch3D convention
CAM_FRAME_PULSAR = 'pulsar_cam'                         # Camera frame in Pulsar convention
WORLD_FRAME_TORCH3D = 'pytorch3d_world'                 # World frame in PyTorch3D convention
WORLD_FRAME_PULSAR = 'pulsar_world'                     # World frame in Pulsar convention

# Paths to subfolders
rgb_folder_path = os.path.join(SAVE_DIR, FILENAME, 'seq00', 'rgb')
if not os.path.exists(rgb_folder_path):
    os.makedirs(rgb_folder_path)
depth_folder_path = os.path.join(SAVE_DIR, FILENAME, 'seq00', 'depth')
if not os.path.exists(depth_folder_path):
    os.makedirs(depth_folder_path)
odom_file_path = os.path.join(SAVE_DIR, FILENAME, 'seq00', 'TransformationMatrices.csv')
pytorch3d_odom_file_path = os.path.join(SAVE_DIR, FILENAME, 'seq00', 'PyTorch3DTransformationMatrices.csv')
pulsar_odom_file_path = os.path.join(SAVE_DIR, FILENAME, 'seq00', 'Pulsar3DTransformationMatrices.csv')

# For quaternion_matrix_torch and translation_matrix_torch functions
trans_mat_torch_3d_homo = torch.eye(4).float().to(device)
rot_mat_torch_3d_homo = torch.eye(4).float().to(device)
trans_mat_pulsar_3d_homo = torch.eye(4).float().to(device)
rot_mat_pulsar_3d_homo = torch.eye(4).float().to(device)
trans_mat_debug_3d_homo = torch.eye(4).float().to(device)
rot_mat_debug_3d_homo = torch.eye(4).float().to(device)

curr_timestep = 0


def callback(rgb, depth):

    global curr_timestep
    global trans_mat_torch_3d_homo
    global rot_mat_torch_3d_homo
    global trans_mat_pulsar_3d_homo
    global rot_mat_pulsar_3d_homo
    global trans_mat_debug_3d_homo
    global rot_mat_debug_3d_homo

    try:

        # Get pose
        (trans_world_2_cam_torch3d,rot_world_2_cam_torch3d) = listener.lookupTransform(CAM_FRAME_TORCH3D, WORLD_FRAME_TORCH3D, rgb.header.stamp)
        (trans_world_2_cam_pulsar,rot_world_2_cam_pulsar) = listener.lookupTransform(CAM_FRAME_PULSAR, WORLD_FRAME_PULSAR, rgb.header.stamp)
        (trans_debug,rot_debug) = listener.lookupTransform(CAM_FRAME, ODOM_FRAME, rgb.header.stamp)

        # Get odom transforms
        # PyTorch3D homogeneous transform
        rot_mat_torch_3d = quaternion_to_matrix(torch.tensor([[rot_world_2_cam_torch3d[3], rot_world_2_cam_torch3d[0], rot_world_2_cam_torch3d[1], rot_world_2_cam_torch3d[2]]]).to(device))[0]
        rot_mat_torch_3d_homo[:3, :3] = rot_mat_torch_3d
        trans_mat_torch_3d_homo[0,3] = trans_world_2_cam_torch3d[0]
        trans_mat_torch_3d_homo[1,3] = trans_world_2_cam_torch3d[1]
        trans_mat_torch_3d_homo[2,3] = trans_world_2_cam_torch3d[2] 
        mat_pose_torch_3d = trans_mat_torch_3d_homo @ rot_mat_torch_3d_homo

        # Pulsar homogeneous transform
        rot_mat_pulsar_3d = quaternion_to_matrix(torch.tensor([[rot_world_2_cam_pulsar[3], rot_world_2_cam_pulsar[0], rot_world_2_cam_pulsar[1], rot_world_2_cam_pulsar[2]]]).to(device))[0]
        rot_mat_pulsar_3d_homo[:3, :3] = rot_mat_pulsar_3d
        trans_mat_pulsar_3d_homo[0,3] = trans_world_2_cam_pulsar[0]
        trans_mat_pulsar_3d_homo[1,3] = trans_world_2_cam_pulsar[1]
        trans_mat_pulsar_3d_homo[2,3] = trans_world_2_cam_pulsar[2] 
        mat_pose_pulsar_3d = trans_mat_pulsar_3d_homo @ rot_mat_pulsar_3d_homo

        # Original homogeneous transform
        rot_mat_debug_3d = quaternion_to_matrix(torch.tensor([[rot_debug[3], rot_debug[0], rot_debug[1], rot_debug[2]]]).to(device))[0]
        rot_mat_debug_3d_homo[:3, :3] = rot_mat_debug_3d
        trans_mat_debug_3d_homo[0,3] = trans_debug[0]
        trans_mat_debug_3d_homo[1,3] = trans_debug[1]
        trans_mat_debug_3d_homo[2,3] = trans_debug[2] 
        mat_pose_debug_3d = trans_mat_debug_3d_homo @ rot_mat_debug_3d_homo

        # Save transforms
        mat_pose_torch_3d_flat = mat_pose_torch_3d.reshape((-1)).cpu().numpy()
        with open(pytorch3d_odom_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(mat_pose_torch_3d_flat)
        mat_pose_pulsar_3d_flat = mat_pose_pulsar_3d.reshape((-1)).cpu().numpy()
        with open(pulsar_odom_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(mat_pose_pulsar_3d_flat)
        mat_pose_debug_3d_flat = mat_pose_debug_3d.reshape((-1)).cpu().numpy()
        with open(odom_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(mat_pose_debug_3d_flat)

        # Save RGB image
        rgb_image = ros_numpy.numpify(rgb)
        cv2.imwrite(os.path.join(rgb_folder_path, '{}.png'.format(str(curr_timestep).zfill(6))), rgb_image[:,:,:3])

        # Save depth image
        depth_image = ros_numpy.numpify(depth)
        np.save(os.path.join(depth_folder_path, '{}.npy'.format(str(curr_timestep).zfill(6))), depth_image)

        # Increment timer
        curr_timestep += 1
        print(curr_timestep)

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return


rospy.init_node('svo2sync')

listener = tf.TransformListener()
rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image)
depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)

ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
ts.registerCallback(callback)

rospy.spin()
