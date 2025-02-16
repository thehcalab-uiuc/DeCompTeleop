# Local Imports
from depth_anything_v2_metric.dpt import TorchDepthAnythingV2
from refinement import TravNetUp3NNRGB

# Standard Library Imports
from collections import deque
import threading
import os
import json

# Third Party Imports
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ROS Imports
import rospy
import ros_numpy
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import message_filters
from cv_bridge import CvBridge
import tf

# PyTorch3D Imports
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, PulsarPointsRenderer
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer.points.pulsar.renderer import Renderer as PulsarDirectRenderer

# PyTorch Imports
import torch
from torchvision import transforms


###################################
# INITIALIZE NECESSARY PARAMETERS #
###################################

# Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Camera parameters
BATCH_SIZE = 1
IMG_HEIGHT = 468
IMG_WIDTH = 832
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
FOCAL_LENGTH_X = 342.6786804199219
FOCAL_LENGTH_Y = 342.6786804199219
FOCAL_LENGTH = (FOCAL_LENGTH_X + FOCAL_LENGTH_Y) / 2.
PRINCIPAL_X = 407.8940734863281
PRINCIPAL_Y = 221.78753662109375
PRINCIPAL = (PRINCIPAL_X, PRINCIPAL_Y)
POSE_PRED_STEP_SIZE = 0.01
U = 1   # Pose prediction coefficients
V = 1   # Pose prediction coefficients

# Resize image to target size
transform_resize_up = transforms.Compose([
    transforms.Resize((IMG_SIZE[0], IMG_SIZE[1]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True),
])

# Transform image to torch tensor float32 of range 0-255
transform_to_img_tensor = transforms.Compose([
    transforms.ToTensor(),
    lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
])

# Normalize image
norm_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# PyTorch3D camera parameters
focal_length_torch = torch.tensor([[FOCAL_LENGTH]], dtype=torch.float32).to(device)                     # (1, 1)
principal_point_torch = torch.tensor([[PRINCIPAL[0], PRINCIPAL[1]]], dtype=torch.float32).to(device)    # (1, 2)
img_size_torch = torch.tensor([[IMG_SIZE[0], IMG_SIZE[1]]], dtype=torch.float32).to(device)             # (1, 2)

# Rendering settings
raster_settings = PointsRasterizationSettings(
    image_size=(IMG_SIZE[0],IMG_SIZE[1]),
    radius = 0.003,
    points_per_pixel = 1,
    bin_size = 0
)
gamma = 1e-1
bg_color = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

# Define pulsar renderer
renderer = PulsarDirectRenderer(
    width=IMG_SIZE[1],
    height=IMG_SIZE[0],
    max_num_balls=int(1e6),
    orthogonal_projection=False,
    right_handed_system=True,
    n_channels=3
).to(device)

# Define depth model
model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 
                'input_width': IMG_SIZE[1], 'input_height': IMG_SIZE[0], 'device': device, 'input_size': 518}
depth_anything = TorchDepthAnythingV2(**{**model_config, 'max_depth': 20.})
depth_anything.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/path/to/depth_anything_v2_metric_terra_vits.pth', map_location='cpu')['model'].items()})
depth_anything = depth_anything.to(device).eval()

# Define inpainting model
with open(os.path.join('/path/to/paint_l1_args.json'), 'r') as f:
    inpainting_train_args = json.load(f)
inpainting_model = TravNetUp3NNRGB(output_size=(IMG_HEIGHT,IMG_WIDTH), activation=inpainting_train_args['model_act']).to(device)
ckpt_path = os.path.join('/path/to/paint_l1.pth')
model_dict = {k.replace('module.', ''): v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
inpainting_model.load_state_dict(model_dict)
inpainting_model = inpainting_model.to(device).eval()
green_value = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)  # Get mask of holes

# ROS constant variables
SPIN_RATE = 30  # 30Hz
bridge = CvBridge()
MAX_SKIP = 10   # 1=15Hz, 5=6Hz, 10=3Hz
DELAY = 0.500   # delay in seconds

# Dynamic base_link -> odom GT Transform Variables
quat_base2odom = torch.zeros((1,4)).float().to(device)
trans_mat_base2odom_3d_homo = torch.eye(4).float().to(device)
rot_mat_base2odom_3d_homo = torch.eye(4).float().to(device)

# Dynamic base_link -> odom Pose Prediction Transform
pred_pose_mat_homo = torch.eye(4).float().to(device)        # base_link -> odom

# Static base_link -> cam Transform
base2cam = torch.tensor([[     0,-1,      0,  0.06],        # base_link -> cam
                         [ -0.05, 0,-0.9987,0.2627],
                         [0.9987, 0,  -0.05,-0.147],
                         [     0, 0,      0,     1]], 
                         dtype=torch.float32,
                         device=device)

# Static PyTorch3D Convention Transforms
pytorch3dworld2odom = torch.tensor([[0,0,1,0],              # pytorch3d world -> odom
                                    [1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,0,1]], 
                                    dtype=torch.float32,
                                    device=device)
cam2pytorch3dcam = torch.tensor([[-1, 0,0,0],               # cam -> pytorch3d cam
                                 [ 0,-1,0,0],
                                 [ 0, 0,1,0],
                                 [ 0, 0,0,1]], 
                                 dtype=torch.float32,
                                 device=device)

# Static Pulsar Convention Transforms
pulsarworld2odom = torch.tensor([[ 0,0,-1,0],               # pulsar world -> odom
                                 [-1,0, 0,0],
                                 [ 0,1, 0,0],
                                 [ 0,0, 0,1]], 
                                 dtype=torch.float32,
                                 device=device)
cam2pulsarcam = torch.tensor([[1, 0, 0,0],                  # cam -> pulsar cam
                              [0,-1, 0,0],
                              [0, 0,-1,0],
                              [0, 0, 0,1]], 
                              dtype=torch.float32,
                              device=device)

# Class to hold timestamp, point cloud, and pose information
class FrameData(object):

    def __init__(self):
        self.timestamp = None
        self.point_cloud_positions = None
        self.point_cloud_features = None
        self.point_cloud_complete = None
        self.ekf_pose = None
        self.torch_image = None
        self.initial_delay = None
        self.mutex = threading.Lock()   # Mutex lock for modifying and accessing frame information

    def update_vars(self, timestamp, point_cloud_positions, point_cloud_features, point_cloud, ekf_pose, torch_image, initial_delay):
        self.timestamp = timestamp
        self.point_cloud_positions = point_cloud_positions
        self.point_cloud_features = point_cloud_features
        self.point_cloud_complete = point_cloud
        self.ekf_pose =  ekf_pose       # base_link -> odom (x,y,z,q_x,q_y,q_z,q_w)
        self.torch_image = torch_image  # (1, 3, H, W), [0,1]
        self.initial_delay = initial_delay

# Instantiate global frame data object
frame_data = FrameData()

# Class to hold timestamp, point cloud, and pose information
class PosePredictionData(object):

    def __init__(self, step_size=0.01):
        self.orig_timestamp = 0
        self.curr_timestamp = 0
        self.curr_pose = None
        self.prediction_step_size = step_size

    def update_vars(self, orig_timestamp, curr_timestamp, curr_pose):
        self.orig_timestamp = orig_timestamp
        self.curr_timestamp = curr_timestamp
        self.curr_pose = curr_pose      # base_link -> odom (x,y,z,q_x,q_y,q_z,q_w)

# Instantiate global pose prediction data object
pose_pred_data = PosePredictionData(step_size=POSE_PRED_STEP_SIZE)

# Class to hold current user action
class UserActionData(object):

    def __init__(self):
        self.linear = []
        self.angular = []
        self.mutex = threading.Lock()   # Mutex lock for modifying and accessing action information

    def update_vars(self, linear, angular):
        self.linear = [linear.x, linear.y, linear.z]
        self.angular = [angular.x, angular.y, angular.z]

# Instantiate global user action data object
user_action = UserActionData()

# Queue for storing renderings to display
display_queue = deque()
display_mutex = threading.Lock()

skip_time = 0
latest_cv_image = None
angular_vel_weight = 0.05
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

##############################################
# IMAGE + EKF SYNCHRONIZED CALLBACK FUNCTION #
##############################################

def callback(rgb, ekf):

    with torch.no_grad():

        global skip_time
        global latest_cv_image

        # Emulating low frame rate
        if skip_time < MAX_SKIP:
            skip_time += 1
            return
        else:
            skip_time = 0

        # Emulate delay
        rospy.sleep(DELAY)

        ## Get timestamp
        current_time_ros = rospy.Time.now()
        curr_time = current_time_ros.secs + current_time_ros.nsecs / 1e9

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        starter.record()

        ## Get all necessary transforms for point cloud creation and later pose prediction
        # base_link -> odom homogeneous transform
        ekf_pos = ekf.pose.pose.position
        trans_base2odom_ros = [ekf_pos.x, ekf_pos.y, ekf_pos.z]
        ekf_quat = ekf.pose.pose.orientation
        quat_base2odom_ros = [ekf_quat.x, ekf_quat.y, ekf_quat.z, ekf_quat.w]
        complete_ekf = trans_base2odom_ros + quat_base2odom_ros # x,y,z,q_x,q_y,q_z,q_w
        quat_base2odom[0,0] = quat_base2odom_ros[3]
        quat_base2odom[0,1] = quat_base2odom_ros[0]
        quat_base2odom[0,2] = quat_base2odom_ros[1]
        quat_base2odom[0,3] = quat_base2odom_ros[2]
        rot_mat_base2odom_3d_homo[:3,:3] = quaternion_to_matrix(quat_base2odom)[0]
        trans_mat_base2odom_3d_homo[0,3] = trans_base2odom_ros[0]
        trans_mat_base2odom_3d_homo[1,3] = trans_base2odom_ros[1]
        trans_mat_base2odom_3d_homo[2,3] = trans_base2odom_ros[2] 
        mat_pose_base2odom = trans_mat_base2odom_3d_homo @ rot_mat_base2odom_3d_homo    # 4x4 base_link -> odom

        # odom -> base_link homogeneous transform
        mat_pose_odom2base = torch.linalg.inv(mat_pose_base2odom)                       # 4x4 odom -> base_link

        # PyTorch3D World -> PyTorch3D Cam homogeneous transform
        mat_pose_pytorch3dworld2pytorch3dcam = cam2pytorch3dcam @ base2cam @ mat_pose_odom2base @ pytorch3dworld2odom   # 4x4 pytorch3d world -> pytorch3d cam

        ## Access RGB image
        cv_image = bridge.compressed_imgmsg_to_cv2(rgb)
        re_rgb_msg = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        pub_orig_img.publish(re_rgb_msg)
        latest_cv_image = cv_image

        ## Convert RGB image to tensor
        torch_image = transform_to_img_tensor(cv_image).unsqueeze(0).to(device).type(torch.float32)  # (1, 3, H, W)
        inpainting_torch_image = torch_image / 255.

        ## Predict depth image
        input_depth = torch_image.squeeze(0).permute(1, 2, 0)
        input_depth = input_depth / 255.0 
        output_depth = depth_anything.infer_image(input_depth)
        output_depth = output_depth.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        ## Publish depth image
        output_depth_np = output_depth.squeeze(0).detach().cpu().permute((1,2,0)).numpy()   # (H, W, 1)
        output_depth_msg = ros_numpy.msgify(Image, output_depth_np, encoding='32FC1')
        pub_depth.publish(output_depth_msg)

        ## Create current PyTorch3D camera
        rot_mat_torch3d_cam = mat_pose_pytorch3dworld2pytorch3dcam[:3,:3].unsqueeze(0)  # (1, 3, 3)
        trans_mat_torch3d_cam = mat_pose_pytorch3dworld2pytorch3dcam[:3,3].unsqueeze(0) # (1, 3)
        torch3d_curr_cam = PerspectiveCameras(focal_length=focal_length_torch.clone(),
                                            principal_point=principal_point_torch.clone(),
                                            R=torch.transpose(rot_mat_torch3d_cam, 1, 2),
                                            T=trans_mat_torch3d_cam,
                                            in_ndc=False,
                                            image_size=img_size_torch.clone(),
                                            device=device)

        ## Create point clouds
        cloud_rgb_in = torch_image      # (1, 3, H, W)
        cloud_depth_in = output_depth   # (1, 1, H, W)

        point_cloud_torch = get_rgbd_point_cloud(camera=torch3d_curr_cam, image_rgb=cloud_rgb_in, depth_map=cloud_depth_in)
        pc_points = point_cloud_torch.points_packed()                               # (720*1280, 3)
        pc_points[:,0] *= -1.
        pc_points[:,2] *= -1.
        pc_colors = point_cloud_torch.features_packed().type(torch.float32) / 255.  # (720*1280, 3)
        pc_colors_alpha = torch.cat([pc_colors, torch.ones((pc_colors.shape[0], 1), device=device)], dim=1)
        pc_combined_alpha = Pointclouds(points=[pc_points], features=[pc_colors_alpha])
        position_list = pc_combined_alpha.points_list()
        features_list = pc_combined_alpha.features_list()

        ender.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        timing = starter.elapsed_time(ender) / 1000.    # seconds

        ## Update global frame information
        frame_data.mutex.acquire(blocking=True)
        frame_data.update_vars(timestamp=curr_time, 
                               point_cloud_positions=position_list, 
                               point_cloud_features=features_list, 
                               point_cloud=pc_combined_alpha, 
                               ekf_pose=complete_ekf, 
                               torch_image=inpainting_torch_image,
                               initial_delay=timing)
        frame_data.mutex.release()


#############################################
# CALLBACK FUNCTION TO SAVE ACTION COMMANDS #
#############################################

def cmd_vel_callback(cmd_vel):
    
    ## Get global variables
    global user_action

    ## Update global user action information
    user_action.mutex.acquire(blocking=True)
    user_action.update_vars(cmd_vel.linear, cmd_vel.angular)
    user_action.mutex.release()


####################################
# FUNCTION TO PREDICT FUTURE POSES #
####################################

def predict_pose(linear_x, angular_z, x_pos, y_pos, theta, u, v, predict_interval):
    pred_x = x_pos + u * linear_x * np.cos(theta) * predict_interval
    pred_y = y_pos + u * linear_x * np.sin(theta) * predict_interval
    pred_theta = theta + v * angular_z * predict_interval
    return pred_x, pred_y, pred_theta


#############################################
# CALLBACK FUNCTION TO RENDER FUTURE IMAGES #
#############################################

def rendering_callback(msg):

    with torch.no_grad():

        ## Get global variables
        global frame_data
        global pose_pred_data
        global user_action
        global pred_pose_mat_homo
        global display_queue
        global display_mutex
        global angular_vel_weight

        ## Access most up to date frame info
        frame_data.mutex.acquire(blocking=True)
        frame_data_latest_timestamp = frame_data.timestamp
        frame_data_latest_point_cloud_positions = frame_data.point_cloud_positions
        frame_data_latest_point_cloud_features = frame_data.point_cloud_features
        frame_data_latest_point_cloud = frame_data.point_cloud_complete
        frame_data_latest_ekf_pose = frame_data.ekf_pose    # [x, y, z, q_x, q_y, q_z, q_w] base_link -> odom
        frame_data_latest_torch_image = frame_data.torch_image  # (1, 3, H, W)
        frame_data_latest_initial_delay = frame_data.initial_delay
        frame_data.mutex.release()

        ## Return if data not initialized yet
        if frame_data_latest_timestamp == None:
            return

        ## Check if pose pred data is out of date compared with frame data
        if pose_pred_data.orig_timestamp < frame_data_latest_timestamp:
            # Clear queue of images to publish
            display_mutex.acquire(blocking=True)
            display_queue.clear()
            display_queue.append(latest_cv_image)
            display_mutex.release()
            # Update timestamp the pose prediction is based off of
            pose_pred_data.orig_timestamp = frame_data_latest_timestamp
            latest_pose_time = frame_data_latest_timestamp + DELAY + frame_data_latest_initial_delay
            latest_ekf = frame_data_latest_ekf_pose
            angular_vel_weight = 0.1
        else:
            latest_pose_time = pose_pred_data.curr_timestamp
            latest_ekf = pose_pred_data.curr_pose

        ## Get current time
        curr_time = latest_pose_time + 1./30. # 30Hz into the future

        ## Compute difference between current time and last pose pred time
        time_difference = curr_time - latest_pose_time

        ## Get user action
        user_action.mutex.acquire(blocking=True)
        linear_vel = user_action.linear
        angular_vel = user_action.angular
        user_action.mutex.release()

        ## Prepare to predict pose
        command_lin_x = float(linear_vel[0]) * 2.5
        command_lin_y = float(linear_vel[1]) * 2.5
        command_ang_z = -1 * float(angular_vel[2]) * angular_vel_weight
        angular_vel_weight = min(0.5, angular_vel_weight * 1.1) # update weight on scaling angular velocity by
        gt_x = float(latest_ekf[0])
        gt_y = float(latest_ekf[1])
        gt_z = float(latest_ekf[2])
        quat_x = float(latest_ekf[3])
        quat_y = float(latest_ekf[4])
        quat_z = float(latest_ekf[5])
        quat_w = float(latest_ekf[6])
        r = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        gt_euler = r.as_euler('zxy', degrees=False) # radians
        gt_theta = gt_euler[0]
        gt_pitch = gt_euler[1]
        gt_roll = gt_euler[2]

        ## Convert user action to world frame
        lin_vels = np.array([[command_lin_x],
                            [command_lin_y],
                            [0.0]])
        rot_matrix = r.as_matrix()
        lin_vels_world = rot_matrix @ lin_vels
        command_lin_x_world = lin_vels_world[0][0]
        command_lin_y_world = lin_vels_world[1][0]

        ang_vels = np.array([[0.0],
                            [0.0],
                            [command_ang_z]])
        ang_vels_world = rot_matrix @ ang_vels
        command_ang_z_world = ang_vels_world[2][0]

        ## Calculate how many timesteps to predict for to get as close as possible to current time
        num_pred_steps = int(time_difference / pose_pred_data.prediction_step_size)
        new_time = num_pred_steps * pose_pred_data.prediction_step_size + latest_pose_time

        ## Predict future poses
        pred_x, pred_y, pred_theta = gt_x, gt_y, gt_theta
        for _ in range(1, num_pred_steps + 1):
            pred_x, pred_y, pred_theta = predict_pose(command_lin_x, command_ang_z, pred_x, pred_y, pred_theta, U, V, pose_pred_data.prediction_step_size)

        ## Update pose pred data
        pose_pred_data.curr_timestamp = new_time
        r = R.from_euler('zxy', [pred_theta, gt_pitch, gt_roll], degrees=False)
        pred_quat = r.as_quat() # x,y,z,w
        new_ekf_pose =  [ pred_x,
                        pred_y,
                        gt_z,
                        pred_quat[0],
                        pred_quat[1],
                        pred_quat[2],
                        pred_quat[3]]
        pose_pred_data.curr_pose = new_ekf_pose

        ## Combine pos and rot into homogenous transformation matrix 
        pred_rot_mat = r.as_matrix()
        pred_pose_mat_homo[0,0]  = pred_rot_mat[0,0]    # setting rotation matrix
        pred_pose_mat_homo[0,1]  = pred_rot_mat[0,1]
        pred_pose_mat_homo[0,2]  = pred_rot_mat[0,2]
        pred_pose_mat_homo[1,0]  = pred_rot_mat[1,0]
        pred_pose_mat_homo[1,1]  = pred_rot_mat[1,1]
        pred_pose_mat_homo[1,2]  = pred_rot_mat[1,2]
        pred_pose_mat_homo[2,0]  = pred_rot_mat[2,0]
        pred_pose_mat_homo[2,1]  = pred_rot_mat[2,1]
        pred_pose_mat_homo[2,2]  = pred_rot_mat[2,2]
        pred_pose_mat_homo[0,3]  = pred_x               # setting translation vector
        pred_pose_mat_homo[1,3]  = pred_y
        pred_pose_mat_homo[2,3]  = gt_z                 # base_link -> odom

        ## Publish predicted odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = WORLD_FRAME_ROBOT
        odom_msg.pose.pose.position.x = pred_x
        odom_msg.pose.pose.position.y = pred_y
        odom_msg.pose.pose.position.z = gt_z
        odom_msg.pose.pose.orientation.w = pred_quat[3]
        odom_msg.pose.pose.orientation.x = pred_quat[0]
        odom_msg.pose.pose.orientation.y = pred_quat[1]
        odom_msg.pose.pose.orientation.z = pred_quat[2]
        pub_odom.publish(odom_msg)

        ## Convert to pulsar convention
        pred_pose_odom2cam_mat_homo = base2cam @ torch.linalg.inv(pred_pose_mat_homo)   # odom -> cam
        pred_pose_pulsarworld2pulsarcam_mat_homo = cam2pulsarcam @ pred_pose_odom2cam_mat_homo @ pulsarworld2odom   # pulsar world -> pulsar cam
        rot_mat_pulsar_cam = pred_pose_pulsarworld2pulsarcam_mat_homo[:3,:3].unsqueeze(0)
        trans_mat_pulsar_cam = pred_pose_pulsarworld2pulsarcam_mat_homo[:3,3].unsqueeze(0)
        pulsar_curr_cam = PerspectiveCameras(focal_length=focal_length_torch.clone(), 
                                             principal_point=principal_point_torch.clone(), 
                                             R=torch.transpose(rot_mat_pulsar_cam, 1, 2), 
                                             T=trans_mat_pulsar_cam, 
                                             in_ndc=False, 
                                             image_size=img_size_torch.clone(), 
                                             device=device)

        ## Render
        renderer = PulsarPointsRenderer(
            rasterizer=PointsRasterizer(cameras=pulsar_curr_cam, raster_settings=raster_settings),
            n_channels=4
        ).to(device)

        images = renderer(frame_data_latest_point_cloud, gamma=(1e-5,),
                        bg_col=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=device),
                            znear=[.1], zfar=[100.0]) * 255.  # (1, H, W, 4)
        
        ## Inpaint
        initial_render = images[:,:,:,:3].squeeze(0)    # (H, W, 3)
        green_mask = torch.all(initial_render == green_value, axis=-1)
        rendered_torch_images = images[:,:,:,:3].permute((0,3,1,2)) / 255.
        if inpainting_train_args['norm_input']:
            initial_torch_image = norm_transform(frame_data_latest_torch_image) # (1, 3, H, W)
            rendered_torch_images = norm_transform(rendered_torch_images)       # (1, 3, H, W)
        else:
            initial_torch_image = frame_data_latest_torch_image                 # (1, 3, H, W)

        inpainted_image = inpainting_model(initial_torch_image, rendered_torch_images).squeeze(0).permute((1,2,0))  # (H, W, 3)
        if inpainting_train_args['model_act'] == 'tanh':
            inpainted_image = (inpainted_image + 1.) * (255./2.)  # [0,255]
        else:
            inpainted_image = inpainted_image * 255.

        initial_render[green_mask] = inpainted_image[green_mask]

        ## Convert rendering to cv2 and push to display queue
        images_np = initial_render.cpu().to(torch.uint8).numpy()
        future_render_np = cv2.flip(images_np, 1)
        display_mutex.acquire(blocking=True)
        display_queue.append(future_render_np)
        display_mutex.release()


##############################################
# CALLBACK FUNCTION TO PUBLISH FUTURE IMAGES #
##############################################

def publishing_callback(msg):

    # Global variables
    global display_queue
    global display_mutex
    queue_valid = False

    # Pop most recent image
    while True:
        display_mutex.acquire(blocking=True)
        if len(display_queue):
            curr_render_np = display_queue.popleft()    # (H, W, 3)
            queue_valid = True
        display_mutex.release()
        if queue_valid:
            break

    # Publish
    if queue_valid:
        curr_render_msg = bridge.cv2_to_imgmsg(curr_render_np, encoding="passthrough")
        pub_img.publish(curr_render_msg)


if __name__ == '__main__':

    print('INITIALIZING NODE')
    rospy.init_node('field_predictor')

    # TOPICS
    RGB_TOPIC = '/terrasentia/zed2/zed_node/left/image_rect_color/compressed'
    EKF_TOPIC = '/terrasentia/zed2/zed_node/odom'
    ACTION_TOPIC = '/terrasentia/motion_command'

    # FRAMES
    CAM_FRAME_TORCH3D = 'pytorch3d_cam'
    CAM_FRAME_PULSAR = 'pulsar_cam'
    CAM_FRAME_ROBOT = 'zed2_left_camera_optical_frame'
    BASE_LINK_ROBOT = 'base_link'
    WORLD_FRAME_TORCH3D = 'pytorch3d_world'
    WORLD_FRAME_PULSAR = 'pulsar_world'
    WORLD_FRAME_ROBOT = 'vio/odom'

    image_sub = message_filters.Subscriber(RGB_TOPIC, CompressedImage)
    ekf_sub = message_filters.Subscriber(EKF_TOPIC, Odometry)
    cmd_vel_sub = rospy.Subscriber(ACTION_TOPIC, Twist, cmd_vel_callback)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, ekf_sub], 1, 0.01)
    ts.registerCallback(callback)
    listener = tf.TransformListener()

    pub_img = rospy.Publisher('/pred/out_img', Image, queue_size=10)
    pub_orig_img = rospy.Publisher('/pred/orig_img', Image, queue_size=10)
    pub_depth = rospy.Publisher('/pred/depth', Image, queue_size=10)
    pub_odom = rospy.Publisher('/pred/odom', Odometry, queue_size=10)

    rospy.Timer(rospy.Duration(1./100.), rendering_callback)
    rospy.Timer(rospy.Duration(1./30.), publishing_callback)
    rospy.spin()
