# PyTorch imports
import torch
import torch.hub
import torch.nn as nn
from torchvision import transforms

# PyTorch3D Imports
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizationSettings
from pytorch3d.renderer.points.pulsar.renderer import Renderer as PulsarDirectRenderer
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import matrix_to_rotation_6d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PulsarDirectGeneralDepthRenderer(nn.Module):

    def __init__(self, img_size=(720,1280), focal_length=529., principal=(631.0499,348.0125), 
                 batch_size=1):

        super().__init__()

        # Image transforms
        self.transform_resize_up = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1]),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
        ])

        # PyTorch3D camera parameters
        focal_length_torch = torch.tensor([[focal_length]], dtype=torch.float32).to(device)                     # (1, 1)
        self.focal_length_torch = torch.tile(focal_length_torch, (batch_size,1))                                # (batch, 1)
        principal_point_torch = torch.tensor([[principal[0], principal[1]]], dtype=torch.float32).to(device)    # (1, 2)
        self.principal_point_torch = torch.tile(principal_point_torch, (batch_size,1))                          # (batch, 2)
        img_size_torch = torch.tensor([[img_size[0], img_size[1]]], dtype=torch.float32).to(device)             # (1, 2)
        self.img_size_torch = torch.tile(img_size_torch, (batch_size,1))                                        # (batch, 2)

        # Pulsar camera intrinsics
        self.renderer_height = img_size[0]
        self.renderer_width = img_size[1]
        self.znear = torch.tensor([0.1], dtype=torch.float32, device=device).reshape((1,))
        self.zfar = torch.tensor([100.], dtype=torch.float32, device=device).reshape((1,))
        focal_length_px = focal_length
        focal_length_px /= self.renderer_width / 2.0
        self.focal_length = torch.tensor([self.znear - 1e-6],
                                          dtype=torch.float32,
                                          device=device,
                                        ).reshape((1,))
        self.sensor_width = self.focal_length / focal_length_px * 2.0
        principal_point_x = principal[0]
        self.principal_point_x = torch.tensor([self.renderer_width / 2.0 - principal_point_x],
                                          dtype=torch.float32,
                                          device=device,
                                        ).reshape((1,))
        principal_point_y = principal[1]
        self.principal_point_y = torch.tensor([principal_point_y - self.renderer_height / 2.0],
                                          dtype=torch.float32,
                                          device=device,
                                        ).reshape((1,))
        
        # Rendering settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(img_size[0],img_size[1]),
            radius = 0.003,
            points_per_pixel = 1,
            bin_size = 0
        )
        self.gamma = 1e-1
        self.bg_color = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

        # Define pulsar renderer
        self.renderer = PulsarDirectRenderer(
            width=img_size[1],
            height=img_size[0],
            max_num_balls=int(1e6),
            orthogonal_projection=False,
            right_handed_system=True,
            n_channels=3
        )

    def forward(self, orig_img, orig_depth, orig_pytorch3d_pose, future_pulsar_poses):

        # Input shapes
        # print(orig_img.shape)               # (batch, 3, H, W)
        # print(orig_depth.shape)             # (batch, 1, H, W)
        # print(orig_pytorch3d_pose.shape)    # (batch, 4, 4)
        # print(future_pulsar_poses.shape)    # (batch, seq_len, 4, 4)

        with torch.no_grad():

            # Create current PyTorch3D camera
            rot_mat_torch3d_cam = orig_pytorch3d_pose[:,:3,:3]  # (batch, 3, 3)
            trans_mat_torch3d_cam = orig_pytorch3d_pose[:,:3,3] # (batch, 3)
            torch3d_curr_cam = PerspectiveCameras(focal_length=self.focal_length_torch.clone(),
                                                principal_point=self.principal_point_torch.clone(),
                                                R=torch.transpose(rot_mat_torch3d_cam, 1, 2),
                                                T=trans_mat_torch3d_cam,
                                                in_ndc=False,
                                                image_size=self.img_size_torch.clone(),
                                                device=device)

            # Create batch of point clouds
            cloud_rgb_in = orig_img         # (batch, 3, H, W)
            cloud_depth_in = orig_depth     # (batch, 1, H, W)
            point_clouds_points_list = []
            point_clouds_features_list = []

            for batch_i, (batch_cloud_rgb_in, batch_cloud_depth_in) in enumerate(zip(cloud_rgb_in, cloud_depth_in)):

                point_cloud_torch = get_rgbd_point_cloud(camera=torch3d_curr_cam.__getitem__(batch_i), 
                                                        image_rgb=batch_cloud_rgb_in.unsqueeze(0), 
                                                        depth_map=batch_cloud_depth_in.unsqueeze(0))
                pc_points = point_cloud_torch.points_packed()                               # (H*W, 3)
                pc_points[:,0] *= -1.
                pc_points[:,2] *= -1.
                pc_colors = point_cloud_torch.features_packed().type(torch.float32) / 255.  # (H*W, 3)
                point_clouds_points_list.append(pc_points)
                point_clouds_features_list.append(pc_colors)

            pc_combined_alpha = Pointclouds(points=point_clouds_points_list, features=point_clouds_features_list)
            position_list = pc_combined_alpha.points_list()
            features_list = pc_combined_alpha.features_list()

            # Convert future poses into pulsar convention
            complete_future_pulsar_rot = future_pulsar_poses[:,:,:3,:3]                             # (batch, seq_len, 3, 3)
            complete_future_pulsar_pos = future_pulsar_poses[:,:,:3,3]                              # (batch, seq_len, 3)
            batch_size = complete_future_pulsar_rot.shape[0]
            seq_len = complete_future_pulsar_rot.shape[1]
            batch_future_pulsar_rot = complete_future_pulsar_rot.reshape((-1, 3, 3))                # (batch*seq_len, 3, 3)
            batch_future_pulsar_pos = complete_future_pulsar_pos.reshape((-1, 3, 1))                # (batch*seq_len, 3, 1)
            batch_future_pulsar_rot = torch.transpose(batch_future_pulsar_rot, 1, 2)                ## 1. Transpose rotation matrix following PyTorch3D convention
            batch_future_pulsar_rot[:, :, :2] *= -1                                                 ## 2. Convert to OpenCV convention
            batch_future_pulsar_rot = batch_future_pulsar_rot.permute(0, 2, 1)
            batch_future_pulsar_pos[:, :2] *= -1
            R_trans = batch_future_pulsar_rot.permute(0, 2, 1)                                      ## 3. Convert to Pulsar convention
            batch_future_pulsar_pos = -torch.bmm(R_trans, batch_future_pulsar_pos).squeeze(2)
            batch_future_pulsar_rot = matrix_to_rotation_6d(R_trans)
            batch_cam_params = torch.cat([batch_future_pulsar_pos, batch_future_pulsar_rot], dim=1) ## 4. Combine extrinsics    (batch*seq_len, 9) - 0:3=pos, 3:9=rot
            batch_future_pulsar_pos = batch_cam_params[:,:3]                                        ## 5. Split up dimensions
            batch_future_pulsar_rot = batch_cam_params[:,3:9]
            future_pulsar_pos = batch_future_pulsar_pos.reshape(((batch_size, seq_len, 3)))
            future_pulsar_rot = batch_future_pulsar_rot.reshape(((batch_size, seq_len, 6)))

            # Render batch of images
            batch_images = []
            for cloud_idx, (vert_pos, vert_col) in enumerate(zip(position_list, features_list)):
                
                # Render each image in current index of batch
                future_images_list = []
                for t in range(seq_len):

                    # Get GT future pose in pulsar convention
                    cam_pos = future_pulsar_pos[cloud_idx, t]   # (3)
                    cam_rot = future_pulsar_rot[cloud_idx, t]   # (6)

                    # Setup pulsar cam params
                    cam_params = torch.cat(
                        (
                            cam_pos,
                            cam_rot,
                            torch.cat(
                                [
                                    self.focal_length.clone(),
                                    self.sensor_width.clone(),
                                    self.principal_point_x.clone(),
                                    self.principal_point_y.clone(),
                                ],
                            ),
                        )
                    )

                    # Get radii of points
                    point_dists = torch.norm((vert_pos - cam_pos), p=2, dim=1, keepdim=False)
                    vert_rad = self.raster_settings.radius / self.focal_length.clone() * point_dists * self.sensor_width.clone()
                    vert_rad = vert_rad / 2.0  # NDC normalization.

                    # Render image with Pulsar
                    future_images_list.append(
                        self.renderer(
                            vert_pos=vert_pos,
                            vert_col=vert_col,
                            vert_rad=vert_rad,
                            cam_params=cam_params,
                            gamma=self.gamma,
                            max_depth=self.zfar,
                            min_depth=self.znear,
                            bg_col=self.bg_color,
                        ).flip(dims=[0]).unsqueeze(0) * 255.
                    )                                   # (1, H, W, 3)

                batch_images.append(torch.cat(future_images_list, dim=0).unsqueeze(0))  # (1, seq_len, H, W, 3)

            pre_flip_rendered_images = torch.cat(batch_images, dim=0)   # (batch, seq_len, H, W, 3)

            # Horizontally flip images following Pulsar convention
            future_images_torch_trans = pre_flip_rendered_images.permute((0,1,4,2,3)).reshape((-1,3,self.renderer_height,self.renderer_width))
            future_images_torch_trans = transforms.functional.hflip(future_images_torch_trans)  # (batch*seq_len, 3, H, W)

        # Return projected images
        out = future_images_torch_trans.reshape((-1,seq_len,3,self.renderer_height,self.renderer_width))   # (batch, seq_len, 3, H, W)
        return out
