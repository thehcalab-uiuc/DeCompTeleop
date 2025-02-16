# Local imports
from model import PulsarDirectGeneralDepthRenderer

# Standard imports
import os
import argparse
import tqdm

# Third party imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib

# PyTorch imports
import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Renderer:

    def __init__(self, args) -> None:

        # Hyperparameters
        self.img_size = (args.height, args.width)
        self.times = args.time

        self.model = PulsarDirectGeneralDepthRenderer(
                                  # Camera params
                                  img_size=(args.height,args.width), 
                                  focal_length=args.focal,
                                  principal=(args.principal_x, args.principal_y),
                                  batch_size=1,
                                  ).to(device)
        
        # Check load folders
        self.load_folder = os.path.join(args.gt_data, args.vid_name, 'seq00')
        print('rendering projections from: ', self.load_folder)
        assert os.path.exists(self.load_folder)
        self.img_folder = os.path.join(self.load_folder, 'rgb')
        self.use_zed = True if args.zed_depth else False
        if self.use_zed:
            self.depth_folder = os.path.join(self.load_folder, 'depth')
        else:
            self.depth_folder = os.path.join(args.depth_data, args.vid_name)
        assert os.path.exists(self.depth_folder)

        # Create save folder
        self.save_folder = os.path.join(args.save_folder, args.model_name, args.vid_name)
        for t_offset in self.times:
            offset_folder = os.path.join(self.save_folder, str(t_offset))
            if not os.path.exists(offset_folder):
                os.makedirs(offset_folder)

        # Transform PIL image to torch tensor float32 of range 0-255
        self.transform_to_img_tensor = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        ])

        # Load odometry
        self.pytorch3d_data = pd.read_csv(os.path.join(self.load_folder, 'PyTorch3DTransformationMatrices.csv'), sep=',', header=None).to_numpy()
        self.pulsar_data = pd.read_csv(os.path.join(self.load_folder, 'Pulsar3DTransformationMatrices.csv'), sep=',', header=None).to_numpy()


    def render(self):

        print('Start Rendering ...')

        # Start of epoch
        self.model.eval()

        # Count number of frames in video
        file_count = 0
        for p in os.listdir(self.img_folder):
            if os.path.isfile(os.path.join(self.img_folder, p)):
                file_count += 1

        # Iterate over images to render from
        for im_num in tqdm.tqdm(range(file_count)):

            # Load conditioning image
            im = Image.open(os.path.join(self.img_folder, '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")
            im = self.transform_to_img_tensor(im).unsqueeze(0).to(device).type(torch.float32)   # (1, 3, 720, 1280)
            im_torch = self.model.transform_resize_up(im)                                       # (1, 3, H, W)

            # Get initial depth image
            init_depth_np = np.load(os.path.join(self.depth_folder, '{}.npy'.format(str(im_num).zfill(6))))
            init_depth_torch = torch.from_numpy(init_depth_np).unsqueeze(0).unsqueeze(0)    # (1, 1, 720, 1280)
            init_depth_torch = self.model.transform_resize_up(init_depth_torch).to(device)  # (1, 1, H, W)

            # Load initial odom transform (pytorch3d convention)
            pytorch3d_transform = self.pytorch3d_data[im_num]
            pytorch3d_transform = pytorch3d_transform.reshape((-1,4,4))
            pytorch3d_transform_torch = torch.from_numpy(pytorch3d_transform).to(device).type(torch.float32)  # (1, 4, 4)

            # Load future odom transform (pulsar convention)
            pulsar_transforms = []
            for t_offset in self.times:
                # Break if invalid future pose timestep
                if im_num + t_offset >= file_count:
                    break
                curr_pulsar_transform = self.pulsar_data[im_num+t_offset]
                curr_pulsar_transform = curr_pulsar_transform.reshape((-1,4,4))
                curr_pulsar_transform_torch = torch.from_numpy(curr_pulsar_transform).to(device).type(torch.float32)    # (1, 4, 4)
                pulsar_transforms.append(curr_pulsar_transform_torch)

            # Skip to next image if no future poses exist
            if not pulsar_transforms:
                continue

            # Combine future poses
            pulsar_transforms_torch = torch.cat(pulsar_transforms, dim=0)   # (seq_len, 4, 4)
            pulsar_transforms_torch = pulsar_transforms_torch.unsqueeze(0)  # (1, seq_len, 4, 4)

            # Predict future image
            future_images_torch = self.model(im_torch, init_depth_torch, pytorch3d_transform_torch, pulsar_transforms_torch)    # (1, seq_len, 3, H, W)

            # Save future images
            future_images_torch = future_images_torch[0].permute((0,2,3,1)) # (seq_len, H, W, 3)
            for i, fut_im in enumerate(future_images_torch):
                future_image_arr = fut_im.detach().cpu().numpy()            # (H, W, 3)
                cv2.imwrite(os.path.join(self.save_folder, str(self.times[i]), '{}.png'.format(str(im_num+self.times[i]).zfill(6))), cv2.cvtColor(future_image_arr, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--gt_data', type=str, help='path to original dataset',
                        default='/path/to/dataset')
    parser.add_argument('--zed_depth', action='store_true', help='use gt zed depth to render')
    parser.add_argument('--depth_data', type=str, help='path to predicted depth dataset (used when zed_depth=False)',
                        default='/path/to/predicted/depth')
    parser.add_argument('--vid_name', type=str, help='name of video to render', required=True)
    parser.add_argument('--save_folder', type=str, help='path to save folder',
                        default='/path/to/save/folder')
    parser.add_argument('--model_name', type=str, help='sub-folder to save to under save_folder',
                        default='dav2')

    # Camera parameters
    parser.add_argument('--focal', default=527.087, type=float, help='camera focal length')
    parser.add_argument('--height', default=720, type=int, help='image height')
    parser.add_argument('--width', default=1280, type=int, help='image width')
    parser.add_argument('--principal_x', default=638.526, type=float, help='camera principal point x')
    parser.add_argument('--principal_y', default=341.214, type=float, help='camera principal point y')

    # Timesteps to render
    parser.add_argument('--time', type=int, nargs='+', default=[1,3,5,7,10,15])

    args = parser.parse_args()

    renderer = Renderer(args)

    with torch.no_grad():
        renderer.render()
