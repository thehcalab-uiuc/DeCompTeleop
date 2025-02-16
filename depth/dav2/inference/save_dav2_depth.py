# Local imports
from depth_anything_v2_metric.dpt import TorchDepthAnythingV2

# Standard imports
import os
import argparse
import tqdm

# Third party imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd

# PyTorch imports
import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthEstimator:

    def __init__(self, args) -> None:

        model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 
                        'input_width': 1280, 'input_height': 720, 'device': device, 'input_size': 518}
        self.depth_anything = TorchDepthAnythingV2(**{**model_config, 'max_depth': 20.})

        ###########################################################
        # UNCOMMENT ONE OF THE FOLLOWING CHECKPOINT LOADING LINES #
        ###########################################################

        # 1) Load finetuned model
        self.depth_anything.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
                        '/path/to/depth_anything_v2_metric_terra_vits.pth', 
                        map_location='cpu')['model'].items()})

        # # 2) Load pretrained model
        # self.depth_anything.load_state_dict(torch.load(
        #                 '/path/to/depth_anything_v2_metric_vkitti_vits.pth', 
        #                 map_location='cpu'))
        
        ###########################################################

        self.depth_anything = self.depth_anything.to(device).eval()

        # Check load folder and create save folder
        self.load_folder = os.path.join(args.data, args.vid_name, 'seq00')
        print('predicting for videos from: ', self.load_folder)
        if not os.path.exists(self.load_folder):
            raise Exception
        self.save_folder = os.path.join(args.save, args.vid_name)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Transform PIL image to torch tensor float32 of range 0-255
        self.transform_to_img_tensor = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        ])


    def estimate(self):

        print('Start Estimating Depths ...')

        with torch.no_grad():

            # Start of epoch
            self.depth_anything.eval()

            # Path to video frames
            load_images_folder = os.path.join(self.load_folder, 'rgb')

            # Count number of frames in video
            file_count = 0
            for p in os.listdir(load_images_folder):
                if os.path.isfile(os.path.join(load_images_folder, p)):
                    file_count += 1

            # Iterate over images to predict from
            for im_num in tqdm.tqdm(range(file_count)):

                # Load input image
                im = Image.open(os.path.join(load_images_folder, '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")
                im = self.transform_to_img_tensor(im)
                im_torch = torch.stack([im], 0).to(device).type(torch.float32)  # (1, 3, 720, 1280)

                # Predict
                input_image = im_torch.squeeze(0).permute(1, 2, 0)
                input_image = input_image / 255.0 
                output_depth = self.depth_anything.infer_image(input_image)     # (720, 1280)

                # Save
                output_depth_arr = output_depth.detach().cpu().numpy()
                np.save(os.path.join(self.save_folder, '{}.npy'.format(str(im_num).zfill(6))), output_depth_arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--data', type=str, help='path to dataset',
                        default='/path/to/dataset')
    parser.add_argument('--vid_name', type=str, help='name of video to predict for', required=True)
    parser.add_argument('--save', type=str, help='path to save folder',
                        default='/path/to/save/folder')

    args = parser.parse_args()

    depth_estimator = DepthEstimator(args)

    with torch.no_grad():
        depth_estimator.estimate()
