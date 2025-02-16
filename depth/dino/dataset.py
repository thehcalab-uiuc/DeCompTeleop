import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


"""
    FilledDepthDataset:
        - Depth maps collected from ZED using 'Neural Fill' mode
        - Relative depth collected using DAV2
        - Using relative depth and constraints to construct a mask for pixels with valid depth values
"""

class FilledDepthDataset(Dataset):
    def __init__(self, device, data_type, data_dir='/path/to/dataset',
                 rel_depth_dir='/path/to/relative/depth/labels'):
        self.train_frames_dict = {
            'early-cornfield1-20220609_cornfield-1654799444.1743867_three_rows': [int('000118'), int('000918')],
            'early-cornfield3-20220711_cornfield3-ts_2022_07_11_12h23m16s_field3_three_rows': [int('000041'), int('000861')],
            'late-cornfield1-20220815_cornfield-ts_2022_08_15_11h20m26s_two_random': [int('000103'), int('000925')],
            'late-cornfield2-20220906_cornfield2-ts_2022_09_06_13h07m47s_one_random_cf2': [int('000018'), int('000830')],
            'late-cornfield4-20220908_cornfield4-ts_2022_09_08_10h16m08s_one_random_cf4': [int('000087'), int('000939')],
            'late-sweet_corn-20220721_sweet_corn-ts_2022_07_21_09h36m36s_two_rows': [int('000053'), int('000877')],
            'middle-cornfield2-20220718_cornfield2-ts_2022_07_18_11h46m31s_three_rows': [int('000108'), int('000917')],
            'middle-cornfield4-20220810_cornfield4-ts_2022_08_10_11h05m57s_double_loop': [int('000048'), int('000858')],
            'middle-cornfield_2023-20230615-ts_2023_06_15_21h07m23s_one_random_row': [int('000003'), int('000827')],
        }

        self.val_frames_dict = {
            'early-cornfield1-20220609_cornfield-1654799444.1743867_three_rows': [int('000919'), int('001117')],
            'early-cornfield3-20220711_cornfield3-ts_2022_07_11_12h23m16s_field3_three_rows': [int('000862'), int('001067')],
            'late-cornfield1-20220815_cornfield-ts_2022_08_15_11h20m26s_two_random': [int('000926'), int('001131')],
            'late-cornfield2-20220906_cornfield2-ts_2022_09_06_13h07m47s_one_random_cf2': [int('000831'), int('001033')],
            'late-cornfield4-20220908_cornfield4-ts_2022_09_08_10h16m08s_one_random_cf4': [int('000940'), int('001152')],
            'late-sweet_corn-20220721_sweet_corn-ts_2022_07_21_09h36m36s_two_rows': [int('000878'), int('001084')],
            'middle-cornfield2-20220718_cornfield2-ts_2022_07_18_11h46m31s_three_rows': [int('000918'), int('001120')],
            'middle-cornfield4-20220810_cornfield4-ts_2022_08_10_11h05m57s_double_loop': [int('000859'), int('001061')],
            'middle-cornfield_2023-20230615-ts_2023_06_15_21h07m23s_one_random_row': [int('000828'), int('001034')],
        }

        self.test_frames_dict = {
            'middle-cornfield_2023-20230804-ts_2023_08_04_12h41m19s_four_rows': [int('000067'), int('001067')],
            'recollected-early-cornfield3-20220714_cornfield3-ts_2022_07_14_12h01m08s_three_rows': [int('000077'), int('001081')],
            'late-cornfield2-20220815_cornfield2-ts_2022_08_15_11h33m57s_one_random': [int('000080'), int('001173')]
        }
        
        # RGB image transformation before feeding into model (from DINOv2)
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

        self.data_dir = data_dir
        self.rel_depth_dir = rel_depth_dir
        self.device = device
        self.data = []
        self.get_data_paths(data_type)
    
    def get_data_paths(self, data_type):
        frames_dict = None
        if data_type == 'train':
            frames_dict = self.train_frames_dict
        elif data_type == 'val':
            frames_dict = self.val_frames_dict
        else:
            frames_dict = self.test_frames_dict
        
        for video_name in frames_dict.keys():
            start_frame = frames_dict[video_name][0]
            end_frame = frames_dict[video_name][1]

            frame_count = 0
            while frame_count <= end_frame - start_frame:
                rgb_frame_name = f'{(start_frame + frame_count):06}.png'
                rgb_frame_path = os.path.join(self.data_dir, video_name, 'seq00', 'rgb', rgb_frame_name)
                depth_frame_name = f'{(start_frame + frame_count):06}.npy'
                depth_frame_path = os.path.join(self.data_dir, video_name, 'seq00', 'depth', depth_frame_name)
                rel_depth_frame_name = f'{(start_frame + frame_count):06}.npy'
                rel_depth_frame_path = os.path.join(self.rel_depth_dir, video_name, rel_depth_frame_name)
                self.data.append((rgb_frame_path, depth_frame_path, rel_depth_frame_path))
                frame_count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load RGB, depth, and relative depth
        rgb_path = self.data[idx][0]
        depth_path = self.data[idx][1]
        rel_depth_path = self.data[idx][2]
        rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
        depth_image = np.load(depth_path)
        rel_depth_image = np.load(rel_depth_path)

        rgb_image_tensor = self.transform_rgb(rgb_image)
        depth_image_tensor = torch.tensor(depth_image)
        rel_depth_image_tensor = torch.tensor(rel_depth_image)
        rgb_image_tensor = rgb_image_tensor.to(self.device)
        depth_image_tensor = depth_image_tensor.to(self.device)
        rel_depth_image_tensor = rel_depth_image_tensor.to(self.device)

        # Mask for valid pixels
        valid_mask = torch.logical_and(torch.logical_and(torch.logical_and(torch.isfinite(depth_image_tensor), depth_image_tensor > 0.0), 
                                                         depth_image_tensor <= 15.0), rel_depth_image_tensor != 0.0)
        valid_mask = valid_mask.to(self.device)

        return rgb_image_tensor, depth_image_tensor, valid_mask
