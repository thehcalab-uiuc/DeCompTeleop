# Standard imports
import os

# Third party imports
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Terra_DAv2_Projected_Inpainting(Dataset):

    def __init__(self, mode, gt_data_path, proj_data_path, offsets, norm_input):
        
        # Init variables
        self.mode = mode
        self.offsets = offsets
        self.gt_data_path = gt_data_path
        self.proj_data_path = proj_data_path
        self.norm_input = norm_input
        
        # Transform PIL image torch tensor float32
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.norm_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Split timesteps
        if self.mode == 'train':
            self.video_times_dict = {
                'early-cornfield1-20220609_cornfield-1654799444.1743867_three_rows'                     : [118, 918],
                'early-cornfield3-20220711_cornfield3-ts_2022_07_11_12h23m16s_field3_three_rows'        : [41 , 861],
                'late-cornfield1-20220815_cornfield-ts_2022_08_15_11h20m26s_two_random'                 : [103, 925],
                'late-cornfield2-20220906_cornfield2-ts_2022_09_06_13h07m47s_one_random_cf2'            : [18 , 830],
                'late-cornfield4-20220908_cornfield4-ts_2022_09_08_10h16m08s_one_random_cf4'            : [87 , 939],
                'late-sweet_corn-20220721_sweet_corn-ts_2022_07_21_09h36m36s_two_rows'                  : [53 , 877],
                'middle-cornfield2-20220718_cornfield2-ts_2022_07_18_11h46m31s_three_rows'              : [108, 917],
                'middle-cornfield4-20220810_cornfield4-ts_2022_08_10_11h05m57s_double_loop'             : [48 , 858],
                'middle-cornfield_2023-20230615-ts_2023_06_15_21h07m23s_one_random_row'                 : [3  , 827]
            }
        elif self.mode == 'val':
            self.video_times_dict = {
                'early-cornfield1-20220609_cornfield-1654799444.1743867_three_rows'                     : [919, 1117],
                'early-cornfield3-20220711_cornfield3-ts_2022_07_11_12h23m16s_field3_three_rows'        : [862, 1067],
                'late-cornfield1-20220815_cornfield-ts_2022_08_15_11h20m26s_two_random'                 : [926, 1131],
                'late-cornfield2-20220906_cornfield2-ts_2022_09_06_13h07m47s_one_random_cf2'            : [831, 1033],
                'late-cornfield4-20220908_cornfield4-ts_2022_09_08_10h16m08s_one_random_cf4'            : [940, 1152],
                'late-sweet_corn-20220721_sweet_corn-ts_2022_07_21_09h36m36s_two_rows'                  : [878, 1084],
                'middle-cornfield2-20220718_cornfield2-ts_2022_07_18_11h46m31s_three_rows'              : [918, 1120],
                'middle-cornfield4-20220810_cornfield4-ts_2022_08_10_11h05m57s_double_loop'             : [859, 1061],
                'middle-cornfield_2023-20230615-ts_2023_06_15_21h07m23s_one_random_row'                 : [828, 1034]
            }
        elif self.mode == 'test':
            self.video_times_dict = {
                'middle-cornfield_2023-20230804-ts_2023_08_04_12h41m19s_four_rows'                      : [67 , 1067],
                'recollected-early-cornfield3-20220714_cornfield3-ts_2022_07_14_12h01m08s_three_rows'   : [77 , 1081],
                'late-cornfield2-20220815_cornfield2-ts_2022_08_15_11h33m57s_one_random'                : [80 , 1173]
            }
        elif self.mode == 'debug':
            self.video_times_dict = {
                'early-cornfield3-20220711_cornfield3-ts_2022_07_11_12h23m16s_field3_three_rows'    : [41 , 861]
            }
        else:
            raise Exception

        # Storing tuples of video name image timesteps
        self.data = []

        # Add paths and odometry to saving variables
        for k,v in self.video_times_dict.items():
            for t in range(v[0], v[1]+1):
                for o in self.offsets:
                    if t - o >= 0:
                        self.data.append([k, o, t-o, t]) # (video_name, offset, conditioning_timestep, future_timestep)

        # Cache to store images as they are loaded
        self.cache = {'gt': {}, 'proj': {}}

    def __getitem__(self, item):

        # Get data
        video_name, offset, start_time, end_time = self.data[item]

        # Get image paths
        gt_vid_folder = os.path.join(self.gt_data_path, video_name, 'seq00', 'rgb')
        gt_orig_image_path = os.path.join(gt_vid_folder, '{}.png'.format(str(start_time).zfill(6)))
        gt_future_image_path = os.path.join(gt_vid_folder, '{}.png'.format(str(end_time).zfill(6)))
        proj_image_path = os.path.join(self.proj_data_path, video_name, str(offset), '{}.png'.format(str(end_time).zfill(6)))

        # Load images (use cache if possible)
        if gt_orig_image_path in self.cache['gt']:
            gt_orig_image = self.cache['gt'][gt_orig_image_path]
        else:
            gt_orig_image = self.image_transform(Image.open(gt_orig_image_path).convert("RGB"))
            self.cache['gt'][gt_orig_image_path] = gt_orig_image
        if self.norm_input:
            gt_orig_image = self.norm_transform(gt_orig_image)

        if gt_future_image_path in self.cache['gt']:
            gt_future_image = self.cache['gt'][gt_future_image_path]
        else:
            gt_future_image = self.image_transform(Image.open(gt_future_image_path).convert("RGB"))
            self.cache['gt'][gt_future_image_path] = gt_future_image

        if proj_image_path in self.cache['proj']:
            proj_image = self.cache['proj'][proj_image_path]
        else:
            proj_image = self.image_transform(Image.open(proj_image_path).convert("RGB"))
            self.cache['proj'][proj_image_path] = proj_image
        if self.norm_input:
            proj_image = self.norm_transform(proj_image)

        return gt_orig_image, gt_future_image, proj_image

    def __len__(self):
        return len(self.data)
