import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class Terra(Dataset):

    def __init__(self, data_folder_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        self.data_folder_path = data_folder_path
        
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
        else:
            raise Exception

        self.data = []
        for k,v in self.video_times_dict.items():
            for t in range(v[0], v[1]+1):
                self.data.append([k, t])

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):

        video_combo = self.data[item]
        video_name = video_combo[0]
        time = video_combo[1]

        img_path = os.path.join(self.data_folder_path, video_name, 'seq00', 'rgb', '{}.png'.format(str(time).zfill(6)))
        depth_path = os.path.join(self.data_folder_path, video_name, 'seq00', 'depth', '{}.npy'.format(str(time).zfill(6)))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        depth = np.load(depth_path)

        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = (torch.logical_and(sample['depth'] < 20., sample['depth'] >= 0.))
        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.data)
