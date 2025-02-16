# Standard imports
import os
import argparse
import tqdm
import csv
from PIL import Image

# Third party imports
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
from torchvision.transforms import ToTensor
import lpips


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Collector:

    def __init__(self, args) -> None:

        # Check load folders and create save csv
        self.gt_load_folder = os.path.join(args.gt_data, args.vid_name, 'seq00')
        print('collecting metrics for: ', self.gt_load_folder)
        if not os.path.exists(self.gt_load_folder):
            raise Exception
        self.pred_load_folder = os.path.join(args.pred_data, args.model_name, args.vid_name, str(args.time_delay))
        if not os.path.exists(self.pred_load_folder):
            raise Exception
        self.save_csv_path = os.path.join(args.pred_data, args.model_name, args.vid_name, '{}_metrics.csv'.format(str(args.time_delay)))
        self.start_time = args.start_time
        self.end_time = args.end_time

        with open(self.save_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'psnr', 'ssim', 'ms_ssim', 'lpips'])

        self.transform = ToTensor()


    def calculate_psnr(self, pred_frame, gt_frame, data_range=1.0):
        # pred_frame: (3, img_w, img_h), range [0, 1]
        # gt_frame: (3, img_w, img_h), range [0, 1]
        mse = torch.mean((pred_frame - gt_frame) ** 2)
        if mse == 0:
            return float('inf')
        psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr_value


    def calculate_ssim(self, pred_frame, gt_frame):
        """
        Get both SSIM and MS-SSIM
        """
        # pred_frame: (3, img_w, img_h), range [0, 1]
        # gt_frame: (3, img_w, img_h), range [0, 1]

        # Add batch dimension
        pred_frame = pred_frame.unsqueeze(0)  # (1, 3, img_w, img_h)
        gt_frame = gt_frame.unsqueeze(0)  # (1, 3, img_w, img_h)

        ssim_value = ssim(pred_frame, gt_frame, data_range=1.0, size_average=True)
        ms_ssim_value = ms_ssim(pred_frame, gt_frame, data_range=1.0, size_average=True)
        return ssim_value, ms_ssim_value


    def calculate_lpips(self, loss_fn, pred_frame, gt_frame):
        # pred_frame: (3, img_w, img_h), range [0, 1]
        # gt_frame: (3, img_w, img_h), range[0, 1]

        # Clip in range [-1, 1]
        pred_frame = (pred_frame * 2) - 1  
        gt_frame = (gt_frame * 2) - 1

        # Add batch dimension
        pred_frame = pred_frame.unsqueeze(0)  # (1, 3, img_w, img_h)
        gt_frame = gt_frame.unsqueeze(0)  # (1, 3, img_w, img_h)

        lpips_value = loss_fn.forward(pred_frame, gt_frame)
        return lpips_value


    def collect(self):

        print('Start Collecting Metrics ...')

        # Path to video frames
        load_images_folder = os.path.join(self.gt_load_folder, 'rgb')

        # Count number of frames in video
        if self.end_time == -1:
            file_count = 0
            for p in os.listdir(load_images_folder):
                if os.path.isfile(os.path.join(load_images_folder, p)):
                    file_count += 1
            self.end_time = file_count - 1

        # Initialize storage and LPIPS loss function
        psnr_list, ssim_list, ms_ssim_list, lpips_list = [], [], [], []
        loss_fn = lpips.LPIPS(net='alex').to(device)

        # Iterate over images
        for im_num in tqdm.tqdm(range(self.start_time, self.end_time+1)):

            # Load gt and projected images
            gt_im = Image.open(os.path.join(load_images_folder, '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")
            pred_im = Image.open(os.path.join(self.pred_load_folder, 'inpainted', '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")

            # Convert to torch tensor [0,1]
            gt_im_torch = self.transform(gt_im).to(device)      # (3, 720, 1280)
            pred_im_torch = self.transform(pred_im).to(device)  # (3, 720, 1280)

            # Get metrics
            curr_psnr = self.calculate_psnr(pred_im_torch, gt_im_torch)
            curr_ssim, curr_ms_ssim = self.calculate_ssim(pred_im_torch, gt_im_torch)
            curr_lpips = self.calculate_lpips(loss_fn, pred_im_torch, gt_im_torch)

            # Save metrics
            curr_psnr = curr_psnr.item()
            curr_ssim = curr_ssim.item()
            curr_ms_ssim = curr_ms_ssim.item()
            curr_lpips = curr_lpips.item()
            psnr_list.append(curr_psnr)
            ssim_list.append(curr_ssim)
            ms_ssim_list.append(curr_ms_ssim)
            lpips_list.append(curr_lpips)
            with open(self.save_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([im_num, curr_psnr, curr_ssim, curr_ms_ssim, curr_lpips])

        print('mean psnr: ', np.mean(psnr_list))
        print('mean ssim: ', np.mean(ssim_list))
        print('mean ms-ssim: ', np.mean(ms_ssim_list))
        print('mean lpips: ', np.mean(lpips_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--gt_data', type=str, help='path to dataset with gt labels',
                        default='/path/to/dataset')
    parser.add_argument('--pred_data', type=str, help='path to folder with predictions',
                        default='/path/to/inpainted/delayed/reprojections')
    parser.add_argument('--model_name', type=str, help='sub-folder to load to under pred_data',
                        default='resnet-l1')
    parser.add_argument('--vid_name', type=str, help='name of video to predict for', required=True)
    parser.add_argument('--time_delay', default=5, type=int, help='delay to compute psnr for')
    parser.add_argument('--start_time', default=0, type=int, help='zero-indexed timestamp to start collecting from')
    parser.add_argument('--end_time', default=-1, type=int, help='zero-indexed timestamp to end collection at (set -1 for end of video)')

    args = parser.parse_args()

    collector = Collector(args)

    with torch.no_grad():
        collector.collect()
