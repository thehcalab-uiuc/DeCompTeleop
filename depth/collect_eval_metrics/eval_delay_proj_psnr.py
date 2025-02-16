# Standard imports
import os
import argparse
import tqdm
import csv
from PIL import Image

# Third party imports
import numpy as np
import cv2


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
            writer.writerow(['time', 'psnr'])


    def calculate_psnr(self, rgb_pixels1, rgb_pixels2):
        # Ensure that both lists have the same length
        if rgb_pixels1.shape != rgb_pixels2.shape:
            raise ValueError("The two lists must have the same shape")

        # Convert lists to NumPy arrays
        array1 = rgb_pixels1.astype(np.float32)
        array2 = rgb_pixels2.astype(np.float32)

        # Calculate the Mean Squared Error (MSE)
        mse = np.mean((array1 - array2) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')  # PSNR is infinite if MSE is 0

        # Calculate PSNR
        max_pixel_value = 255.0  # Assuming the pixel values are in the range [0, 255]
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        
        return psnr


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

        psnr_list = []

        # Iterate over images
        for im_num in tqdm.tqdm(range(self.start_time, self.end_time+1)):

            # Load gt and projected images
            gt_im = Image.open(os.path.join(load_images_folder, '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")
            proj_im = Image.open(os.path.join(self.pred_load_folder, '{}.png'.format(str(im_num).zfill(6)))).convert("RGB")

            # Convert to np array
            gt_im_np = np.array(gt_im)      # (720, 1280, 3)
            proj_im_np = np.array(proj_im)  # (720, 1280, 3)

            # Get mask of holes
            green_value = np.array([0, 255, 0], dtype=np.uint8)
            green_mask = np.all(proj_im_np == green_value, axis=-1)

            # Get non-hole values
            gt_im_np_masked = gt_im_np[~green_mask]     # (num_valid_pixels, 3) uint8
            proj_im_np_masked = proj_im_np[~green_mask] # (num_valid_pixels, 3) uint8

            if proj_im_np_masked.shape[0]:

                # Compute PSNR
                curr_psnr = self.calculate_psnr(gt_im_np_masked, proj_im_np_masked)
                psnr_list.append(curr_psnr)

                # Add to csv
                with open(self.save_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([im_num, curr_psnr])

        print('mean psnr: ', np.mean(psnr_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--gt_data', type=str, help='path to dataset with gt labels',
                        default='/path/to/dataset')
    parser.add_argument('--pred_data', type=str, help='path to folder with predictions',
                        default='/path/to/predicted/delayed/reprojections')
    parser.add_argument('--model_name', type=str, help='sub-folder to load to under pred_data',
                        default='dav2')
    parser.add_argument('--vid_name', type=str, help='name of video to predict for', required=True)
    parser.add_argument('--time_delay', default=5, type=int, help='delay to compute psnr for')
    parser.add_argument('--start_time', default=0, type=int, help='zero-indexed timestamp to start collecting from')
    parser.add_argument('--end_time', default=-1, type=int, help='zero-indexed timestamp to end collection at (set -1 for end of video)')

    args = parser.parse_args()

    collector = Collector(args)

    collector.collect()
