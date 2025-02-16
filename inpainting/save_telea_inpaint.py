# Standard imports
import os
import argparse

# Third party imports
import cv2
from PIL import Image
import numpy as np
import tqdm
from time import time


class Painter:

    def __init__(self, args) -> None:

        # Hyperparameters
        self.times = args.time

        # Check load folders
        self.proj_folder = os.path.join(args.proj_data, args.vid_name)
        print('inpainting projections from: ', self.proj_folder)
        for t_offset in self.times:
            offset_folder = os.path.join(self.proj_folder, str(t_offset))
            assert os.path.exists(offset_folder)

        # Create save folder
        self.save_folder = os.path.join(args.save_folder, args.model_name, args.vid_name)
        for t_offset in self.times:
            offset_folder = os.path.join(self.save_folder, str(t_offset))
            if not os.path.exists(offset_folder):
                os.makedirs(offset_folder)


    def paint(self):

        print('Start Painting ...')

        timing = []

        # Iterate over delays
        for t_offset in self.times:

            # Iterate over images to render from
            for file_name in tqdm.tqdm(os.listdir(os.path.join(self.proj_folder, str(t_offset)))):

                # Check if image
                proj_image_path = os.path.join(self.proj_folder, str(t_offset), file_name)
                if os.path.isfile(proj_image_path):

                    # Load reprojected image to be inpainted
                    proj_image = cv2.imread(proj_image_path)

                    # Create mask
                    green_value = np.array([0, 255, 0], dtype=np.uint8)     # Get mask of holes
                    green_mask = np.all(proj_image == green_value, axis=-1).astype(np.uint8) 
                    green_mask[green_mask==1] = 255

                    # Inpaint
                    init_time = time()
                    pred_image = cv2.inpaint(proj_image, green_mask, 3, cv2.INPAINT_TELEA)
                    end_time = time()
                    timing.append(end_time - init_time)

                    # Save inpainted image
                    inpainted_save_folder = os.path.join(self.save_folder, str(t_offset), 'inpainted')
                    if not os.path.exists(inpainted_save_folder):
                        os.makedirs(inpainted_save_folder)
                    cv2.imwrite(os.path.join(inpainted_save_folder, file_name), pred_image)

        print('mean timing: ', np.mean(timing), ' std timing: ', np.std(timing))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--proj_data', type=str, help='path to reprojected dataset folder',
                        default='/path/to/predicted/delayed/reprojections')
    parser.add_argument('--vid_name', type=str, help='name of video to render', required=True)
    parser.add_argument('--save_folder', type=str, help='path to save folder',
                        default='/path/to/save/folder')
    parser.add_argument('--model_name', type=str, help='sub-folder to save to under save_folder',
                        default='telea')

    # Delays to inpaint
    parser.add_argument('--time', type=int, nargs='+', default=[1,3,5,7,10,15])

    args = parser.parse_args()

    painter = Painter(args)

    painter.paint()
