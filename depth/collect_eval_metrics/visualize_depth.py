# Standard imports
import os
import argparse
import tqdm

# Third party imports
import cv2
import numpy as np
import matplotlib


class Renderer:

    def __init__(self, args) -> None:

        # Check load folders
        self.use_zed = True if args.zed_depth else False
        if self.use_zed:
            self.load_folder = os.path.join(args.gt_data, args.vid_name, 'seq00')
            self.depth_folder = os.path.join(self.load_folder, 'depth')
        else:
            self.depth_folder = os.path.join(args.depth_data, args.vid_name)
        assert os.path.exists(self.depth_folder)
        print('rendering projections from: ', self.depth_folder)

        # Create save folder
        self.save_folder = os.path.join(args.save_folder, args.model_name, args.vid_name)
        if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)


    def render(self):

        print('Start Rendering ...')

        # Count number of frames in video
        file_count = 0
        for p in os.listdir(self.depth_folder):
            if os.path.isfile(os.path.join(self.depth_folder, p)):
                file_count += 1

        # Iterate over images to render from
        for im_num in tqdm.tqdm(range(file_count)):

            # Get depth image
            init_depth_np = np.load(os.path.join(self.depth_folder, '{}.npy'.format(str(im_num).zfill(6))))

            # Save depth image
            cmap = matplotlib.colormaps.get_cmap('Spectral')
            invalid_mask = ~np.isfinite(init_depth_np)
            pred_depth_np = (init_depth_np - 0.) / (20. - 0.) * 255.0
            pred_depth_np[invalid_mask] = 0
            pred_depth_np = pred_depth_np.astype(np.uint8)
            pred_depth_np = (cmap(pred_depth_np)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            pred_depth_np[invalid_mask] = [0, 255, 0]
            cv2.imwrite(os.path.join(self.save_folder, 'depth_{}.png'.format(str(im_num).zfill(6))), pred_depth_np)


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

    args = parser.parse_args()

    renderer = Renderer(args)

    renderer.render()
