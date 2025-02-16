# Standard imports
import os
import argparse
import tqdm
import csv

# Third party imports
import numpy as np
import torch


class Collector:

    def __init__(self, args) -> None:

        # Check load folders and create save csv
        self.gt_load_folder = os.path.join(args.gt_data, args.vid_name, 'seq00')
        print('collecting metrics for: ', self.gt_load_folder)
        if not os.path.exists(self.gt_load_folder):
            raise Exception
        self.pred_load_folder = os.path.join(args.pred_data, args.vid_name)
        if not os.path.exists(self.pred_load_folder):
            raise Exception
        self.save_csv_path = os.path.join(args.pred_data, '{}_metrics.csv'.format(args.vid_name))
        self.start_time = args.start_time
        self.end_time = args.end_time

        with open(self.save_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'd1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog'])


    def eval_depth(self, pred, target):
        assert pred.shape == target.shape

        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)
        d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
        d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)

        abs_rel = torch.mean(torch.abs(diff) / target)
        sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
                'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


    def collect(self):

        print('Start Collecting Metrics ...')

        # Path to video frames
        load_images_folder = os.path.join(self.gt_load_folder, 'depth')

        # Count number of frames in video
        if self.end_time == -1:
            file_count = 0
            for p in os.listdir(load_images_folder):
                if os.path.isfile(os.path.join(load_images_folder, p)):
                    file_count += 1
            self.end_time = file_count - 1

        # Iterate over images
        for im_num in tqdm.tqdm(range(self.start_time, self.end_time+1)):

            # Load gt and prediction 
            gt_depth_np = np.load(os.path.join(load_images_folder, '{}.npy'.format(str(im_num).zfill(6))))
            pred_depth_np = np.load(os.path.join(self.pred_load_folder, '{}.npy'.format(str(im_num).zfill(6))))

            # Convert to torch
            gt_depth = torch.from_numpy(gt_depth_np)
            pred_depth = torch.from_numpy(pred_depth_np)

            # Get valid pixels
            valid_mask = (gt_depth >= 0.) & (gt_depth <= 20.)

            # Evaluate
            curr_results = self.eval_depth(pred_depth[valid_mask], gt_depth[valid_mask])

            # Add to csv
            with open(self.save_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([im_num, curr_results['d1'], curr_results['d2'], curr_results['d3'], 
                                 curr_results['abs_rel'], curr_results['sq_rel'], curr_results['rmse'], 
                                 curr_results['rmse_log'], curr_results['log10'], curr_results['silog']])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--gt_data', type=str, help='path to dataset with gt labels',
                        default='/path/to/dataset')
    parser.add_argument('--pred_data', type=str, help='path to folder with predictions',
                        default='/path/to/predicted/depth')
    parser.add_argument('--vid_name', type=str, help='name of video to predict for', required=True)
    parser.add_argument('--start_time', default=0, type=int, help='zero-indexed timestamp to start collecting from')
    parser.add_argument('--end_time', default=-1, type=int, help='zero-indexed timestamp to end collection at (set -1 for end of video)')

    args = parser.parse_args()

    collector = Collector(args)

    collector.collect()
