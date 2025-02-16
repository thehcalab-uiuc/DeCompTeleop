# Local imports
from refinement import TravNetUp3NNRGB

# Standard imports
import os
import argparse
from PIL import Image
import json

# Third party imports
import numpy as np
import cv2

# PyTorch imports
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--run_name', type=str, required=True, help='name of model folder to load from')
    parser.add_argument('--gt_data', type=str, required=True, help='path to dataset folder with original images')
    parser.add_argument('--proj_data', type=str, required=True, help='path to dataset folder with reprojected images')
    parser.add_argument('--gamma_sub_folder', type=str, default='pulsar_gamma_1e-1', help='name of sub-folder within proj_data to train on')

    # Hyperparameters
    parser.add_argument('--vid_name', type=str, required=True, help='name of video to predict')
    parser.add_argument('--start_time', type=int, default=0, help='conditioning frame timestep')
    parser.add_argument('--offset', type=int, default=1, help='applied time delay')

    args = parser.parse_args()

    # Create results folder
    model_folder_path = os.path.join('/path/to/weights/inpainting/folder', args.run_name)
    assert os.path.exists(model_folder_path)
    assert os.path.exists(os.path.join(args.proj_data, args.gamma_sub_folder, args.vid_name, str(args.offset)))
    results_folder_path = os.path.join(model_folder_path, 'pred', args.ckpt_name, args.gamma_sub_folder, args.vid_name, str(args.offset))
    os.makedirs(results_folder_path, exist_ok=True)

    # Load original args
    with open(os.path.join(model_folder_path, 'paint_{}_args.json'.format(args.run_name)), 'r') as f:
        train_args = json.load(f)

    with torch.no_grad():

        # Load model
        model = TravNetUp3NNRGB(output_size=(720,1280), activation=train_args['model_act']).to(device)
        ckpt_path = os.path.join(model_folder_path, 'paint_{}.pth'.format(args.run_name))
        model_dict = {k.replace('module.', ''): v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
        model.load_state_dict(model_dict)
        model = model.to(device).eval()

        # Image transform
        image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        norm_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load images
        gt_vid_folder = os.path.join(args.gt_data, args.vid_name, 'seq00', 'rgb')
        gt_orig_image_path = os.path.join(gt_vid_folder, '{}.png'.format(str(args.start_time).zfill(6)))
        gt_future_image_path = os.path.join(gt_vid_folder, '{}.png'.format(str(args.start_time+args.offset).zfill(6)))
        proj_image_path = os.path.join(args.proj_data, args.gamma_sub_folder, args.vid_name, str(args.offset), '{}.png'.format(str(args.start_time+args.offset).zfill(6)))
        gt_orig_image = image_transform(Image.open(gt_orig_image_path).convert("RGB")).unsqueeze(0) # (1, 3, 720, 1280)
        proj_image = image_transform(Image.open(proj_image_path).convert("RGB")).unsqueeze(0)       # (1, 3, 720, 1280)
        if train_args['norm_input']:
            gt_orig_image = norm_transform(gt_orig_image)
            proj_image = norm_transform(proj_image)
        gt_orig_image = gt_orig_image.to(device)
        proj_image = proj_image.to(device)

        # Predict image
        pred_image = model(gt_orig_image, proj_image).squeeze(0).permute((1,2,0))   # (720, 1280, 3)
        if train_args['model_act'] == 'tanh':
            pred_image = (pred_image + 1.) * (255./2.)                              # [0,255]
        else:
            pred_image = pred_image * 255.

        # Save image
        pred_image_np = pred_image.detach().cpu().numpy()
        cv2.imwrite(os.path.join(results_folder_path, '{}.png'.format(str(args.start_time+args.offset).zfill(6))), cv2.cvtColor(pred_image_np, cv2.COLOR_BGR2RGB))
