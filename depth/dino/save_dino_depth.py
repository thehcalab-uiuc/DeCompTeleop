import argparse
import torch
import os
import cv2
import numpy as np

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dinovit import DINOv2DPT


"""
    Generate predictions of DINOv2 monocular depth estimation model for baseline results
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = '/path/to/save/folder'
input_dir = '/path/to/dataset'
os.makedirs(output_dir, exist_ok=True)

# Load model and weights
runs = 28
model = DINOv2DPT().to(device)
encoder_path = f'/path/to/encoder_epoch100_0.089727_0.255787'
decoder_path = f'/path/to/decoder_epoch100_0.089727_0.255787'
final_layer_path = f'/path/to/final_layer_epoch100_0.089727_0.255787'
model.backbone.load_state_dict(torch.load(encoder_path))
model.decoder.load_state_dict(torch.load(decoder_path))
model.final_layer.load_state_dict(torch.load(final_layer_path))
model.eval()

# RGB image transformation before feeding into model (from DINOv2)
transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
    transforms.Normalize(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
    ),
])

test_video_names = ['middle-cornfield_2023-20230804-ts_2023_08_04_12h41m19s_four_rows',
                    'recollected-early-cornfield3-20220714_cornfield3-ts_2022_07_14_12h01m08s_three_rows',
                    'late-cornfield2-20220815_cornfield2-ts_2022_08_15_11h33m57s_one_random']

with torch.no_grad():
    for video_name in tqdm(test_video_names, desc='Loop through test videos ...'):
        frame_dir = os.path.join(input_dir, video_name, 'seq00', 'rgb')
        frames = os.listdir(frame_dir)
        for frame_name in tqdm(frames, desc="Loop through frames ...", leave=False):
            frame_path = os.path.join(frame_dir, frame_name)
            frame_num = frame_name[:-4]  # without .png

            rgb_image = np.array(Image.open(frame_path).convert("RGB"))
            rgb_image_tensor = transform_rgb(rgb_image).to(device)
            rgb_image_tensor = rgb_image_tensor.unsqueeze(0)  # Add batch dimension
            output_depth = model(rgb_image_tensor)

            # Resize to (1280, 720)
            output_depth_numpy = output_depth.squeeze(0).cpu().numpy()
            output_depth_numpy = cv2.resize(output_depth_numpy, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Save predicted depth
            save_dir = os.path.join(output_dir, video_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{frame_num}.npy')
            np.save(save_path, output_depth_numpy)
