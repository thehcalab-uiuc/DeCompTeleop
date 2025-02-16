import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FilledDepthDataset
from dinovit import DINOv2DPT


'''
    Visualize predicted images of DINOv2 monocular depth estimation model
'''

# Scale-invariant loss
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(torch.maximum(pred[valid_mask], torch.tensor(0.3)))  # Avoid 0 being passed to log
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Hyperparameters
batch_size = 4
runs = 28
out_dir = f'./test_results/{runs}'
os.makedirs(out_dir, exist_ok=True)

test_dataset = FilledDepthDataset(device=device, data_type='test')  # Dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Dataloader

# Model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINOv2DPT().to(device)
loss_func = SiLogLoss()

# Load weights
encoder_path = '/path/to/encoder_epoch100_0.089727_0.255787'
decoder_path = '/path/to/decoder_epoch100_0.089727_0.255787'
final_layer_path = '/path/to/final_layer_epoch100_0.089727_0.255787'
model.backbone.load_state_dict(torch.load(encoder_path))
model.decoder.load_state_dict(torch.load(decoder_path))
model.final_layer.load_state_dict(torch.load(final_layer_path))
model.eval()

# Parameters to unnormalize RGB images
mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])

test_loss = 0.0
test_loader = tqdm(test_loader, desc='Test')
with torch.no_grad():
    for i, (rgb, depth, valid_mask) in enumerate(test_loader):
        output = model(rgb) 
        si_loss = loss_func(output, depth, valid_mask)  # Scale-invariant loss
        test_loss += si_loss.item()

        # Save images
        output_arr = output[0].cpu().numpy()  # Predicted depth select from batch
        depth[0][~valid_mask[0]] = torch.inf  # Set invalid pixels to infinity
        gt_arr = depth[0].cpu().numpy()  # Ground-truth depth
        rgb_arr = rgb[0].cpu().numpy()  # RGB image
        rgb_arr = (rgb_arr * std[:, None, None] + mean[:, None, None]).astype(int)
        rgb_arr = np.clip(rgb_arr, 0, 255)
        rgb_arr = np.transpose(rgb_arr, (1, 2, 0))

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        depth_plot_0 = axes[0].imshow(output_arr, cmap='viridis')
        axes[0].set_title('Predicted Depth')
        axes[0].axis('off')
        cbar_0 = fig.colorbar(depth_plot_0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar_0.set_label('Depth in meters')

        depth_plot_1 = axes[1].imshow(gt_arr, cmap='viridis')
        axes[1].set_title('Ground-truth Depth')
        axes[1].axis('off')
        cbar_1 = fig.colorbar(depth_plot_1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar_1.set_label('Depth in meters')

        axes[2].imshow(rgb_arr)
        axes[2].set_title('RGB Image')
        axes[2].axis('off')
        plt.savefig(os.path.join(out_dir, f'{i:06}.png'))

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.6f}')
