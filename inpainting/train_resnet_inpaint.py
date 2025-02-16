# Local imports
from dataset import Terra_DAv2_Projected_Inpainting
from refinement import TravNetUp3NNRGB

# Standard imports
import os
import argparse
import math
import json

# Third party imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

class Trainer:
    def __init__(self, args):

        self.args = args

        # Dataset
        self.train_dataset = Terra_DAv2_Projected_Inpainting(mode='debug' if args.debug else 'train',
                                                             gt_data_path=args.gt_data,
                                                             proj_data_path=os.path.join(args.proj_data, args.gamma_sub_folder),
                                                             offsets=args.offsets,
                                                             norm_input=args.norm_input)

        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

        # Load model
        self.model = TravNetUp3NNRGB(output_size=(720,1280), activation=args.model_act)
        self.model = self.model.to(device)

        # Freeze weights
        if args.freeze_nav:
            for param in self.model.block1.parameters():
                param.requires_grad = False
            for param in self.model.block2.parameters():
                param.requires_grad = False
            for param in self.model.block3.parameters():
                param.requires_grad = False
            for param in self.model.block4.parameters():
                param.requires_grad = False
            for param in self.model.block5.parameters():
                param.requires_grad = False

        # Loss
        if args.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif args.loss == 'ssim':
            self.criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        elif args.loss == 'ms_ssim':
            self.criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        else:
            raise Exception

        # Optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-5)

        # Create subfolder for latest run of saves
        model_dir = os.path.join('checkpoints', args.run_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        run_num = 0
        while os.path.exists(os.path.join(model_dir, str(run_num))):
            run_num += 1
        run_dir = os.path.join(model_dir, str(run_num))
        os.makedirs(run_dir, exist_ok=False)

        # Logging and saving paths
        self.logger_path = os.path.join(run_dir, 'logs')
        os.makedirs(self.logger_path)
        self.logger = SummaryWriter(self.logger_path)
        self.ckpt_save_path = os.path.join(run_dir, 'ckpt')
        os.makedirs(self.ckpt_save_path)

        # Save args
        with open(os.path.join(run_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, ensure_ascii=False, indent=4)
    
    def save_model(self, epoch, train_loss):
        model_save_path = os.path.join(self.ckpt_save_path, f'model_epoch{epoch + 1}_{train_loss:.6f}.pth')
        torch.save(self.model.state_dict(), model_save_path)
        optim_save_path = os.path.join(self.ckpt_save_path, f'optim_epoch{epoch + 1}_{train_loss:.6f}.pth')
        torch.save(self.optimizer.state_dict(), optim_save_path)

    def plot_training_curve(self, train_losses):
        plt.figure(num=0, figsize=(10, 5), dpi=300)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join(self.logger_path, 'loss_curve.png'))
    
    def get_learning_rate(self, step, step_per_epoch):
        mul = np.cos(step / (self.args.num_epochs * step_per_epoch) * math.pi) * 0.5 + 0.5
        return (self.args.start_lr - self.args.end_lr) * mul + self.args.end_lr

    def train(self):

        # Training loop
        print('Start Training ...')

        # Losses lists
        train_losses = []

        step_per_epoch = self.train_loader.__len__()
        step = 0

        # Loop over epochs
        for epoch in range(self.args.num_epochs):

            # Start of epoch
            self.model.train()
            train_loss = 0.0

            # Iterate over batches
            train_loader = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.args.num_epochs} [Train]')

            for gt_orig_image, gt_future_image, proj_image in train_loader:
                
                # Get data
                gt_orig_image = gt_orig_image.to(device)        # (batch_size, 3, 720, 1280) float32, input normalized
                gt_future_image = gt_future_image.to(device)    # (batch_size, 3, 720, 1280) float32, label normalized [0,1]
                proj_image = proj_image.to(device)              # (batch_size, 3, 720, 1280) float32, input normalized

                # Zero gradients
                self.optimizer.zero_grad()

                # Predict images
                pred_image = self.model(gt_orig_image, proj_image)  # (batch_size, 3, 720, 1280)

                # Compute loss
                if self.args.loss == 'l1':
                    if self.args.model_act == 'tanh':
                        gt_future_image = 2 * gt_future_image - 1.      # [-1,1]
                elif self.args.loss in ['ssim', 'ms_ssim']:
                    if self.args.model_act == 'tanh':
                        pred_image = (pred_image + 1.) / 2.             # [0,1]
                iter_loss = self.criterion(pred_image, gt_future_image)

                # Update network
                iter_loss.backward()
                self.optimizer.step()
                train_loss += iter_loss.item()

            # Average losses over epoch
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.logger.add_scalar('Loss/train', avg_train_loss, epoch)

            print(f'Epoch [{epoch + 1}/{self.args.num_epochs}]| Training Loss: {avg_train_loss:.6f}')

            if ((epoch + 1) % self.args.save_freq == 0) or not epoch:
                self.save_model(epoch, avg_train_loss)  # Save model
                self.plot_training_curve(train_losses)
            
        self.logger.close()
        self.plot_training_curve(train_losses)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # OS setup
    parser.add_argument('--run_name', type=str, default='debug', help='name of model folder to save to')
    parser.add_argument('--gt_data', type=str, required=True, help='path to dataset folder with original images')
    parser.add_argument('--proj_data', type=str, required=True, help='path to dataset folder with reprojected images')
    parser.add_argument('--gamma_sub_folder', type=str, default='pulsar_gamma_1e-1', help='name of sub-folder within proj_data to train on')
    parser.add_argument('--debug', action="store_true", help='use debug data split to overfit')

    # Hyperparameters
    parser.add_argument('--model_act', choices=('sig', 'tanh', ''), default='tanh', help='final activation of model')
    parser.add_argument('--loss', choices=('l1', 'ssim', 'ms_ssim'), default='l1', help='training loss type')
    parser.add_argument('--norm_input', action="store_true", help='normalize input data to mean and std')
    parser.add_argument('--offsets', type=int, nargs='+', default=[1,3,5,7,10,15], help='list of time delays to train with')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='epochs to train')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of saving model by epoch')
    parser.add_argument('--freeze_nav', action="store_true", help='freeze navigation cnn')
    
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
