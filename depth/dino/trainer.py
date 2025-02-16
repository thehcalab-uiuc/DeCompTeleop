import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FilledDepthDataset
from dinovit import DINOv2DPT


"""
    Trainer for DINOv2 monocular depth estimation

    - Arguments for training:
        - mode = train (required)
        - runs: The recorded label for the current run (required)
        - lr: learning rate (required)
        - batch_size: batch size for the training and validation dataset (required)
        - num_epochs: Total number of epochs for training (required)
    - Arguments for testing:
        - mode = test (required)
        - runs: The recorded label for the current run (required)
        - batch_size: batch size for the test dataset (required)
        - encoder_path: The path of the DINOv2 backbone weight (required)
        - decoder_path: The path of the decoder weight (required)
        - final_layer_path: The path of the final depth head weight (required)

    Encoder weights saved in ./checkpoints/{runs}/depth/encoder
    Decoder weights saved in ./checkpoints/{runs}/depth/decoder
    Final layer weights saved in ./checkpoints/{runs}/depth/final_layer
"""

# Scale-invariant loss class
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(torch.maximum(pred[valid_mask], torch.tensor(0.3)))  # Avoid 0 being passed to log
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss

# Scheduler function
def lambda_epoch(epoch):
    if epoch < 40:
        return 1.0
    else:
        return 0.99 ** (epoch - 40)  # Decay rate

class Trainer:
    def __init__(self, run_idx, lr, batch_size, num_epochs) -> None:
        self.run_idx = run_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.batch_size = batch_size
        self.lr =lr
        self.num_epochs = num_epochs

        # Dataset
        self.train_dataset = FilledDepthDataset(device=self.device, data_type='train')
        self.validation_dataset = FilledDepthDataset(device=self.device, data_type='val')
        self.test_dataset = FilledDepthDataset(device=self.device, data_type='test')

        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Model, loss function, optimizer, scheduler
        self.model = DINOv2DPT().to(self.device)
        self.loss_func = SiLogLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_epoch)

        # Logger
        self.logger_path = f'./runs/logs/{self.run_idx}/depth'
        os.makedirs(self.logger_path, exist_ok=True)
        self.logger = SummaryWriter(self.logger_path)
    
    def save_model(self, epoch, train_loss, validation_loss):
        # Save decoder
        decoder_save_dir = f'./checkpoints/{self.run_idx}/depth/decoder'
        os.makedirs(decoder_save_dir, exist_ok=True)
        decoder_save_path = os.path.join(decoder_save_dir, f'epoch{epoch + 1}_{train_loss:.6f}_{validation_loss:.6f}.pth')
        torch.save(self.model.decoder.state_dict(), decoder_save_path)

        # Save final layer (depth head)
        final_layer_save_dir = f'./checkpoints/{self.run_idx}/depth/final_layer'
        os.makedirs(final_layer_save_dir, exist_ok=True)
        final_layer_save_path = os.path.join(final_layer_save_dir, f'epoch{epoch + 1}_{train_loss:.6f}_{validation_loss:.6f}.pth')
        torch.save(self.model.final_layer.state_dict(), final_layer_save_path)

        # Save encoder (dinov2 backbone)
        encoder_save_dir = f'./checkpoints/{self.run_idx}/depth/encoder'
        os.makedirs(encoder_save_dir, exist_ok=True)
        encoder_save_path = os.path.join(encoder_save_dir, f'epoch{epoch + 1}_{train_loss:.6f}_{validation_loss:.6f}.pth')
        torch.save(self.model.backbone.state_dict(), encoder_save_path)

    def plot_training_curve(self, train_losses, validation_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(self.logger_path + '/loss_curve.png')

    def train(self):
        # Training loop
        print('Start Training ...')
        train_losses = []
        validation_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_loader = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]')
            for rgb, depth, valid_mask in train_loader:
                self.optimizer.zero_grad()
                output = self.model(rgb)
                si_loss = self.loss_func(output, depth, valid_mask)  # Scale-invariant loss
                si_loss.backward()
                self.optimizer.step()
                train_loss += si_loss.item()
                
            self.scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.logger.add_scalar('Loss/train', avg_train_loss, epoch)

            # Validation
            print('Start Validation ...')
            self.model.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_loader = tqdm(self.validation_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Validation]')
                for rgb, depth, valid_mask in validation_loader:
                    output = self.model(rgb)
                    si_loss = self.loss_func(output, depth, valid_mask)  # Scale-invariant loss
                    validation_loss += si_loss.item()

                avg_validation_loss = validation_loss / len(self.validation_loader)
                validation_losses.append(avg_validation_loss)
                self.logger.add_scalar('Loss/validation', avg_validation_loss, epoch)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}]| Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_validation_loss:.6f}')

            if (epoch + 1) % 5 == 0:
                self.save_model(epoch, avg_train_loss, avg_validation_loss)  # Save model
            
        self.logger.close()
        self.plot_training_curve(train_losses, validation_losses)
    
    def test(self, encoder_path, decoder_path, final_layer_path):
        self.model.backbone.load_state_dict(torch.load(encoder_path))
        self.model.decoder.load_state_dict(torch.load(decoder_path))
        self.model.final_layer.load_state_dict(torch.load(final_layer_path))
        self.model.eval()

        test_loss = 0.0
        test_loader = tqdm(self.test_loader, desc='Test')
        with torch.no_grad():
            for rgb, depth, valid_mask in test_loader:
                output = self.model(rgb) 
                si_loss = self.loss_func(output, depth, valid_mask)  # Scale-invariant loss
                test_loss += si_loss.item()

        avg_test_loss = test_loss / len(self.test_loader)
        print(f'Test Loss: {avg_test_loss:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('--runs', type=int, help='training num')
    
    # Training
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=40, type=int, help='epochs to train')
    
    # Test
    parser.add_argument('--encoder_path', default=None, type=str, help='saved encoder weights path')
    parser.add_argument('--decoder_path', default=None, type=str, help='saved decoder weights path')
    parser.add_argument('--final_layer_path', default=None, type=str, help='saved final_layer weights path')
    
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer(args.runs, args.lr, args.batch_size, args.num_epochs)
        trainer.train()
    else:
        trainer = Trainer(args.runs, args.lr, args.batch_size, args.num_epochs)
        trainer.test(args.encoder_path, args.decoder_path, args.final_layer_path)
