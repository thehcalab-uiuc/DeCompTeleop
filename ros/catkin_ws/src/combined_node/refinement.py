# Modified from https://github.com/andreschreiber/W-RIZZ/blob/main/architectures/travnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VisionNavNet(nn.Module):
    """ Visual Navigation Model """
    def __init__(self):
        """ Constructs vision navigation model """
        super().__init__()

        resnet18 = models.resnet18()

        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.conv     = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.fc       = nn.Linear(64*4*5, 64)
        self.dropout  = nn.Dropout(p=0.5)
        self.reg      = nn.Linear(64, 2)

    def forward(self, img):
        """ Computes heading distance (robot heading, distance ratio) from input image (resolution of 240x320)"""
        x = self.features(img)
        x = self.conv(x)
        x = x.view(-1, 64*4*5)
        x = self.fc(x)
        x = self.dropout(x)
        heading_distance = self.reg(x)
        return heading_distance

class TravNetUp3NNRGB(nn.Module):
    def __init__(self, output_size=(360, 640), bottleneck_dim=512, output_channels=3, activation='tanh'):
        super().__init__()

        # load ResNet encoder with pretrained nav weights
        vision_nav_net = VisionNavNet()
        checkpoint = torch.load('/path/to/VisionNavNet_state_hd.pth.tar')
        vision_nav_net.load_state_dict(checkpoint['state_dict'])
        model = models.resnet18()
        model.conv1.load_state_dict(vision_nav_net.features[0].state_dict())
        model.layer1.load_state_dict(vision_nav_net.features[4].state_dict())
        model.layer2.load_state_dict(vision_nav_net.features[5].state_dict())
        model.layer3.load_state_dict(vision_nav_net.features[6].state_dict())
        model.layer4.load_state_dict(vision_nav_net.features[7].state_dict())

        # output size
        self.out_dim = (output_size[0], output_size[1])

        # encoder
        self.block1 = nn.Sequential(*(list(model.children())[:3]))
        self.block2 = nn.Sequential(model.maxpool, model.layer1)
        self.block3 = model.layer2
        self.block4 = model.layer3
        self.block5 = model.layer4
        
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.convUp1A = nn.Sequential(
            nn.Conv2d(in_channels=256+512+512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.convUp1B = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.convUp2A = nn.Sequential(
            nn.Conv2d(in_channels=128+256+256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.convUp2B = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.convUp3A = nn.Sequential(
            nn.Conv2d(in_channels=64+128+128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.convUp3B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.convUp4A = nn.Sequential(
            nn.Conv2d(in_channels=32+64+64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.convUp4B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.convUp5A = nn.Sequential(
            nn.Conv2d(in_channels=16+64+64, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.convUp5B = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(in_channels=8, out_channels=output_channels, kernel_size=1, stride=1)

        self.activation_name = activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sig':
            self.activation = nn.Sigmoid()
        
    def forward(self, orig, proj):
        with torch.no_grad():
            # Encoder
            out1_orig = self.block1(orig)
            out2_orig = self.block2(out1_orig)
            out3_orig = self.block3(out2_orig)
            out4_orig = self.block4(out3_orig)
            out5_orig = self.block5(out4_orig)

            out1_proj = self.block1(proj)
            out2_proj = self.block2(out1_proj)
            out3_proj = self.block3(out2_proj)
            out4_proj = self.block4(out3_proj)
            out5_proj = self.block5(out4_proj)

        # Bottleneck
        x = torch.cat((out5_orig, out5_proj), dim=1)
        x = self.bottleneck(x)

        # Decoder
        x = torch.cat((x, out5_orig, out5_proj), dim=1)
        x = self.convUp1B(F.interpolate(self.convUp1A(x), out4_orig.shape[2:]))
        x = torch.cat((x, out4_orig, out4_proj), dim=1)
        x = self.convUp2B(F.interpolate(self.convUp2A(x), out3_orig.shape[2:]))
        x = torch.cat((x, out3_orig, out3_proj), dim=1)
        x = self.convUp3B(F.interpolate(self.convUp3A(x), out2_orig.shape[2:]))
        x = torch.cat((x, out2_orig, out2_proj), dim=1)
        x = self.convUp4B(F.interpolate(self.convUp4A(x), out1_orig.shape[2:]))
        x = torch.cat((x, out1_orig, out1_proj), dim=1)
        x = self.convUp5B(F.interpolate(self.convUp5A(x), self.out_dim))
        output = self.final(x)
        if self.activation_name:
            output = self.activation(output)

        return output
