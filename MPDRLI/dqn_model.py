import os.path as osp
import numpy as np
import torch
from torch import nn

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)

class DQN(nn.Module): # for atari
    def __init__(self, in_channels, num_actions):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        super(DQN, self).__init__()
        print("in_channels", in_channels)
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.Softplus(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.Softplus(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Softplus(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.Softplus(),
            nn.Linear(in_features=512, out_features=num_actions),
        )

        self.apply(weights_init)
        
    def forward(self, obs):
        
        # print("obs size", obs.size())

        out = obs.float() / 255 # convert 8-bits RGB color to float in [0, 1]
        out = out.permute(0, 3, 1, 2) # reshape to [batch_size, img_c * frames, img_h, img_w]

        out = self.convnet(out)
        out = out.view(out.size(0), -1) # flatten feature maps to a big vector
        # print("out size", out.size())
        out = self.classifier(out)
        return out