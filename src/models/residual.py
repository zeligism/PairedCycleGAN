
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 dilation=(1,1),
                 downsample=None,
                 dropout_p=0.0):
        super().__init__()

        self.dilation = dilation
        self.downsample = downsample

        self.main = nn.Sequential(
            ### Conv 3x3 ###
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation[0], dilation=dilation[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            ### Conv 3x3 ###
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=dilation[1], dilation=dilation[1], bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):

        residual = x if self.downsample is None else self.downsample(x)

        return self.main(x) + residual


class ResidualBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels,
                 downsample=None,
                 dilation=1,
                 dropout_p=0.0):
        super().__init__()

        self.downsample = downsample
        self.dilation = dilation

        self.main = nn.Sequential(

            ### Conv 1x1 ###
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            ### Conv 3x3 ###
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=dilation[1], dilation=dilation[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            ### Conv 1x1 ###
            nn.Conv2d(out_channels, out_channels * 4, 1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )


    def forward(self, x):

        residual = x if self.downsample is None else self.downsample(x)

        return self.main(x) + residual

