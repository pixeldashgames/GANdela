import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, kernel_size=4, stride=2,
                 padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels= 1, features=64, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, padding, padding_mode='reflect'),
        )

        # ------------------------------------------------------------------------------
        # ---------------------------------- ENCODER -----------------------------------
        # ------------------------------------------------------------------------------
        self.down1 = Block(features, features * 2, down=True, act='leaky', use_dropout=False, kernel_size=kernel_size,
                           stride=stride, padding=padding)  # 64 X 64
        self.down2 = Block(features * 2, features * 4, down=True, act='leaky', use_dropout=False,
                           kernel_size=kernel_size, stride=stride, padding=padding)  # 32 X 32
        self.down3 = Block(features * 4, features * 8, down=True, act='leaky', use_dropout=False,
                           kernel_size=kernel_size, stride=stride, padding=padding)  # 16 X 16
        self.down4 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False,
                           kernel_size=kernel_size, stride=stride, padding=padding)  # 8 X 8
        self.down5 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False,
                           kernel_size=kernel_size, stride=stride, padding=padding)  # 4 X 4
        self.down6 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False,
                           kernel_size=kernel_size, stride=stride, padding=padding)  # 2 X 2

        # ----------------------------------------------------------------------------
        # --------------------------------BOTTLENECK ---------------------------------
        # ----------------------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),  # 1 X 1
            nn.ReLU()
        )

        # ----------------------------------------------------------------------------
        # ---------------------------------DECODER -----------------------------------
        # ----------------------------------------------------------------------------
        self.up = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size, stride, padding),  # 2x2 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(features * 4, out_channels, kernel_size, stride, padding),  # 4x4 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 8x8 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 16x16 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 32x32 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 64x64 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 128x128 spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding),  # 256x256 spatial size
            nn.ReLU(),
            nn.Tanh()
        )
        

        # self.final_up = nn.Sequential(
        #     nn.ConvTranspose2d(features * 2, in_channels, kernel_size, stride, padding),
        #     nn.Tanh()
        

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        output = self.up(bottleneck)
        # output = output.permute(0, 2, 3, 1)
        return output
