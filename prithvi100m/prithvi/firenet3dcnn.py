import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM3D

class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 4, 8, 12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                Conv2Plus1D(
                    in_planes=in_channels,
                    out_planes=out_channels // len(dilations),
                    spatial_dilation=d
                ),
                nn.BatchNorm3d(out_channels // len(dilations)),
                nn.ReLU(inplace=True)
            )
            for d in dilations
        ])
        self.project = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(out)


class Conv2Plus1D(nn.Module):
    def __init__(self, in_planes, out_planes, mid_planes=None, spatial_dilation=1, temporal_dilation=1):
        super().__init__()
        if mid_planes is None:
            mid_planes = (in_planes * out_planes) // (in_planes + out_planes)
        
        spatial_padding = (0, spatial_dilation, spatial_dilation)
        temporal_padding = (temporal_dilation, 0, 0)

        self.spatial = nn.Conv3d(in_planes,
                                 mid_planes,
                                 kernel_size=(1, 3, 3),
                                 padding=spatial_padding,
                                 dilation=(1, spatial_dilation, spatial_dilation))

        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.temporal = nn.Conv3d(mid_planes,
                                  out_planes,
                                  kernel_size=(3, 1, 1),
                                  padding=temporal_padding,
                                  dilation=(temporal_dilation, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.spatial(x)))
        x = self.relu2(self.bn2(self.temporal(x)))
        return x

class FireNet3DCNN(nn.Module):
    def __init__(self, in_channels, base_filters=32):
        super(FireNet3DCNN, self).__init__()

        self.encoder = nn.Sequential(
            Conv2Plus1D(in_channels, base_filters),
            Conv2Plus1D(base_filters, base_filters * 2),
            nn.MaxPool3d(kernel_size=2),  # Downsample
        )

        self.bottleneck = nn.Sequential(
            Conv2Plus1D(base_filters * 2, base_filters * 4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),  # Upsample

            Conv2Plus1D(base_filters * 2, base_filters),

            nn.Conv3d(base_filters, 1, kernel_size=1),  # Output: 1 channel for binary mask
        )

    def forward(self, x):
        """
        Input: x shape (B, C, T, H, W)
        Output: logits shape (B, 1, T, H, W)
        """
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = x.mean(dim=2)
        return x.squeeze(1)

class FireNet3DCNNSplit(nn.Module):
    def __init__(self, in_channels, env_filters=32, fire_filters=8, spatial_dilation=1, temporal_dilation=1):
        super().__init__()

        self.env_encoder = nn.Sequential(
            Conv2Plus1D(in_channels - 1, env_filters),
            Conv2Plus1D(env_filters,
                        env_filters * 2,
                        spatial_dilation=spatial_dilation,
                        temporal_dilation=temporal_dilation),
            nn.MaxPool3d(kernel_size=2),
        )

        self.fire_encoder = nn.Sequential(
            Conv2Plus1D(1, fire_filters, mid_planes=1),
            Conv2Plus1D(fire_filters,
                        fire_filters * 2,
                        mid_planes=1,
                        spatial_dilation=spatial_dilation,
                        temporal_dilation=temporal_dilation),
            nn.MaxPool3d(kernel_size=2),
        )

        self.bottleneck = nn.Sequential(
            Conv2Plus1D((env_filters + fire_filters) * 2,
                        (env_filters + fire_filters) * 4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((env_filters + fire_filters) * 4,
                               (env_filters + fire_filters) * 2,
                               kernel_size=2, stride=2), # Upsample

            Conv2Plus1D((env_filters + fire_filters) * 2, env_filters + fire_filters),

            nn.Conv3d(env_filters + fire_filters, 1, kernel_size=1), # Output: 1 channel for binary mask
        )

    def forward(self, x):
        """
        Input: x shape (B, C, T, H, W)
        Output: logits shape (B, 1, T, H, W)
        """
        x_env = x[:, :-1, ...]  
        x_fire = x[:, -1, ...].unsqueeze(1)
        x_env = self.env_encoder(x_env)
        x_fire = self.fire_encoder(x_fire)
        x_combined = torch.cat([x_env, x_fire], dim=1)
        z = self.bottleneck(x_combined)
        out = self.decoder(z)
        out = out.mean(dim=2)
        return out.squeeze(1)


class FireNet3DCNNSplitCBAM(nn.Module):
    def __init__(self, in_channels, env_filters=32, fire_filters=8):
        super().__init__()

        self.env_encoder = nn.Sequential(
            Conv2Plus1D(in_channels - 1, env_filters),
            CBAM3D(env_filters),
            nn.Dropout3d(p=0.3),

            Conv2Plus1D(env_filters, env_filters * 2),
            CBAM3D(env_filters * 2),
            nn.Dropout3d(p=0.3),

            nn.MaxPool3d(kernel_size=2),
        )

        self.fire_encoder = nn.Sequential(
            Conv2Plus1D(1, fire_filters, mid_planes=1),
            CBAM3D(fire_filters, sam_only=True),
            Conv2Plus1D(fire_filters, fire_filters * 2, mid_planes=1),
            CBAM3D(fire_filters * 2, sam_only=True),
            nn.MaxPool3d(kernel_size=2),
        )

        self.bottleneck = nn.Sequential(
            Conv2Plus1D((env_filters + fire_filters) * 2,
                        (env_filters + fire_filters) * 4),
            CBAM3D((env_filters + fire_filters) * 4),
            nn.Dropout3d(p=0.5),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((env_filters + fire_filters) * 4,
                               (env_filters + fire_filters) * 2,
                               kernel_size=2, stride=2), # Upsample

            Conv2Plus1D((env_filters + fire_filters) * 2, env_filters + fire_filters),
            CBAM3D(env_filters + fire_filters),
            nn.Conv3d(env_filters + fire_filters, 1, kernel_size=1), # Output: 1 channel for binary mask
        )

    def forward(self, x):
        """
        Input: x shape (B, C, T, H, W)
        Output: logits shape (B, 1, T, H, W)
        """
        x_env = x[:, :-1, ...]  
        x_fire = x[:, -1, ...].unsqueeze(1)
        x_env = self.env_encoder(x_env)
        x_fire = self.fire_encoder(x_fire)
        x_combined = torch.cat([x_env, x_fire], dim=1)
        z = self.bottleneck(x_combined)
        out = self.decoder(z)
        out = out.mean(dim=2)
        return out.squeeze(1)



class FireNet3DCNNSplitASPP(nn.Module):
    def __init__(self, in_channels, env_filters=32, fire_filters=8):
        super().__init__()

        self.env_encoder = nn.Sequential(
            Conv2Plus1D(in_channels - 1, env_filters),
            Conv2Plus1D(env_filters,
                        env_filters * 2),
            nn.MaxPool3d(kernel_size=2),
        )

        self.fire_encoder = nn.Sequential(
            Conv2Plus1D(1, fire_filters, mid_planes=1),
            Conv2Plus1D(fire_filters,
                        fire_filters * 2,
                        mid_planes=1),
            nn.MaxPool3d(kernel_size=2),
        )

        self.bottleneck = ASPP3D(
            in_channels=(env_filters + fire_filters) * 2,
            out_channels=(env_filters + fire_filters) * 4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((env_filters + fire_filters) * 4,
                               (env_filters + fire_filters) * 2,
                               kernel_size=2, stride=2),  # Upsample

            Conv2Plus1D((env_filters + fire_filters) * 2, env_filters + fire_filters),

            nn.Conv3d(env_filters + fire_filters, 1, kernel_size=1),  # Output: 1 channel for binary mask
        )

    def forward(self, x):
        """
        Input: x shape (B, C, T, H, W)
        Output: logits shape (B, 1, T, H, W)
        """
        x_env = x[:, :-1, ...]
        x_fire = x[:, -1, ...].unsqueeze(1)
        x_env = self.env_encoder(x_env)
        x_fire = self.fire_encoder(x_fire)
        x_combined = torch.cat([x_env, x_fire], dim=1)
        z = self.bottleneck(x_combined)
        out = self.decoder(z)
        out = out[:, :, 0, :, :]
        return out.squeeze(1)
