import torch 
import torch.nn as nn
import torch.nn.functional as F

class SAM3D(nn.Module):
    def __init__(self, bias=False):
        super(SAM3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7,
                              padding=3, bias=bias)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, T, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)  # (B, 2, T, H, W)
        attn = torch.sigmoid(self.conv(concat))
        return attn * x

class CAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAM3D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        max_pool = F.adaptive_max_pool3d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(b, c)
        attn = self.mlp(max_pool) + self.mlp(avg_pool)
        attn = torch.sigmoid(attn).view(b, c, 1, 1, 1)
        return attn * x

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, sam_only=False):
        super(CBAM3D, self).__init__()
        self.sam_only = sam_only
        if not self.sam_only:
            self.cam = CAM3D(channels, reduction)
        self.sam = SAM3D()

    def forward(self, x):
        if self.sam_only:
            out = self.sam(x)
            return out + x

        out = self.cam(x)
        out = self.sam(out)
        return out + x  # Residual connection
