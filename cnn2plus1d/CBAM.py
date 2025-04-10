import torch 
import torch.nn as nn
import torch.nn.functional as F

class SAM3D(nn.Module):
    def __init__(self, bias=False):
        """Spatial Attention Module for 3D data.
        Args:
            bias (bool): If True, adds a learnable bias to the output.
        """
         # Initialize the parent class
        super(SAM3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=2, out_channels=1, kernel_size=7,
            padding=3, bias=bias)

    def forward(self, x):
        """Forward pass of the module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)."""
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, T, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)  # (B, 2, T, H, W)
        attn = torch.sigmoid(self.conv(concat))
        return attn * x

class CAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        """Channel Attention Module for 3D data.
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the MLP.
        """
         # Initialize the parent class
         # Define the MLP for channel attention
        super(CAM3D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True)
        )

    def forward(self, x):
        """Forward pass of the module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)."""
        b, c, t, h, w = x.size()
        max_pool = F.adaptive_max_pool3d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(b, c)
        attn = self.mlp(max_pool) + self.mlp(avg_pool)
        attn = torch.sigmoid(attn).view(b, c, 1, 1, 1)
        return attn * x

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, sam_only=False):
        """CBAM for 3D data.
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the MLP.
            sam_only (bool): If True, only use the Spatial Attention Module.
        """
         # Initialize the parent class
         # Define the modules
        super(CBAM3D, self).__init__()
        self.sam_only = sam_only
        if not self.sam_only:
            self.cam = CAM3D(channels, reduction)
        self.sam = SAM3D()

    def forward(self, x):
        """Forward pass of the module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)."""
        if self.sam_only:
            out = self.sam(x)
            return out + x

        out = self.cam(x)
        out = self.sam(out)
        return out + x  # Residual connection
