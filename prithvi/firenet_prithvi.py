import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LinearMappingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # Use Conv3d to map 39 input channels to 6 output channels
        self.linear = nn.Sequential(
            nn.Conv3d(39, 5, kernel_size=1),
            nn.BatchNorm3d(5) # normalize across batch and spatial dims
        )
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.linear(x)


class PrithviEncoder(nn.Module):
    def __init__(self, model, device, num_frames=3):
        super().__init__()
        self.model = model
        self.device = device
        self.num_frames = num_frames

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        returns: list of 3 feature maps: [(B, embedding_dim, H, W), ...]
        """
        x = x.to(self.device)

        with torch.no_grad():
            hidden_states = self.model.forward_features(x)  # ← name it hidden_states

        processed_features = []
        for feature in hidden_states[-3:]:
            # Remove CLS token
            feature = feature[:, 1:, :]  # (B, num_patches, dim)
            B, N, C = feature.shape

            H = int(np.sqrt(N // self.num_frames))
            assert H * H * self.num_frames == N, "Patch dimensions don't match expected frame count."

            # Reshape: (B, num_frames, H, H, C) → (B, C, T, H, W)
            feature = feature.view(B, self.num_frames, H, H, C)
            feature = feature[:, -1, :, :, :]  # Only keep last frame → (B, H, H, C)
            feature = feature.permute(0, 3, 1, 2)  # → (B, C, H, H)
            processed_features.append(feature)

        return processed_features



class CNNDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Added sigmoid for segmentation
        )

    def forward(self, x):
        """
        x: (B, embedding_dim, T, H, W)
        output: (B, 224, 224) with pixel-wise probabilities
        """
        B, C, T, H, W = x.shape

        # Collapse over time dimension
        x = x.mean(dim=2)        # (B, C, H, W)

        x = self.decoder(x)      # (B, 1, 224, 224)
        x = x.squeeze(1)         # (B, 224, 224)
        return x



class SegformerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)
        hidden_states = self.proj(hidden_states)                  # (B, H*W, decoder_dim)
        return hidden_states


class SegformerDecoderHead(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: a configuration object with attributes:
                - num_encoder_stages: number of hidden states (stages) from the encoder to use.
                - hidden_sizes: list of encoder hidden dimensions for each stage.
                - decoder_hidden_size: target hidden dimension in the decoder.
                - classifier_dropout_prob: dropout probability.
                - num_labels: number of segmentation classes.
                - (optional) decoder_output_size: desired final output spatial size (H_out, W_out).
        """
        super().__init__()
        assert len(config.decoder_hidden_sizes) == config.num_encoder_stages
        self.mlps = nn.ModuleList([
            SegformerMLP(input_dim, output_dim)
            for input_dim, output_dim in zip(config.hidden_sizes, config.decoder_hidden_sizes)
        ])

        total_decoder_channels = sum(config.decoder_hidden_sizes)
        self.fuse_conv = nn.Conv2d(
            in_channels=total_decoder_channels,
            out_channels=config.decoder_hidden_sizes[-1],  # or any target
            kernel_size=1,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_sizes[-1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_sizes[-1], config.num_labels, kernel_size=1)
        self.config = config


    def forward(self, encoder_hidden_states: list) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: List of tensors, each of shape (B, embedding_dim_i, H_i, W_i).
        Returns:
            logits: Segmentation logits of shape (B, num_labels, H_out, W_out)
        """
        batch_size = encoder_hidden_states[0].size(0)
        target_size = encoder_hidden_states[0].shape[2:]  # Highest-res stage

        processed_states = []
        for state, mlp, out_dim in zip(encoder_hidden_states, self.mlps, self.config.decoder_hidden_sizes):
            H, W = state.shape[2], state.shape[3]
            state = mlp(state)  # (B, H*W, decoder_dim)
            state = state.transpose(1, 2).view(batch_size, out_dim, H, W)  # (B, decoder_dim, H, W)
            state = F.interpolate(state, size=target_size, mode="bilinear", align_corners=False)
            processed_states.append(state)

        x = torch.cat(processed_states, dim=1)  # (B, sum(decoder_dims), H, W)
        x = self.fuse_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        if hasattr(self.config, "decoder_output_size"):
            x = F.interpolate(x, size=self.config.decoder_output_size, mode="bilinear", align_corners=False)

        logits = self.classifier(x)
        return logits



class FireNet(nn.Module):
    def __init__(self, linear_map, encoder, decoder):
        super().__init__()
        self.linear_map = linear_map
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_env = x[:, :39, ...]  
        x_fire = x[:, 39:, ...]
        x_env_mapped = self.linear_map(x_env) # (B, 6, T, H, W)
        x_combined = torch.cat([x_env_mapped, x_fire], dim=1)
        z = self.encoder(x_combined) # (B, embedding_dim, T, H', W')
        out = self.decoder(z)
        return out



