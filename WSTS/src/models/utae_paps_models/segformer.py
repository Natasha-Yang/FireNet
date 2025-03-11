import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, TFSegformerModel


class SegformerEncoder(nn.Module):
    def __init__(self, pretrained_model = "nvidia/mit-b0"):
        super(SegformerEncoder, self).__init__()
        self.processor = SegformerImageProcessor.from_pretrained(pretrained_model)
        self.encoder = TFSegformerModel.from_pretrained(pretrained_model) 
    
    def forward(self, x):
        with torch.no_grad:
            out = self.encoder(x)
        return out