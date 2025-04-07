from typing import Any

import torch
import torch.nn as nn

from .BaseModel import BaseModel

class Segformer(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        *args: Any,
        **kwargs: Any
    ):
        super(Segformer, self).__init__()
        