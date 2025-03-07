from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from firenet.config import MODELS_DIR, NDWS_PROCESSED_DATA_DIR
from firenet.dataset_NDWS import *

app = typer.Typer()

