import torch
import torch.nn as nn
import numpy as np
from .cnn2plus1d.firenet3dcnn import FireNet3DCNN, FireNet3DCNNSplit, FireNet3DCNNSplitCBAM
from .prithvi.prithvi_dataloader import FireNetDataset
from segmentation_models_pytorch.losses import JaccardLoss, FocalLoss
from torchmetrics.classification import JaccardIndex
from monai.losses import HausdorffDTLoss, TverskyLoss
import torchmetrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml
import os
import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pos_weight = torch.tensor([1000.0], device=device)
weightedBCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = JaccardLoss(mode="binary", from_logits=True)
hausdorff_loss = HausdorffDTLoss(include_background=True, reduction='mean')
focal_loss = FocalLoss(mode='binary', alpha=0.99, gamma=3.0)
tversky_loss = TverskyLoss(alpha=0.45, beta=0.55, sigmoid=True)

def jaccard_hausdorff(preds, targets):
    """Combined Jaccard and Hausdorff loss.
    Args:
        preds (torch.Tensor): Predicted logits.
        targets (torch.Tensor): Target labels.
    Returns:
        torch.Tensor: Combined loss value."""
    # preds: raw logits → must apply sigmoid for Hausdorff
    return 0.7 * dice_loss(preds, targets) + 0.3 * hausdorff_loss(torch.sigmoid(preds), targets)

def jaccard_focal(preds, targets):
    """Combined Jaccard and Focal loss.
    Args:
        preds (torch.Tensor): Predicted logits.
        targets (torch.Tensor): Target labels.
    Returns:
        torch.Tensor: Combined loss value."""
    # preds: raw logits → must apply sigmoid for Focal
    return 0.5 * dice_loss(preds, targets) + 0.5 * focal_loss(torch.sigmoid(preds), targets)


def train_firenet(model, train_loader, val_loader, criterion, device, model_name, num_epochs=10,
                  lr_frozen=1e-3, lr_unfrozen=1e-4,
                  log_file="train_logs/training_log.txt", model_dir="saved_models",
                  unfreeze_epoch=None, early_stop_patience=5):
    """Train the FireNet model.
    Args:
        model (nn.Module): FireNet model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (callable): Loss function."""

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model.to(device)

    # If unfreeze_epoch is 0, immediately unfreeze encoder
    if unfreeze_epoch == 0 and hasattr(model, 'encoder'):
        print("Unfreezing encoder from the start.")
        for param in model.encoder.parameters():
            param.requires_grad = True
        model.encoder.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        wandb.log({"encoder_unfrozen_epoch": 1})
        with open(log_file, "a") as f:
            f.write(f"Encoder unfrozen at epoch 1\n")
    elif unfreeze_epoch != None:
        # Freeze encoder initially
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.encoder.eval()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=1e-5)
        scheduler = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_frozen, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Metrics
    train_f1 = torchmetrics.F1Score(task="binary").to(device)
    val_f1 = torchmetrics.F1Score(task="binary").to(device)
    train_iou = JaccardIndex(task="binary", num_classes=2).to(device)
    val_iou = JaccardIndex(task="binary", num_classes=2).to(device)

    best_val_f1 = float('-inf')
    early_stop_counter = 0

    #Train loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_f1.reset()
        train_iou.reset()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = preds.squeeze(1)
            binary_preds = (torch.sigmoid(preds) > 0.5).int()
            train_f1(binary_preds, y.int())
            train_iou(binary_preds, y.int())

        avg_train_loss = total_loss / len(train_loader)
        train_f1_score = train_f1.compute().item()
        train_iou_score = train_iou.compute().item()

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | F1: {train_f1_score:.4f} | IoU: {train_iou_score:.4f}")

        avg_val_loss, val_f1_score, val_iou_score = validate_firenet(model,
                                                                     val_loader,
                                                                     device,
                                                                     criterion,
                                                                     val_f1,
                                                                     val_iou,
                                                                     log_samples=True,
                                                                     epoch=epoch)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_f1": train_f1_score,
            "val_loss": avg_val_loss,
            "val_f1": val_f1_score,
            "train_iou": train_iou_score,
            "val_iou": val_iou_score,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            early_stop_counter = 0
            best_model_path = os.path.join(model_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), best_model_path)
            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_epoch"] = epoch + 1
            print(f"Model improved. Saved to {best_model_path}")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epoch(s)")

        # Unfreeze encoder if scheduled
        if epoch == unfreeze_epoch and unfreeze_epoch != 0:
            print(f"Unfreezing encoder and lowering LR at epoch {epoch+1}")
            wandb.log({"encoder_unfrozen_epoch": epoch + 1})
            with open(log_file, "a") as f:
                f.write(f"Encoder unfrozen at epoch {epoch+1}\n")

            for param in model.encoder.parameters():
                param.requires_grad = True
            model.encoder.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
            early_stop_counter = 0

        scheduler.step(val_f1_score)

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            wandb.log({"early_stopped_epoch": epoch + 1})
            break

        with open(log_file, "a") as f:
            f.write(
                f"Epoch {epoch+1}: "
                f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1_score:.4f}, Train IoU: {train_iou_score:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}, Val IoU: {val_iou_score:.4f}\n"
            )




def validate_firenet(model, val_loader, device, criterion, val_f1, val_iou, log_samples=False, epoch=None):
    """Validate the FireNet model.
    Args:
        model (nn.Module): FireNet model.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run the model on.
        criterion (callable): Loss function.
        log_samples (bool): Whether to log sample images."""
    
    model.eval()
    total_loss = 0.0
    val_f1.reset()
    val_iou.reset()
    log_done = False 

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).float()
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
            preds = preds.squeeze(1)
            binary_preds = (torch.sigmoid(preds) > 0.5).int()
            val_f1(binary_preds, y.int())
            val_iou(binary_preds, y.int())

    avg_loss = total_loss / len(val_loader)
    val_f1_score = val_f1.compute().item()
    val_iou_score = val_iou.compute().item()

    print(f"Validation Loss: {avg_loss:.4f} | F1: {val_f1_score:.4f} | IoU: {val_iou_score:.4f}")
    model.train()
    return avg_loss, val_f1_score, val_iou_score


def main():
    print(f"Using {device} device.\n")


    # Loading data ---------------------------------------------------------------------------------

    with open("prithvi/prithvi.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    dataset = FireNetDataset(**data_config)

    print(f"Dataset loaded from: {data_config['data_dir']}")
    dataset.setup()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    val_loader = dataset.val_dataloader()

    # Create model and load checkpoint -------------------------------------------------------------
    model = FireNet3DCNNSplit(40)
    checkpoint = torch.load('saved_models/3dcnn_split_64_jaccard_-1e-3.pt', map_location=device)
    model.load_state_dict(checkpoint)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    train_firenet(model, train_loader, val_loader, dice_loss, device, "3dcnn_split_64_jaccard_-1e-3", num_epochs = 50)

if __name__ == '__main__':
    wandb.init(project="firenet-training", name="3d-CNN-split-64_jaccard-1e-3")
    main()



