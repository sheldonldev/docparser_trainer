from pathlib import Path

import torch
from torch import optim
from transformers import AutoModelForSequenceClassification  # type: ignore
from util_common.path import ensure_parent


def _save_checkpoint(
    model, optimizer, epoch: int, accuracy: float, save_path: Path
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    ensure_parent(save_path)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def save_checkpoint(
    model, optimizer, epoch: int, accuracy: float, save_path: Path
):
    """
    Save the model and optimizer state at a given epoch to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        epoch (int): The current epoch number.
    """
    if not save_path.exists():
        _save_checkpoint(model, optimizer, epoch, accuracy, save_path)
    else:
        checkpoint = torch.load(save_path)
        old_accuracy = checkpoint['accuracy']
        if accuracy > old_accuracy:
            _save_checkpoint(model, optimizer, epoch, accuracy, save_path)


def load_checkpoint(save_path: Path):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=2
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epoch = 0
    if save_path.exists():
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    return model, optimizer, epoch
