import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


"""
Wound segmentation trainer for leoscode/wound-segmentation-images.

The dataset is loaded with KaggleHub's pandas adapter from:
data_wound_seg/correspondence_table.xlsx

The script then uses the downloaded folders:
- data_wound_seg/train_images
- data_wound_seg/train_masks
- data_wound_seg/test_images
- data_wound_seg/test_masks

It trains a small U-Net on the training split, validates on a holdout split
from train_images, and reports final test Dice/IoU on test_images.
"""


DATASET_HANDLE = "leoscode/wound-segmentation-images"
CORRESPONDENCE_FILE = "data_wound_seg/correspondence_table.xlsx"
DEFAULT_DATA_SUBDIR = "data_wound_seg"


@dataclass
class Config:
    kaggle_dataset: str = DATASET_HANDLE
    correspondence_file: str = CORRESPONDENCE_FILE
    data_root: Optional[str] = None
    batch_size: int = 8
    image_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    save_path: str = "wound_segmentation_unet.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_dataset(cfg: Config) -> Path:
    root = Path(kagglehub.dataset_download(cfg.kaggle_dataset))
    data_root = root / DEFAULT_DATA_SUBDIR
    if not data_root.exists():
        raise FileNotFoundError(f"Expected dataset folder not found: {data_root}")
    return data_root


def load_correspondence_table(kaggle_dataset: str, file_path: str) -> pd.DataFrame:
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        kaggle_dataset,
        file_path,
    )


def build_split_frame(data_root: Path, split_name: str, mapping_df: pd.DataFrame) -> pd.DataFrame:
    image_dir = data_root / f"{split_name}_images"
    mask_dir = data_root / f"{split_name}_masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing split folders: {image_dir} or {mask_dir}")

    new_id_to_origin = dict(zip(mapping_df["new_id"].astype(str), mapping_df["origin_id"].astype(str)))

    rows = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "new_id": image_path.name,
                "origin_id": new_id_to_origin.get(image_path.name, ""),
            }
        )

    if not rows:
        raise ValueError(f"No image-mask pairs found in {split_name} split")

    return pd.DataFrame(rows)


def make_image_transform(train: bool, image_size: int) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ]
    )


class WoundSegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_transform = make_image_transform(train=train, image_size=image_size)
        self.mask_transform = make_mask_transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: Optional[List[int]] = None):
        super().__init__()
        features = features or [32, 64, 128, 256]

        self.down1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(features[2], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(features[3], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(features[2], features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv1 = DoubleConv(features[1], features[0])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool1(skip1))
        skip3 = self.down3(self.pool2(skip2))
        bottleneck = self.bottom(self.pool3(skip3))

        x = self.up3(bottleneck)
        x = torch.cat((x, skip3), dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.conv1(x)
        return self.final(x)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    score = (2.0 * intersection + eps) / (probs.sum(dim=1) + targets.sum(dim=1) + eps)
    return 1.0 - score.mean()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    score = (2.0 * intersection + 1e-6) / (preds.sum(dim=1) + targets.sum(dim=1) + 1e-6)
    return float(score.mean().item())


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    score = (intersection + 1e-6) / (union + 1e-6)
    return float(score.mean().item())


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
        loss = bce + dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
        loss = bce + dice_loss(logits, masks)

        total_loss += loss.item() * images.size(0)
        dice_scores.append(dice_score(logits, masks))
        iou_scores.append(iou_score(logits, masks))

    return total_loss / len(loader.dataset), float(np.mean(dice_scores)), float(np.mean(iou_scores))


def build_dataloaders(cfg: Config, data_root: Path, mapping_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_frame = build_split_frame(data_root, "train", mapping_df)
    test_frame = build_split_frame(data_root, "test", mapping_df)

    train_frame, val_frame = train_test_split(train_frame, test_size=0.15, random_state=cfg.seed, shuffle=True)
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    train_dataset = WoundSegmentationDataset(train_frame, image_size=cfg.image_size, train=True)
    val_dataset = WoundSegmentationDataset(val_frame, image_size=cfg.image_size, train=False)
    test_dataset = WoundSegmentationDataset(test_frame, image_size=cfg.image_size, train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def run_training(cfg: Config) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = download_dataset(cfg)
    mapping_df = load_correspondence_table(cfg.kaggle_dataset, cfg.correspondence_file)
    print(f"Loaded correspondence table with {len(mapping_df)} rows")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, data_root, mapping_df)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_dice = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"Test  loss={test_loss:.4f} | Dice={test_dice:.4f} | IoU={test_iou:.4f}")

    save_path = Path(cfg.save_path)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path.resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Wound segmentation trainer for Kaggle wound dataset")
    parser.add_argument("--kaggle_dataset", type=str, default=DATASET_HANDLE)
    parser.add_argument("--correspondence_file", type=str, default=CORRESPONDENCE_FILE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="wound_segmentation_unet.pth")
    args = parser.parse_args()

    return Config(
        kaggle_dataset=args.kaggle_dataset,
        correspondence_file=args.correspondence_file,
        batch_size=args.batch_size,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    run_training(parse_args())