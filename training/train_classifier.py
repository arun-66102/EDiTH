"""
EfficientNet Training Script for DR Severity Classification.

Trains on the APTOS 2019 Blindness Detection dataset with:
- Ben Graham preprocessing + CLAHE
- Class-weighted focal loss (handles severe class imbalance)
- Data augmentation (flips, rotation, color jitter)
- Evaluation metrics: QWK, Sensitivity, Specificity, AUC-ROC, Confusion Matrix
- Fast training with Mixed Precision (AMP) & Early Stopping

Usage:
    python train_classifier.py --data-dir /path/to/aptos2019 --epochs 15 --batch-size 32

Expected APTOS directory structure:
    aptos2019/
    ├── train_images/
    │   ├── 000c1434d8d7.png
    │   └── ...
    └── train.csv  (columns: id_code, diagnosis)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Add parent directory to path for preprocessing imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# ============================================================
# Dataset
# ============================================================
class APTOSDataset(Dataset):
    """APTOS 2019 Diabetic Retinopathy dataset."""

    def __init__(self, df, image_dir, transform=None, target_size=300):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['id_code']}.png"

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Try jpg extension
            img_path = self.image_dir / f"{row['id_code']}.jpg"
            image = cv2.imread(str(img_path))

        if image is None:
            # Return a blank image if file not found
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        # Ben Graham preprocessing
        image = self._ben_graham(image, self.target_size)

        # CLAHE
        image = self._apply_clahe(image)

        # BGR → RGB → PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        if self.transform:
            pil_image = self.transform(pil_image)

        label = int(row["diagnosis"])
        return pil_image, label

    @staticmethod
    def _crop_fundus(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(image.shape[1] - x, w + 2 * pad)
        h = min(image.shape[0] - y, h + 2 * pad)
        return image[y : y + h, x : x + w]

    @staticmethod
    def _ben_graham(image, target_size):
        image = APTOSDataset._crop_fundus(image)
        image = cv2.resize(image, (target_size, target_size))
        image = image.astype(np.float32)
        blur = cv2.GaussianBlur(image, (0, 0), target_size / 30.0)
        image = cv2.addWeighted(image, 4, blur, -4, 128)
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# Focal Loss (handles class imbalance)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in DR classification.
    Down-weights easy examples and focuses on hard misclassified samples.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true, y_pred, y_probs, num_classes=5):
    """Compute all evaluation metrics."""
    metrics = {}

    # Quadratic Weighted Kappa (PRIMARY METRIC)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # Accuracy
    metrics["accuracy"] = (np.array(y_true) == np.array(y_pred)).mean()

    # Per-class metrics
    report = classification_report(
        y_true, y_pred, target_names=[f"Grade {i}" for i in range(num_classes)],
        output_dict=True, zero_division=0,
    )

    # Sensitivity (Recall) — macro average and per-class
    metrics["sensitivity_macro"] = report["macro avg"]["recall"]
    for i in range(num_classes):
        metrics[f"sensitivity_grade_{i}"] = report[f"Grade {i}"]["recall"]

    # Specificity — compute per-class from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f"specificity_grade_{i}"] = spec
        specificities.append(spec)
    metrics["specificity_macro"] = np.mean(specificities)

    # AUC-ROC (one-vs-rest)
    if y_probs is not None and len(y_probs) > 0:
        try:
            y_true_arr = np.array(y_true)
            y_probs_arr = np.array(y_probs)
            metrics["auc_roc_macro"] = roc_auc_score(
                y_true_arr, y_probs_arr, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auc_roc_macro"] = 0.0

    # Confusion Matrix
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100. * correct / total:.1f}%")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics["loss"] = avg_loss

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet for DR Classification")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to APTOS 2019 dataset directory")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0",
                        help="Architecture (e.g. efficientnet_b0, efficientnet_b3)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without QWK improvement)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model_name} | Image size: {args.target_size}x{args.target_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "train.csv")
    print(f"Total images: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")

    # Train/Val split (stratified)
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, df["diagnosis"]))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    image_dir = data_dir / "train_images"
    train_ds = APTOSDataset(train_df, image_dir, train_transform, args.target_size)
    val_ds = APTOSDataset(val_df, image_dir, val_transform, args.target_size)

    # Weighted sampler for class imbalance
    class_counts = train_df["diagnosis"].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_df["diagnosis"].values]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = timm.create_model(args.model_name, pretrained=True, num_classes=5)
    model.to(device)

    # Focal Loss with class weights
    alpha = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (not args.no_amp and device.type == "cuda") else None

    # Model save name
    weights_filename = f"{args.model_name}_dr.pth"
    best_weights_path = output_dir / weights_filename

    # Training
    best_qwk = 0.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print(f"Starting training ({args.model_name}) with AMP={'ON' if scaler else 'OFF'}...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}")
        print(f"  Val   QWK:  {val_metrics['qwk']:.4f}  (PRIMARY)")
        print(f"  Val   Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"  Val   Sensitivity:  {val_metrics['sensitivity_macro']:.4f}")
        print(f"  Val   Specificity:  {val_metrics['specificity_macro']:.4f}")
        if "auc_roc_macro" in val_metrics:
            print(f"  Val   AUC-ROC:      {val_metrics['auc_roc_macro']:.4f}")

        # Save best model (by QWK)
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  ★ New best QWK! Model saved to {best_weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n✋ Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
                break

    print("\n" + "=" * 60)
    print(f"Training complete. Best QWK: {best_qwk:.4f}")
    print(f"Best model saved at: {best_weights_path}")
    print("=" * 60)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    final_metrics = evaluate(model, val_loader, criterion, device)

    print(f"  QWK:         {final_metrics['qwk']:.4f}")
    print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {final_metrics['sensitivity_macro']:.4f}")
    print(f"  Specificity: {final_metrics['specificity_macro']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = np.array(final_metrics["confusion_matrix"])
    for i, row in enumerate(cm):
        print(f"    Grade {i}: {row}")

    # Copy best weights to the expected backend location
    target_weights = Path(__file__).parent.parent / "backend" / "models" / "weights"
    target_weights.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(best_weights_path, target_weights / weights_filename)
    print(f"\nWeights copied to {target_weights / weights_filename}")

    # For backward compatibility, also copy as efficientnet_b3_dr.pth if training b0
    if args.model_name == "efficientnet_b0":
        shutil.copy(best_weights_path, target_weights / "efficientnet_b3_dr.pth")


if __name__ == "__main__":
    main()
