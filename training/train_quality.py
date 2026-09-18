"""
MobileNetV2 Training Script for Fundus Image Quality Assessment.

Trains a binary classifier (Gradable / Ungradable) on labeled fundus images.

Usage:
    python train_quality.py --data-dir /path/to/iqa_dataset --epochs 15

Expected dataset structure:
    iqa_dataset/
    ├── gradable/
    │   ├── img001.png
    │   └── ...
    └── ungradable/
        ├── img100.png
        └── ...
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Fundus IQA")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to IQA dataset (with gradable/ and ungradable/ subdirs)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset — expects folder structure: data_dir/gradable/, data_dir/ungradable/
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Apply val transform to val split
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy().tolist())
                val_labels_list.extend(labels.cpu().numpy().tolist())

        scheduler.step()

        val_acc = (np.array(val_preds) == np.array(val_labels_list)).mean()
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss / train_total:.4f} | "
              f"Train Acc: {train_correct / train_total:.4f}")
        print(f"  Val   Loss: {val_loss / val_size:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = output_dir / "mobilenetv2_iqa.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ★ New best! Model saved to {save_path}")

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / "mobilenetv2_iqa.pth", map_location=device, weights_only=True))
    model.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy().tolist())
            val_labels_list.extend(labels.cpu().numpy().tolist())

    print("\nFinal Classification Report:")
    print(classification_report(val_labels_list, val_preds, target_names=full_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels_list, val_preds))

    # Copy weights
    target = Path(__file__).parent.parent / "backend" / "models" / "weights"
    target.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(output_dir / "mobilenetv2_iqa.pth", target / "mobilenetv2_iqa.pth")
    print(f"Weights copied to {target / 'mobilenetv2_iqa.pth'}")


if __name__ == "__main__":
    main()
