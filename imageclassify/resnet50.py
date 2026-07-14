from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 3


def set_seed(seed: int = 42) -> None:
    """设置随机种子，增强实验可复现性。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 完全确定性可能略微降低训练速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境未检测到可用的 CUDA GPU。")
    return device


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """构建训练和验证/测试预处理。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def check_three_classes(dataset: datasets.ImageFolder, split_name: str) -> None:
    if len(dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"{split_name} 中检测到 {len(dataset.classes)} 个类别：{dataset.classes}；"
            f"本代码要求恰好 {NUM_CLASSES} 个类别。"
        )


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], datasets.ImageFolder]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"未找到训练目录：{train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"未找到验证目录：{val_dir}")

    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    check_three_classes(train_dataset, "训练集")
    check_three_classes(val_dataset, "验证集")

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "训练集与验证集的类别文件夹不一致。\n"
            f"训练集：{train_dataset.class_to_idx}\n"
            f"验证集：{val_dataset.class_to_idx}"
        )

    generator = torch.Generator()
    generator.manual_seed(42)

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    test_loader: Optional[DataLoader] = None
    if test_dir.is_dir():
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        check_three_classes(test_dataset, "测试集")
        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise ValueError(
                "训练集与测试集的类别文件夹不一致。\n"
                f"训练集：{train_dataset.class_to_idx}\n"
                f"测试集：{test_dataset.class_to_idx}"
            )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_args,
        )

    return train_loader, val_loader, test_loader, train_dataset


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """创建 ResNet50，并将分类头替换为三分类输出。"""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    根据训练集类别数量计算反频率权重。
    权重经过归一化，使平均值约为 1。
    """
    counts = torch.bincount(
        torch.tensor(dataset.targets, dtype=torch.long),
        minlength=len(dataset.classes),
    ).float()

    if torch.any(counts == 0):
        raise ValueError(f"存在没有样本的类别，类别计数为：{counts.tolist()}")

    weights = counts.sum() / (len(counts) * counts)
    return weights


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> None:
    indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    confusion_matrix += torch.bincount(
        indices.cpu(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confusion_matrix: torch.Tensor) -> Dict[str, object]:
    cm = confusion_matrix.float()
    total = cm.sum().clamp_min(1.0)
    correct = torch.diag(cm)

    accuracy = correct.sum() / total

    precision_per_class = correct / cm.sum(dim=0).clamp_min(1.0)
    recall_per_class = correct / cm.sum(dim=1).clamp_min(1.0)
    f1_per_class = (
        2.0
        * precision_per_class
        * recall_per_class
        / (precision_per_class + recall_per_class).clamp_min(1e-12)
    )

    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(precision_per_class.mean().item()),
        "macro_recall": float(recall_per_class.mean().item()),
        "macro_f1": float(f1_per_class.mean().item()),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": confusion_matrix.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
) -> Tuple[float, Dict[str, object]]:
    """
    optimizer 不为 None 时执行训练，否则执行验证/测试。
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        update_confusion_matrix(
            confusion_matrix,
            targets.detach(),
            predictions.detach(),
            NUM_CLASSES,
        )

    average_loss = total_loss / max(total_samples, 1)
    metrics = metrics_from_confusion_matrix(confusion_matrix)
    return average_loss, metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_accuracy: float,
    class_to_idx: Dict[str, int],
    image_size: int,
    dropout: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "class_to_idx": class_to_idx,
        "num_classes": NUM_CLASSES,
        "image_size": image_size,
        "dropout": dropout,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, int], int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到模型权重：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = int(checkpoint.get("num_classes", len(class_to_idx)))
    image_size = int(checkpoint.get("image_size", 224))
    dropout = float(checkpoint.get("dropout", 0.2))

    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"权重文件包含 {num_classes} 个类别，与当前三分类代码不一致。"
        )

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx, image_size


def print_metrics(
    split_name: str,
    loss: float,
    metrics: Dict[str, object],
    class_names: Iterable[str],
) -> None:
    print(
        f"{split_name:<6s} "
        f"loss={loss:.4f}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"macro-P={metrics['macro_precision']:.4f}  "
        f"macro-R={metrics['macro_recall']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )

    precision = metrics["precision_per_class"]
    recall = metrics["recall_per_class"]
    f1 = metrics["f1_per_class"]

    for index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"P={precision[index]:.4f}, "
            f"R={recall[index]:.4f}, "
            f"F1={f1[index]:.4f}"
        )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    class_names = train_dataset.classes
    print(f"使用设备：{device}")
    print(f"类别映射：{train_dataset.class_to_idx}")
    print(
        f"样本数：train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset) if test_loader is not None else 0}"
    )

    model = build_model(
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(train_dataset).to(device)
        print(f"类别权重：{class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_learning_rate,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_model.pth"
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )

        scheduler.step()
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d}/{args.epochs:03d}  "
            f"lr={current_lr:.3e}  time={elapsed:.1f}s"
        )
        print_metrics("train", train_loss, train_metrics, class_names)
        print_metrics("val", val_loss, val_metrics, class_names)

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        val_accuracy = float(val_metrics["accuracy"])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                class_to_idx=train_dataset.class_to_idx,
                image_size=args.image_size,
                dropout=args.dropout,
            )
            print(f"已保存最佳模型：{best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        with open(output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"\n验证集准确率连续 {args.early_stopping_patience} 个 epoch "
                "未提升，提前停止训练。"
            )
            break

    print(f"\n训练完成。最佳验证集准确率：{best_val_accuracy:.4f}")
    print(f"最佳模型：{best_checkpoint_path}")

    if test_loader is not None:
        best_model, _, _ = load_checkpoint_model(best_checkpoint_path, device)
        test_loss, test_metrics = run_epoch(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        print("\n最佳模型测试结果：")
        print_metrics("test", test_loss, test_metrics, class_names)

        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(test_metrics, file, ensure_ascii=False, indent=2)


def test(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp

    data_dir = Path(args.data_dir)
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"未找到测试集目录：{test_dir}")

    model, class_to_idx, image_size = load_checkpoint_model(
        Path(args.checkpoint),
        device,
    )
    _, eval_transform = build_transforms(image_size)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    check_three_classes(test_dataset, "测试集")

    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "测试集类别映射与模型权重不一致。\n"
            f"模型：{class_to_idx}\n"
            f"测试集：{test_dataset.class_to_idx}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
    )

    print(f"使用设备：{device}")
    print_metrics("test", test_loss, test_metrics, test_dataset.classes)
    print("混淆矩阵（行是真实类别，列是预测类别）：")
    print(torch.tensor(test_metrics["confusion_matrix"]))


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, class_to_idx, image_size = load_checkpoint_model(
        Path(args.checkpoint),
        device,
    )

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"未找到图片：{image_path}")

    _, eval_transform = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu()

    idx_to_class = {index: name for name, index in class_to_idx.items()}
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
    )

    print(f"图片：{image_path}")
    print(f"预测类别：{idx_to_class[int(sorted_indices[0])]}")
    print("各类别概率：")
    for probability, index in zip(sorted_probabilities, sorted_indices):
        print(
            f"  {idx_to_class[int(index)]}: "
            f"{float(probability) * 100:.2f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 ResNet50 的图像三分类程序"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data-dir", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, default="outputs")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--label-smoothing", type=float, default=0.1)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="设为 0 表示关闭早停",
    )
    train_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 ResNet50 主干，仅训练最后的分类头",
    )
    train_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不加载 ImageNet 预训练权重",
    )
    train_parser.add_argument(
        "--class-weights",
        action="store_true",
        help="根据各类别样本数量自动设置交叉熵类别权重",
    )
    train_parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="关闭 CUDA 自动混合精度",
    )
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test", help="在测试集上评估")
    test_parser.add_argument("--data-dir", type=str, required=True)
    test_parser.add_argument("--checkpoint", type=str, required=True)
    test_parser.add_argument("--batch-size", type=int, default=16)
    test_parser.add_argument("--num-workers", type=int, default=4)
    test_parser.add_argument("--device", type=str, default="auto")
    test_parser.add_argument("--disable-amp", action="store_true")
    test_parser.set_defaults(func=test)

    predict_parser = subparsers.add_parser("predict", help="预测单张图片")
    predict_parser.add_argument("--image", type=str, required=True)
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--device", type=str, default="auto")
    predict_parser.set_defaults(func=predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()