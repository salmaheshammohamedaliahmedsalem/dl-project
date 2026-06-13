import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config_lge_task2 import (
    DATASET_CONFIGS_LGE_2D,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    DEFAULT_VAL_SPLIT,
    SOURCE_ORDER_LGE,
)
from .dataset_2d_cmr_models import MultiSourceRoundRobinLoader
from .losses_2d import OneVsRestCombinedLoss
from .model_2d_resunetpp_multihead import MultiHeadResUNetPP2D
from .postprocess_2d import decode_with_rules, masks_to_one_vs_rest


def get_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def dice_score(pred, target, num_classes):
    scores = []
    for class_id in range(num_classes):
        pred_bin = (pred == class_id).float()
        target_bin = (target == class_id).float()
        intersection = (pred_bin * target_bin).sum()
        denom = pred_bin.sum() + target_bin.sum()
        scores.append(float((2 * intersection + 1e-6) / (denom + 1e-6)))
    return scores


def mean_foreground_dice(per_class_scores):
    foreground = per_class_scores[1:]
    return float(np.mean(foreground)) if foreground else 0.0


def load_partial_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = len(state_dict) - len(compatible)
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(
        f"Warm-start loaded {len(compatible)} compatible tensors from {checkpoint_path}; "
        f"skipped {skipped} mismatched tensors.",
        flush=True,
    )


def build_pos_class_weights(loss_cfg, device):
    pos_weight = loss_cfg.get("pos_weight_fg")
    class_weights = loss_cfg.get("class_weights_fg")
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight else None
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    return pos_weight, class_weights


@torch.no_grad()
def validate(model, val_loader, criterion, device, max_val_batches=None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    dice_records = {source: [] for source in DATASET_CONFIGS_LGE_2D.keys()}

    iterator = tqdm(val_loader, desc="Validation", leave=False)
    for batch_idx, (images, masks, source) in enumerate(iterator):
        if max_val_batches and batch_idx >= max_val_batches:
            break
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images, source)
        cfg = DATASET_CONFIGS_LGE_2D[source]
        num_classes = cfg["num_classes"]
        target = masks_to_one_vs_rest(masks, num_classes)
        pos_weight, class_weights = build_pos_class_weights(cfg.get("loss", {}), device)
        loss = criterion(logits, target, pos_weight=pos_weight, class_weights=class_weights)
        total_loss += float(loss.item())
        total_count += 1

        preds = decode_with_rules(logits, cfg["postprocess_rules"])
        dice_records[source].append(dice_score(preds, masks, num_classes))

    dice_map = {}
    for source, records in dice_records.items():
        if not records:
            dice_map[source] = 0.0
            continue
        per_class = [sum(values) / len(values) for values in zip(*records)]
        dice_map[source] = mean_foreground_dice(per_class)
    return total_loss / max(total_count, 1), dice_map


def parse_args():
    parser = argparse.ArgumentParser(description="Quick Task 2 LGE 2D OVR training.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="OUTPUTS_ORGANIZED/outputs/task2_lge_quick_ovr2d")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--bce-weight", type=float, default=0.45)
    parser.add_argument("--dice-weight", type=float, default=0.55)
    parser.add_argument("--focal-weight", type=float, default=0.45)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--warm-start-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--reset-lr-on-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config_snapshot.json", "w") as file:
        json.dump({**vars(args), "device": str(device)}, file, indent=2)

    print("Building Task 2 LGE training loader...", flush=True)
    train_loader = MultiSourceRoundRobinLoader(
        DATASET_CONFIGS_LGE_2D,
        "train",
        args.batch_size,
        args.image_size,
        args.val_split,
        args.num_workers,
        args.seed,
        args.use_weighted_sampler,
    )
    print("Building Task 2 LGE validation loader...", flush=True)
    val_loader = MultiSourceRoundRobinLoader(
        DATASET_CONFIGS_LGE_2D,
        "val",
        args.batch_size,
        args.image_size,
        args.val_split,
        args.num_workers,
        args.seed,
        False,
    )

    num_classes_by_source = {
        source: cfg["num_classes"]
        for source, cfg in DATASET_CONFIGS_LGE_2D.items()
    }
    model = MultiHeadResUNetPP2D(
        in_channels=3,
        source_order=SOURCE_ORDER_LGE,
        num_classes_by_source=num_classes_by_source,
    ).to(device)
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model params: {total_params:,}", flush=True)

    criterion = OneVsRestCombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        focal_gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_mean_dice = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    history = []

    if args.warm_start_checkpoint:
        load_partial_state(model, args.warm_start_checkpoint, device)

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.reset_lr_on_resume:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print(f"Reset optimizer LR to {args.lr:.2e} after resume.", flush=True)
        best_mean_dice = float(checkpoint.get("best_mean_dice", best_mean_dice))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}", flush=True)

    log_path = checkpoint_dir / "train_log.csv"
    if log_path.exists() and args.resume_checkpoint:
        history = pd.read_csv(log_path).to_dict("records")

    run_start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.max_minutes > 0 and (time.time() - run_start_time) >= args.max_minutes * 60.0:
            print(f"Reached max runtime of {args.max_minutes:.1f} minutes before epoch {epoch}.", flush=True)
            break
        start_time = time.time()
        model.train()
        train_losses = []
        iterator = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
        for batch_idx, (images, masks, source) in enumerate(iterator):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, source)
            cfg = DATASET_CONFIGS_LGE_2D[source]
            target = masks_to_one_vs_rest(masks, cfg["num_classes"])
            pos_weight, class_weights = build_pos_class_weights(cfg.get("loss", {}), device)
            loss = criterion(logits, target, pos_weight=pos_weight, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            iterator.set_postfix(loss=f"{np.mean(train_losses):.4f}", source=source)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, dice_map = validate(
            model,
            val_loader,
            criterion,
            device,
            max_val_batches=args.max_val_batches or None,
        )
        mean_val_dice = float(np.mean(list(dice_map.values()))) if dice_map else 0.0
        scheduler.step(mean_val_dice)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_val_dice": mean_val_dice,
            "lr": lr,
            "time_s": elapsed,
        }
        for source, value in dice_map.items():
            row[f"dice_{source}"] = value
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        dice_str = " | ".join(f"{source}={value:.4f}" for source, value in dice_map.items())
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | Mean Dice {mean_val_dice:.4f} | {dice_str} | "
            f"LR {lr:.2e} | Time {elapsed:.1f}s",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mean_dice": max(best_mean_dice, mean_val_dice),
            "num_classes_by_source": num_classes_by_source,
            "source_order": SOURCE_ORDER_LGE,
            "image_size": args.image_size,
            "task": "task2_lge_quick_ovr2d",
        }
        torch.save(checkpoint, checkpoint_dir / "last_model.pth")
        if mean_val_dice > best_mean_dice:
            best_mean_dice = mean_val_dice
            epochs_without_improvement = 0
            checkpoint["best_mean_dice"] = best_mean_dice
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  Saved best Task 2 model (Dice={best_mean_dice:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{args.patience} epochs", flush=True)
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    print(f"Done. Best Task 2 mean Dice: {best_mean_dice:.4f}", flush=True)


if __name__ == "__main__":
    main()
