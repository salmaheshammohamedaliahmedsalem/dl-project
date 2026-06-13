# train_2d_cmr_models.py
# Model-aligned 2D multi-view segmentation training with round-robin loading.

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

from .config_cmr import DATASET_CONFIGS_2D, SOURCE_ORDER, DEFAULT_SEED, DEFAULT_VAL_SPLIT
from .dataset_2d_cmr_models import MultiSourceRoundRobinLoader
from .model_2d_resunetpp_multihead import MultiHeadResUNetPP2D
from .losses_2d import OneVsRestCombinedLoss
from .postprocess_2d import masks_to_one_vs_rest, decode_with_rules


def get_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def dice_score(pred, target, num_classes):
    """Per-class Dice scores (including background)."""
    scores = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append(float((2 * inter + 1e-6) / (denom + 1e-6)))
    return scores


def mean_dice(per_class_scores, ignore_background=True):
    start = 1 if ignore_background else 0
    fg = per_class_scores[start:]
    return float(np.mean(fg)) if fg else 0.0


@torch.no_grad()
def validate(model, val_loader, criterion, device, dataset_configs, max_val_batches=None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    dice_records = {k: [] for k in dataset_configs.keys()}

    for batch_idx, (images, masks, source) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        if max_val_batches and batch_idx >= max_val_batches:
            break
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images, source)
        num_classes = dataset_configs[source]["num_classes"]
        target = masks_to_one_vs_rest(masks, num_classes)

        loss_cfg = dataset_configs[source].get("loss", {})
        pos_weight = loss_cfg.get("pos_weight_fg")
        class_weights = loss_cfg.get("class_weights_fg")
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight else None
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None

        loss = criterion(logits, target, pos_weight=pos_weight, class_weights=class_weights)
        total_loss += loss.item()
        total_count += 1

        preds = decode_with_rules(logits, dataset_configs[source]["postprocess_rules"])
        ds = dice_score(preds, masks, num_classes)
        dice_records[source].append(ds)

    dice_map = {}
    for source, records in dice_records.items():
        if not records:
            dice_map[source] = 0.0
            continue
        per_class = [sum(vals) / len(vals) for vals in zip(*records)]
        dice_map[source] = mean_dice(per_class, ignore_background=True)

    return total_loss / max(total_count, 1), dice_map


def set_encoder_trainable(model, trainable):
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/model_2d")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--reset-lr-on-resume", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_snap = vars(args).copy()
    config_snap["device"] = str(device)
    with open(ckpt_dir / "config_snapshot.json", "w") as f:
        json.dump(config_snap, f, indent=2)

    dataset_configs = DATASET_CONFIGS_2D

    print("Building training loader...")
    train_loader = MultiSourceRoundRobinLoader(
        dataset_configs, "train", args.batch_size, args.image_size,
        args.val_split, args.num_workers, args.seed, args.use_weighted_sampler,
    )

    print("Building validation loader...")
    val_loader = MultiSourceRoundRobinLoader(
        dataset_configs, "val", args.batch_size, args.image_size,
        args.val_split, args.num_workers, args.seed, False,
    )

    num_classes_by_source = {s: cfg["num_classes"] for s, cfg in dataset_configs.items()}
    model = MultiHeadResUNetPP2D(
        in_channels=3,
        source_order=SOURCE_ORDER,
        num_classes_by_source=num_classes_by_source,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    criterion = OneVsRestCombinedLoss(
        bce_weight=args.bce_weight, dice_weight=args.dice_weight,
        focal_weight=args.focal_weight, focal_gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_mean_dice = -1.0
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    if args.warm_start_checkpoint:
        checkpoint = torch.load(args.warm_start_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        best_mean_dice = float(checkpoint.get("best_mean_dice", best_mean_dice))
        print(f"Warm-started model from {args.warm_start_checkpoint} (best Dice={best_mean_dice:.4f})")

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.reset_lr_on_resume:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print(f"Reset optimizer LR to {args.lr:.2e} after resume.")
        best_mean_dice = float(checkpoint.get("best_mean_dice", best_mean_dice))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch} (best Dice={best_mean_dice:.4f})")

    existing_log = ckpt_dir / "train_log.csv"
    if existing_log.exists() and args.resume_checkpoint:
        history = pd.read_csv(existing_log).to_dict("records")

    run_start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.max_minutes > 0 and (time.time() - run_start_time) >= args.max_minutes * 60.0:
            print(f"Reached max runtime of {args.max_minutes:.1f} minutes before epoch {epoch}.")
            break
        t0 = time.time()
        model.train()
        freeze_encoder = args.freeze_encoder_epochs > 0 and epoch < start_epoch + args.freeze_encoder_epochs
        set_encoder_trainable(model, not freeze_encoder)
        if freeze_encoder:
            print(f"Epoch {epoch}: encoder frozen; training decoder heads only.")
        train_losses = []

        for batch_idx, (images, masks, source) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images, source)
            num_classes = dataset_configs[source]["num_classes"]
            target = masks_to_one_vs_rest(masks, num_classes)

            loss_cfg = dataset_configs[source].get("loss", {})
            pw = loss_cfg.get("pos_weight_fg")
            cw = loss_cfg.get("class_weights_fg")
            pw = torch.tensor(pw, dtype=torch.float32, device=device) if pw else None
            cw = torch.tensor(cw, dtype=torch.float32, device=device) if cw else None

            loss = criterion(logits, target, pos_weight=pw, class_weights=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, dice_map = validate(
            model,
            val_loader,
            criterion,
            device,
            dataset_configs,
            max_val_batches=args.max_val_batches or None,
        )
        mean_val_dice = float(np.mean(list(dice_map.values()))) if dice_map else 0.0
        scheduler.step(mean_val_dice)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "mean_val_dice": mean_val_dice, "lr": lr, "time_s": elapsed,
        }
        for s, d in dice_map.items():
            row[f"dice_{s}"] = d
        history.append(row)

        pd.DataFrame(history).to_csv(ckpt_dir / "train_log.csv", index=False)

        dice_str = " | ".join(f"{s}={d:.4f}" for s, d in dice_map.items())
        print(f"Epoch {epoch:03d} | TrLoss {train_loss:.4f} | VlLoss {val_loss:.4f} | "
              f"MeanDice {mean_val_dice:.4f} | {dice_str} | LR {lr:.2e} | {elapsed:.0f}s")

        latest_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mean_dice": max(best_mean_dice, mean_val_dice),
            "num_classes_by_source": num_classes_by_source,
            "image_size": args.image_size,
        }
        torch.save(latest_checkpoint, ckpt_dir / "last_model.pth")

        if mean_val_dice > best_mean_dice:
            best_mean_dice = mean_val_dice
            epochs_without_improvement = 0
            latest_checkpoint["best_mean_dice"] = best_mean_dice
            torch.save(latest_checkpoint, ckpt_dir / "best_model.pth")
            print(f"  ✓ Saved best model (Dice={best_mean_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Done. Best mean Dice: {best_mean_dice:.4f}")


if __name__ == "__main__":
    main()
