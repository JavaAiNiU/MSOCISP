"""Evaluate MGCC on MCR with PSNR, SSIM and DeltaE2000."""

import argparse
import csv
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from skimage import color
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.MCRDataset import MCRDataset
from models.model_MSO_CCA_RAWLoss import MGCC
from utils.metrics import get_psnr_torch, get_ssim_torch


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a state dictionary.")
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)


def tensor_to_uint8_hwc(image):
    image = image.detach().float().clamp(0.0, 1.0)
    return (
        image.mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def calculate_delta_e2000(prediction, target):
    pred_rgb = tensor_to_uint8_hwc(prediction)
    target_rgb = tensor_to_uint8_hwc(target)
    if pred_rgb.shape != target_rgb.shape:
        min_h = min(pred_rgb.shape[0], target_rgb.shape[0])
        min_w = min(pred_rgb.shape[1], target_rgb.shape[1])
        pred_rgb = pred_rgb[:min_h, :min_w]
        target_rgb = target_rgb[:min_h, :min_w]

    pred_lab = color.rgb2lab(pred_rgb, illuminant="D65", observer="2")
    target_lab = color.rgb2lab(target_rgb, illuminant="D65", observer="2")
    delta_e = color.deltaE_ciede2000(
        target_lab, pred_lab, kL=1, kC=1, kH=1
    )
    return float(np.mean(delta_e))


def validate_metric(name, value, sample_name, logger):
    value = float(value)
    if np.isnan(value):
        message = f"{name} is NaN for sample: {sample_name}"
        logger.critical(message)
        raise FloatingPointError(message)
    return value


def metric_or_error(name, function, sample_name, logger):
    try:
        value = function()
    except Exception as error:
        logger.critical(
            "%s calculation failed for %s: %s",
            name,
            sample_name,
            error,
            exc_info=True,
        )
        raise
    return validate_metric(name, value, sample_name, logger)


def strict_mean(values, metric_name):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise RuntimeError(f"Cannot calculate the mean of empty {metric_name} values.")
    nan_indices = np.flatnonzero(np.isnan(values))
    if nan_indices.size:
        raise FloatingPointError(
            f"{metric_name} contains NaN at indices: {nan_indices.tolist()}"
        )
    return float(np.mean(values))


def build_loader(args, inference_device):
    dataset = MCRDataset(
        data_dir=args.data_dir,
        image_list_file=args.test_list_file,
        patch_size=None,
        split="test",
        transpose=False,
        h_flip=False,
        v_flip=False,
        ratio=args.use_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=inference_device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def save_prediction(prediction, input_path, index, images_dir):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    save_path = os.path.join(images_dir, f"{index:04d}_{stem}.png")
    Image.fromarray(tensor_to_uint8_hwc(prediction)).save(save_path)
    return save_path


@torch.inference_mode()
def evaluate(args):
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.result_dir, exist_ok=True)
    images_dir = os.path.join(args.result_dir, "images")
    logs_dir = os.path.join(args.result_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(images_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logs_dir, "test.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("test_mcr")

    dataset, loader = build_loader(args, device)
    model = MGCC().to(device)
    load_checkpoint(model, args.model, device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.model}")
    print(f"Test images: {len(dataset)}")
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.6f} M)")
    print(f"Trainable parameters: {trainable_params:,}")
    if hasattr(model, "w_r"):
        print(
            "Luminance weights: "
            f"w_r={model.w_r.item():.6f}, "
            f"w_g={model.w_g.item():.6f}, "
            f"w_b={model.w_b.item():.6f}"
        )

    rows = []
    sample_index = 0
    amp_enabled = args.amp and device.type == "cuda"
    progress = tqdm(loader, desc="Testing MCR", dynamic_ncols=True)

    for batch in progress:
        input_raw = batch["input_raw"].to(device, non_blocking=True)
        gt_rgb = batch["gt_rgb"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(input_raw)
            pred_rgb = output[0] if isinstance(output, (tuple, list)) else output
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_time = time.perf_counter() - start_time

        pred_rgb = pred_rgb.float().clamp(0.0, 1.0)
        gt_rgb = gt_rgb.float().clamp(0.0, 1.0)
        if pred_rgb.shape != gt_rgb.shape:
            raise ValueError(
                f"Prediction {tuple(pred_rgb.shape)} and target "
                f"{tuple(gt_rgb.shape)} have different shapes."
            )

        pred_255 = pred_rgb * 255.0
        gt_255 = gt_rgb * 255.0
        psnr_batch = get_psnr_torch(pred_255, gt_255)
        ssim_batch = get_ssim_torch(pred_255, gt_255)

        for batch_index in range(pred_rgb.shape[0]):
            prediction = pred_rgb[batch_index]
            target = gt_rgb[batch_index]
            input_path = batch["input_path"][batch_index]

            psnr_value = validate_metric(
                "PSNR", psnr_batch[batch_index].item(), input_path, logger
            )
            ssim_value = validate_metric(
                "SSIM", ssim_batch[batch_index].item(), input_path, logger
            )
            delta_e_value = metric_or_error(
                "DeltaE2000",
                lambda: calculate_delta_e2000(prediction, target),
                input_path,
                logger,
            )

            row = {
                "index": sample_index,
                "input_path": input_path,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "delta_e_00": delta_e_value,
                "inference_time_ms": 1000.0 * batch_time / pred_rgb.shape[0],
                "saved_path": "",
            }

            if args.save_images:
                row["saved_path"] = save_prediction(
                    prediction, input_path, sample_index, images_dir
                )

            rows.append(row)
            logger.info(json.dumps(row, ensure_ascii=False))
            sample_index += 1

        progress.set_postfix(
            PSNR=f"{strict_mean([item['psnr'] for item in rows], 'PSNR'):.3f}",
            SSIM=f"{strict_mean([item['ssim'] for item in rows], 'SSIM'):.4f}",
        )

    if not rows:
        raise RuntimeError("The MCR test loader produced no samples.")

    metric_names = (
        "psnr",
        "ssim",
        "delta_e_00",
        "inference_time_ms",
    )
    summary = {
        "device": str(device),
        "checkpoint": os.path.abspath(args.model),
        "num_images": len(rows),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    summary.update(
        {
            name: strict_mean([row[name] for row in rows], name)
            for name in metric_names
        }
    )

    csv_path = os.path.join(args.result_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.result_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print("MCR average results")
    print("=" * 62)
    print(f"PSNR       : {summary['psnr']:.6f} dB  (higher is better)")
    print(f"SSIM       : {summary['ssim']:.6f}     (higher is better)")
    print(f"DeltaE2000 : {summary['delta_e_00']:.6f}     (lower is better)")
    print(f"Inference  : {summary['inference_time_ms']:.3f} ms/image")
    print(f"Per-image metrics: {csv_path}")
    print(f"Summary: {summary_path}")
    print("=" * 62)
    logger.info("Average results: %s", json.dumps(summary, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model_MSO_CCA_RAWLoss on the MCR test set."
    )
    parser.add_argument("--data_dir", default="/mnt/data/zpc/data/MCR")
    parser.add_argument("--test_list_file", default="MCR_test_list.txt")
    parser.add_argument(
        "--model", required=True, help="Checkpoint saved by train_mcr.py."
    )
    parser.add_argument(
        "--result_dir",
        default=(
            "/mnt/data/zpc/MSOCO/MSOCO/Results/result_MSO_CCA/MCR/"
            "model_MSO_CCA_RAWLoss_Perceptual/test"
        ),
    )
    parser.add_argument("--gpu", type=int, default=0, help="Use -1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--no-ratio", dest="use_ratio", action="store_false"
    )
    parser.add_argument(
        "--no-save-images", dest="save_images", action="store_false"
    )
    parser.set_defaults(
        use_ratio=True,
        save_images=True,
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if not os.path.isfile(args.model):
        parser.error(f"checkpoint not found: {args.model}")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
