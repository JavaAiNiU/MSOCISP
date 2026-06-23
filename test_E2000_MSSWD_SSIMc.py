"""Evaluate image results with DeltaE2000, MSSWD and SSIMc.

Dependencies:
    pip install numpy scikit-image tqdm torch pyiqa

Metric directions:
    DeltaE2000: lower is better
    MSSWD:       lower is better
    SSIMc:       higher is better
"""

import glob
import os

import numpy as np
import pyiqa
import torch
from skimage import color, io
from tqdm import tqdm


# ================= Configuration =================
# Ground-truth PNG folder.
GT_FOLDER = "D:\\个人论文\\ISPresult\\MAIgt"

# One or more folders containing algorithm output PNG images.
ALGO_FOLDERS = [
            'trainZRR_testMAI\\best_psnr_model\\images_zrr2mai',
            'test_MAI\\best_psnr_model\\images_mai',
]

def ensure_rgb(image, image_path):
    """Return an H x W x 3 RGB image loaded by scikit-image."""
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image, but got shape {image.shape}: {image_path}"
        )
    return image


def calculate_delta_e2000(gt_path, dist_path):
    """Calculate mean CIEDE2000 exactly as in testE2000.py."""
    img_gt = ensure_rgb(io.imread(gt_path), gt_path)
    img_dist = ensure_rgb(io.imread(dist_path), dist_path)

    # Keep the original top-left alignment for a possible one-pixel size error.
    if img_gt.shape != img_dist.shape:
        min_h = min(img_gt.shape[0], img_dist.shape[0])
        min_w = min(img_gt.shape[1], img_dist.shape[1])
        img_gt = img_gt[:min_h, :min_w, :]
        img_dist = img_dist[:min_h, :min_w, :]

    # scikit-image converts uint8 RGB data to Lab using D65 and a 2-degree
    # observer by default. Standard CIEDE2000 uses kL=kC=kH=1.
    gt_lab = color.rgb2lab(img_gt, illuminant="D65", observer="2")
    dist_lab = color.rgb2lab(img_dist, illuminant="D65", observer="2")
    delta_e_map = color.deltaE_ciede2000(
        gt_lab,
        dist_lab,
        kL=1,
        kC=1,
        kH=1,
    )
    return float(np.mean(delta_e_map))


def create_pyiqa_metrics(device):
    """Create MSSWD and SSIMc once and reuse them for all image pairs."""
    print("Loading PyIQA metrics...")
    metrics = {
        "MSSWD": pyiqa.create_metric("msswd", device=device),
        "SSIMc": pyiqa.create_metric("ssimc", device=device),
    }
    for metric in metrics.values():
        metric.eval()
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    metrics = create_pyiqa_metrics(device)
    average_results = []

    print("Starting DeltaE2000, MSSWD and SSIMc evaluation...")
    print(f"Number of algorithm folders: {len(ALGO_FOLDERS)}")

    for algo_dir in ALGO_FOLDERS:
        algo_name = os.path.basename(os.path.normpath(algo_dir))
        print(f"\nProcessing algorithm: {algo_name}\nPath: {algo_dir}")

        # 只在内存中保留计算最终平均值所需的分数，不写入文件。
        metric_scores = {
            "DeltaE_00": [],
            "MSSWD": [],
            "SSIMc": [],
        }

        # Keep the original loading method: enumerate PNG files in each folder.
        dist_imgs = sorted(glob.glob(os.path.join(algo_dir, "*.png")))
        if not dist_imgs:
            print(f"Warning: no PNG images found in {algo_dir}")
            continue

        for dist_path in tqdm(dist_imgs, desc=algo_name):
            image_name_png = os.path.basename(dist_path)
            image_name_base = os.path.splitext(image_name_png)[0]

            # Keep the original matching rule: prediction xxx.png -> GT xxx.png.
            gt_path = os.path.join(GT_FOLDER, image_name_base + ".png")
            if not os.path.exists(gt_path):
                continue

            row_result = {
                "DeltaE_00": None,
                "MSSWD": None,
                "SSIMc": None,
            }

            # A failure in one metric does not prevent the other metrics from
            # being calculated and recorded for this image pair.
            try:
                row_result["DeltaE_00"] = calculate_delta_e2000(
                    gt_path, dist_path
                )
            except Exception as exc:
                print(f"\nDeltaE2000 error on {image_name_base}: {exc}")

            with torch.inference_mode():
                for metric_name, metric_model in metrics.items():
                    try:
                        # Preserve test_me.py's PyIQA input order: dist, reference.
                        score = metric_model(dist_path, gt_path)
                        row_result[metric_name] = float(score.item())
                    except Exception as exc:
                        print(
                            f"\n{metric_name} error on {image_name_base}: {exc}"
                        )

            for metric_name, score in row_result.items():
                if score is not None:
                    metric_scores[metric_name].append(score)

        # 每个算法文件夹只保留三项指标的最终平均值。
        if any(metric_scores.values()):
            average_results.append(
                {
                    "Algorithm": algo_name,
                    "DeltaE_00": (
                        float(np.mean(metric_scores["DeltaE_00"]))
                        if metric_scores["DeltaE_00"]
                        else None
                    ),
                    "MSSWD": (
                        float(np.mean(metric_scores["MSSWD"]))
                        if metric_scores["MSSWD"]
                        else None
                    ),
                    "SSIMc": (
                        float(np.mean(metric_scores["SSIMc"]))
                        if metric_scores["SSIMc"]
                        else None
                    ),
                }
            )

    if not average_results:
        print("No results were calculated. Check the configured folder paths.")
        return

    # DeltaE2000 越小越好，因此按其平均值升序打印。
    average_results.sort(
        key=lambda result: (
            result["DeltaE_00"] is None,
            result["DeltaE_00"] if result["DeltaE_00"] is not None else np.inf,
        )
    )

    def format_score(value):
        return "N/A" if value is None else f"{value:.6f}"

    # 仅在命令行输出每个算法的最终平均结果。
    print("\n" + "=" * 72)
    print("Final average results")
    print("=" * 72)
    print(f"{'Algorithm':<28}{'DeltaE_00':>14}{'MSSWD':>14}{'SSIMc':>14}")
    print("-" * 72)
    for result in average_results:
        print(
            f"{result['Algorithm']:<28}"
            f"{format_score(result['DeltaE_00']):>14}"
            f"{format_score(result['MSSWD']):>14}"
            f"{format_score(result['SSIMc']):>14}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
