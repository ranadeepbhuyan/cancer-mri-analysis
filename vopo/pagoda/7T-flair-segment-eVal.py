#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PAGODA-Net: 7T FLAIR Tumor Segmentation Accuracy & Metrics Calculator     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                            ║
║  This script performs end-to-end evaluation of tumor segmentation on       ║
║  the projected 7T FLAIR image:                                             ║
║                                                                            ║
║    1. Extracts ground truth tumor masks from the color overlay image       ║
║    2. Segments tumors from the FLAIR image using anomaly detection         ║
║    3. Computes per-tumor and overall metrics:                              ║
║       • Accuracy, Precision, Recall, Specificity                           ║
║       • Dice Score (F1), IoU (Jaccard), F2 Score                           ║
║       • Hausdorff Distance (HD), HD95, Normalized Surface Dice (NSD)       ║
║    4. Performs instance-level matching (per-tumor IoU-based)                ║
║    5. Generates a visual confusion report and CSV summary                  ║
║                                                                            ║
║  Usage:                                                                    ║
║    python flair_segmentation_eval.py                                       ║
║    python flair_segmentation_eval.py --flair path.png --overlay path.png   ║
║    python flair_segmentation_eval.py --flair path.png --gt-mask mask.png   ║
║                                                                            ║
║  Dependencies: numpy, scipy, Pillow                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import csv
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    label as ndimage_label,
    distance_transform_edt,
)
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default file paths (used when running without CLI arguments)
DEFAULT_FLAIR_PATH = "7T8_FLAIR_realistic.png"
DEFAULT_OVERLAY_PATH = "7T8_overlay_realistic.png"
DEFAULT_7T_TUMOR_PATH = "7T8_tumors_realistic.png"
DEFAULT_OUTPUT_DIR = "flair_eval_results"

EPS = 1e-8

# Colors for confusion overlay
COLOR_TP = (0, 200, 0)       # Green  — correctly detected tumor
COLOR_FP = (255, 50, 50)     # Red    — false alarm
COLOR_FN = (50, 100, 255)    # Blue   — missed tumor
COLOR_TN = (0, 0, 0)         # Black  — correct background


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationMetrics:
    """All metrics for a single binary segmentation evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    dice: float = 0.0
    iou: float = 0.0
    f2_score: float = 0.0
    hausdorff: float = -1.0
    hausdorff_95: float = -1.0
    nsd: float = -1.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    pred_volume: int = 0
    gt_volume: int = 0


@dataclass
class InstanceMatch:
    """A matched pair of GT and predicted tumor instances."""
    gt_id: int = 0
    pred_id: int = 0
    iou: float = 0.0
    dice: float = 0.0
    gt_area: int = 0
    pred_area: int = 0
    gt_centroid: Tuple[float, float] = (0, 0)
    pred_centroid: Tuple[float, float] = (0, 0)


@dataclass
class FullEvaluation:
    """Complete evaluation results."""
    overall: SegmentationMetrics = field(default_factory=SegmentationMetrics)
    per_instance: List[InstanceMatch] = field(default_factory=list)
    n_gt_tumors: int = 0
    n_pred_tumors: int = 0
    n_matched: int = 0
    n_false_positive: int = 0
    n_false_negative: int = 0
    mean_instance_dice: float = 0.0
    detection_rate: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, gt: np.ndarray, nsd_tol: float = 2.0) -> SegmentationMetrics:
    """
    Compute all binary segmentation metrics.

    Parameters
    ----------
    pred : np.ndarray (bool)
        Predicted binary tumor mask.
    gt : np.ndarray (bool)
        Ground truth binary tumor mask.
    nsd_tol : float
        Tolerance in pixels for Normalized Surface Dice.

    Returns
    -------
    SegmentationMetrics
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    total = tp + fp + fn + tn

    m = SegmentationMetrics(
        tp=tp, fp=fp, fn=fn, tn=tn,
        pred_volume=int(pred.sum()),
        gt_volume=int(gt.sum()),
        accuracy=(tp + tn) / (total + EPS),
        precision=tp / (tp + fp + EPS),
        recall=tp / (tp + fn + EPS),
        specificity=tn / (tn + fp + EPS),
        dice=(2.0 * tp) / (2.0 * tp + fp + fn + EPS),
        iou=tp / (tp + fp + fn + EPS),
    )

    # F2 score (recall-weighted)
    beta = 2.0
    m.f2_score = ((1 + beta**2) * m.precision * m.recall) / (
        beta**2 * m.precision + m.recall + EPS
    )

    # Distance-based metrics
    if tp > 0:
        m.hausdorff, m.hausdorff_95 = _hausdorff(pred, gt)
        m.nsd = _nsd(pred, gt, nsd_tol)

    return m


def _hausdorff(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Compute Hausdorff Distance and HD95."""
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)

    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return -1.0, -1.0

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d1 = dt_gt[pred_surf]
    d2 = dt_pred[gt_surf]
    all_d = np.concatenate([d1, d2])

    return float(np.max(all_d)), float(np.percentile(all_d, 95))


def _nsd(pred: np.ndarray, gt: np.ndarray, tol: float) -> float:
    """Compute Normalized Surface Dice at given tolerance."""
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)

    n_p = pred_surf.sum()
    n_g = gt_surf.sum()
    if n_p == 0 or n_g == 0:
        return 0.0

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    p_ok = np.sum(dt_gt[pred_surf] <= tol)
    g_ok = np.sum(dt_pred[gt_surf] <= tol)

    return float((p_ok + g_ok) / (n_p + n_g))


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH EXTRACTION (from color overlay)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_gt_from_overlay(overlay_path: str, original_7t_path: Optional[str] = None) -> np.ndarray:
    """
    Extract ground truth tumor mask from the color overlay image.

    The overlay has colored regions (high saturation) where tumors are.
    We detect these by looking for pixels with high color saturation
    relative to the original grayscale 7T image.

    Parameters
    ----------
    overlay_path : str
        Path to the color-coded tumor overlay image.
    original_7t_path : str, optional
        Path to original 7T image (used to subtract base intensity).

    Returns
    -------
    np.ndarray (bool)
        Binary ground truth tumor mask.
    """
    print("  Extracting ground truth from overlay...")

    overlay = np.array(Image.open(overlay_path).convert("RGB")).astype(float)
    h, w = overlay.shape[:2]
    r, g, b = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]

    # Colored tumor regions have high saturation
    # Convert to HSV-like saturation
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = (max_rgb - min_rgb) / (max_rgb + EPS)
    brightness = max_rgb / 255.0

    # Tumor overlay regions: high saturation AND reasonable brightness
    gt_mask = (saturation > 0.25) & (brightness > 0.1)

    # Clean up
    gt_mask = binary_fill_holes(gt_mask)
    gt_mask = binary_erosion(gt_mask, iterations=1)
    gt_mask = binary_dilation(gt_mask, iterations=1)

    # Remove very small fragments
    labeled, n = ndimage_label(gt_mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < 15:
            gt_mask[labeled == i] = False

    print(f"    GT tumor pixels: {gt_mask.sum()}")
    print(f"    GT tumor regions: {ndimage_label(gt_mask)[1]}")

    return gt_mask


def extract_gt_from_difference(tumor_7t_path: str, original_7t_path: str) -> np.ndarray:
    """
    Extract ground truth by differencing the tumor-projected 7T and original 7T.

    Parameters
    ----------
    tumor_7t_path : str
        Path to the 7T image WITH projected tumors.
    original_7t_path : str
        Path to the ORIGINAL 7T image (no tumors).

    Returns
    -------
    np.ndarray (bool)
        Binary ground truth tumor mask.
    """
    print("  Extracting ground truth from image difference...")

    tumor = np.array(Image.open(tumor_7t_path).convert("L")).astype(float)
    original = np.array(Image.open(original_7t_path).convert("L")).astype(float)

    # Resize if needed
    if tumor.shape != original.shape:
        from PIL import Image as PILImage
        original_img = PILImage.open(original_7t_path).convert("L")
        original_img = original_img.resize((tumor.shape[1], tumor.shape[0]), PILImage.LANCZOS)
        original = np.array(original_img).astype(float)

    diff = np.abs(tumor - original)
    gt_mask = diff > 8  # threshold for meaningful difference

    gt_mask = binary_fill_holes(gt_mask)
    gt_mask = binary_erosion(gt_mask, iterations=1)
    gt_mask = binary_dilation(gt_mask, iterations=1)

    labeled, n = ndimage_label(gt_mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < 15:
            gt_mask[labeled == i] = False

    print(f"    GT tumor pixels: {gt_mask.sum()}")
    print(f"    GT tumor regions: {ndimage_label(gt_mask)[1]}")

    return gt_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  FLAIR TUMOR SEGMENTATION (prediction)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_flair_tumors(
    flair_path: str,
    min_area: int = 30,
    hyper_z_thresh: float = 2.0,
    hypo_z_thresh: float = -2.0,
) -> np.ndarray:
    """
    Segment tumors from a T2-FLAIR image using multi-scale anomaly detection.

    In FLAIR images:
      • Hyperintense tumors appear as BRIGHT spots (edema, gliosis)
      • Hypointense tumors appear as DARK spots (necrosis, hemorrhage)
      • Both are detected as deviations from local tissue mean

    Parameters
    ----------
    flair_path : str
        Path to the T2-FLAIR image.
    min_area : int
        Minimum tumor area in pixels to keep (filters noise).
    hyper_z_thresh : float
        Z-score threshold for hyperintense detection.
    hypo_z_thresh : float
        Z-score threshold for hypointense detection.

    Returns
    -------
    np.ndarray (bool)
        Binary predicted tumor mask.
    """
    print("  Segmenting tumors from FLAIR image...")

    img = Image.open(flair_path).convert("L")
    arr = np.array(img).astype(float)
    h, w = arr.shape

    # ── Brain mask ──
    brain = arr > 10
    brain = binary_fill_holes(brain)
    brain = binary_erosion(brain, iterations=3)
    brain = binary_dilation(brain, iterations=3)
    brain = binary_fill_holes(brain)
    lb, nb = ndimage_label(brain)
    if nb > 1:
        sizes = [np.sum(lb == i) for i in range(1, nb + 1)]
        brain = lb == (np.argmax(sizes) + 1)

    inner = binary_erosion(brain, iterations=6)

    # ── Multi-scale anomaly detection ──
    z_hyper = np.zeros_like(arr)
    z_hypo = np.zeros_like(arr)

    for sigma in [3, 5, 8, 12]:
        local_mean = gaussian_filter(arr, sigma=sigma)
        local_var = gaussian_filter((arr - local_mean) ** 2, sigma=sigma * 1.5)
        local_std = np.sqrt(local_var + EPS)

        z = (arr - local_mean) / (local_std + EPS)
        z_hyper = np.maximum(z_hyper, z)
        z_hypo = np.minimum(z_hypo, z)

    # Hyperintense (bright spots)
    hyper_mask = (z_hyper > hyper_z_thresh) & inner

    # Hypointense (dark spots) — exclude ventricles
    hypo_raw = (z_hypo < hypo_z_thresh) & inner & (arr > 5)
    # Filter out ventricles (large dark connected regions)
    hypo_labeled, n_hypo = ndimage_label(hypo_raw)
    hypo_mask = np.zeros_like(hypo_raw)
    for i in range(1, n_hypo + 1):
        comp = hypo_labeled == i
        if comp.sum() < 500:  # not a ventricle
            hypo_mask |= comp

    # Combine
    tumor_raw = hyper_mask | hypo_mask

    # Morphological cleanup
    tumor_raw = binary_dilation(tumor_raw, iterations=1)
    tumor_raw = binary_fill_holes(tumor_raw)
    tumor_raw = binary_erosion(tumor_raw, iterations=1)

    # Filter by minimum area
    labeled, n = ndimage_label(tumor_raw)
    pred_mask = np.zeros_like(tumor_raw)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_area:
            pred_mask |= comp

    print(f"    Predicted tumor pixels: {pred_mask.sum()}")
    print(f"    Predicted tumor regions: {ndimage_label(pred_mask)[1]}")

    return pred_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTANCE-LEVEL MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def match_instances(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1,
) -> Tuple[List[InstanceMatch], int, int]:
    """
    Match predicted and ground truth tumor instances using IoU.

    Parameters
    ----------
    pred : np.ndarray (bool)
        Predicted tumor mask.
    gt : np.ndarray (bool)
        Ground truth tumor mask.
    iou_threshold : float
        Minimum IoU for a valid match.

    Returns
    -------
    Tuple[List[InstanceMatch], int, int]
        (matched_pairs, num_false_positives, num_false_negatives)
    """
    pred_labeled, n_pred = ndimage_label(pred)
    gt_labeled, n_gt = ndimage_label(gt)

    print(f"  Matching instances: {n_gt} GT × {n_pred} Pred...")

    # IoU matrix
    iou_mat = np.zeros((n_gt, n_pred))
    for gi in range(1, n_gt + 1):
        gm = gt_labeled == gi
        for pi in range(1, n_pred + 1):
            pm = pred_labeled == pi
            inter = np.sum(gm & pm)
            union = np.sum(gm | pm)
            if union > 0:
                iou_mat[gi - 1, pi - 1] = inter / union

    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    matches = []

    iou_work = iou_mat.copy()
    while True:
        if iou_work.size == 0:
            break
        best = np.unravel_index(np.argmax(iou_work), iou_work.shape)
        if iou_work[best] < iou_threshold:
            break

        gi, pi = best
        gm = gt_labeled == (gi + 1)
        pm = pred_labeled == (pi + 1)

        tp = np.sum(gm & pm)
        dice = (2.0 * tp) / (gm.sum() + pm.sum() + EPS)

        g_ys, g_xs = np.where(gm)
        p_ys, p_xs = np.where(pm)

        matches.append(InstanceMatch(
            gt_id=gi + 1,
            pred_id=pi + 1,
            iou=float(iou_work[best]),
            dice=float(dice),
            gt_area=int(gm.sum()),
            pred_area=int(pm.sum()),
            gt_centroid=(float(g_ys.mean()), float(g_xs.mean())),
            pred_centroid=(float(p_ys.mean()), float(p_xs.mean())),
        ))

        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_work[gi, :] = 0
        iou_work[:, pi] = 0

    fp_count = n_pred - len(matched_pred)
    fn_count = n_gt - len(matched_gt)

    return matches, fp_count, fn_count


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_visual_report(
    flair_arr: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    evaluation: FullEvaluation,
    output_path: str,
):
    """
    Create a comprehensive visual evaluation report.

    Layout:
      Row 1: FLAIR | GT Overlay | Pred Overlay | Confusion Map
      Row 2: Metrics table
    """
    print("  Generating visual report...")

    h, w = flair_arr.shape[:2]
    if flair_arr.ndim == 2:
        flair_rgb = np.stack([flair_arr] * 3, axis=2)
    else:
        flair_rgb = flair_arr.copy()

    def get_font(sz):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
        except:
            return ImageFont.load_default()

    def get_font_regular(sz):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
        except:
            return ImageFont.load_default()

    # ── Build sub-images ──

    # 1. FLAIR original
    img_flair = flair_rgb.astype(np.uint8)

    # 2. GT overlay (blue)
    gt_overlay = flair_rgb.copy()
    gt_overlay[gt] = gt_overlay[gt] * 0.4 + np.array([50, 100, 255]) * 0.6
    gt_contour = binary_dilation(gt, iterations=2) & ~gt
    gt_overlay[gt_contour] = [50, 100, 255]
    gt_overlay = gt_overlay.astype(np.uint8)

    # 3. Pred overlay (green)
    pred_overlay = flair_rgb.copy()
    pred_overlay[pred] = pred_overlay[pred] * 0.4 + np.array([0, 220, 0]) * 0.6
    pred_contour = binary_dilation(pred, iterations=2) & ~pred
    pred_overlay[pred_contour] = [0, 220, 0]
    pred_overlay = pred_overlay.astype(np.uint8)

    # 4. Confusion map
    confusion = flair_rgb.copy() * 0.3
    tp_mask = pred & gt
    fp_mask = pred & ~gt
    fn_mask = ~pred & gt
    confusion[tp_mask] = confusion[tp_mask] * 0.2 + np.array(COLOR_TP) * 0.8
    confusion[fp_mask] = confusion[fp_mask] * 0.2 + np.array(COLOR_FP) * 0.8
    confusion[fn_mask] = confusion[fn_mask] * 0.2 + np.array(COLOR_FN) * 0.8
    confusion = confusion.astype(np.uint8)

    # ── Assemble panel ──
    scale = min(1.0, 600 / w)
    sw, sh = int(w * scale), int(h * scale)
    gap = 8
    label_h = 20

    # Metrics table height
    metrics_h = 420

    panel_w = sw * 4 + gap * 3 + 20
    panel_h = sh + label_h + 55 + metrics_h

    panel = Image.new("RGB", (panel_w, panel_h), (15, 23, 42))
    dp = ImageDraw.Draw(panel)

    ft = get_font(15)
    fl = get_font(11)
    fm = get_font_regular(11)
    fs = get_font_regular(10)

    # Title
    dp.text((10, 5), "7T FLAIR Tumor Segmentation — Evaluation Report",
            fill=(251, 191, 36), font=ft)

    # Images row
    ty = 30
    labels = ["T2-FLAIR Input", "Ground Truth", "Prediction", "Confusion (TP/FP/FN)"]
    imgs = [img_flair, gt_overlay, pred_overlay, confusion]
    label_colors = [(200, 200, 200), (50, 100, 255), (0, 220, 0), (251, 191, 36)]

    x = 10
    for im, lbl, lc in zip(imgs, labels, label_colors):
        pil_im = Image.fromarray(im).resize((sw, sh), Image.LANCZOS)
        panel.paste(pil_im, (x, ty))
        dp.text((x, ty + sh + 2), lbl, fill=lc, font=fl)
        x += sw + gap

    # ── Metrics section ──
    my = ty + sh + label_h + 15
    m = evaluation.overall

    # Left column: Overall metrics
    dp.text((10, my), "Overall Segmentation Metrics", fill=(251, 191, 36), font=ft)
    my += 22

    metrics_data = [
        ("Accuracy",         f"{m.accuracy:.4f}",     (200, 200, 200)),
        ("Precision",        f"{m.precision:.4f}",     (200, 200, 200)),
        ("Recall (Sens.)",   f"{m.recall:.4f}",        (100, 255, 100) if m.recall > 0.7 else (255, 100, 100)),
        ("Specificity",      f"{m.specificity:.4f}",   (200, 200, 200)),
        ("Dice Score (F1)",  f"{m.dice:.4f}",          (100, 255, 100) if m.dice > 0.7 else (255, 200, 50)),
        ("IoU (Jaccard)",    f"{m.iou:.4f}",           (200, 200, 200)),
        ("F2 Score",         f"{m.f2_score:.4f}",      (200, 200, 200)),
        ("Hausdorff (HD)",   f"{m.hausdorff:.2f} px" if m.hausdorff >= 0 else "N/A", (200, 200, 200)),
        ("HD95",             f"{m.hausdorff_95:.2f} px" if m.hausdorff_95 >= 0 else "N/A", (200, 200, 200)),
        ("NSD",              f"{m.nsd:.4f}" if m.nsd >= 0 else "N/A", (200, 200, 200)),
    ]

    for label, value, color in metrics_data:
        dp.text((20, my), f"{label}:", fill=(148, 163, 184), font=fm)
        dp.text((200, my), value, fill=color, font=ft)
        my += 18

    my += 8
    dp.text((20, my), f"TP: {m.tp}   FP: {m.fp}   FN: {m.fn}   TN: {m.tn}",
            fill=(148, 163, 184), font=fs)
    my += 16
    dp.text((20, my), f"Pred Volume: {m.pred_volume} px   GT Volume: {m.gt_volume} px",
            fill=(148, 163, 184), font=fs)

    # Right column: Instance metrics
    rx = panel_w // 2 + 20
    iy = ty + sh + label_h + 15
    dp.text((rx, iy), "Instance-Level Tumor Matching", fill=(251, 191, 36), font=ft)
    iy += 22

    inst_data = [
        ("GT Tumors",        str(evaluation.n_gt_tumors)),
        ("Pred Tumors",      str(evaluation.n_pred_tumors)),
        ("Matched Pairs",    str(evaluation.n_matched)),
        ("False Positives",  str(evaluation.n_false_positive)),
        ("False Negatives",  str(evaluation.n_false_negative)),
        ("Detection Rate",   f"{evaluation.detection_rate:.1%}"),
        ("Mean Dice (match)", f"{evaluation.mean_instance_dice:.4f}"),
    ]

    for label, value in inst_data:
        dp.text((rx, iy), f"{label}:", fill=(148, 163, 184), font=fm)
        dp.text((rx + 180, iy), value, fill=(200, 200, 200), font=ft)
        iy += 18

    # Per-instance table
    if evaluation.per_instance:
        iy += 10
        dp.text((rx, iy), "Per-Tumor Results:", fill=(148, 163, 184), font=fl)
        iy += 16

        # Header
        headers = ["GT#", "Pred#", "IoU", "Dice", "GT px", "Pred px"]
        hx = rx
        for hdr in headers:
            dp.text((hx, iy), hdr, fill=(251, 191, 36), font=fs)
            hx += 65
        iy += 14

        for match in evaluation.per_instance[:12]:  # max 12 rows
            vals = [
                str(match.gt_id), str(match.pred_id),
                f"{match.iou:.3f}", f"{match.dice:.3f}",
                str(match.gt_area), str(match.pred_area),
            ]
            hx = rx
            dice_color = (100, 255, 100) if match.dice > 0.5 else (255, 200, 50)
            for vi, v in enumerate(vals):
                c = dice_color if vi == 3 else (200, 200, 200)
                dp.text((hx, iy), v, fill=c, font=fs)
                hx += 65
            iy += 13

    # Legend
    ly = panel_h - 25
    dp.rectangle([(10, ly), (25, ly + 12)], fill=COLOR_TP)
    dp.text((30, ly - 1), "True Positive", fill=(200, 200, 200), font=fs)
    dp.rectangle([(160, ly), (175, ly + 12)], fill=COLOR_FP)
    dp.text((180, ly - 1), "False Positive", fill=(200, 200, 200), font=fs)
    dp.rectangle([(320, ly), (335, ly + 12)], fill=COLOR_FN)
    dp.text((340, ly - 1), "False Negative", fill=(200, 200, 200), font=fs)

    panel.save(output_path)
    print(f"    Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_csv(evaluation: FullEvaluation, output_path: str):
    """Export evaluation results to CSV."""
    m = evaluation.overall

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall metrics
        writer.writerow(["=== OVERALL SEGMENTATION METRICS ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{m.accuracy:.6f}"])
        writer.writerow(["Precision", f"{m.precision:.6f}"])
        writer.writerow(["Recall", f"{m.recall:.6f}"])
        writer.writerow(["Specificity", f"{m.specificity:.6f}"])
        writer.writerow(["Dice Score", f"{m.dice:.6f}"])
        writer.writerow(["IoU", f"{m.iou:.6f}"])
        writer.writerow(["F2 Score", f"{m.f2_score:.6f}"])
        writer.writerow(["Hausdorff Distance", f"{m.hausdorff:.4f}"])
        writer.writerow(["HD95", f"{m.hausdorff_95:.4f}"])
        writer.writerow(["NSD", f"{m.nsd:.6f}"])
        writer.writerow(["TP", m.tp])
        writer.writerow(["FP", m.fp])
        writer.writerow(["FN", m.fn])
        writer.writerow(["TN", m.tn])
        writer.writerow(["Pred Volume (px)", m.pred_volume])
        writer.writerow(["GT Volume (px)", m.gt_volume])
        writer.writerow([])

        # Instance metrics
        writer.writerow(["=== INSTANCE-LEVEL METRICS ==="])
        writer.writerow(["GT Tumors", evaluation.n_gt_tumors])
        writer.writerow(["Pred Tumors", evaluation.n_pred_tumors])
        writer.writerow(["Matched", evaluation.n_matched])
        writer.writerow(["False Positives", evaluation.n_false_positive])
        writer.writerow(["False Negatives", evaluation.n_false_negative])
        writer.writerow(["Detection Rate", f"{evaluation.detection_rate:.4f}"])
        writer.writerow(["Mean Instance Dice", f"{evaluation.mean_instance_dice:.6f}"])
        writer.writerow([])

        # Per-instance
        if evaluation.per_instance:
            writer.writerow(["=== PER-TUMOR RESULTS ==="])
            writer.writerow(["GT_ID", "Pred_ID", "IoU", "Dice", "GT_Area", "Pred_Area"])
            for match in evaluation.per_instance:
                writer.writerow([
                    match.gt_id, match.pred_id,
                    f"{match.iou:.6f}", f"{match.dice:.6f}",
                    match.gt_area, match.pred_area,
                ])

    print(f"    Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINT
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(evaluation: FullEvaluation):
    """Print all results to console."""
    m = evaluation.overall

    print(f"\n{'═' * 60}")
    print(f"  OVERALL SEGMENTATION METRICS")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<22s} {'Value':>12s}")
    print(f"  {'─' * 36}")
    print(f"  {'Accuracy':<22s} {m.accuracy:>12.4f}")
    print(f"  {'Precision':<22s} {m.precision:>12.4f}")
    print(f"  {'Recall (Sensitivity)':<22s} {m.recall:>12.4f}")
    print(f"  {'Specificity':<22s} {m.specificity:>12.4f}")
    print(f"  {'Dice Score (F1)':<22s} {m.dice:>12.4f}")
    print(f"  {'IoU (Jaccard)':<22s} {m.iou:>12.4f}")
    print(f"  {'F2 Score':<22s} {m.f2_score:>12.4f}")
    if m.hausdorff >= 0:
        print(f"  {'Hausdorff (HD)':<22s} {m.hausdorff:>10.2f} px")
        print(f"  {'HD95':<22s} {m.hausdorff_95:>10.2f} px")
    if m.nsd >= 0:
        print(f"  {'NSD':<22s} {m.nsd:>12.4f}")
    print(f"  {'─' * 36}")
    print(f"  TP={m.tp:>8d}    FP={m.fp:>8d}")
    print(f"  FN={m.fn:>8d}    TN={m.tn:>8d}")
    print(f"  Pred Vol={m.pred_volume} px    GT Vol={m.gt_volume} px")

    print(f"\n{'═' * 60}")
    print(f"  INSTANCE-LEVEL TUMOR MATCHING")
    print(f"{'═' * 60}")
    print(f"  GT Tumors          : {evaluation.n_gt_tumors}")
    print(f"  Predicted Tumors   : {evaluation.n_pred_tumors}")
    print(f"  Matched Pairs      : {evaluation.n_matched}")
    print(f"  False Positives    : {evaluation.n_false_positive}")
    print(f"  False Negatives    : {evaluation.n_false_negative}")
    print(f"  Detection Rate     : {evaluation.detection_rate:.1%}")
    print(f"  Mean Dice (matched): {evaluation.mean_instance_dice:.4f}")

    if evaluation.per_instance:
        print(f"\n  {'GT#':>4s} {'Pred#':>6s} {'IoU':>8s} {'Dice':>8s} {'GT px':>8s} {'Pred px':>8s}")
        print(f"  {'─' * 48}")
        for match in evaluation.per_instance:
            print(f"  {match.gt_id:>4d} {match.pred_id:>6d} "
                  f"{match.iou:>8.4f} {match.dice:>8.4f} "
                  f"{match.gt_area:>8d} {match.pred_area:>8d}")
    print(f"{'═' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    flair_path: str,
    gt_mask: Optional[np.ndarray] = None,
    overlay_path: Optional[str] = None,
    tumor_7t_path: Optional[str] = None,
    original_7t_path: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_tumor_area: int = 30,
) -> FullEvaluation:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    flair_path : str
        Path to the 7T FLAIR image to evaluate.
    gt_mask : np.ndarray, optional
        Pre-computed ground truth mask. If None, extracted from overlay or difference.
    overlay_path : str, optional
        Path to color overlay (for GT extraction).
    tumor_7t_path : str, optional
        Path to tumor-projected 7T (for GT extraction via difference).
    original_7t_path : str, optional
        Path to original 7T (for GT extraction via difference).
    output_dir : str
        Directory for output files.
    min_tumor_area : int
        Minimum tumor area in pixels.

    Returns
    -------
    FullEvaluation
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'╔' + '═' * 58 + '╗'}")
    print(f"{'║'} 7T FLAIR Tumor Segmentation Evaluation {'║':>19s}")
    print(f"{'╚' + '═' * 58 + '╝'}")

    # ── Step 1: Get ground truth ──
    print("\n[1/5] Ground Truth Extraction")
    if gt_mask is not None:
        print("  Using provided ground truth mask.")
    elif overlay_path and os.path.exists(overlay_path):
        gt_mask = extract_gt_from_overlay(overlay_path)
    elif tumor_7t_path and original_7t_path:
        gt_mask = extract_gt_from_difference(tumor_7t_path, original_7t_path)
    else:
        raise ValueError("Need gt_mask, overlay_path, or tumor_7t_path+original_7t_path")

    # ── Step 2: Segment FLAIR ──
    print("\n[2/5] FLAIR Tumor Segmentation")
    pred_mask = segment_flair_tumors(flair_path, min_area=min_tumor_area)

    # Ensure same shape
    assert pred_mask.shape == gt_mask.shape, (
        f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}"
    )

    # ── Step 3: Compute overall metrics ──
    print("\n[3/5] Computing Overall Metrics")
    overall = compute_metrics(pred_mask, gt_mask)

    # ── Step 4: Instance matching ──
    print("\n[4/5] Instance-Level Matching")
    matches, n_fp, n_fn = match_instances(pred_mask, gt_mask)

    n_gt = ndimage_label(gt_mask)[1]
    n_pred = ndimage_label(pred_mask)[1]

    evaluation = FullEvaluation(
        overall=overall,
        per_instance=matches,
        n_gt_tumors=n_gt,
        n_pred_tumors=n_pred,
        n_matched=len(matches),
        n_false_positive=n_fp,
        n_false_negative=n_fn,
        mean_instance_dice=float(np.mean([m.dice for m in matches])) if matches else 0.0,
        detection_rate=len(matches) / (n_gt + EPS),
    )

    # ── Step 5: Generate outputs ──
    print("\n[5/5] Generating Outputs")

    flair_arr = np.array(Image.open(flair_path).convert("RGB")).astype(float)

    create_visual_report(
        flair_arr, pred_mask, gt_mask, evaluation,
        os.path.join(output_dir, "evaluation_report.png"),
    )
    export_csv(evaluation, os.path.join(output_dir, "evaluation_metrics.csv"))

    # Save masks
    Image.fromarray((gt_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "gt_mask.png")
    )
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "pred_mask.png")
    )

    # Print results
    print_results(evaluation)

    return evaluation


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="7T FLAIR Tumor Segmentation Metrics Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using color overlay as ground truth
  python flair_segmentation_eval.py --flair 7T8_FLAIR.png --overlay 7T8_overlay.png

  # Using image difference as ground truth
  python flair_segmentation_eval.py --flair 7T8_FLAIR.png \\
      --tumor-7t 7T8_tumors.png --original-7t 7T8.png

  # With pre-made ground truth mask
  python flair_segmentation_eval.py --flair 7T8_FLAIR.png --gt-mask gt.png

  # Custom parameters
  python flair_segmentation_eval.py --flair flair.png --overlay overlay.png \\
      --min-area 50 --output-dir ./results
        """,
    )

    parser.add_argument("--flair", type=str, default=DEFAULT_FLAIR_PATH,
                        help="Path to FLAIR image")
    parser.add_argument("--overlay", type=str, default=DEFAULT_OVERLAY_PATH,
                        help="Path to color overlay (for GT extraction)")
    parser.add_argument("--tumor-7t", type=str, default=None,
                        help="Path to tumor-projected 7T image")
    parser.add_argument("--original-7t", type=str, default=None,
                        help="Path to original 7T image")
    parser.add_argument("--gt-mask", type=str, default=None,
                        help="Path to pre-made ground truth mask")
    parser.add_argument("--min-area", type=int, default=30,
                        help="Minimum tumor area in pixels (default: 30)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    gt = None
    if args.gt_mask:
        gt = np.array(Image.open(args.gt_mask).convert("L")) > 127

    evaluate(
        flair_path=args.flair,
        gt_mask=gt,
        overlay_path=args.overlay,
        tumor_7t_path=args.tumor_7t,
        original_7t_path=args.original_7t,
        output_dir=args.output_dir,
        min_tumor_area=args.min_area,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Demo: run on the generated images
        evaluate(
            flair_path=DEFAULT_FLAIR_PATH,
            overlay_path=DEFAULT_OVERLAY_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
        )
