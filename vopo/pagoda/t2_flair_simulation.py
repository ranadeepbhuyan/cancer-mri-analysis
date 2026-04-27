#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  T2-FLAIR MRI Simulation Toolkit                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Two conversion pipelines for simulating T2-FLAIR MRI appearance:          ║
║                                                                            ║
║    1. SWI  → T2-FLAIR  (Susceptibility Weighted Imaging to FLAIR)          ║
║    2. 3D Brain Render → T2-FLAIR  (Surface rendering to FLAIR)             ║
║                                                                            ║
║  Author : PAGODA-Net Research Pipeline                                     ║
║  Usage  : python t2_flair_simulation.py                                    ║
║  Deps   : numpy, scipy, Pillow                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

T2-FLAIR (Fluid-Attenuated Inversion Recovery) Key Characteristics:
───────────────────────────────────────────────────────────────────
  • Gray matter  → LIGHTER (higher signal ~120–165)
  • White matter → DARKER  (mid-gray signal ~70–110)
  • CSF          → SUPPRESSED (dark/black) — the defining FLAIR feature
  • Pathology    → HYPERINTENSE (bright white: edema, gliosis, demyelination)
  • Veins        → NOT prominently visible (unlike SWI)
  • Noise        → Rician (magnitude of complex Gaussian)
  • Artifacts    → B1 bias field inhomogeneity, Gibbs ringing at edges

References:
───────────
  • Buxton, R. B. (2013). The physics of functional MRI (fMRI).
  • Bhuyan, R., & Nandi, G. (2024). Multimodal MRI augmentation for brain
    tumor detection with loss-aware exchange and residual networks.
"""

import os
import argparse
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    sobel,
    label as ndimage_label,
)
from PIL import Image, ImageEnhance


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def add_rician_noise(image: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """
    Add Rician noise to simulate MRI acquisition noise.

    In MRI, the signal is complex-valued. The magnitude image follows a
    Rician distribution: |S + n_real + j*n_imag| where n ~ N(0, sigma).

    Parameters
    ----------
    image : np.ndarray
        Input image (float).
    sigma : float
        Standard deviation of Gaussian noise in real and imaginary channels.

    Returns
    -------
    np.ndarray
        Noisy image with Rician noise characteristics.
    """
    h, w = image.shape[:2]
    noise_real = np.random.normal(0, sigma, (h, w))
    noise_imag = np.random.normal(0, sigma, (h, w))
    noisy = np.sqrt(np.maximum((image + noise_real) ** 2 + noise_imag ** 2, 0))
    return noisy


def add_bias_field(image: np.ndarray, strength: float = 0.07) -> np.ndarray:
    """
    Simulate B1 inhomogeneity (bias field) artifact.

    MRI images have spatially varying intensity due to imperfect RF coil
    sensitivity profiles. This is modeled as a smooth multiplicative field.

    Parameters
    ----------
    image : np.ndarray
        Input image (float).
    strength : float
        Maximum fractional deviation from unity (e.g., 0.07 = ±7%).

    Returns
    -------
    np.ndarray
        Image with bias field applied.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    bias = 1.0
    bias += strength * np.sin(1.1 * np.pi * yy / h)
    bias += (strength * 0.6) * np.cos(0.7 * np.pi * xx / w)
    bias += (strength * 0.4) * np.sin(2.3 * np.pi * yy / h) * np.cos(1.8 * np.pi * xx / w)

    return image * bias


def add_gibbs_ringing(image: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """
    Simulate Gibbs ringing artifact at sharp edges.

    Truncation of k-space data in MRI causes oscillatory artifacts near
    high-contrast boundaries (Gibbs phenomenon).

    Parameters
    ----------
    image : np.ndarray
        Input image (float).
    strength : float
        Intensity of ringing artifact.

    Returns
    -------
    np.ndarray
        Image with Gibbs ringing.
    """
    ring = gaussian_filter(image, sigma=0.4) - gaussian_filter(image, sigma=1.2)
    return image + ring * strength


def create_brain_mask(image_gray: np.ndarray, threshold: float = 12) -> np.ndarray:
    """
    Generate a binary brain mask from a grayscale MRI image.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale image (0–255 float or uint8).
    threshold : float
        Intensity threshold separating brain from background.

    Returns
    -------
    np.ndarray
        Boolean brain mask.
    """
    mask = image_gray > threshold
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask)
    return mask


def to_grayscale_output(flair: np.ndarray) -> np.ndarray:
    """
    Convert single-channel float image to 3-channel uint8 grayscale.

    Parameters
    ----------
    flair : np.ndarray
        2D float array.

    Returns
    -------
    np.ndarray
        3-channel uint8 image (H, W, 3).
    """
    flair_clipped = np.clip(flair, 0, 255).astype(np.uint8)
    return np.stack([flair_clipped] * 3, axis=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSION 1: SWI → T2-FLAIR
# ═══════════════════════════════════════════════════════════════════════════════

def swi_to_t2_flair(
    input_path: str,
    output_path: str,
    noise_sigma: float = 2.2,
    bias_strength: float = 0.07,
    smoothing_sigma: float = 1.8,
    contrast_enhance: float = 1.25,
    brightness_adjust: float = 0.95,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert a Susceptibility Weighted Image (SWI) to simulated T2-FLAIR.

    SWI Characteristics (input):
    ─────────────────────────────
      • Veins appear VERY DARK (hypointense, deoxygenated blood susceptibility)
      • White matter is brighter than gray matter
      • Ultra-high spatial resolution with fine venous detail
      • Background is black

    T2-FLAIR Characteristics (target):
    ───────────────────────────────────
      • Gray matter LIGHTER than white matter (reversed from SWI/T1)
      • CSF is SUPPRESSED (dark) — inversion recovery nulling
      • Veins NOT prominently visible
      • Smoother appearance than SWI

    Pipeline Steps:
    ───────────────
      1. Generate brain mask
      2. Suppress venous structures via iterative inpainting
      3. Normalize brain intensities
      4. Compute cortex-depth map for GM/WM differentiation
      5. Remap tissue contrast (invert GM/WM relationship)
      6. Enhance cortical ribbon
      7. Suppress sulcal CSF
      8. Add midline fissure
      9. Smooth to FLAIR resolution
      10. Add MRI artifacts (bias field, Rician noise, Gibbs ringing)

    Parameters
    ----------
    input_path : str
        Path to input SWI image (.png, .jpg, etc.).
    output_path : str
        Path to save the output T2-FLAIR image.
    noise_sigma : float
        Rician noise standard deviation (default: 2.2).
    bias_strength : float
        Bias field inhomogeneity strength (default: 0.07).
    smoothing_sigma : float
        Gaussian smoothing to match FLAIR resolution (default: 1.8).
    contrast_enhance : float
        Post-processing contrast enhancement factor (default: 1.25).
    brightness_adjust : float
        Post-processing brightness adjustment factor (default: 0.95).
    verbose : bool
        Print progress messages.

    Returns
    -------
    np.ndarray
        The simulated T2-FLAIR image as a 2D float array.
    """
    if verbose:
        print("=" * 60)
        print("  SWI → T2-FLAIR Conversion")
        print("=" * 60)

    # ── Load image ──
    img = Image.open(input_path).convert("L")
    arr = np.array(img).astype(float)
    h, w = arr.shape
    if verbose:
        print(f"  Input size     : {w} × {h}")

    # ── Step 1: Brain mask ──
    if verbose:
        print("  [1/10] Generating brain mask...")
    brain_mask = create_brain_mask(arr, threshold=12)

    # ── Step 2: Venous structure suppression ──
    if verbose:
        print("  [2/10] Suppressing venous structures...")

    # In SWI, veins are abnormally dark compared to surrounding tissue.
    # We detect them as pixels significantly darker than the local mean.
    local_mean = gaussian_filter(arr, sigma=5)
    vein_mask = (arr < local_mean - 25) & brain_mask & (arr < 110)

    # Inpaint veins with local tissue average
    filled = arr.copy()
    filled[vein_mask] = local_mean[vein_mask]

    # Iterative smoothing to clean residual vessel traces
    for i in range(3):
        smooth = gaussian_filter(filled, sigma=2)
        dilated_veins = binary_dilation(vein_mask, iterations=1)
        filled[dilated_veins] = smooth[dilated_veins]

    # ── Step 3: Normalize brain intensities ──
    if verbose:
        print("  [3/10] Normalizing brain intensities...")
    brain_vals = filled[brain_mask]
    p1, p99 = np.percentile(brain_vals, [1, 99])
    norm = np.clip((filled - p1) / (p99 - p1 + 1e-8), 0, 1)

    # ── Step 4: Cortex depth map ──
    if verbose:
        print("  [4/10] Computing cortex depth map...")
    dist = distance_transform_edt(brain_mask)
    max_d = np.percentile(dist[brain_mask], 95)
    dist_norm = np.clip(dist / (max_d + 1e-8), 0, 1)

    # ── Step 5: Remap tissue contrast ──
    if verbose:
        print("  [5/10] Remapping tissue contrast (GM/WM inversion)...")

    # In SWI:  WM bright (~0.7–1.0), GM slightly darker (~0.4–0.7)
    # In FLAIR: GM bright (~130–165),  WM darker (~75–110)
    #
    # dist_norm: 0 = cortex (outer), 1 = deep white matter (center)
    gm_weight = 1.0 - dist_norm  # high at cortex
    wm_weight = dist_norm          # high at center

    gm_signal = 125 + gm_weight * 45 + norm * 10   # ~125–180
    wm_signal = 70 + wm_weight * 30 + norm * 15    # ~70–115

    flair = gm_signal * (1 - dist_norm) + wm_signal * dist_norm

    # Add anatomical texture from original (subtle)
    texture = (norm - gaussian_filter(norm, sigma=8)) * 25
    flair += texture

    # ── Step 6: Cortical ribbon enhancement ──
    if verbose:
        print("  [6/10] Enhancing cortical ribbon...")
    cortex_band = (dist_norm > 0.02) & (dist_norm < 0.35) & brain_mask
    cortex_boost = gaussian_filter(cortex_band.astype(float), sigma=3)
    flair += cortex_boost * 20

    # ── Step 7: Sulcal CSF suppression ──
    if verbose:
        print("  [7/10] Suppressing sulcal CSF (FLAIR characteristic)...")
    local_smooth = gaussian_filter(flair, sigma=6)
    sulci_score = np.clip((local_smooth - flair) / 30, 0, 1) * brain_mask

    # Also use original SWI dark lines (non-vein sulci)
    original_sulci = (arr < local_mean - 10) & brain_mask & (~vein_mask)
    original_sulci_score = gaussian_filter(original_sulci.astype(float), sigma=1.5)
    sulci_score = np.maximum(sulci_score, original_sulci_score * 0.6)

    flair = flair * (1 - sulci_score * 0.7) + 15 * sulci_score * 0.7

    # ── Step 8: Midline interhemispheric fissure ──
    if verbose:
        print("  [8/10] Adding midline fissure...")
    mid_x = w // 2
    fissure = np.zeros((h, w))
    fissure[:, mid_x - 3 : mid_x + 4] = 1.0
    fissure = gaussian_filter(fissure, sigma=4) * brain_mask
    flair = flair * (1 - fissure * 0.4)

    # ── Step 9: Smooth to FLAIR resolution ──
    if verbose:
        print("  [9/10] Smoothing to FLAIR resolution...")
    flair = gaussian_filter(flair, sigma=smoothing_sigma)

    # ── Step 10: MRI artifacts ──
    if verbose:
        print("  [10/10] Adding MRI artifacts (bias field, noise, ringing)...")

    flair = add_bias_field(flair, strength=bias_strength)
    flair = add_rician_noise(flair, sigma=noise_sigma)
    flair = add_gibbs_ringing(flair, strength=0.10)

    # Apply soft brain mask
    soft_mask = gaussian_filter(brain_mask.astype(float), sigma=2.0)
    flair *= soft_mask
    flair = np.clip(flair, 0, 255)

    # ── Save output ──
    output = to_grayscale_output(flair)
    result = Image.fromarray(output)
    result = ImageEnhance.Contrast(result).enhance(contrast_enhance)
    result = ImageEnhance.Brightness(result).enhance(brightness_adjust)
    result.save(output_path)

    if verbose:
        print(f"  Output saved   : {output_path}")
        print(f"  Output size    : {result.size[0]} × {result.size[1]}")
        print("=" * 60)

    return flair


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSION 2: 3D Brain Rendering → T2-FLAIR
# ═══════════════════════════════════════════════════════════════════════════════

def render3d_to_t2_flair(
    input_path: str,
    output_path: str,
    noise_sigma: float = 2.2,
    bias_strength: float = 0.06,
    smoothing_sigma: float = 2.0,
    contrast_enhance: float = 1.15,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert a 3D brain surface rendering to simulated T2-FLAIR.

    This handles RGBA images of 3D brain models (e.g., from Neurotorium,
    BrainSuite, FreeSurfer surface renders) where:
      • The brain model sits on a colored/gray background
      • Golden/tan highlighted regions represent functional or pathological areas
      • White/cream regions represent normal brain parenchyma
      • The image has an alpha channel or distinct background color

    Pipeline Steps:
    ───────────────
      1. Load RGBA image and extract alpha mask
      2. Detect brain region via edge detection (if no alpha)
      3. Segment tissue types by color:
         - Golden/tan → pathological/functional (FLAIR hyperintense)
         - White/cream → normal parenchyma
         - Dark grooves → sulci (CSF suppression)
      4. Build FLAIR intensity map with GM/WM contrast
      5. Apply CSF suppression at sulci
      6. Map pathological regions to hyperintense signal
      7. Smooth to FLAIR resolution
      8. Add MRI artifacts

    Parameters
    ----------
    input_path : str
        Path to input 3D brain rendering (.png with alpha recommended).
    output_path : str
        Path to save the output T2-FLAIR image.
    noise_sigma : float
        Rician noise standard deviation (default: 2.2).
    bias_strength : float
        Bias field strength (default: 0.06).
    smoothing_sigma : float
        Gaussian smoothing sigma (default: 2.0).
    contrast_enhance : float
        Post-processing contrast factor (default: 1.15).
    verbose : bool
        Print progress messages.

    Returns
    -------
    np.ndarray
        The simulated T2-FLAIR image as a 2D float array.
    """
    if verbose:
        print("=" * 60)
        print("  3D Brain Rendering → T2-FLAIR Conversion")
        print("=" * 60)

    # ── Load image ──
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img).astype(float)
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0
    if verbose:
        print(f"  Input size     : {w} × {h}")

    # ── Step 1: Brain mask ──
    if verbose:
        print("  [1/8] Determining brain mask...")

    has_alpha = alpha.max() > 0 and alpha.min() < 0.5
    if has_alpha:
        # Use alpha channel directly
        brain_mask = alpha > 0.15
        if verbose:
            print("         → Using alpha channel")
    else:
        # Detect brain via edge detection (for images without alpha)
        gray = rgb.mean(axis=2)
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        grad = np.hypot(sx, sy)

        edge_mask = grad > 12
        struct = np.ones((10, 10))
        dilated = binary_dilation(edge_mask, structure=struct, iterations=2)
        filled = binary_fill_holes(dilated)
        filled = binary_erosion(filled, structure=np.ones((8, 8)), iterations=2)
        filled = binary_dilation(filled, structure=np.ones((5, 5)), iterations=1)
        filled = binary_fill_holes(filled)

        labeled, num_features = ndimage_label(filled)
        if num_features > 0:
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest = np.argmax(sizes) + 1
            brain_mask = labeled == largest
        else:
            brain_mask = filled

        # Build a soft alpha from the mask
        alpha = gaussian_filter(brain_mask.astype(float), sigma=2)
        if verbose:
            print("         → Detected via edge analysis")

    # ── Step 2: Color-based tissue segmentation ──
    if verbose:
        print("  [2/8] Segmenting tissue types by color...")

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = rgb.mean(axis=2) / 255.0

    # Golden/tan pathological regions:
    # These have warm hue: R > G > B with noticeable difference
    warm_diff = r - b
    gold_score = np.clip(warm_diff / 60.0, 0, 1) * np.clip((r - g) / 40.0 + 0.5, 0, 1)
    gold_score = gold_score * brain_mask
    gold_score = gaussian_filter(gold_score, sigma=2)  # smooth boundaries

    # Sulci/fissures: dark grooves between gyri (shadows on surface)
    local_brightness = gaussian_filter(brightness, sigma=8)
    sulci_score = np.clip((local_brightness - brightness) * 4, 0, 1) * brain_mask

    if verbose:
        gold_pct = (gold_score > 0.3).sum() / brain_mask.sum() * 100
        print(f"         → Gold/pathological area: {gold_pct:.1f}% of brain")

    # ── Step 3: Distance from brain center (cortex vs deep structures) ──
    if verbose:
        print("  [3/8] Computing cortex depth map...")
    dist = distance_transform_edt(brain_mask)
    max_d = np.percentile(dist[brain_mask], 95) if brain_mask.any() else 1
    dist_norm = np.clip(dist / (max_d + 1e-8), 0, 1)

    # ── Step 4: Build T2-FLAIR intensity map ──
    if verbose:
        print("  [4/8] Building FLAIR tissue intensity map...")

    # Cortical (outer) = gray matter = lighter in FLAIR
    # Central = white matter = darker in FLAIR
    gm_weight = 1.0 - dist_norm
    wm_weight = dist_norm

    base = 65 + gm_weight * 55  # WM~65, GM~120
    base = base * brain_mask

    # Subtle anatomical texture from original brightness
    tissue_texture = (brightness - 0.5) * 30
    base += tissue_texture * brain_mask
    base = gaussian_filter(base, sigma=smoothing_sigma)

    flair = base.copy()

    # ── Step 5: CSF suppression at sulci ──
    if verbose:
        print("  [5/8] Suppressing sulcal CSF...")
    sulci_dark = 12 + np.random.normal(0, 2, (h, w))
    flair = flair * (1 - sulci_score * 0.85) + sulci_dark * sulci_score * 0.85

    # ── Step 6: Pathological hyperintensity ──
    if verbose:
        print("  [6/8] Mapping pathological regions to hyperintense signal...")
    # In FLAIR: edema, gliosis, demyelination → bright (180–255)
    hyperintense = 175 + gold_score * 80
    flair = flair * (1 - gold_score * 0.9) + hyperintense * gold_score * 0.9

    # ── Step 7: Smooth to FLAIR resolution ──
    if verbose:
        print("  [7/8] Smoothing to FLAIR resolution...")
    flair = gaussian_filter(flair, sigma=smoothing_sigma)

    # ── Step 8: MRI artifacts ──
    if verbose:
        print("  [8/8] Adding MRI artifacts (bias field, noise, ringing)...")

    flair = add_bias_field(flair, strength=bias_strength)
    flair = add_rician_noise(flair, sigma=noise_sigma)
    flair = add_gibbs_ringing(flair, strength=0.15)

    # Apply brain mask with soft edges
    soft_alpha = gaussian_filter((alpha > 0.1).astype(float), sigma=1.5)
    flair *= soft_alpha
    flair = np.clip(flair, 0, 255)

    # ── Save output ──
    output = to_grayscale_output(flair)
    result = Image.fromarray(output)
    result = ImageEnhance.Contrast(result).enhance(contrast_enhance)
    result.save(output_path)

    if verbose:
        print(f"  Output saved   : {output_path}")
        print(f"  Output size    : {result.size[0]} × {result.size[1]}")
        print("=" * 60)

    return flair


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDE-BY-SIDE COMPARISON GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def create_comparison(
    original_path: str,
    flair_path: str,
    output_path: str,
    original_label: str = "Original",
    flair_label: str = "T2-FLAIR (Simulated)",
):
    """
    Create a side-by-side comparison image.

    Parameters
    ----------
    original_path : str
        Path to original image.
    flair_path : str
        Path to simulated FLAIR image.
    output_path : str
        Path to save the comparison image.
    original_label : str
        Label for the original image.
    flair_label : str
        Label for the FLAIR image.
    """
    from PIL import ImageDraw, ImageFont

    orig = Image.open(original_path).convert("RGB")
    flair = Image.open(flair_path).convert("RGB")

    # Match heights
    target_h = max(orig.height, flair.height)
    orig_r = orig.resize(
        (int(orig.width * target_h / orig.height), target_h), Image.LANCZOS
    )
    flair_r = flair.resize(
        (int(flair.width * target_h / flair.height), target_h), Image.LANCZOS
    )

    gap = 20
    label_h = 40
    total_w = orig_r.width + gap + flair_r.width
    canvas = Image.new("RGB", (total_w, target_h + label_h), (0, 0, 0))

    canvas.paste(orig_r, (0, label_h))
    canvas.paste(flair_r, (orig_r.width + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text(
        (orig_r.width // 2 - 40, 8), original_label, fill=(200, 200, 200), font=font
    )
    draw.text(
        (orig_r.width + gap + flair_r.width // 2 - 80, 8),
        flair_label,
        fill=(251, 191, 36),
        font=font,
    )

    canvas.save(output_path)
    print(f"  Comparison saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="T2-FLAIR MRI Simulation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
─────────
  # Convert SWI image to T2-FLAIR
  python t2_flair_simulation.py --mode swi --input swi7T3.png --output swi_flair.png

  # Convert 3D brain render to T2-FLAIR
  python t2_flair_simulation.py --mode render --input brain_3d.png --output render_flair.png

  # Both conversions with comparison images
  python t2_flair_simulation.py --mode both \\
      --swi-input swi7T3.png \\
      --render-input Neurotorium_3_cropped.png \\
      --output-dir ./flair_outputs \\
      --compare

  # Custom noise and smoothing
  python t2_flair_simulation.py --mode swi --input swi.png --output flair.png \\
      --noise 3.0 --smoothing 2.5 --contrast 1.3
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["swi", "render", "both"],
        default="both",
        help="Conversion mode: 'swi', 'render', or 'both' (default: both)",
    )
    parser.add_argument("--input", type=str, help="Input image path (for swi/render mode)")
    parser.add_argument("--output", type=str, help="Output image path (for swi/render mode)")
    parser.add_argument("--swi-input", type=str, help="SWI input path (for 'both' mode)")
    parser.add_argument("--render-input", type=str, help="3D render input path (for 'both' mode)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory (for 'both' mode)")
    parser.add_argument("--compare", action="store_true", help="Generate side-by-side comparison images")
    parser.add_argument("--noise", type=float, default=2.2, help="Rician noise sigma (default: 2.2)")
    parser.add_argument("--smoothing", type=float, default=1.8, help="FLAIR smoothing sigma (default: 1.8)")
    parser.add_argument("--contrast", type=float, default=1.25, help="Contrast enhancement factor (default: 1.25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.mode == "swi":
        if not args.input or not args.output:
            parser.error("--input and --output are required for 'swi' mode")
        swi_to_t2_flair(
            args.input, args.output,
            noise_sigma=args.noise,
            smoothing_sigma=args.smoothing,
            contrast_enhance=args.contrast,
        )
        if args.compare:
            comp_path = args.output.replace(".png", "_comparison.png")
            create_comparison(args.input, args.output, comp_path, "SWI (Original)")

    elif args.mode == "render":
        if not args.input or not args.output:
            parser.error("--input and --output are required for 'render' mode")
        render3d_to_t2_flair(
            args.input, args.output,
            noise_sigma=args.noise,
            smoothing_sigma=args.smoothing,
            contrast_enhance=args.contrast,
        )
        if args.compare:
            comp_path = args.output.replace(".png", "_comparison.png")
            create_comparison(args.input, args.output, comp_path, "3D Render (Original)")

    elif args.mode == "both":
        os.makedirs(args.output_dir, exist_ok=True)

        if args.swi_input:
            swi_out = os.path.join(args.output_dir, "swi_T2_FLAIR.png")
            swi_to_t2_flair(
                args.swi_input, swi_out,
                noise_sigma=args.noise,
                smoothing_sigma=args.smoothing,
                contrast_enhance=args.contrast,
            )
            if args.compare:
                create_comparison(
                    args.swi_input, swi_out,
                    os.path.join(args.output_dir, "swi_comparison.png"),
                    "SWI (Original)",
                )
            print()

        if args.render_input:
            render_out = os.path.join(args.output_dir, "render_T2_FLAIR.png")
            render3d_to_t2_flair(
                args.render_input, render_out,
                noise_sigma=args.noise,
                smoothing_sigma=args.smoothing,
                contrast_enhance=args.contrast,
            )
            if args.compare:
                create_comparison(
                    args.render_input, render_out,
                    os.path.join(args.output_dir, "render_comparison.png"),
                    "3D Render (Original)",
                )

        if not args.swi_input and not args.render_input:
            parser.error("Provide --swi-input and/or --render-input for 'both' mode")

    print("\n✓ All conversions complete.")


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST (runs when executed without CLI arguments)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # Default demo: process both images if they exist
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  T2-FLAIR Simulation Toolkit — Demo Run                 ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        np.random.seed(42)

        swi_path = "swi7T3.png"
        render_path = "Neurotorium_3_cropped.png"

        if os.path.exists(swi_path):
            swi_to_t2_flair(swi_path, "swi7T3_T2_FLAIR.png")
            create_comparison(swi_path, "swi7T3_T2_FLAIR.png", "swi_comparison.png", "SWI")
            print()
        else:
            print(f"  ⚠ SWI image not found: {swi_path}")
            print(f"    → Run: python t2_flair_simulation.py --mode swi --input <your_swi.png> --output flair.png")
            print()

        if os.path.exists(render_path):
            render3d_to_t2_flair(render_path, "Neurotorium_3_T2_FLAIR.png")
            create_comparison(render_path, "Neurotorium_3_T2_FLAIR.png", "render_comparison.png", "3D Render")
            print()
        else:
            print(f"  ⚠ 3D render not found: {render_path}")
            print(f"    → Run: python t2_flair_simulation.py --mode render --input <your_render.png> --output flair.png")
            print()

        print("✓ Demo complete.")
