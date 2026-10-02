"""
Step 4: Run inference on new HSI data.
- Loads trained model + PCA
- Runs segmentation on any new HSI sample folder
- Outputs:
  * 2D overlay image with tumor highlighted
  * JSON file with tumor bounding box and centroid coordinates
  * tumor_coords.json for 3D visualization
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import pickle
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model definition (must match step3_train.py) ────────────────────────────
import torch.nn.functional as F

class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)

class UNetResidual(nn.Module):
    def __init__(self, in_channels=30, num_classes=1):
        super().__init__()
        self.enc1 = ResConvBlock(in_channels, 64)
        self.enc2 = ResConvBlock(64, 128)
        self.enc3 = ResConvBlock(128, 256)
        self.enc4 = ResConvBlock(256, 512)
        self.pool  = nn.MaxPool2d(2)
        self.bridge = ResConvBlock(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ResConvBlock(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResConvBlock(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResConvBlock(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResConvBlock(128, 64)
        self.out  = nn.Conv2d(64, num_classes, 1)
        self.aux3 = nn.Conv2d(256, num_classes, 1)
        self.aux2 = nn.Conv2d(128, num_classes, 1)
    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.bridge(self.pool(s4))
        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out(d1)  # inference: main head only


def parse_hdr(hdr_path):
    info = {}
    with open(hdr_path, "r", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")
    for line in content.split("\n"):
        if "=" in line and not line.strip().startswith("{"):
            key, _, val = line.partition("=")
            info[key.strip()] = val.strip()
    return info

def load_envi_raw(folder):
    raw_path = folder / "raw"
    hdr_path = folder / "raw.hdr"
    if not raw_path.exists():
        return None
    info = parse_hdr(hdr_path)
    samples = int(info.get("samples", 345))
    lines   = int(info.get("lines", 389))
    bands   = int(info.get("bands", 826))
    dtype_id = int(info.get("data type", 12))
    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = dtype_map.get(dtype_id, np.uint16)
    raw_data = np.fromfile(str(raw_path), dtype=dtype)
    cube = raw_data.reshape((lines, bands, samples))
    cube = np.transpose(cube, (0, 2, 1))
    return cube.astype(np.float32)

def load_reference(folder, name):
    ref_path = folder / name
    hdr_path = folder / f"{name}.hdr"
    if not ref_path.exists():
        return None
    info = parse_hdr(hdr_path)
    samples = int(info.get("samples", 345))
    lines   = int(info.get("lines", 1))
    bands   = int(info.get("bands", 826))
    dtype_id = int(info.get("data type", 12))
    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = dtype_map.get(dtype_id, np.uint16)
    ref = np.fromfile(str(ref_path), dtype=dtype)
    ref = ref.reshape((lines, bands, samples))
    ref = np.transpose(ref, (0, 2, 1))
    return ref.mean(axis=0).astype(np.float32)

def calibrate(cube, dark, white):
    dark_bc = dark[np.newaxis, :, :]
    white_bc = white[np.newaxis, :, :]
    denom = np.where(white_bc - dark_bc == 0, 1e-6, white_bc - dark_bc)
    return np.clip((cube - dark_bc) / denom, 0, 1)

def generate_hsi_plots(calibrated, sample_id, mask=None):
    print("Generating advanced HSI analysis maps...")
    
    # 1. Mean Spectral Intensity Heatmap
    mean_intensity = calibrated.mean(axis=2)
    plt.imsave(RESULTS_DIR / f"{sample_id}_hsi_mean.png", mean_intensity, cmap="hot")
    
    # 2. Spectral Profile
    mean_profile = calibrated.mean(axis=(0,1))
    fig = plt.figure(figsize=(10, 4))
    plt.plot(mean_profile, color='navy')
    plt.fill_between(range(len(mean_profile)), mean_profile, color='navy', alpha=0.3)
    plt.title("Mean Spectral Profile")
    plt.xlabel("Band Index")
    plt.ylabel("Mean Intensity")
    plt.grid(True, alpha=0.3)
    
    fig.savefig(RESULTS_DIR / f"{sample_id}_hsi_profile.png", bbox_inches="tight")
    plt.close(fig)
    
    # 3. Spectral Variance Heatmap
    variance = calibrated.var(axis=2)
    plt.imsave(RESULTS_DIR / f"{sample_id}_hsi_variance.png", variance, cmap="viridis")
    
    # 4. RGB Composite Approx
    B = calibrated.shape[2]
    # Simple approx picking bands for R, G, B
    r_band = int(B * 0.15)
    g_band = int(B * 0.3)
    b_band = int(B * 0.5)
    rgb = np.stack([calibrated[:, :, r_band], calibrated[:, :, g_band], calibrated[:, :, b_band]], axis=-1)
    rgb = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)
    plt.imsave(RESULTS_DIR / f"{sample_id}_hsi_rgb.png", rgb)

def predict_from_folder(sample_folder_path, threshold=0.5):
    """Run inference on a raw HSI sample folder."""
    folder = Path(sample_folder_path)
    sample_id = folder.name

    print(f"Loading HSI data from: {folder}")
    cube  = load_envi_raw(folder)
    dark  = load_reference(folder, "darkReference")
    white = load_reference(folder, "whiteReference")

    if cube is None or dark is None or white is None:
        raise FileNotFoundError(f"Missing files in {folder}")

    calibrated = calibrate(cube, dark, white)
    H_orig, W_orig, B = calibrated.shape

    # Load PCA model
    pca_path = PROCESSED_DIR / "pca_model.pkl"
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
    scaler = pca_data["scaler"]
    pca    = pca_data["pca"]
    n_comp = pca_data["n_components"]

    # Transform with PCA
    pixels_flat   = calibrated.reshape(-1, B)
    pixels_scaled = scaler.transform(pixels_flat)
    pixels_pca    = pca.transform(pixels_scaled).astype(np.float32)
    cube_pca      = pixels_pca.reshape(H_orig, W_orig, n_comp)

    # Resize to 256x256
    cube_resized = cv2.resize(cube_pca, (256, 256), interpolation=cv2.INTER_LINEAR)
    hsi_tensor   = torch.from_numpy(cube_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Load model
    model_path = MODEL_DIR / "best_model.pth"
    model = UNetResidual(in_channels=n_comp, num_classes=1).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(hsi_tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (prob > threshold).astype(np.uint8)

    # Upscale mask back to original size
    mask_orig = cv2.resize(binary_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    prob_orig = cv2.resize(prob, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    generate_hsi_plots(calibrated, sample_id, mask=mask_orig)

    return prob_orig, mask_orig, (H_orig, W_orig)

def predict_from_npz(npz_path, threshold=0.5):
    """Run inference on a preprocessed .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    hsi  = data["hsi"]         # (256, 256, 30)
    rgb  = data["rgb"]         # (256, 256, 3)
    gt   = data.get("tumor_mask", None)
    sample_id = str(data.get("sample_id", Path(npz_path).stem))

    # Load PCA model for n_components
    pca_path = PROCESSED_DIR / "pca_model.pkl"
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
    n_comp = pca_data["n_components"]

    # Load model
    model_path = MODEL_DIR / "best_model.pth"
    model = UNetResidual(in_channels=n_comp, num_classes=1).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hsi_tensor = torch.from_numpy(hsi).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = model(hsi_tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (prob > threshold).astype(np.uint8)
    return prob, binary_mask, rgb, gt, sample_id

def extract_tumor_coords(binary_mask, original_size=None):
    """
    Extract tumor coordinates from binary mask.
    Returns dict with bounding boxes, centroid, contours.
    """
    H, W = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:   # Filter tiny noise regions
            continue

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

        x, y, w, h = cv2.boundingRect(cnt)

        # Normalize to [0, 1] for 3D mapping
        regions.append({
            "centroid_px": [cx, cy],
            "centroid_norm": [cx / W, cy / H],
            "bbox_px": [x, y, w, h],
            "bbox_norm": [x / W, y / H, w / W, h / H],
            "area_px": int(area),
            "area_pct": float(area / (H * W) * 100)
        })

    # Sort by area (largest first)
    regions.sort(key=lambda r: r["area_px"], reverse=True)

    total_tumor_px = int(binary_mask.sum())
    total_brain_px = int((binary_mask >= 0).sum())

    return {
        "regions": regions,
        "num_regions": len(regions),
        "total_tumor_pixels": total_tumor_px,
        "tumor_coverage_pct": float(total_tumor_px / total_brain_px * 100),
        "image_size": [H, W]
    }

def create_2d_overlay(rgb_img, binary_mask, prob_map, gt_mask=None, sample_id=""):
    """Create a 2D overlay visualization containing ONLY the Prediction Overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="#0d1117")

    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # Create Overlay
    overlay = rgb_img.copy()
    tumor_color = np.array([255, 50, 50], dtype=np.uint8)  # Red for tumor
    overlay[binary_mask == 1] = (
        0.4 * overlay[binary_mask == 1] + 0.6 * tumor_color
    ).astype(np.uint8)

    # Draw contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 50, 50), 2)

    # Draw bounding boxes and centroids
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay_bgr, (x, y), (x + w, y + h), (255, 255, 0), 1)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(overlay_bgr, (cx, cy), 4, (0, 255, 255), -1)

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # Plot Prediction Overlay
    ax.imshow(overlay_rgb)
    ax.set_title(f"Prediction Overlay: {sample_id}", color="white", fontsize=15, fontweight="bold")
    ax.axis("off")

    red_patch = mpatches.Patch(color=(1, 0.2, 0.2), label="Tumor Region")
    ax.legend(handles=[red_patch], loc="lower right",
                   facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="white", fontsize=12)

    plt.tight_layout()
    return fig

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Inference")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to preprocessed .npz file")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to raw HSI folder")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.npz:
        npz = Path(args.npz)
        prob, mask, rgb, gt, sample_id = predict_from_npz(npz, args.threshold)
    elif args.folder:
        prob, mask, orig_size = predict_from_folder(args.folder, args.threshold)
        folder = Path(args.folder)
        rgb_path = folder / "image.jpg"
        rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else np.zeros((*orig_size, 3), np.uint8)
        if rgb_path.exists():
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = None
        sample_id = folder.name
    else:
        # Demo: run on all val samples
        npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
        if not npz_files:
            print("No .npz files found. Run step2_preprocess.py first.")
            exit(1)
        npz = npz_files[0]
        prob, mask, rgb, gt, sample_id = predict_from_npz(npz, args.threshold)

    # Extract coordinates
    coords = extract_tumor_coords(mask)
    print(f"\nTumor Detection Results for: {sample_id}")
    print(f"  Tumor regions found: {coords['num_regions']}")
    print(f"  Tumor coverage: {coords['tumor_coverage_pct']:.2f}%")
    for i, region in enumerate(coords["regions"]):
        print(f"  Region {i+1}:")
        print(f"    Centroid (px): {region['centroid_px']}")
        print(f"    Centroid (normalized): {region['centroid_norm']}")
        print(f"    Bounding box (px): {region['bbox_px']}")
        print(f"    Area: {region['area_px']} px ({region['area_pct']:.2f}%)")

    # Save coordinates JSON
    coords["sample_id"] = sample_id
    coords_path = RESULTS_DIR / f"{sample_id}_tumor_coords.json"
    with open(coords_path, "w") as f:
        json.dump(coords, f, indent=2)
    print(f"\nCoordinates saved: {coords_path}")

    # Save 3D mapping coords (for the Three.js viewer)
    if coords["regions"]:
        primary = coords["regions"][0]
        three_d_coords = {
            "sample_id": sample_id,
            "tumor_detected": True,
            "primary_region": {
                "u": primary["centroid_norm"][0],  # 0-1 normalized X
                "v": primary["centroid_norm"][1],  # 0-1 normalized Y
                "size": primary["area_pct"] / 100, # relative size
                "confidence": float(prob[mask == 1].mean()) if mask.sum() > 0 else 0
            },
            "all_regions": [
                {"u": r["centroid_norm"][0], "v": r["centroid_norm"][1],
                 "size": r["area_pct"] / 100, "bbox_norm": r["bbox_norm"]}
                for r in coords["regions"]
            ],
            "tumor_mask_b64": None  # Filled by web app when needed
        }
    else:
        three_d_coords = {"sample_id": sample_id, "tumor_detected": False}

    mapping_path = RESULTS_DIR / f"{sample_id}_3d_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(three_d_coords, f, indent=2)
    print(f"3D mapping saved: {mapping_path}")

    # Generate 2D overlay
    fig = create_2d_overlay(rgb, mask, prob, gt, sample_id)
    overlay_path = RESULTS_DIR / f"{sample_id}_overlay.png"
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"2D overlay saved: {overlay_path}")
    
    # Save Heatmap and Spectral Maps individually
    heatmap_path = RESULTS_DIR / f"{sample_id}_heatmap.png"
    plt.imsave(heatmap_path, prob, cmap="hot")
    print(f"Heatmap saved: {heatmap_path}")
    
    spectral_path = RESULTS_DIR / f"{sample_id}_spectral.png"
    plt.imsave(spectral_path, rgb)
    print(f"Spectral map saved: {spectral_path}")

    print("\nDone!")
