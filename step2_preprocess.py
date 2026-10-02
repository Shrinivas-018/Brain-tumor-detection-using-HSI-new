"""
Step 2: Preprocess HSI Brain Tumor Dataset.
- Reads ENVI .raw + .hdr files
- Applies dark/white reference calibration (reflectance = (raw - dark) / (white - dark))
- Reduces 826 bands to key spectral features using PCA or band selection
- Saves numpy arrays for training: X (H x W x features), y (H x W) labels
"""

import numpy as np
import spectral.io.envi as envi
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import cv2

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Benchmark subset as specified in the README paper
BENCHMARK_IDS = [
    "004-02", "005-01", "007-01", "008-01", "008-02",
    "010-03", "012-01", "012-02", "013-01", "014-01",
    "015-01", "016-01", "016-02", "016-03", "016-04",
    "016-05", "017-01", "018-01", "018-02", "019-01",
    "020-01", "021-01", "021-02", "021-05", "022-01",
    "022-02", "022-03"
]

NUM_PCA_COMPONENTS = 30  # Reduce from 826 -> 30 principal components

def parse_hdr(hdr_path):
    """Parse ENVI .hdr file to get image dimensions."""
    info = {}
    with open(hdr_path, "r", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")
    for line in content.split("\n"):
        if "=" in line and not line.strip().startswith("{"):
            key, _, val = line.partition("=")
            info[key.strip()] = val.strip()
    return info

def load_envi_raw(folder):
    """
    Load raw HSI cube using manual parsing.
    Returns (H x W x B) float32 array.
    """
    raw_path = folder / "raw"
    hdr_path = folder / "raw.hdr"

    if not raw_path.exists() or not hdr_path.exists():
        return None, None, None

    info = parse_hdr(hdr_path)
    samples = int(info.get("samples", 345))    # W
    lines   = int(info.get("lines", 389))      # H
    bands   = int(info.get("bands", 826))      # B
    dtype_id = int(info.get("data type", 12))

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = dtype_map.get(dtype_id, np.uint16)

    # BIL interleave: lines x bands x samples
    raw_data = np.fromfile(str(raw_path), dtype=dtype)
    cube = raw_data.reshape((lines, bands, samples))
    cube = np.transpose(cube, (0, 2, 1))  # -> H x W x B
    return cube.astype(np.float32), samples, lines

def load_reference(folder, name):
    """Load dark or white reference (1 x W x B after BIL)."""
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
    ref = np.transpose(ref, (0, 2, 1))  # -> lines x W x B
    # Average across lines -> (W x B)
    return ref.mean(axis=0).astype(np.float32)

def load_gt_map(folder):
    """Load ground truth label map -> H x W uint8."""
    gt_path = folder / "gtMap"
    hdr_path = folder / "gtMap.hdr"
    if not gt_path.exists():
        return None

    info = parse_hdr(hdr_path)
    samples = int(info.get("samples", 345))
    lines   = int(info.get("lines", 389))
    bands   = int(info.get("bands", 1))
    dtype_id = int(info.get("data type", 12))
    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = dtype_map.get(dtype_id, np.uint16)

    gt = np.fromfile(str(gt_path), dtype=dtype)
    gt = gt.reshape((lines, bands, samples))  # BIL: lines x 1 x samples
    gt = np.transpose(gt, (0, 2, 1)).squeeze()  # -> H x W
    return gt.astype(np.uint8)

def calibrate(cube, dark, white):
    """Reflectance calibration: R = (raw - dark) / (white - dark)."""
    dark_bc = dark[np.newaxis, :, :]    # 1 x W x B
    white_bc = white[np.newaxis, :, :]  # 1 x W x B
    denom = white_bc - dark_bc
    denom = np.where(denom == 0, 1e-6, denom)
    calibrated = (cube - dark_bc) / denom
    calibrated = np.clip(calibrated, 0, 1)
    return calibrated

# ─── PASS 1: Load all data and fit PCA ───────────────────────────────────────
print("=" * 60)
print("PASS 1: Loading all samples to fit PCA...")
print("=" * 60)

all_spectra = []   # Store flat pixel spectra for PCA fitting
sample_ids = []    # Track which folders are valid

folders = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
# Filter to benchmark IDs only
folders = [f for f in folders if f.name in BENCHMARK_IDS]
print(f"Processing {len(folders)} benchmark samples.")

meta = {}  # Store metadata per sample

for folder in tqdm(folders, desc="Loading HSI cubes"):
    cube, W, H = load_envi_raw(folder)
    if cube is None:
        print(f"  [SKIP] {folder.name}: raw file missing")
        continue

    dark  = load_reference(folder, "darkReference")
    white = load_reference(folder, "whiteReference")
    gt    = load_gt_map(folder)

    if dark is None or white is None or gt is None:
        print(f"  [SKIP] {folder.name}: reference or gt missing")
        continue

    # Calibrate
    calibrated = calibrate(cube, dark, white)  # H x W x B

    # Only use labeled pixels (label > 0) for PCA fitting
    mask = (gt > 0) & (gt < 4)  # Exclude background=4 and unlabeled=0
    pixels = calibrated[mask]    # N_labeled x 826

    all_spectra.append(pixels)
    meta[folder.name] = {
        "shape": calibrated.shape,
        "gt_shape": gt.shape,
        "n_pixels": pixels.shape[0]
    }
    sample_ids.append(folder.name)

all_spectra_np = np.vstack(all_spectra)
print(f"\nTotal labeled pixels collected: {all_spectra_np.shape[0]:,}")
print(f"Spectral dimension: {all_spectra_np.shape[1]} bands")

# ─── Fit PCA ──────────────────────────────────────────────────────────────────
print(f"\nFitting PCA (826 -> {NUM_PCA_COMPONENTS} components)...")
# Sample up to 500k pixels for efficiency
if all_spectra_np.shape[0] > 500_000:
    idx = np.random.choice(all_spectra_np.shape[0], 500_000, replace=False)
    pca_train = all_spectra_np[idx]
else:
    pca_train = all_spectra_np

scaler = StandardScaler()
pca_train_scaled = scaler.fit_transform(pca_train)

pca = PCA(n_components=NUM_PCA_COMPONENTS, whiten=True)
pca.fit(pca_train_scaled)

var_explained = pca.explained_variance_ratio_.cumsum()[-1]
print(f"Variance explained by {NUM_PCA_COMPONENTS} PCs: {var_explained*100:.1f}%")

# Save PCA model
pca_path = PROCESSED_DIR / "pca_model.pkl"
with open(pca_path, "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca, "n_components": NUM_PCA_COMPONENTS}, f)
print(f"PCA model saved to {pca_path}")

# ─── PASS 2: Transform and save all samples ───────────────────────────────────
print("\n" + "=" * 60)
print("PASS 2: Transforming and saving processed samples...")
print("=" * 60)

TARGET_H, TARGET_W = 256, 256  # Resize to fixed size for batching

for folder in tqdm(folders, desc="Processing samples"):
    if folder.name not in sample_ids:
        continue

    out_path = PROCESSED_DIR / f"{folder.name}.npz"
    if out_path.exists():
        print(f"  [SKIP] {folder.name} already processed.")
        continue

    cube, W, H = load_envi_raw(folder)
    dark  = load_reference(folder, "darkReference")
    white = load_reference(folder, "whiteReference")
    gt    = load_gt_map(folder)

    calibrated = calibrate(cube, dark, white)   # H x W x 826
    H_orig, W_orig, B = calibrated.shape

    # Reshape to pixels, scale, PCA
    pixels_flat = calibrated.reshape(-1, B)
    pixels_scaled = scaler.transform(pixels_flat)
    pixels_pca = pca.transform(pixels_scaled).astype(np.float32)  # N x 30
    cube_pca = pixels_pca.reshape(H_orig, W_orig, NUM_PCA_COMPONENTS)  # H x W x 30

    # Resize to target size
    cube_resized = cv2.resize(cube_pca, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_LINEAR)  # 256x256x30
    gt_resized = cv2.resize(gt, (TARGET_W, TARGET_H),
                             interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # Binary tumor mask: 1=tumor, 0=everything else
    # Labels: 0=unlabeled, 1=normal, 2=tumor, 3=hypervascularized, 4=background
    tumor_mask = (gt_resized == 2).astype(np.uint8)

    # Also load the RGB image
    rgb_path = folder / "image.jpg"
    if rgb_path.exists():
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (TARGET_W, TARGET_H))
    else:
        # Synthesize RGB from PCA bands 0,1,2
        rgb = (cube_resized[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

    np.savez_compressed(
        out_path,
        hsi=cube_resized,           # (256, 256, 30) float32 PCA features
        gt=gt_resized,               # (256, 256) uint8 multi-class labels
        tumor_mask=tumor_mask,       # (256, 256) uint8 binary tumor mask
        rgb=rgb,                     # (256, 256, 3) uint8 RGB image
        sample_id=folder.name
    )

print("\n[OK] All samples preprocessed and saved to 'processed/'")
print(f"Files saved: {len(list(PROCESSED_DIR.glob('*.npz')))}")

# Print class distribution
print("\nClass distribution across dataset:")
total_pixels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for npz in PROCESSED_DIR.glob("*.npz"):
    data = np.load(npz)
    gt = data["gt"]
    for cls in range(5):
        total_pixels[cls] += (gt == cls).sum()

class_names = {0: "Unlabeled", 1: "Normal", 2: "Tumor", 3: "Hypervasc.", 4: "Background"}
total = sum(total_pixels.values())
for cls, count in total_pixels.items():
    print(f"  Class {cls} ({class_names[cls]}): {count:,} pixels ({count/total*100:.1f}%)")
