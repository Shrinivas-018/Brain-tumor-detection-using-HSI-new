"""
Step 1: Extract all zip files from the HSI Brain Tumor Dataset.
Extracts all 36 zips into a 'dataset/' subfolder.
"""

import zipfile
import os
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

zips = sorted(BASE_DIR.glob("*.zip"))
print(f"Found {len(zips)} zip files to extract.\n")

total_bytes = sum(z.stat().st_size for z in zips)
print(f"Total compressed size: {total_bytes / 1e9:.2f} GB\n")

for zip_path in tqdm(zips, desc="Extracting zips"):
    folder_name = zip_path.stem  # e.g. "004-02"
    out_dir = DATASET_DIR / folder_name

    if out_dir.exists():
        print(f"  [SKIP] {zip_path.name} already extracted.")
        continue

    print(f"  Extracting {zip_path.name} -> dataset/{folder_name}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract and flatten the folder structure
        for member in z.infolist():
            member_path = Path(member.filename)
            parts = member_path.parts
            if len(parts) > 1:
                rel = Path(*parts[1:])
            else:
                continue  # Skip the folder entry itself

            target = out_dir / rel
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

print("\n[OK] All zips extracted to 'dataset/' folder.")
print("\nDataset structure:")
for folder in sorted(DATASET_DIR.iterdir()):
    if folder.is_dir():
        files = list(folder.iterdir())
        print(f"  {folder.name}/ -> {len(files)} files")
