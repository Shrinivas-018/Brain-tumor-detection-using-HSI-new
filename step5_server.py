"""
Step 5: Flask web server for the Brain Tumor Digital Twin Viewer.
- Serves the 3D GLB brain viewer
- Accepts GLB file uploads
- Accepts inference results JSON
- Serves the interactive Three.js visualization
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from pathlib import Path
import json, os, subprocess, sys, zipfile

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR / "web"), static_url_path="")

@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "web", "index.html")

@app.route("/upload_glb", methods=["POST"])
def upload_glb():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".glb"):
        return jsonify({"error": "Only .glb files accepted"}), 400
    save_path = UPLOADS_DIR / "brain.glb"
    f.save(save_path)
    return jsonify({"status": "ok", "path": "/brain_model"})

@app.route("/brain_model")
def brain_model():
    glb_path = UPLOADS_DIR / "brain.glb"
    if not glb_path.exists():
        return jsonify({"error": "No GLB uploaded yet"}), 404
    return send_file(glb_path, mimetype="model/gltf-binary")

@app.route("/upload_hsi", methods=["POST"])
def upload_hsi():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".zip"):
        return jsonify({"error": "Only .zip HSI files accepted"}), 400
    
    sample_id = f.filename.replace(".zip", "")
    zip_path = UPLOADS_DIR / f.filename
    f.save(zip_path)
    
    # Extract
    sample_folder = UPLOADS_DIR / sample_id
    sample_folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(sample_folder)
        
    # Handle if zip contains a subfolder with the same name
    if (sample_folder / sample_id).exists():
        sample_folder = sample_folder / sample_id
        
    # Run Inference
    print(f"Running inference on uploaded folder: {sample_folder}")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "step4_inference.py"),
         "--folder", str(sample_folder), "--threshold", "0.5"],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500
        
    mapping_file = RESULTS_DIR / f"{sample_id}_3d_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as mf:
            mapping = json.load(mf)
        return jsonify({"status": "ok", "result": mapping})
    return jsonify({"error": "Inference finished but no mapping JSON generated."}), 500


@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/list_results")
def list_results():
    jsons = sorted(RESULTS_DIR.glob("*_3d_mapping.json"))
    items = []
    for j in jsons:
        with open(j) as f:
            data = json.load(f)
        overlay = j.with_name(j.name.replace("_3d_mapping.json", "_overlay.png"))
        items.append({
            "sample_id": data.get("sample_id", j.stem),
            "tumor_detected": data.get("tumor_detected", False),
            "mapping_url": f"/results/{j.name}",
            "overlay_url": f"/results/{overlay.name}" if overlay.exists() else None
        })
    return jsonify(items)

@app.route("/run_inference", methods=["POST"])
def run_inference():
    """Trigger inference on a specific sample."""
    data = request.get_json()
    npz_path = data.get("npz_path")
    threshold = data.get("threshold", 0.5)
    if not npz_path or not Path(npz_path).exists():
        return jsonify({"error": "Invalid npz_path"}), 400
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "step4_inference.py"),
         "--npz", npz_path, "--threshold", str(threshold)],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500
    # Find the output JSON
    sample_id = Path(npz_path).stem
    mapping_file = RESULTS_DIR / f"{sample_id}_3d_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            mapping = json.load(f)
        return jsonify({"status": "ok", "result": mapping})
    return jsonify({"status": "ok", "output": result.stdout})

if __name__ == "__main__":
    print("Starting Brain Tumor Digital Twin Server...")
    print("Open: http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
