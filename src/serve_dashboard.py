"""
Local training console: dataset stats, training curves, add data (drag-and-drop),
add new class, update splits, start/continue training.
Run from project root:  python -m src.serve_dashboard
Then open http://127.0.0.1:5001
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import io

from src.dataset_stats import get_dataset_stats
from src.inference import predict_image, DEFAULT_CKPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=None)
DASHBOARD_DIR = PROJECT_ROOT / "dashboard_static"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _sanitize_class_name(name: str) -> str:
    if not name or not name.strip():
        return ""
    s = name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return re.sub(r"[^\w\-]", "", s) or ""


@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/api/dataset_stats")
def api_dataset_stats():
    stats = get_dataset_stats(DATA_PROCESSED, SPLITS_DIR)
    return jsonify(stats)


@app.route("/api/training_log")
def api_training_log():
    log_path = PROJECT_ROOT / "reports" / "training_log.json"
    if not log_path.exists():
        resp = jsonify([])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp
    try:
        with open(log_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, OSError):
        data = []
    resp = jsonify(data)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


def _is_training_process_running(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


@app.route("/api/training_status")
def api_training_status():
    pid_path = PROJECT_ROOT / "reports" / "training_pid.txt"
    running = _is_training_process_running(pid_path)
    if not running and pid_path.exists():
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass
    resp = jsonify({"running": running})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/classes", methods=["GET"])
def api_list_classes():
    if not DATA_PROCESSED.exists():
        return jsonify([])
    classes = sorted([d.name for d in DATA_PROCESSED.iterdir() if d.is_dir()])
    return jsonify(classes)


@app.route("/api/classes", methods=["POST"])
def api_create_class():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    safe = _sanitize_class_name(name)
    if not safe:
        return jsonify({"error": "Invalid or empty class name"}), 400
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    target = DATA_PROCESSED / safe
    if target.exists():
        return jsonify({"error": f"Class already exists: {safe}"}), 400
    target.mkdir(parents=True)
    return jsonify({"class_name": safe})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    class_name = (request.form.get("class_name") or "").strip()
    safe = _sanitize_class_name(class_name)
    if not safe:
        return jsonify({"error": "Invalid or empty class name"}), 400
    target_dir = DATA_PROCESSED / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    saved = 0
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        dest = target_dir / (Path(f.filename).name or f"img_{saved}{ext}")
        try:
            f.save(str(dest))
            saved += 1
        except Exception:
            pass
    return jsonify({"class_name": safe, "saved": saved})


@app.route("/api/update_splits", methods=["POST"])
def api_update_splits():
    if not DATA_PROCESSED.exists():
        return jsonify({"error": "No processed data. Add data first."}), 400
    try:
        subprocess.run(
            [
                sys.executable, "-m", "src.make_splits",
                "--data_dir", str(DATA_PROCESSED),
                "--out_dir", str(SPLITS_DIR),
                "--seed", "42",
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return jsonify({"ok": True})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": (e.stderr or e.stdout or str(e))}), 500


@app.route("/predict")
@app.route("/predict/")
def predict_page():
    path = DASHBOARD_DIR / "predict.html"
    if not path.exists():
        return jsonify({"error": "predict.html not found", "path": str(path)}), 500
    return send_from_directory(str(DASHBOARD_DIR), "predict.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept one image file, return prediction and top-k."""
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "No image file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format. Use {ALLOWED_EXT}"}), 400
    if not DEFAULT_CKPT.exists():
        return jsonify({"error": "Model not found. Train first (checkpoints/best.pt)."}), 503
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400
    try:
        top_k = min(int(request.form.get("top_k", 5)), 20)
    except (TypeError, ValueError):
        top_k = 5
    try:
        result = predict_image(img, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "detail": tb}), 500


@app.route("/api/start_training", methods=["POST"])
def api_start_training():
    if not (SPLITS_DIR / "train.txt").exists():
        return jsonify({"error": "Splits not found. Click 'Update splits' first."}), 400
    try:
        subprocess.Popen(
            [
                sys.executable, "-m", "src.train",
                "--config", "configs/cnn.yaml",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"ok": True, "message": "Training started. Refresh to see progress."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use 5001: port 5000 is often taken by AirPlay on macOS
    app.run(host="127.0.0.1", port=5001, debug=False)
