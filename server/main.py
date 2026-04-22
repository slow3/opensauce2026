"""
Open Sauce 2026 Server — Flask API for the scanning pipeline.
Runs on each rig PC. Receives guest registrations from the iPad kiosk,
watches the Camera2Cloud drop folder, and dispatches the pipeline.

Start: python main.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline.job_manager import job_manager
from pipeline.watcher import FolderWatcher
from pipeline.runner import dispatch

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("opensauce2026.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("main")

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RIG     = CONFIG["rig"]
log.info(f"Open Sauce 2026 starting — RIG {RIG}: {CONFIG['rig_label']} ({CONFIG['station_id']})")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../kiosk", static_url_path="/")
CORS(app)  # allow iPad (different IP) to POST

# ── pending guest sessions ────────────────────────────────────────────────────
# When iPad registers a guest, we store the session here.
# When Camera2Cloud signals images are ready, we match by rig and dispatch.
_pending: dict[str, dict] = {}


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "ok":         True,
        "rig":        RIG,
        "rig_label":  CONFIG["rig_label"],
        "station_id": CONFIG["station_id"],
        "active_jobs": len(job_manager.active()),
    })


@app.post("/api/register")
def register():
    """Receive guest registration from iPad kiosk."""
    data = request.get_json(force=True)

    session_id = data.get("session_id")
    rig        = data.get("rig")
    guest      = data.get("guest", {})

    if not session_id or not rig or not guest.get("name"):
        return jsonify({"ok": False, "error": "Missing required fields"}), 400

    if rig != RIG:
        return jsonify({
            "ok": False,
            "error": f"This PC handles RIG {RIG} only. Guest selected RIG {rig}."
        }), 400

    # Create a job and store pending session
    job = job_manager.create(session_id, rig, guest)
    _pending[session_id] = {"job_id": job.job_id, "data": data}

    log.info(f"Registered: {guest['name']} | session={session_id} | job={job.job_id}")

    return jsonify({
        "ok":          True,
        "job_id":      job.job_id,
        "session_id":  session_id,
        "queue_position": len(job_manager.active()),
    })


@app.post("/api/images-ready")
def images_ready():
    """
    Called by Camera2Cloud webhook (if supported) OR by the internal watcher
    when a drop folder settles. Matches to a pending session and dispatches.
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    src_folder = data.get("folder")

    if not src_folder:
        return jsonify({"ok": False, "error": "Missing folder path"}), 400

    # If session_id given, match to pending job; otherwise use most recent pending
    if session_id and session_id in _pending:
        pending = _pending.pop(session_id)
    elif _pending:
        # Take the oldest pending session (FIFO)
        session_id, pending = next(iter(_pending.items()))
        del _pending[session_id]
    else:
        log.warning(f"Images arrived at {src_folder} but no pending guest session")
        return jsonify({"ok": False, "error": "No pending guest session"}), 404

    job = job_manager.get(pending["job_id"])
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404

    log.info(f"Images ready for job {job.job_id} — dispatching pipeline")
    dispatch(job, src_folder, CONFIG)

    return jsonify({"ok": True, "job_id": job.job_id})


@app.get("/api/jobs")
def list_jobs():
    """Return all jobs for the Jobs tab on the kiosk."""
    return jsonify({"ok": True, "jobs": job_manager.all()})


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Not found"}), 404
    return jsonify({"ok": True, "job": job.to_dict()})


@app.post("/api/sync")
def sync():
    """Batch sync endpoint for pending offline submissions from iPad."""
    data         = request.get_json(force=True)
    submissions  = data.get("submissions", [])
    results      = []

    for sub in submissions:
        try:
            session_id = sub.get("session_id")
            rig        = sub.get("rig")
            guest      = sub.get("guest", {})
            job        = job_manager.create(session_id, rig, guest)
            _pending[session_id] = {"job_id": job.job_id, "data": sub}
            results.append({"session_id": session_id, "ok": True, "job_id": job.job_id})
        except Exception as e:
            results.append({"session_id": sub.get("session_id"), "ok": False, "error": str(e)})

    return jsonify({"ok": True, "results": results})


# ── Camera2Cloud folder watcher ───────────────────────────────────────────────

def _on_images_settled(folder: str, image_count: int):
    """Called by FolderWatcher when a batch of images finishes transferring."""
    log.info(f"Drop folder ready: {folder} ({image_count} images)")

    if not _pending:
        log.warning("Images arrived but no pending guest — queuing for manual match")
        return

    # Take oldest pending session
    session_id = next(iter(_pending))
    pending    = _pending.pop(session_id)
    job        = job_manager.get(pending["job_id"])

    if job:
        log.info(f"Matched images to job {job.job_id} ({job.guest.get('name')})")
        dispatch(job, folder, CONFIG)


def _start_watcher():
    min_images = CONFIG["pipeline"].get(
        "image_count_rig1" if RIG == 1 else "image_count_rig2", 1
    )
    watcher = FolderWatcher(
        drop_folder    = CONFIG["paths"]["camera2cloud_drop"],
        callback       = _on_images_settled,
        settle_seconds = CONFIG["pipeline"].get("image_settle_seconds", 5),
        min_images     = max(1, min_images // 2),  # trigger at 50% expected images
    )
    watcher.start()
    return watcher


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CONFIG["paths"]["projects_root"],     exist_ok=True)
    os.makedirs(CONFIG["paths"]["camera2cloud_drop"], exist_ok=True)

    watcher = _start_watcher()
    log.info(f"Server starting on port {CONFIG['server']['port']}")

    try:
        app.run(
            host  = CONFIG["server"]["host"],
            port  = CONFIG["server"]["port"],
            debug = False,
            use_reloader = False,
        )
    finally:
        watcher.stop()
