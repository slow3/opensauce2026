"""
Pipeline Runner — executes the correct pipeline steps for a given job.
Supports modular modes: FULL, SPLAT, MESH, SPLAT_NO_EMAIL, FROM_COLMAP, COMPRESS_ONLY.
Each step updates the job's status so the kiosk can show live progress.
Steps run sequentially; any failure marks the job failed.
"""

import os
import re
import shutil
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .job_manager import Job, JobStatus, StepStatus, PipelineMode, job_manager

log = logging.getLogger("runner")


def _slug(name: str) -> str:
    """Turn a guest name into a safe folder-name component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())[:32]


def _project_dir(projects_root: str, session_id: str, guest_name: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    folder    = f"{timestamp}_{_slug(guest_name)}_{session_id[-6:]}"
    return os.path.join(projects_root, folder)


def _make_env(extra_path_dirs: list[str] = None) -> dict:
    """Build an env dict with optional extra dirs prepended to PATH.
    Used to inject bundled DLL directories (e.g. COLMAP's lib/) so that
    executables find their dependencies without a system-wide install.
    """
    env = os.environ.copy()
    if extra_path_dirs:
        env["PATH"] = ";".join(extra_path_dirs) + ";" + env.get("PATH", "")
    return env


def _run(cmd: list[str], cwd: Optional[str] = None, timeout: int = 3600,
         extra_path_dirs: list[str] = None) -> subprocess.CompletedProcess:
    log.info(f"$ {' '.join(str(c) for c in cmd)}")
    env = _make_env(extra_path_dirs)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          timeout=timeout, env=env)


# ── PLY crop helpers ──────────────────────────────────────────────────────────

def _crop_ply_box(src: str, dst: str, min_xyz: list, max_xyz: list):
    """Keep Gaussians whose XYZ centre is inside the axis-aligned box."""
    try:
        from plyfile import PlyData, PlyElement
        import numpy as np
    except ImportError:
        raise RuntimeError("plyfile and numpy are required for manual crop — pip install plyfile numpy")

    plydata = PlyData.read(src)
    verts = plydata["vertex"]
    x = np.array(verts["x"], dtype=float)
    y = np.array(verts["y"], dtype=float)
    z = np.array(verts["z"], dtype=float)
    mask = (
        (x >= min_xyz[0]) & (x <= max_xyz[0]) &
        (y >= min_xyz[1]) & (y <= max_xyz[1]) &
        (z >= min_xyz[2]) & (z <= max_xyz[2])
    )
    kept = verts.data[mask]
    el = PlyElement.describe(kept, "vertex")
    PlyData([el], text=False).write(dst)
    log.info(f"Box crop: {int(mask.sum())}/{len(mask)} Gaussians kept")


def _crop_ply_sphere(src: str, dst: str, center: list, radius: float):
    """Keep Gaussians whose XYZ centre is within the sphere."""
    try:
        from plyfile import PlyData, PlyElement
        import numpy as np
    except ImportError:
        raise RuntimeError("plyfile and numpy are required for manual crop — pip install plyfile numpy")

    plydata = PlyData.read(src)
    verts = plydata["vertex"]
    x = np.array(verts["x"], dtype=float)
    y = np.array(verts["y"], dtype=float)
    z = np.array(verts["z"], dtype=float)
    dist2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    mask = dist2 <= float(radius) ** 2
    kept = verts.data[mask]
    el = PlyElement.describe(kept, "vertex")
    PlyData([el], text=False).write(dst)
    log.info(f"Sphere crop: {int(mask.sum())}/{len(mask)} Gaussians kept")


# ── individual steps ─────────────────────────────────────────────────────────

def step_receive_images(job: Job, src_folder: str, config: dict) -> str:
    """Move images from Camera2Cloud drop folder into colmap/images/.

    Images are placed at <project>/colmap/images/ so LichtFeld's --data-path
    (<project>/colmap) resolves the standard COLMAP dataset layout:
        colmap/images/    ← image files
        colmap/sparse/0/  ← COLMAP mapper output (cameras, images, points3D)
    Source files are moved (not copied) to avoid filling the drop folder.
    """
    project_dir = _project_dir(
        config["paths"]["projects_root"],
        job.session_id,
        job.guest.get("name", "guest"),
    )
    colmap_dir = os.path.join(project_dir, "colmap")
    photos_dir = os.path.join(colmap_dir, "images")
    os.makedirs(photos_dir, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".tiff", ".tif", ".png", ".raw", ".cr2", ".nef"}
    moved = 0
    for f in Path(src_folder).iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            shutil.move(str(f), photos_dir)
            moved += 1

    rig_key  = f"image_count_rig{job.rig}"
    expected = config["pipeline"].get(rig_key, 0)
    minimum  = max(1, expected // 2)
    if moved < minimum:
        raise RuntimeError(
            f"Only {moved} images received; expected ~{expected} (minimum {minimum}). "
            f"Drop folder: {src_folder}"
        )

    log.info(f"Received {moved}/{expected} images → {photos_dir}")
    job.project_dir = project_dir
    return project_dir


def step_colmap(job: Job, config: dict):
    """Run COLMAP alignment headlessly.

    Expects images at <project>/colmap/images/.
    COLMAP mapper outputs to <project>/colmap/sparse/0/.

    config["paths"]["colmap_exe"] must point to colmap.exe directly, not COLMAP.bat —
    subprocess cannot invoke .bat files without shell=True.
    """
    colmap_dir = os.path.join(job.project_dir, "colmap")
    photos_dir = os.path.join(colmap_dir, "images")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    db_path    = os.path.join(colmap_dir, "database.db")
    os.makedirs(sparse_dir, exist_ok=True)

    colmap = config["paths"]["colmap_exe"]
    # COLMAP ships bundled DLLs in ../lib — inject into PATH so the exe finds them
    colmap_lib = str(Path(colmap).parent.parent / "lib")
    colmap_env = [colmap_lib] if Path(colmap_lib).exists() else []

    steps = [
        [colmap, "feature_extractor",
         "--database_path", db_path,
         "--image_path", photos_dir,
         "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", "1"],

        [colmap, "exhaustive_matcher",
         "--database_path", db_path,
         "--SiftMatching.use_gpu", "1"],

        [colmap, "mapper",
         "--database_path", db_path,
         "--image_path", photos_dir,
         "--output_path", sparse_dir,
         "--Mapper.ba_global_function_tolerance", "0.000001"],
    ]

    for cmd in steps:
        result = _run(cmd, extra_path_dirs=colmap_env)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP failed:\n{result.stderr}")

    subdirs = [d for d in Path(sparse_dir).iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError(f"COLMAP mapper produced no sparse reconstruction in {sparse_dir}")


def step_lichtfeld(job: Job, config: dict):
    """Run LichtFeld Studio headlessly to produce a .ply splat.

    Correct CLI flags (confirmed from LichtFeld-Studio v0.4.2):
      --data-path   COLMAP dataset root
      --output-path output directory (LichtFeld creates it)
      --iter        training iteration count
      --headless    disable visualisation
      --undistort   required when COLMAP uses SIMPLE_RADIAL or any distorted model
    """
    colmap_dir  = os.path.join(job.project_dir, "colmap")
    splats_dir  = os.path.join(job.project_dir, "Splats")
    os.makedirs(splats_dir, exist_ok=True)

    lichtfeld  = config["paths"]["lichtfeld_exe"]
    iterations = config["pipeline"].get("lichtfeld_iterations", 10000)

    cmd = [
        lichtfeld,
        "--data-path",   colmap_dir,
        "--output-path", splats_dir,
        "--iter",        str(iterations),
        "--headless",
        "--undistort",
    ]

    result = _run(cmd, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(f"LichtFeld failed:\n{result.stderr}")

    plys = sorted(Path(splats_dir).glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise RuntimeError(f"LichtFeld finished but no .ply found in {splats_dir} — check logs")
    job.outputs["raw_ply"] = str(plys[-1])


def step_crop_splat(job: Job, config: dict):
    """Crop the raw .ply splat.

    Decision tree:
    1. If an .rcbox calibration file is configured and exists → auto-crop via rcbox_converter.
    2. Else → pause the pipeline (AWAITING_CROP) until staff POSTs crop config to
       /api/jobs/<id>/crop, then apply box, sphere, or skip as instructed.
    """
    raw_ply = job.outputs.get("raw_ply")
    if not raw_ply or not os.path.exists(raw_ply):
        raise RuntimeError("No raw .ply to crop")

    rcbox = config["paths"].get("rcbox_file", "")

    if rcbox and os.path.exists(rcbox):
        # Automatic crop via .rcbox calibration file
        from .rcbox_converter import trim_ply_with_rcbox
        raw_path   = Path(raw_ply)
        cropped_ply = str(raw_path.with_stem(raw_path.stem + "_cropped"))
        trim_ply_with_rcbox(raw_ply, rcbox, cropped_ply)
        job.outputs["cropped_ply"] = cropped_ply
        log.info(f"[{job.job_id}] Auto-cropped via .rcbox → {cropped_ply}")
        return

    # No .rcbox — pause and wait for staff to provide crop parameters
    log.info(f"[{job.job_id}] No .rcbox found — pausing for manual crop input")
    job.status = JobStatus.AWAITING_CROP
    job.set_step("splat_crop", StepStatus.AWAITING_CROP)

    # Block until main.py calls job.crop_event.set() after /api/jobs/<id>/crop
    job.crop_event.wait()

    # Resumed — set status back to running before continuing
    job.status = JobStatus.RUNNING
    job.set_step("splat_crop", StepStatus.RUNNING)

    crop = job.crop_config
    crop_mode = crop.get("mode", "none")

    if crop_mode == "none":
        log.info(f"[{job.job_id}] Staff chose to skip crop — using raw .ply")
        job.outputs["cropped_ply"] = raw_ply
        return

    raw_path    = Path(raw_ply)
    cropped_ply = str(raw_path.with_stem(raw_path.stem + "_cropped"))

    if crop_mode == "box":
        _crop_ply_box(raw_ply, cropped_ply, crop["min"], crop["max"])
    elif crop_mode == "sphere":
        _crop_ply_sphere(raw_ply, cropped_ply, crop["center"], crop["radius"])
    else:
        log.warning(f"[{job.job_id}] Unknown crop mode '{crop_mode}' — skipping crop")
        job.outputs["cropped_ply"] = raw_ply
        return

    job.outputs["cropped_ply"] = cropped_ply
    log.info(f"[{job.job_id}] Manual crop ({crop_mode}) → {cropped_ply}")


def step_compress_splat(job: Job, config: dict):
    """Compress the cropped .ply using splat-transform CLI.

    splat-transform (@playcanvas/splat-transform, npm) uses POSITIONAL syntax:
        splat-transform [GLOBAL OPTIONS] input [ACTIONS] output
    Action flags (-N, -H) must appear BETWEEN input and output, not before input.

    Flags confirmed from github.com/playcanvas/splat-transform README:
        -N / --filter-nan           Remove NaN/Inf Gaussians
        -H / --filter-harmonics 0   Strip all SH (band 0 only = smallest file)
    Output format is inferred from extension:
        .compressed.ply → PlayCanvas compressed PLY (smaller than plain PLY)
        .html           → self-contained HTML viewer
    """
    source_ply      = job.outputs.get("cropped_ply") or job.outputs.get("raw_ply")
    splat_transform = config["paths"].get("splat_transform", "splat-transform")
    strip_sh        = config["pipeline"].get("splat_strip_sh", True)

    if not source_ply or not os.path.exists(source_ply):
        raise RuntimeError("No .ply to compress")

    source_path  = Path(source_ply)
    compressed   = str(source_path.with_suffix(".compressed.ply"))

    # Correct positional syntax: input [actions] output
    actions = ["--filter-nan"]
    if strip_sh:
        actions += ["--filter-harmonics", "0"]

    cmd = [splat_transform, source_ply] + actions + [compressed]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"splat-transform failed:\n{result.stderr}")

    job.outputs["compressed_ply"] = compressed

    # HTML viewer — best-effort, non-fatal
    stem = source_path.stem.replace("_cropped", "")
    viewer_html = str(source_path.with_stem(stem + "_viewer").with_suffix(".html"))
    result_html = _run([splat_transform, compressed, viewer_html])
    if result_html.returncode == 0 and os.path.exists(viewer_html):
        job.outputs["viewer_html"] = viewer_html
    else:
        log.warning(f"Viewer HTML export failed (non-fatal): {result_html.stderr}")


def step_autorc_mesh(job: Job, config: dict):
    """
    Run AutoRC for mesh reconstruction (RIG 2 only).
    AutoRC has no CLI — we write a config pointing at this job's Photos dir,
    then launch the exe and watch for output in the Models folder.
    """
    import json, time

    autorc_exe    = config["paths"]["autorc_exe"]
    autorc_dir    = os.path.dirname(autorc_exe)
    settings_path = os.path.join(autorc_dir, "data", "settings", "autorc.json")
    preset_path   = config["paths"]["autorc_preset"]
    models_dir    = os.path.join(job.project_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)

    with open(settings_path) as f:
        settings = json.load(f)

    settings["General"]["Preset_Name"] = "opensauce-2026"
    settings["General"]["Preset_Path"] = preset_path

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    proc = subprocess.Popen([autorc_exe], cwd=autorc_dir)

    deadline = time.time() + 7200
    while time.time() < deadline:
        fbx_files = list(Path(models_dir).glob("*.fbx"))
        if fbx_files:
            log.info(f"AutoRC produced {len(fbx_files)} model(s)")
            job.outputs["mesh_fbx"] = str(fbx_files[0])
            proc.terminate()
            return
        time.sleep(30)

    proc.terminate()
    raise RuntimeError("AutoRC timed out — no FBX produced after 2 hours")


def step_prusa_handoff(job: Job, config: dict):
    """Copy the mesh FBX to a Prusa-monitored hot folder (RIG 2 only)."""
    mesh_fbx  = job.outputs.get("mesh_fbx")
    prusa_dir = config["paths"].get("prusa_hotfolder", "")

    if not mesh_fbx:
        raise RuntimeError("No mesh FBX for Prusa handoff")
    if not prusa_dir:
        log.warning("No prusa_hotfolder configured — skipping Prusa handoff")
        return

    os.makedirs(prusa_dir, exist_ok=True)
    dest = os.path.join(prusa_dir, os.path.basename(mesh_fbx))
    shutil.copy2(mesh_fbx, dest)
    log.info(f"Sent to Prusa hot folder: {dest}")
    job.outputs["prusa_file"] = dest


def step_email(job: Job, config: dict):
    """Send the guest their splat link and/or mesh file."""
    from .email_sender import send_delivery_email

    guest = job.guest
    if guest.get("email_declined") or not guest.get("email"):
        log.info("Guest declined email — skipping")
        return

    send_delivery_email(
        config     = config,
        guest      = guest,
        rig        = job.rig,
        session_id = job.session_id,
        outputs    = job.outputs,
    )


# ── step runner ───────────────────────────────────────────────────────────────

def _run_step(job: Job, step_id: str, fn, *args):
    """Set step RUNNING, call fn, set DONE — or FAILED on exception."""
    job.set_step(step_id, StepStatus.RUNNING)
    try:
        fn(*args)
        job.set_step(step_id, StepStatus.DONE)
    except Exception as e:
        log.error(f"[{job.job_id}] Step {step_id} failed: {e}")
        job.set_step(step_id, StepStatus.FAILED)
        raise


# ── pipeline functions ────────────────────────────────────────────────────────

def run_rig1_pipeline(job: Job, src_folder: str, config: dict):
    log.info(f"[{job.job_id}] Starting RIG 1 pipeline for {job.guest.get('name')}")
    job.status = JobStatus.RUNNING
    try:
        _run_step(job, "images_received", step_receive_images, job, src_folder, config)
        _run_step(job, "colmap_align",    step_colmap,         job, config)
        _run_step(job, "lichtfeld_splat", step_lichtfeld,      job, config)
        _run_step(job, "splat_crop",      step_crop_splat,     job, config)
        _run_step(job, "splat_compress",  step_compress_splat, job, config)
        _run_step(job, "email_delivery",  step_email,          job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] RIG 1 pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


def run_rig2_pipeline(job: Job, src_folder: str, config: dict):
    log.info(f"[{job.job_id}] Starting RIG 2 pipeline for {job.guest.get('name')}")
    job.status = JobStatus.RUNNING
    try:
        _run_step(job, "images_received", step_receive_images, job, src_folder, config)
        _run_step(job, "colmap_align",    step_colmap,         job, config)
        _run_step(job, "autorc_mesh",     step_autorc_mesh,    job, config)
        _run_step(job, "lichtfeld_splat", step_lichtfeld,      job, config)
        _run_step(job, "splat_crop",      step_crop_splat,     job, config)
        _run_step(job, "splat_compress",  step_compress_splat, job, config)
        _run_step(job, "prusa_handoff",   step_prusa_handoff,  job, config)
        _run_step(job, "email_delivery",  step_email,          job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] RIG 2 pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


def run_splat_pipeline(job: Job, src_folder: str, config: dict, send_email: bool = True):
    """Splat only: images → COLMAP → splat → crop → compress → (email)."""
    log.info(f"[{job.job_id}] Starting SPLAT pipeline for {job.guest.get('name')}")
    job.status = JobStatus.RUNNING
    try:
        _run_step(job, "images_received", step_receive_images, job, src_folder, config)
        _run_step(job, "colmap_align",    step_colmap,         job, config)
        _run_step(job, "lichtfeld_splat", step_lichtfeld,      job, config)
        _run_step(job, "splat_crop",      step_crop_splat,     job, config)
        _run_step(job, "splat_compress",  step_compress_splat, job, config)
        if send_email:
            _run_step(job, "email_delivery", step_email, job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] SPLAT pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


def run_mesh_pipeline(job: Job, src_folder: str, config: dict):
    """Mesh only: images → AutoRC mesh → email."""
    log.info(f"[{job.job_id}] Starting MESH pipeline for {job.guest.get('name')}")
    job.status = JobStatus.RUNNING

    # Mesh needs a project dir even though we skip COLMAP
    job.project_dir = _project_dir(
        config["paths"]["projects_root"],
        job.session_id,
        job.guest.get("name", "guest"),
    )
    os.makedirs(job.project_dir, exist_ok=True)

    try:
        _run_step(job, "images_received", step_receive_images, job, src_folder, config)
        _run_step(job, "autorc_mesh",     step_autorc_mesh,    job, config)
        _run_step(job, "email_delivery",  step_email,          job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] MESH pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


def run_from_colmap_pipeline(job: Job, colmap_dir: str, config: dict):
    """Skip to LichtFeld using an existing COLMAP output directory.

    colmap_dir should be the dataset root containing images/ and sparse/.
    The job's project_dir is derived from the parent of colmap_dir.
    """
    log.info(f"[{job.job_id}] Starting FROM_COLMAP pipeline")
    job.status = JobStatus.RUNNING

    # Derive project_dir from parent of the supplied colmap dir
    job.project_dir = str(Path(colmap_dir).parent)

    # Ensure the colmap dir is correctly structured
    if not Path(colmap_dir).exists():
        job.status = JobStatus.FAILED
        job.error  = f"COLMAP directory not found: {colmap_dir}"
        return

    try:
        _run_step(job, "lichtfeld_splat", step_lichtfeld,      job, config)
        _run_step(job, "splat_crop",      step_crop_splat,     job, config)
        _run_step(job, "splat_compress",  step_compress_splat, job, config)
        _run_step(job, "email_delivery",  step_email,          job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] FROM_COLMAP pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


def run_compress_only_pipeline(job: Job, ply_path: str, config: dict):
    """Crop + compress an existing .ply, then email.

    ply_path is the raw input .ply.
    """
    log.info(f"[{job.job_id}] Starting COMPRESS_ONLY pipeline for {ply_path}")
    job.status = JobStatus.RUNNING

    job.project_dir = str(Path(ply_path).parent)
    job.outputs["raw_ply"] = ply_path

    if not os.path.exists(ply_path):
        job.status = JobStatus.FAILED
        job.error  = f".ply not found: {ply_path}"
        return

    try:
        _run_step(job, "splat_crop",     step_crop_splat,     job, config)
        _run_step(job, "splat_compress", step_compress_splat, job, config)
        _run_step(job, "email_delivery", step_email,          job, config)
        job.status = JobStatus.COMPLETED
        log.info(f"[{job.job_id}] COMPRESS_ONLY pipeline complete")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)


# ── main dispatch ─────────────────────────────────────────────────────────────

def dispatch(job: Job, config: dict, src_folder: str = None):
    """Launch the correct pipeline in a background thread based on job.mode.

    src_folder   — images drop folder (FULL / SPLAT / MESH modes)
    job.start_path — colmap dir (FROM_COLMAP) or .ply path (COMPRESS_ONLY)
    """
    mode = job.mode

    if mode in (PipelineMode.FULL,):
        fn   = run_rig1_pipeline if job.rig == 1 else run_rig2_pipeline
        args = (job, src_folder, config)

    elif mode == PipelineMode.SPLAT:
        fn   = run_splat_pipeline
        args = (job, src_folder, config)

    elif mode == PipelineMode.SPLAT_NO_EMAIL:
        fn   = run_splat_pipeline
        args = (job, src_folder, config, False)  # send_email=False

    elif mode == PipelineMode.MESH:
        fn   = run_mesh_pipeline
        args = (job, src_folder, config)

    elif mode == PipelineMode.FROM_COLMAP:
        colmap_dir = job.start_path or src_folder
        fn   = run_from_colmap_pipeline
        args = (job, colmap_dir, config)

    elif mode == PipelineMode.COMPRESS_ONLY:
        ply_path = job.start_path or src_folder
        fn   = run_compress_only_pipeline
        args = (job, ply_path, config)

    else:
        log.error(f"[{job.job_id}] Unknown pipeline mode: {mode}")
        job.status = JobStatus.FAILED
        job.error  = f"Unknown pipeline mode: {mode}"
        return

    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
