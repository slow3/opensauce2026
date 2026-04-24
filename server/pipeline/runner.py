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
         extra_path_dirs: list[str] = None,
         hide_window: bool = False) -> subprocess.CompletedProcess:
    log.info(f"$ {' '.join(str(c) for c in cmd)}")
    env = _make_env(extra_path_dirs)
    # CREATE_NO_WINDOW (0x08000000) keeps GUI apps from flashing a window on screen.
    # Safe on Windows; ignored on other platforms via the hasattr guard.
    flags = subprocess.CREATE_NO_WINDOW if hide_window and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          timeout=timeout, env=env, creationflags=flags)


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


# ── preview generation helpers ────────────────────────────────────────────────

def _preview_dir(job: Job) -> str:
    d = os.path.join(job.project_dir, "previews")
    os.makedirs(d, exist_ok=True)
    return d


def _scatter_svg(xs: list, ys: list, *, width=320, height=200,
                 dot_r=1.5, color="#1D9E75", bg="#0a0a0f",
                 box=None, title="") -> str:
    """Minimal SVG scatter plot (top-down projection).

    box: (x_min, y_min, x_max, y_max) in data coordinates — drawn as amber dashed rect.
    """
    if not xs:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'style="background:{bg};border-radius:6px"></svg>')
    pad = 14
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span_x = mx_x - mn_x or 1.0
    span_y = mx_y - mn_y or 1.0
    W, H = width - 2 * pad, height - 2 * pad

    def tx(v): return pad + (v - mn_x) / span_x * W
    def ty(v): return height - pad - (v - mn_y) / span_y * H

    dots = "".join(
        f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="{dot_r}" '
        f'fill="{color}" fill-opacity="0.6"/>'
        for x, y in zip(xs, ys)
    )
    box_svg = ""
    if box:
        bx0, by0, bx1, by1 = box
        sx0, sy0 = tx(bx0), ty(by1)
        sx1, sy1 = tx(bx1), ty(by0)
        bw, bh = sx1 - sx0, sy1 - sy0
        box_svg = (f'<rect x="{sx0:.1f}" y="{sy0:.1f}" width="{max(bw,1):.1f}" height="{max(bh,1):.1f}" '
                   f'fill="#EF9F2720" stroke="#EF9F27" stroke-width="1.5" stroke-dasharray="5,3"/>')

    title_svg = (f'<text x="{width//2}" y="11" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="9" fill="#666">{title}</text>') if title else ""

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'style="background:{bg};border-radius:6px">'
            f'{title_svg}{dots}{box_svg}</svg>')


def _quat_to_center(qw, qx, qy, qz, tx, ty, tz):
    """Convert COLMAP quaternion + translation to camera center in world space."""
    r00 = 1 - 2*(qy*qy + qz*qz);  r10 = 2*(qx*qy + qz*qw);  r20 = 2*(qx*qz - qy*qw)
    r01 = 2*(qx*qy - qz*qw);       r11 = 1 - 2*(qx*qx + qz*qz); r21 = 2*(qy*qz + qx*qw)
    r02 = 2*(qx*qz + qy*qw);       r12 = 2*(qy*qz - qx*qw);  r22 = 1 - 2*(qx*qx + qy*qy)
    cx = -(r00*tx + r10*ty + r20*tz)
    cy = -(r01*tx + r11*ty + r21*tz)
    cz = -(r02*tx + r12*ty + r22*tz)
    return cx, cy, cz


def _read_colmap_images_bin(path: str) -> list:
    """Parse COLMAP images.bin → list of (cx, cy, cz) camera world positions."""
    import struct
    centers = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            f.read(4)                                         # IMAGE_ID
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz      = struct.unpack("<ddd",  f.read(24))
            f.read(4)                                         # CAMERA_ID
            while f.read(1) != b"\x00":                      # null-terminated name
                pass
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)                              # skip POINTS2D (x,y,id)
            centers.append(_quat_to_center(qw, qx, qy, qz, tx, ty, tz))
    return centers


def _filter_colmap_points(sparse_dir: str) -> int:
    """Remove statistical outliers from COLMAP points3D.bin (3-sigma in XYZ).

    COLMAP routinely produces a few mismatched-feature triangulations that land
    far outside the scene (e.g. 100× the scene scale). Those outlier seeds cause
    NaN/Inf gradient explosions in 3DGS training within the first 10 iterations.
    Filtering them out before passing the dataset to LichtFeld prevents this.

    Returns the number of points removed.
    """
    import struct
    import numpy as np

    bin_path = os.path.join(sparse_dir, "0", "points3D.bin")
    if not os.path.exists(bin_path):
        return 0

    points = []
    with open(bin_path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid    = struct.unpack("<Q",   f.read(8))[0]
            xyz    = struct.unpack("<ddd", f.read(24))
            rgb    = struct.unpack("<BBB", f.read(3))
            error  = struct.unpack("<d",   f.read(8))[0]
            tlen   = struct.unpack("<Q",   f.read(8))[0]
            track  = f.read(tlen * 8)          # (IMAGE_ID u32, POINT2D_IDX u32) * n
            points.append((pid, xyz, rgb, error, tlen, track))

    if not points:
        return 0

    coords = np.array([p[1] for p in points])   # (N, 3)
    median = np.median(coords, axis=0)
    std    = np.std(coords, axis=0)
    std[std < 1e-10] = 1.0                       # avoid division by zero
    z_max  = np.abs((coords - median) / std).max(axis=1)
    keep   = z_max <= 3.0
    removed = int((~keep).sum())

    if removed == 0:
        return 0

    kept = [p for p, k in zip(points, keep) if k]
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for pid, xyz, rgb, error, tlen, track in kept:
            f.write(struct.pack("<Q",   pid))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *rgb))
            f.write(struct.pack("<d",   error))
            f.write(struct.pack("<Q",   tlen))
            f.write(track)

    return removed


def _save_capture_preview(job: Job, photos_dir: str):
    """Copy (and thumbnail) the middle camera image as previews/capture.jpg."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".tiff", ".tif", ".png"}
    imgs = sorted(p for p in Path(photos_dir).iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        return
    mid = imgs[len(imgs) // 2]
    dest = os.path.join(_preview_dir(job), "capture.jpg")
    try:
        from PIL import Image as PILImage
        img = PILImage.open(str(mid))
        img.thumbnail((960, 720))
        img.convert("RGB").save(dest, "JPEG", quality=82)
    except Exception:
        shutil.copy2(str(mid), dest)
    job.previews["capture"] = "capture.jpg"
    log.info(f"[{job.job_id}] Capture preview → {dest}")


def _save_colmap_preview(job: Job, sparse_dir: str):
    """Render COLMAP camera positions as a top-down SVG scatter."""
    bin_path = os.path.join(sparse_dir, "0", "images.bin")
    if not os.path.exists(bin_path):
        return
    try:
        centers = _read_colmap_images_bin(bin_path)
        if not centers:
            return
        xs = [c[0] for c in centers]
        zs = [c[2] for c in centers]
        svg = _scatter_svg(xs, zs, width=320, height=200, dot_r=3,
                           color="#1D9E75", title=f"COLMAP — {len(centers)} cameras")
        dest = os.path.join(_preview_dir(job), "colmap.svg")
        with open(dest, "w") as f:
            f.write(svg)
        job.previews["colmap"] = "colmap.svg"
        log.info(f"[{job.job_id}] COLMAP preview → {dest} ({len(centers)} cameras)")
    except Exception as e:
        log.warning(f"[{job.job_id}] COLMAP preview failed (non-fatal): {e}")


def _save_splat_preview(job: Job, box: dict = None):
    """Render raw .ply as a top-down XZ scatter SVG.

    box: job.crop_config dict (optional) — draws the crop region as an amber overlay.
    """
    raw_ply = job.outputs.get("raw_ply")
    if not raw_ply or not os.path.exists(raw_ply):
        return
    try:
        from plyfile import PlyData
        import numpy as np
        verts = PlyData.read(raw_ply)["vertex"]
        n = len(verts)
        step = max(1, n // 5000)
        xs = np.array(verts["x"][::step], dtype=float).tolist()
        zs = np.array(verts["z"][::step], dtype=float).tolist()

        box_2d = None
        fname  = "splat.svg"
        title  = "Splat — top-down XZ"
        if box:
            mode = box.get("mode", "none")
            if mode == "box":
                mn, mx = box["min"], box["max"]
                box_2d = (mn[0], mn[2], mx[0], mx[2])
                fname  = "crop.svg"
                title  = f"Crop (box) — top-down XZ"
            elif mode == "sphere":
                c, r = box["center"], float(box["radius"])
                box_2d = (c[0]-r, c[2]-r, c[0]+r, c[2]+r)
                fname  = "crop.svg"
                title  = f"Crop (sphere r={r:.2f})"

        svg = _scatter_svg(xs, zs, width=320, height=200, dot_r=1.2,
                           color="#1D9E75", box=box_2d, title=title)
        dest = os.path.join(_preview_dir(job), fname)
        with open(dest, "w") as f:
            f.write(svg)
        key = "crop" if box else "splat"
        job.previews[key] = fname
        log.info(f"[{job.job_id}] {'Crop' if box else 'Splat'} preview → {dest}")
    except Exception as e:
        log.warning(f"[{job.job_id}] Splat preview failed (non-fatal): {e}")


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
    _save_capture_preview(job, photos_dir)
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

    removed = _filter_colmap_points(sparse_dir)
    if removed:
        log.info(f"[{job.job_id}] Filtered {removed} outlier point(s) from COLMAP sparse model")

    _save_colmap_preview(job, sparse_dir)


def step_lichtfeld(job: Job, config: dict):
    """Run LichtFeld Studio headlessly to produce a .ply splat.

    CLI flags used (confirmed from LichtFeld-Studio tag 5a92bff):
      --data-path   COLMAP dataset root (must contain images/ and sparse/)
      --output-path output directory (created automatically)
      --iter        training iteration count
      --headless    disable the visualisation window (true headless)
      --train       start training immediately without waiting for GUI input
      --log-file    write log to a file instead of stdout (avoids pipe buffer block)
      --strategy    adc (recommended for non-360° rigs) or mcmc
      --gut         handle lens distortion analytically in-training (GUT mode);
                    unlike --undistort (which pre-undistorts pixels), --gut is
                    stable at resize_factor 2. Use --gut, not --undistort.
      --resize_factor downscale factor; "auto" (default) causes NaN/Inf at iter 10
                    for large sensors (e.g. 7728×5152). Set to 2 for best
                    quality/stability tradeoff, 4 for faster training.
      --config      optional JSON config file (export from File → Export Config in GUI)
    """
    colmap_dir  = os.path.join(job.project_dir, "colmap")
    splats_dir  = os.path.join(job.project_dir, "Splats")
    os.makedirs(splats_dir, exist_ok=True)

    lichtfeld     = config["paths"]["lichtfeld_exe"]
    iterations    = config["pipeline"].get("lichtfeld_iterations", 10000)
    lichtfeld_cfg = config["paths"].get("lichtfeld_config", "")
    strategy      = config["pipeline"].get("lichtfeld_strategy", "adc")
    resize_factor = str(config["pipeline"].get("lichtfeld_resize_factor", 2))
    lf_log_path   = os.path.join(job.project_dir, "lichtfeld.log")

    cmd = [
        lichtfeld,
        "--data-path",     colmap_dir,
        "--output-path",   splats_dir,
        "--iter",          str(iterations),
        "--strategy",      strategy,
        "--headless",      # disable the visualisation window — no GPU interop needed
        "--train",         # start training immediately without waiting for GUI
        "--log-file",      lf_log_path,
        "--gut",           # GUT mode — handles COLMAP lens distortion analytically
        "--resize_factor", resize_factor,
    ]

    # Optional JSON config file — lets staff tune training params via LichtFeld GUI:
    #   1. Open LichtFeld GUI, set preferred parameters
    #   2. File → Export Config → save to the path in config.json paths.lichtfeld_config
    if lichtfeld_cfg and os.path.exists(lichtfeld_cfg):
        cmd += ["--config", lichtfeld_cfg]
        log.info(f"LichtFeld: using config file {lichtfeld_cfg}")

    log.info(f"[{job.job_id}] LichtFeld log → {lf_log_path}")
    proc = subprocess.run(cmd, timeout=7200)

    if proc.returncode != 0:
        # Read last 2000 chars of log for the error message
        try:
            with open(lf_log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-2000:]
        except Exception:
            tail = "(could not read log)"
        raise RuntimeError(f"LichtFeld failed (rc={proc.returncode}):\n{tail}")

    plys = sorted(Path(splats_dir).glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise RuntimeError(f"LichtFeld finished but no .ply found in {splats_dir} — check logs")
    job.outputs["raw_ply"] = str(plys[-1])
    _save_splat_preview(job)


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
    _save_splat_preview(job, box=crop)
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


def _write_simplify_params(project_dir: str, count: int) -> str:
    """Write a temporary RealityCapture simplify params XML for the given triangle count.

    Uses the same format as the AutoRC PowerShell scripts (simplify-400k-params.xml etc.).
    mvsFltSimplificationType 0 = absolute triangle count.
    """
    import uuid
    cfg_id = str(uuid.uuid4()).upper()
    xml = (
        f'<Configuration id="{{{cfg_id}}}">\n'
        f'  <entry key="mvsFltSimplificationType" value="0"/>\n'
        f'  <entry key="mvsFltTargetTrisCountAbs" value="{count}"/>\n'
        f'  <entry key="mvsFltBorderDecimationStyle" value="1"/>\n'
        f'  <entry key="simplPreserveParts" value="2"/>\n'
        f'</Configuration>'
    )
    path = os.path.join(project_dir, f"simplify-{count}-params.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def step_rc_mesh(job: Job, config: dict):
    """Run RealityCapture headlessly via CLI to produce a textured mesh.

    Direct replacement for AutoRC. Uses the exact CLI flag sequence confirmed
    from AutoRC's _AutoRC_Create.ps1 PowerShell source.

    Workflow:
      addFolder → detectMarkers → align → [importGCP → update] →
      selectMaximalComponent → setReconstructionRegion →
      calculateHighModel → removeMarginalTriangles →
      simplify(web_lod) → unwrap → calculateTexture → export FBX →
      selectHigh → simplify(print_lod) → unwrap → reprojectTexture → export OBJ →
      save → quit

    Outputs stored in job.outputs:
      "mesh_fbx"  — web LOD FBX  (passed to Prusa handoff / email)
      "mesh_obj"  — print LOD OBJ (lower triangle count, Prusa-ready)
      "mesh_high" — full-res FBX  (archive, skipped if rc_skip_highpoly=true)
      "rc_project"— .rcproj path

    config["paths"]["rc_exe"]             — path to RealityCapture.exe
    config["rc"]["markers_params"]         — markers-params.xml path
    config["rc"]["gcp_params"]            — gcp-params.xml path (optional)
    config["rc"]["reproject_params"]      — reproject-params.xml path
    config["rc"]["unwrap_params"]         — unwrap-params.xml path
    config["rc"]["reconstruction_region"] — .rcbox path (optional, falls back to auto)
    config["rc"]["gcp_file"]             — GCP CSV path (optional, skipped if empty)
    config["rc"]["web_lod_count"]        — triangle count for web/email LOD (default 500000)
    config["rc"]["print_lod_count"]      — triangle count for print LOD (default 100000)
    config["rc"]["skip_highpoly_export"] — bool, skip exporting the full-res mesh (default false)
    config["rc"]["detail"]               — "high" | "normal" | "preview" (default "high")
    """
    rc_exe  = config["paths"]["rc_exe"]
    rc_cfg  = config.get("rc", {})

    # Default param file locations from the AutoRC data directory
    autorc_settings = config["paths"].get(
        "autorc_data_dir",
        "D:\\Reconstructions\\20221223-AutoRC\\data\\settings"
    )

    markers_params        = rc_cfg.get("markers_params",
                            os.path.join(autorc_settings, "markers-params.xml"))
    gcp_params            = rc_cfg.get("gcp_params",
                            os.path.join(autorc_settings, "gcp-params.xml"))
    reproject_params      = rc_cfg.get("reproject_params",
                            os.path.join(autorc_settings, "reproject-params.xml"))
    unwrap_params         = rc_cfg.get("unwrap_params",
                            os.path.join(autorc_settings, "unwrap-params.xml"))
    reconstruction_region = rc_cfg.get("reconstruction_region", "")
    gcp_file              = rc_cfg.get("gcp_file", "")
    web_lod_count         = int(rc_cfg.get("web_lod_count",   500_000))
    print_lod_count       = int(rc_cfg.get("print_lod_count", 100_000))
    skip_highpoly_export  = rc_cfg.get("skip_highpoly_export", False)
    detail                = rc_cfg.get("detail", "high")  # high | normal | preview

    # Detail → RC CLI command
    detail_cmd = {
        "high":    "-calculateHighModel",
        "normal":  "-calculateNormalModel",
        "preview": "-calculatePreviewModel",
    }.get(detail, "-calculateHighModel")

    # Paths
    images_dir   = os.path.join(job.project_dir, "colmap", "images")
    models_dir   = os.path.join(job.project_dir, "Models")
    project_path = os.path.join(job.project_dir, f"{job.job_id}.rcproj")
    os.makedirs(models_dir, exist_ok=True)

    # Model names
    name_high  = f"{job.job_id}_High"
    name_web   = f"{job.job_id}_{web_lod_count // 1000}k"
    name_print = f"{job.job_id}_{print_lod_count // 1000}k"

    path_high  = os.path.join(models_dir, name_high  + ".fbx")
    path_web   = os.path.join(models_dir, name_web   + ".fbx")
    path_print = os.path.join(models_dir, name_print + ".obj")

    # Simplify params XMLs (generated from our triangle counts)
    simplify_web   = _write_simplify_params(job.project_dir, web_lod_count)
    simplify_print = _write_simplify_params(job.project_dir, print_lod_count)

    # ── Build CLI command sequence ─────────────────────────────────────────
    cmd = [
        rc_exe,
        "-set", "appIncSubdirs=true",
        "-addFolder", images_dir,
    ]

    # Marker detection (skip gracefully if XML missing)
    if os.path.exists(markers_params):
        cmd += ["-detectMarkers", markers_params]
    else:
        log.warning("markers-params.xml not found — skipping marker detection")

    cmd += ["-align"]

    # GCP import (optional — skip if no file configured)
    if gcp_file and os.path.exists(gcp_file):
        cmd += ["-importGroundControlPoints", gcp_file, gcp_params, "-update"]
    elif gcp_file:
        log.warning(f"GCP file not found: {gcp_file} — skipping GCP import")

    cmd += ["-selectMaximalComponent"]

    # Reconstruction region
    if reconstruction_region and os.path.exists(reconstruction_region):
        cmd += ["-setReconstructionRegion", reconstruction_region]
    else:
        log.info("No reconstruction region configured — RC will use auto bounds")

    # High-poly reconstruction + clean up marginal triangles
    cmd += [
        detail_cmd,
        "-selectMarginalTriangles", "-removeSelectedTriangles",
        "-renameSelectedModel", name_high,
        "-save", project_path,
    ]

    # ── Web LOD: simplify → unwrap → texture → export FBX ─────────────────
    cmd += [
        "-selectModel",        name_high,
        "-simplify",           simplify_web,
        "-renameSelectedModel", name_web,
        "-cleanModel", "-closeHoles",
        "-unwrap",             unwrap_params,
        "-calculateTexture",
        "-exportModel",        name_web, path_web,
        "-save",               project_path,
    ]

    # ── Print LOD: simplify → unwrap → reproject from web LOD → export OBJ ─
    cmd += [
        "-selectModel",        name_high,
        "-simplify",           simplify_print,
        "-renameSelectedModel", name_print,
        "-unwrap",             unwrap_params,
        "-reprojectTexture",   name_web, name_print, reproject_params,
        "-selectModel",        name_print,
        "-exportModel",        name_print, path_print,
        "-save",               project_path,
    ]

    # ── Optional full-res archive export ──────────────────────────────────
    if not skip_highpoly_export:
        cmd += [
            "-selectModel", name_high,
            "-exportModel", name_high, path_high,
            "-save",        project_path,
        ]

    cmd += ["-quit"]

    # ── Run RC ─────────────────────────────────────────────────────────────
    log.info(f"[{job.job_id}] Starting RealityCapture ({detail} detail, "
             f"web={web_lod_count//1000}k, print={print_lod_count//1000}k)")
    result = _run(cmd, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(f"RealityCapture failed (rc={result.returncode}):\n{result.stderr}")

    # Verify output
    produced = [p for p in (path_web, path_print, path_high) if os.path.exists(p)]
    if not produced:
        raise RuntimeError(
            f"RealityCapture exited cleanly but no model files found in {models_dir}. "
            f"Check the RC log for errors.")

    job.outputs["mesh_fbx"]   = path_web    # Prusa handoff + email attachment
    job.outputs["mesh_obj"]   = path_print  # print-ready OBJ
    job.outputs["rc_project"] = project_path
    if os.path.exists(path_high):
        job.outputs["mesh_high"] = path_high

    log.info(f"[{job.job_id}] RC mesh complete: {len(produced)} model(s) → {models_dir}")


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
        _run_step(job, "autorc_mesh",     step_rc_mesh,        job, config)
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
    """Mesh only: images → RC mesh → email."""
    log.info(f"[{job.job_id}] Starting MESH pipeline for {job.guest.get('name')}")
    job.status = JobStatus.RUNNING

    try:
        _run_step(job, "images_received", step_receive_images, job, src_folder, config)
        _run_step(job, "autorc_mesh",     step_rc_mesh,        job, config)
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

    If cloud is enabled (config["cloud"]["enabled"] == true) and the mode
    involves processing images from scratch (FULL / SPLAT / SPLAT_NO_EMAIL),
    the heavy steps (COLMAP + 3DGS training) are offloaded to the cloud GPU.
    The local pipeline resumes for crop → compress → email after the PLY
    is downloaded.

    src_folder   — images drop folder (FULL / SPLAT / MESH modes)
    job.start_path — colmap dir (FROM_COLMAP) or .ply path (COMPRESS_ONLY)
    """
    mode = job.mode

    # ── Cloud offload ──────────────────────────────────────────────────────
    # Activate for image-based modes when a cloud backend is configured.
    # FROM_COLMAP and COMPRESS_ONLY skip cloud (they start mid-pipeline).
    # MESH skips cloud (AutoRC is a local Windows GUI app).
    _cloud_eligible = mode in (
        PipelineMode.FULL,
        PipelineMode.SPLAT,
        PipelineMode.SPLAT_NO_EMAIL,
    )
    if _cloud_eligible and src_folder:
        try:
            from .cloud_dispatcher import CloudDispatcher
            cloud = CloudDispatcher(config)
            if cloud.is_enabled():
                log.info(f"[{job.job_id}] Cloud dispatch enabled ({cloud.backend}) — offloading")
                t = threading.Thread(
                    target=cloud.run_remote,
                    args=(job, src_folder),
                    daemon=True,
                )
                t.start()
                return
        except ImportError:
            log.warning("cloud_dispatcher not available — falling back to local pipeline")
        except Exception as e:
            log.error(f"[{job.job_id}] Cloud init failed, falling back to local: {e}")

    # ── Local pipeline ─────────────────────────────────────────────────────
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
