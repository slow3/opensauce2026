"""
RunPod Serverless Handler — executes COLMAP + Gaussian Splatting training in the cloud.

This file runs INSIDE the Docker container on RunPod. It:
  1. Downloads images from S3
  2. Runs COLMAP (feature extraction → matching → mapper)
  3. Trains a Gaussian Splat (original Inria gaussian-splatting repo)
  4. Uploads the output .ply back to S3

Input payload (job["input"]):
  {
    "job_id":               "abc12345",
    "s3_bucket":            "my-bucket",
    "s3_prefix":            "opensauce/jobs/abc12345",
    "iterations":           10000,
    "aws_access_key_id":    "AKIA...",
    "aws_secret_access_key":"...",
    "aws_region":           "us-east-1"
  }

Output (returned when job completes):
  {
    "status":      "done",
    "ply_s3_key":  "opensauce/jobs/abc12345/output/point_cloud.ply",
    "ply_size_mb": 248.3,
    "gaussians":   1023456
  }

Progress stream (yielded during execution, visible via RunPod status endpoint):
  {"status": "downloading_images", "count": 40}
  {"status": "colmap_feature_extraction"}
  {"status": "colmap_matching"}
  {"status": "colmap_mapper"}
  {"status": "splat_training", "iteration": 0}
  {"status": "uploading_output"}
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

import boto3
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("handler")

WORKSPACE    = Path("/workspace")
GAUSS_SPLAT  = Path("/app/gaussian-splatting/train.py")
IMAGE_EXTS   = {".jpg", ".jpeg", ".tiff", ".tif", ".png", ".raw", ".cr2", ".nef"}


def _run(cmd: list, cwd=None, timeout=7200):
    log.info("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if result.stdout:
        log.info(result.stdout[-2000:])  # last 2000 chars to avoid log flooding
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}):\n{result.stderr[-2000:]}")
    return result


def handler(job):
    inp = job["input"]

    job_id    = inp["job_id"]
    s3_bucket = inp["s3_bucket"]
    s3_prefix = inp["s3_prefix"]
    iterations = int(inp.get("iterations", 10000))

    # Set up AWS credentials from payload (injected by cloud_dispatcher, never logged)
    s3 = boto3.client(
        "s3",
        region_name    = inp.get("aws_region", "us-east-1"),
        aws_access_key_id     = inp.get("aws_access_key_id"),
        aws_secret_access_key = inp.get("aws_secret_access_key"),
    )

    # Working directories
    work_dir   = WORKSPACE / job_id
    colmap_dir = work_dir / "colmap"
    images_dir = colmap_dir / "images"
    sparse_dir = colmap_dir / "sparse"
    db_path    = colmap_dir / "database.db"
    model_dir  = work_dir / "splat_model"

    for d in [images_dir, sparse_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Download images from S3 ─────────────────────────────────────────

    log.info(f"Downloading images from s3://{s3_bucket}/{s3_prefix}/images/")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=f"{s3_prefix}/images/")

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = Path(key).name
            if Path(name).suffix.lower() in IMAGE_EXTS:
                s3.download_file(s3_bucket, key, str(images_dir / name))
                count += 1

    if count == 0:
        raise RuntimeError(f"No images found at s3://{s3_bucket}/{s3_prefix}/images/")

    log.info(f"Downloaded {count} images")
    yield {"status": "downloading_images", "count": count}

    # ── 2. COLMAP ──────────────────────────────────────────────────────────

    yield {"status": "colmap_feature_extraction"}
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu",    "1",
    ])

    yield {"status": "colmap_matching"}
    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ])

    yield {"status": "colmap_mapper"}
    _run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--output_path",   str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001",
    ])

    subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError(f"COLMAP mapper produced no sparse reconstruction in {sparse_dir}")
    log.info(f"COLMAP produced {len(subdirs)} reconstruction(s)")

    # ── 3. Gaussian Splatting training ─────────────────────────────────────
    # gaussian-splatting train.py expects -s to point at the COLMAP dataset root
    # (must contain images/ and sparse/0/).

    yield {"status": "splat_training", "iteration": 0, "total": iterations}
    _run([
        "python", str(GAUSS_SPLAT),
        "-s", str(colmap_dir),
        "-m", str(model_dir),
        "--iterations",     str(iterations),
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--checkpoint_iterations",  # empty = no checkpoints
    ], timeout=14400)  # 4 hour hard cap

    # Find output PLY: gaussian-splatting saves to
    # {model_dir}/point_cloud/iteration_{N}/point_cloud.ply
    plys = sorted(model_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise RuntimeError(f"No .ply found under {model_dir} after training")
    output_ply = plys[-1]

    # Count Gaussians (each line in the PLY element header is one vertex)
    gaussians = 0
    try:
        from plyfile import PlyData
        pd = PlyData.read(str(output_ply))
        gaussians = len(pd["vertex"])
    except Exception:
        pass

    log.info(f"Training complete — {gaussians:,} Gaussians, {output_ply.stat().st_size / 1024**2:.1f} MB")

    # ── 4. Upload output PLY to S3 ─────────────────────────────────────────

    yield {"status": "uploading_output"}
    s3_ply_key = f"{s3_prefix}/output/{output_ply.name}"
    s3.upload_file(str(output_ply), s3_bucket, s3_ply_key)
    log.info(f"Uploaded to s3://{s3_bucket}/{s3_ply_key}")

    # Clean up workspace to avoid filling the volume between jobs
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    yield {
        "status":      "done",
        "ply_s3_key":  s3_ply_key,
        "ply_size_mb": round(output_ply.stat().st_size / 1024**2, 1),
        "gaussians":   gaussians,
    }


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
