"""
Cloud Dispatcher — offloads COLMAP + 3DGS training to a remote GPU instance.

Supports two backends (configured in config.json → "cloud" → "backend"):
  "runpod"    — RunPod Serverless (Linux, cheapest, best API)
  "aws_batch" — AWS Batch with a custom AMI (supports Windows for LichtFeld)

The dispatcher handles steps: images_received + colmap_align + lichtfeld_splat.
After cloud completes and the PLY is downloaded, the local pipeline resumes
with: splat_crop → splat_compress → email_delivery.

config.json "cloud" section:
{
  "cloud": {
    "enabled":    false,         // set true to activate
    "backend":    "runpod",      // "runpod" | "aws_batch"
    "s3_bucket":  "my-bucket",
    "s3_prefix":  "opensauce",
    "aws_region": "us-east-1",

    // RunPod:
    "runpod_api_key":    "rpa_xxxx",
    "runpod_endpoint_id":"xxxxxxxx",

    // AWS Batch (Windows AMI with LichtFeld pre-installed):
    "batch_job_queue":       "opensauce-gpu-queue",
    "batch_job_definition":  "opensauce-pipeline:1",
    "batch_status_bucket":   "my-bucket",   // where batch worker writes status.json
    "batch_status_prefix":   "opensauce/batch-status"
  }
}

AWS credentials are read from environment variables (standard boto3 precedence):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
Or from ~/.aws/credentials on the rig PC.
"""

import os
import re
import time
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import requests

from .job_manager import Job, JobStatus, StepStatus

log = logging.getLogger("cloud")

IMAGE_EXTS = {".jpg", ".jpeg", ".tiff", ".tif", ".png", ".raw", ".cr2", ".nef"}

# Maps RunPod/Batch progress status strings → (local_step_id, set_previous_done)
_RUNPOD_STEP_MAP = [
    # (runpod_status_key, step_to_mark_running, step_to_mark_done_first)
    ("downloading_images",        "images_received", None),
    ("colmap_feature_extraction", "colmap_align",    "images_received"),
    ("colmap_matching",           "colmap_align",    None),
    ("colmap_mapper",             "colmap_align",    None),
    ("splat_training",            "lichtfeld_splat", "colmap_align"),
    ("uploading_output",          "lichtfeld_splat", None),
]
_RUNPOD_STATUS_INDEX = {s[0]: i for i, s in enumerate(_RUNPOD_STEP_MAP)}


class CloudDispatcher:
    """Manage a single cloud backend for GPU-heavy pipeline steps."""

    def __init__(self, config: dict):
        self._config    = config
        self._cloud_cfg = config.get("cloud", {})
        self.backend    = self._cloud_cfg.get("backend", "runpod")
        self.s3_bucket  = self._cloud_cfg.get("s3_bucket", "")
        self.s3_prefix  = self._cloud_cfg.get("s3_prefix", "opensauce")
        self.aws_region = self._cloud_cfg.get("aws_region", "us-east-1")

        self._s3 = boto3.client("s3", region_name=self.aws_region)

    def is_enabled(self) -> bool:
        return (
            self._cloud_cfg.get("enabled", False)
            and bool(self.s3_bucket)
        )

    # ── Public entry point ────────────────────────────────────────────────

    def run_remote(self, job: Job, src_folder: str):
        """Upload images, dispatch cloud job, poll, download PLY, resume locally.

        Runs in a background thread (called by runner.dispatch).
        Resumes with step_crop_splat → step_compress_splat → step_email after
        the cloud job completes and the PLY is available locally.
        """
        from .runner import _run_step, step_crop_splat, step_compress_splat, step_email, step_prusa_handoff

        job.status = JobStatus.RUNNING

        try:
            s3_job_prefix = f"{self.s3_prefix}/jobs/{job.job_id}"

            # ── Step 1: upload images → S3, move locally to clear drop folder ──
            job.set_step("images_received", StepStatus.RUNNING)
            n_uploaded = self._upload_images(src_folder, s3_job_prefix)
            self._move_images_locally(src_folder, job)
            job.set_step("images_received", StepStatus.DONE)
            log.info(f"[{job.job_id}] Uploaded {n_uploaded} images to S3")

            # ── Steps 2+3: COLMAP + 3DGS training — remote ───────────────────
            job.set_step("colmap_align", StepStatus.RUNNING)

            if self.backend == "runpod":
                cloud_result = self._run_runpod(job, s3_job_prefix)
            elif self.backend == "aws_batch":
                cloud_result = self._run_aws_batch(job, s3_job_prefix)
            else:
                raise RuntimeError(f"Unknown cloud backend: {self.backend!r}")

            job.set_step("colmap_align",    StepStatus.DONE)
            job.set_step("lichtfeld_splat", StepStatus.DONE)

            # ── Download PLY back to local projects folder ────────────────────
            ply_s3_key = cloud_result["ply_s3_key"]
            local_ply  = self._download_ply(ply_s3_key, job)
            job.outputs["raw_ply"] = local_ply
            log.info(f"[{job.job_id}] PLY downloaded: {local_ply} "
                     f"({cloud_result.get('ply_size_mb', '?')} MB, "
                     f"{cloud_result.get('gaussians', '?'):,} Gaussians)")

            # ── Local continuation: crop → compress → email ───────────────────
            _run_step(job, "splat_crop",    step_crop_splat,     job, self._config)
            _run_step(job, "splat_compress", step_compress_splat, job, self._config)

            # Rig 2 only: Prusa handoff (if step exists on this job)
            if "prusa_handoff" in job.steps:
                _run_step(job, "prusa_handoff", step_prusa_handoff, job, self._config)

            _run_step(job, "email_delivery", step_email, job, self._config)

            job.status = JobStatus.COMPLETED
            log.info(f"[{job.job_id}] Cloud pipeline complete")

        except Exception as e:
            log.error(f"[{job.job_id}] Cloud pipeline failed: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error  = str(e)
            # Mark any still-running steps as failed
            for step_id, status in job.steps.items():
                if status == StepStatus.RUNNING:
                    job.set_step(step_id, StepStatus.FAILED)

    # ── S3 helpers ────────────────────────────────────────────────────────

    def _upload_images(self, src_folder: str, s3_job_prefix: str) -> int:
        """Upload all images from src_folder to S3. Returns count."""
        uploaded = 0
        for f in Path(src_folder).iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                key = f"{s3_job_prefix}/images/{f.name}"
                self._s3.upload_file(str(f), self.s3_bucket, key)
                uploaded += 1
        if uploaded == 0:
            raise RuntimeError(f"No images found in {src_folder}")
        return uploaded

    def _move_images_locally(self, src_folder: str, job: Job):
        """Move images from drop folder into project dir/colmap/images/ (clears drop)."""
        projects_root = self._config["paths"]["projects_root"]
        timestamp     = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        slug          = re.sub(r"[^a-zA-Z0-9_-]", "_",
                               job.guest.get("name", "guest").strip())[:32]
        project_dir   = os.path.join(projects_root,
                                     f"{timestamp}_{slug}_{job.session_id[-6:]}")
        photos_dir    = os.path.join(project_dir, "colmap", "images")
        os.makedirs(photos_dir, exist_ok=True)

        for f in Path(src_folder).iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                shutil.move(str(f), photos_dir)

        job.project_dir = project_dir

    def _download_ply(self, s3_key: str, job: Job) -> str:
        """Download the PLY from S3 into the local project's Splats/ directory."""
        splats_dir = os.path.join(job.project_dir, "Splats")
        os.makedirs(splats_dir, exist_ok=True)

        ply_name   = Path(s3_key).name
        local_path = os.path.join(splats_dir, ply_name)

        log.info(f"[{job.job_id}] Downloading s3://{self.s3_bucket}/{s3_key}")
        self._s3.download_file(self.s3_bucket, s3_key, local_path)
        return local_path

    # ── RunPod backend ────────────────────────────────────────────────────

    def _run_runpod(self, job: Job, s3_job_prefix: str) -> dict:
        """Submit job to RunPod serverless endpoint and poll until complete."""
        api_key     = self._cloud_cfg["runpod_api_key"]
        endpoint_id = self._cloud_cfg["runpod_endpoint_id"]
        headers     = {"Authorization": f"Bearer {api_key}",
                       "Content-Type":  "application/json"}

        iterations  = self._config["pipeline"].get("lichtfeld_iterations", 10000)

        payload = {
            "input": {
                "job_id":               job.job_id,
                "s3_bucket":            self.s3_bucket,
                "s3_prefix":            s3_job_prefix,
                "iterations":           iterations,
                # Inject AWS credentials so the RunPod container can access S3.
                # Prefer explicit cloud config keys; fall back to env vars.
                "aws_access_key_id":     (
                    self._cloud_cfg.get("aws_access_key_id")
                    or os.environ.get("AWS_ACCESS_KEY_ID", "")
                ),
                "aws_secret_access_key": (
                    self._cloud_cfg.get("aws_secret_access_key")
                    or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
                ),
                "aws_region": self.aws_region,
            }
        }

        log.info(f"[{job.job_id}] Submitting RunPod job to endpoint {endpoint_id}")
        resp = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/run",
            json    = payload,
            headers = headers,
            timeout = 30,
        )
        resp.raise_for_status()
        runpod_job_id = resp.json()["id"]
        log.info(f"[{job.job_id}] RunPod job ID: {runpod_job_id}")

        return self._poll_runpod(job, runpod_job_id, endpoint_id, api_key, headers)

    def _poll_runpod(self, job: Job, runpod_job_id: str,
                     endpoint_id: str, api_key: str, headers: dict) -> dict:
        """Poll RunPod status endpoint, update local job steps, return final output."""
        status_url  = f"https://api.runpod.ai/v2/{endpoint_id}/status/{runpod_job_id}"
        cancel_url  = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{runpod_job_id}"
        deadline    = time.time() + 7200  # 2-hour hard cap
        last_rp_status = None
        seen_stream_indices: set = set()

        while time.time() < deadline:
            try:
                resp = requests.get(status_url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning(f"[{job.job_id}] RunPod poll error (retrying): {e}")
                time.sleep(15)
                continue

            rp_status = data.get("status", "")

            if rp_status == "COMPLETED":
                output = data.get("output")
                if not output:
                    # Aggregate stream: last item is the final result
                    stream = data.get("stream", [])
                    output = stream[-1].get("output") if stream else None
                if not output:
                    raise RuntimeError("RunPod job completed but output is empty")
                if output.get("error"):
                    raise RuntimeError(f"Cloud job error: {output['error']}")
                return output

            elif rp_status == "FAILED":
                err = data.get("error") or data.get("output", {}).get("error", "unknown error")
                raise RuntimeError(f"RunPod job failed: {err}")

            elif rp_status in ("IN_PROGRESS", "IN_QUEUE", "RUNNING"):
                # Parse stream to update local step status
                stream = data.get("stream", [])
                for idx, item in enumerate(stream):
                    if idx in seen_stream_indices:
                        continue
                    seen_stream_indices.add(idx)
                    item_output = item.get("output", {})
                    rp_step = item_output.get("status", "")
                    if rp_step and rp_step != last_rp_status:
                        self._apply_runpod_step(job, rp_step)
                        last_rp_status = rp_step

                if rp_status == "IN_QUEUE" and last_rp_status is None:
                    log.info(f"[{job.job_id}] RunPod job queued, waiting for worker...")

            time.sleep(10)

        # Timed out — cancel the cloud job
        try:
            requests.post(cancel_url, headers=headers, timeout=10)
        except Exception:
            pass
        raise RuntimeError("Cloud job timed out after 2 hours")

    def _apply_runpod_step(self, job: Job, rp_status: str):
        """Map a RunPod progress status string to local job step updates."""
        idx = _RUNPOD_STATUS_INDEX.get(rp_status)
        if idx is None:
            return
        _, step_running, step_done_first = _RUNPOD_STEP_MAP[idx]
        if step_done_first:
            job.set_step(step_done_first, StepStatus.DONE)
        if step_running and job.steps.get(step_running) == StepStatus.WAITING:
            job.set_step(step_running, StepStatus.RUNNING)
        log.info(f"[{job.job_id}] Cloud progress: {rp_status}")

    # ── AWS Batch backend ─────────────────────────────────────────────────

    def _run_aws_batch(self, job: Job, s3_job_prefix: str) -> dict:
        """Submit job to AWS Batch and poll until complete.

        The Batch job definition must be pre-configured with:
        - A Windows or Linux AMI that has COLMAP + LichtFeld + splat-transform installed
        - IAM role with S3 read/write access to s3_bucket
        - The entrypoint script reads JOB_ID, S3_BUCKET, S3_PREFIX from env vars,
          runs the pipeline, and writes status.json to s3_prefix/batch-status/JOB_ID.json
        """
        batch = boto3.client("batch", region_name=self.aws_region)

        iterations = self._config["pipeline"].get("lichtfeld_iterations", 10000)

        response = batch.submit_job(
            jobName         = f"opensauce-{job.job_id}",
            jobQueue        = self._cloud_cfg["batch_job_queue"],
            jobDefinition   = self._cloud_cfg["batch_job_definition"],
            containerOverrides={
                "environment": [
                    {"name": "JOB_ID",            "value": job.job_id},
                    {"name": "S3_BUCKET",          "value": self.s3_bucket},
                    {"name": "S3_PREFIX",          "value": s3_job_prefix},
                    {"name": "ITERATIONS",         "value": str(iterations)},
                    {"name": "COLMAP_EXE",         "value": "colmap"},
                    {"name": "STATUS_S3_BUCKET",   "value": self.s3_bucket},
                    {"name": "STATUS_S3_PREFIX",   "value": self._cloud_cfg.get(
                        "batch_status_prefix", f"{self.s3_prefix}/batch-status")},
                ]
            },
        )
        batch_job_id = response["jobId"]
        log.info(f"[{job.job_id}] AWS Batch job ID: {batch_job_id}")

        return self._poll_aws_batch(job, batch_job_id, s3_job_prefix)

    def _poll_aws_batch(self, job: Job, batch_job_id: str, s3_job_prefix: str) -> dict:
        """Poll AWS Batch job status + S3 status file for step-level progress."""
        batch    = boto3.client("batch", region_name=self.aws_region)
        deadline = time.time() + 7200
        last_step_reported = None

        # Step reporting: the Batch worker writes a status JSON to S3 after each step
        status_prefix = self._cloud_cfg.get(
            "batch_status_prefix", f"{self.s3_prefix}/batch-status")
        status_key = f"{status_prefix}/{job.job_id}.json"

        while time.time() < deadline:
            try:
                resp  = batch.describe_jobs(jobs=[batch_job_id])
                bjob  = resp["jobs"][0] if resp["jobs"] else None
            except Exception as e:
                log.warning(f"[{job.job_id}] Batch poll error (retrying): {e}")
                time.sleep(20)
                continue

            if not bjob:
                time.sleep(20)
                continue

            b_status = bjob.get("status", "")

            if b_status == "SUCCEEDED":
                # Read result from S3 status file
                try:
                    import json
                    obj = self._s3.get_object(Bucket=self.s3_bucket, Key=status_key)
                    result = json.loads(obj["Body"].read())
                    if result.get("error"):
                        raise RuntimeError(f"Batch job error: {result['error']}")
                    return result
                except self._s3.exceptions.NoSuchKey:
                    raise RuntimeError(
                        f"Batch job succeeded but no status file at s3://{self.s3_bucket}/{status_key}")

            elif b_status == "FAILED":
                reason = bjob.get("statusReason", "unknown")
                raise RuntimeError(f"AWS Batch job failed: {reason}")

            elif b_status in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
                # Try to read intermediate step progress from S3
                try:
                    import json
                    obj = self._s3.get_object(Bucket=self.s3_bucket, Key=status_key)
                    progress = json.loads(obj["Body"].read())
                    current_step = progress.get("current_step")
                    if current_step and current_step != last_step_reported:
                        self._apply_runpod_step(job, current_step)
                        last_step_reported = current_step
                except Exception:
                    pass  # Status file not written yet — normal during startup

            time.sleep(15)

        # Timed out — terminate the Batch job
        try:
            batch.terminate_job(jobId=batch_job_id, reason="opensauce timeout")
        except Exception:
            pass
        raise RuntimeError("Cloud job timed out after 2 hours")
