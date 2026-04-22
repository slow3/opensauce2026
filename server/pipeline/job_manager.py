"""
Job Manager — tracks pipeline jobs and their step-by-step status.
Supports modular pipeline modes so any stage can be run independently.
"""

import uuid
import threading
from enum import Enum
from datetime import datetime
from typing import Optional


class StepStatus(str, Enum):
    WAITING       = "waiting"
    RUNNING       = "running"
    DONE          = "done"
    FAILED        = "failed"
    SKIPPED       = "skipped"
    AWAITING_CROP = "awaiting_crop"


class JobStatus(str, Enum):
    QUEUED        = "queued"
    RUNNING       = "running"
    AWAITING_CROP = "awaiting_crop"   # pipeline paused, waiting for staff crop input
    COMPLETED     = "completed"
    FAILED        = "failed"


class PipelineMode(str, Enum):
    """Which parts of the pipeline to execute."""
    FULL            = "full"            # full rig pipeline (default)
    SPLAT           = "splat"           # images → COLMAP → splat → crop → compress → email
    MESH            = "mesh"            # images → AutoRC mesh → email (rig 2 only)
    SPLAT_NO_EMAIL  = "splat_no_email"  # splat pipeline without email (staff test)
    FROM_COLMAP     = "from_colmap"     # skip to lichtfeld using existing COLMAP output
    COMPRESS_ONLY   = "compress_only"   # crop + compress an existing .ply


# ── Step definitions per mode ─────────────────────────────────────────────────

_FULL_RIG1 = [
    "images_received",
    "colmap_align",
    "lichtfeld_splat",
    "splat_crop",
    "splat_compress",
    "email_delivery",
]

_FULL_RIG2 = [
    "images_received",
    "colmap_align",
    "autorc_mesh",
    "lichtfeld_splat",
    "splat_crop",
    "splat_compress",
    "prusa_handoff",
    "email_delivery",
]

_SPLAT_ONLY = [
    "images_received",
    "colmap_align",
    "lichtfeld_splat",
    "splat_crop",
    "splat_compress",
    "email_delivery",
]

_MESH_ONLY = [
    "images_received",
    "autorc_mesh",
    "email_delivery",
]

_FROM_COLMAP = [
    "lichtfeld_splat",
    "splat_crop",
    "splat_compress",
    "email_delivery",
]

_COMPRESS_ONLY = [
    "splat_crop",
    "splat_compress",
    "email_delivery",
]

STEP_LABELS = {
    "images_received":  "Images received",
    "colmap_align":     "COLMAP alignment",
    "lichtfeld_splat":  "Gaussian splat",
    "splat_crop":       "Crop bounding region",
    "splat_compress":   "Compress & export",
    "autorc_mesh":      "Mesh reconstruction",
    "prusa_handoff":    "Send to Prusa",
    "email_delivery":   "Email guest",
}


def _steps_for(mode: PipelineMode, rig: int) -> list[str]:
    if mode == PipelineMode.FULL:
        return _FULL_RIG1 if rig == 1 else _FULL_RIG2
    if mode == PipelineMode.SPLAT:
        return _SPLAT_ONLY
    if mode == PipelineMode.SPLAT_NO_EMAIL:
        return [s for s in _SPLAT_ONLY if s != "email_delivery"]
    if mode == PipelineMode.MESH:
        return _MESH_ONLY
    if mode == PipelineMode.FROM_COLMAP:
        return _FROM_COLMAP
    if mode == PipelineMode.COMPRESS_ONLY:
        return _COMPRESS_ONLY
    return _FULL_RIG1 if rig == 1 else _FULL_RIG2


# ── Job ───────────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, session_id: str, rig: int, guest: dict,
                 mode: PipelineMode = PipelineMode.FULL,
                 start_path: Optional[str] = None):
        self.job_id     = str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.rig        = rig
        self.guest      = guest
        self.mode       = mode
        self.status     = JobStatus.QUEUED
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.updated_at = self.created_at
        self.project_dir: Optional[str] = None
        self.start_path: Optional[str] = start_path  # images folder / colmap dir / ply path
        self.outputs: dict = {}
        self.error: Optional[str] = None

        # Crop configuration — set by staff via /api/jobs/<id>/crop
        self.crop_config: dict = {}         # {"mode":"box","min":[x,y,z],"max":[X,Y,Z]}
                                            # {"mode":"sphere","center":[x,y,z],"radius":r}
                                            # {"mode":"none"} = skip crop
        self.crop_event = threading.Event() # signalled when crop_config is set by staff

        step_ids = _steps_for(mode, rig)
        self.steps = {s: StepStatus.WAITING for s in step_ids}

    def set_step(self, step: str, status: StepStatus):
        if step in self.steps:
            self.steps[step] = status
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "session_id":  self.session_id,
            "rig":         self.rig,
            "mode":        self.mode,
            "guest_name":  self.guest.get("name", ""),
            "status":      self.status,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "project_dir": self.project_dir,
            "outputs":     self.outputs,
            "error":       self.error,
            "crop_config": self.crop_config,
            "steps": [
                {
                    "id":     s,
                    "label":  STEP_LABELS.get(s, s),
                    "status": self.steps[s],
                }
                for s in self.steps
            ],
        }


# ── Job Manager ───────────────────────────────────────────────────────────────

class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, rig: int, guest: dict,
               mode: PipelineMode = PipelineMode.FULL,
               start_path: Optional[str] = None) -> Job:
        job = Job(session_id, rig, guest, mode=mode, start_path=start_path)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in reversed(list(self._jobs.values()))]

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values()
                    if j.status in (JobStatus.QUEUED, JobStatus.RUNNING,
                                    JobStatus.AWAITING_CROP)]


job_manager = JobManager()
