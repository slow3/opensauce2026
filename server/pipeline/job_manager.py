"""
Job Manager — tracks pipeline jobs and their step-by-step status.
Jobs are stored in memory and exposed via the /api/jobs endpoint
so the ScanOps kiosk can show live pipeline progress.
"""

import uuid
import time
import threading
from enum import Enum
from datetime import datetime
from typing import Optional

class StepStatus(str, Enum):
    WAITING  = "waiting"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"

class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"

# Pipeline steps per rig
RIG1_STEPS = [
    "images_received",
    "colmap_align",
    "lichtfeld_splat",
    "splat_trim",
    "splat_compress",
    "email_delivery",
]

RIG2_STEPS = [
    "images_received",
    "colmap_align",
    "autorc_mesh",
    "lichtfeld_splat",
    "splat_trim",
    "splat_compress",
    "prusa_handoff",
    "email_delivery",
]

STEP_LABELS = {
    "images_received":  "Images received",
    "colmap_align":     "COLMAP alignment",
    "lichtfeld_splat":  "Gaussian splat",
    "splat_trim":       "Trim bounding region",
    "splat_compress":   "Compress splat",
    "autorc_mesh":      "Mesh reconstruction",
    "prusa_handoff":    "Send to Prusa",
    "email_delivery":   "Email guest",
}

class Job:
    def __init__(self, session_id: str, rig: int, guest: dict):
        self.job_id     = str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.rig        = rig
        self.guest      = guest
        self.status     = JobStatus.QUEUED
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.updated_at = self.created_at
        self.project_dir: Optional[str] = None
        self.outputs: dict = {}
        self.error: Optional[str] = None

        steps = RIG1_STEPS if rig == 1 else RIG2_STEPS
        self.steps = {s: StepStatus.WAITING for s in steps}

    def set_step(self, step: str, status: StepStatus):
        self.steps[step] = status
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "session_id":  self.session_id,
            "rig":         self.rig,
            "guest_name":  self.guest.get("name", ""),
            "status":      self.status,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "project_dir": self.project_dir,
            "outputs":     self.outputs,
            "error":       self.error,
            "steps": [
                {
                    "id":     s,
                    "label":  STEP_LABELS.get(s, s),
                    "status": self.steps[s],
                }
                for s in self.steps
            ],
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, rig: int, guest: dict) -> Job:
        job = Job(session_id, rig, guest)
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
                    if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)]


job_manager = JobManager()
