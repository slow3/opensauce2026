"""
Folder Watcher — monitors the Camera2Cloud drop folder for new image batches.
Uses watchdog to detect new files, then waits for the transfer to settle
before handing off to the pipeline runner.
"""

import os
import time
import logging
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

log = logging.getLogger("watcher")

class DropFolderHandler(FileSystemEventHandler):
    """
    Watches for new image files arriving in the drop folder.
    Waits until no new files have arrived for `settle_seconds` before
    triggering the callback — prevents starting pipeline mid-transfer.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".tiff", ".tif", ".png", ".raw", ".cr2", ".nef"}

    def __init__(self, callback, settle_seconds: int = 5, min_images: int = 1):
        self._callback       = callback
        self._settle_seconds = settle_seconds
        self._min_images     = min_images
        self._pending_dirs: dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in self.IMAGE_EXTS:
            return
        folder = str(path.parent)
        with self._lock:
            self._pending_dirs[folder] = time.time()
        self._reset_timer()

    def on_modified(self, event):
        self.on_created(event)

    def _reset_timer(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._settle_seconds, self._check_settled)
        self._timer.daemon = True
        self._timer.start()

    def _check_settled(self):
        now = time.time()
        with self._lock:
            ready = [
                folder for folder, last_seen in self._pending_dirs.items()
                if now - last_seen >= self._settle_seconds
            ]
            for folder in ready:
                del self._pending_dirs[folder]

        for folder in ready:
            images = self._count_images(folder)
            if images >= self._min_images:
                log.info(f"Drop folder settled: {folder} ({images} images)")
                try:
                    self._callback(folder, images)
                except Exception as e:
                    log.error(f"Callback error for {folder}: {e}")
            else:
                log.warning(f"Skipping {folder} — only {images} images (min {self._min_images})")

    def _count_images(self, folder: str) -> int:
        try:
            return sum(
                1 for f in Path(folder).iterdir()
                if f.suffix.lower() in self.IMAGE_EXTS
            )
        except Exception:
            return 0


class FolderWatcher:
    def __init__(self, drop_folder: str, callback, settle_seconds: int = 5, min_images: int = 1):
        self._drop_folder    = drop_folder
        self._observer       = Observer()
        self._handler        = DropFolderHandler(callback, settle_seconds, min_images)

    def start(self):
        os.makedirs(self._drop_folder, exist_ok=True)
        self._observer.schedule(self._handler, self._drop_folder, recursive=True)
        self._observer.start()
        log.info(f"Watching: {self._drop_folder}")

    def stop(self):
        self._observer.stop()
        self._observer.join()
