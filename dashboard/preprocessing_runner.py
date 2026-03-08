"""Background preprocessing session manager for the integrated dashboard."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_LOG_LINES = 200


@dataclass
class PreprocessingState:
    state: str = "idle"          # idle | running | completed | failed | stopping
    step: str = ""               # "downsample" | "edges" | ""
    elapsed_sec: float = 0.0
    error_message: str = ""
    log_lines: list = field(default_factory=list)


class PreprocessingSession:
    """Thread-safe manager for running preprocessing subprocess jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._state = PreprocessingState()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            return asdict(self._state)

    def request_stop(self):
        self._stop_flag.set()
        with self._lock:
            self._state.state = "stopping"
        # Kill the subprocess if it's still running
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def start_downsample(self, config_dict: dict):
        if self.is_running:
            raise RuntimeError("A preprocessing step is already running.")
        self._stop_flag.clear()
        with self._lock:
            self._state = PreprocessingState(state="running", step="downsample")
        self._thread = threading.Thread(
            target=self._run_step,
            args=("downsample", "scripts/downsample.py", config_dict),
            daemon=True,
        )
        self._thread.start()

    def start_edges(self, config_dict: dict):
        if self.is_running:
            raise RuntimeError("A preprocessing step is already running.")
        self._stop_flag.clear()
        with self._lock:
            self._state = PreprocessingState(state="running", step="edges")
        self._thread = threading.Thread(
            target=self._run_step,
            args=("edges", "scripts/build_edges.py", config_dict),
            daemon=True,
        )
        self._thread.start()

    def _build_cmd(self, script_path: str, config_dict: dict) -> list[str]:
        cmd = [sys.executable, str(PROJECT_ROOT / script_path)]
        for key, val in config_dict.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
                # False bool flags are omitted (store_true semantics)
            elif val is not None and val != "":
                cmd.extend([flag, str(val)])
        return cmd

    def _run_step(self, step_name: str, script_path: str, config_dict: dict):
        start_time = time.time()
        try:
            cmd = self._build_cmd(script_path, config_dict)
            self._append_log(f"[dashboard] Running: {' '.join(cmd)}")

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )

            for line in self._proc.stdout:
                line = line.rstrip("\n")
                self._append_log(line)
                with self._lock:
                    self._state.elapsed_sec = time.time() - start_time
                if self._stop_flag.is_set():
                    self._proc.terminate()
                    break

            self._proc.wait()
            returncode = self._proc.returncode

            with self._lock:
                self._state.elapsed_sec = time.time() - start_time
                if self._stop_flag.is_set():
                    self._state.state = "idle"
                    self._state.step = ""
                elif returncode == 0:
                    self._state.state = "completed"
                else:
                    self._state.state = "failed"
                    self._state.error_message = f"Process exited with code {returncode}"
        except Exception as exc:
            with self._lock:
                self._state.state = "failed"
                self._state.error_message = str(exc)
                self._state.elapsed_sec = time.time() - start_time
        finally:
            self._proc = None

    def _append_log(self, line: str):
        with self._lock:
            self._state.log_lines.append(line)
            if len(self._state.log_lines) > MAX_LOG_LINES:
                self._state.log_lines = self._state.log_lines[-MAX_LOG_LINES:]
