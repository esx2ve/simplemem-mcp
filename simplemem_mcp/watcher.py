"""Cloud-integrated file watcher for automatic code index updates.

Monitors file changes and sends updates through the BackendClient to the cloud API.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any

import requests
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from simplemem_mcp import DEFAULT_BACKEND_URL
from simplemem_mcp.projects_utils import (
    get_project_id as generate_project_id,
    infer_project_from_session_path,
)

if TYPE_CHECKING:
    from simplemem_mcp.client import BackendClient
    from simplemem_mcp.local_reader import LocalReader

log = logging.getLogger("simplemem_mcp.watcher")

# Default file patterns to watch
DEFAULT_PATTERNS = ["*.py", "*.ts", "*.js", "*.tsx", "*.jsx"]

# State file for watcher persistence (auto-resume on startup)
STATE_FILE = Path.home() / ".simplemem" / "watcher_state.json"

# Module-level lock for state file writes (prevents race conditions between managers)
_state_lock = threading.Lock()

# Cache limits
MAX_INDEXED_CACHE_ENTRIES = 50  # Max projects in indexed sessions cache
MAX_INDEXED_CACHE_SESSIONS = 10000  # Max sessions per project


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHER STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


def _load_watcher_state() -> dict:
    """Load watcher state from file.

    Returns:
        Dict with 'code_watchers' (list of project paths) and 'trace_watcher' (bool)
    """
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load watcher state: {e}")
    return {"code_watchers": [], "trace_watcher": False}


def _save_watcher_state(code_watchers: list[str], trace_watcher: bool) -> None:
    """Save watcher state to file atomically.

    Uses tmp file + rename pattern for atomic writes, and module-level lock
    to prevent race conditions between CloudWatcherManager and TraceWatcherManager.

    Args:
        code_watchers: List of project root paths being watched
        trace_watcher: Whether trace watcher is running
    """
    with _state_lock:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Write to temp file first
            tmp_file = STATE_FILE.with_suffix(".json.tmp")
            with open(tmp_file, "w") as f:
                json.dump({
                    "code_watchers": code_watchers,
                    "trace_watcher": trace_watcher,
                }, f, indent=2)
                f.flush()
                # Note: os.fsync(f.fileno()) could be added for extra durability
            # Atomic rename
            tmp_file.rename(STATE_FILE)
            log.debug(f"Saved watcher state: {len(code_watchers)} code watchers, trace={trace_watcher}")
        except Exception as e:
            log.warning(f"Failed to save watcher state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FileEvent:
    """Represents a file change event."""

    path: Path
    event_type: str  # "created", "modified", "deleted", "moved"
    timestamp: float = field(default_factory=time.time)
    dest_path: Path | None = None  # For move events


class DebouncedQueue:
    """Queue with debouncing - waits for quiet period before yielding events.

    Coalesces rapid file changes into single events per file.
    """

    def __init__(self, quiet_period: float = 1.5):
        """Initialize the debounced queue.

        Args:
            quiet_period: Seconds to wait after last event before processing
        """
        self.quiet_period = quiet_period
        self._pending: dict[str, FileEvent] = {}  # path -> latest event
        self._lock = threading.Lock()
        self._last_event_time = 0.0

    def put(self, event: FileEvent) -> None:
        """Add an event to the queue (coalesces with pending events)."""
        with self._lock:
            path_key = str(event.path)

            # For deletes, just record it
            if event.event_type == "deleted":
                self._pending[path_key] = event
            # For moves, track as delete + create
            elif event.event_type == "moved" and event.dest_path:
                # Delete from old path
                self._pending[path_key] = FileEvent(
                    path=event.path,
                    event_type="deleted",
                    timestamp=event.timestamp,
                )
                # Create at new path
                dest_key = str(event.dest_path)
                self._pending[dest_key] = FileEvent(
                    path=event.dest_path,
                    event_type="created",
                    timestamp=event.timestamp,
                )
            else:
                # Create/modify: coalesce to latest
                self._pending[path_key] = event

            self._last_event_time = time.time()

    def get_ready_events(self) -> list[FileEvent]:
        """Get events that are ready (quiet period elapsed).

        Returns:
            List of events ready for processing (empty if still in quiet period)
        """
        with self._lock:
            if not self._pending:
                return []

            # Check if quiet period has elapsed
            elapsed = time.time() - self._last_event_time
            if elapsed < self.quiet_period:
                return []

            # Return all pending events and clear
            events = list(self._pending.values())
            self._pending.clear()
            return events

    def clear(self) -> None:
        """Clear all pending events."""
        with self._lock:
            self._pending.clear()


def _event_path_to_path(event_path: bytes | str) -> Path:
    """Convert watchdog event path to Path, handling bytes properly."""
    if isinstance(event_path, bytes):
        return Path(event_path.decode("utf-8", errors="replace"))
    return Path(event_path)


class CodeWatchHandler(FileSystemEventHandler):
    """Watchdog event handler that filters and queues file events."""

    def __init__(
        self,
        queue: DebouncedQueue,
        patterns: list[str],
        project_root: Path,
    ):
        """Initialize the handler.

        Args:
            queue: DebouncedQueue to add events to
            patterns: Glob patterns for files to watch (e.g., ["*.py", "*.ts"])
            project_root: Project root for relative path matching
        """
        super().__init__()
        self.queue = queue
        self.patterns = patterns
        self.project_root = project_root

    def _matches_patterns(self, path: Path) -> bool:
        """Check if path matches any of our patterns."""
        for pattern in self.patterns:
            if path.match(pattern):
                return True
        return False

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        # Ignore common non-code directories
        ignore_dirs = {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        }
        parts = path.parts
        for part in parts:
            if part in ignore_dirs or part.endswith(".egg-info"):
                return True
        return False

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = _event_path_to_path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File created: {path}")
            self.queue.put(FileEvent(path=path, event_type="created"))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = _event_path_to_path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File modified: {path}")
            self.queue.put(FileEvent(path=path, event_type="modified"))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = _event_path_to_path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File deleted: {path}")
            self.queue.put(FileEvent(path=path, event_type="deleted"))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = _event_path_to_path(event.src_path)
        dest_path = _event_path_to_path(event.dest_path)
        if self._should_ignore(src_path) and self._should_ignore(dest_path):
            return
        src_matches = self._matches_patterns(src_path)
        dest_matches = self._matches_patterns(dest_path)
        if src_matches or dest_matches:
            log.debug(f"File moved: {src_path} -> {dest_path}")
            self.queue.put(FileEvent(
                path=src_path,
                event_type="moved",
                dest_path=dest_path,
            ))


# ═══════════════════════════════════════════════════════════════════════════════
# CLOUD WATCHER
# ═══════════════════════════════════════════════════════════════════════════════


class CloudWatcherWorker(threading.Thread):
    """Background thread that processes file events and sends to cloud backend."""

    def __init__(
        self,
        queue: DebouncedQueue,
        client: "BackendClient",
        reader: "LocalReader",
        project_root: Path,
        project_id: str | None = None,
        poll_interval: float = 0.5,
    ):
        """Initialize the worker.

        Args:
            queue: DebouncedQueue with file events
            client: BackendClient for sending updates to cloud
            reader: LocalReader for reading file contents
            project_root: Project root directory
            project_id: Explicit project ID (config:xxx). If None, generates from path.
            poll_interval: How often to check for events (seconds)
        """
        super().__init__(daemon=True, name=f"cloud-watcher-{project_root.name}")
        self.queue = queue
        self.client = client
        self.reader = reader
        self.project_root = project_root
        self.project_id = project_id  # Use explicit ID if provided
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        """Main worker loop - process events from queue."""
        log.info(f"CloudWatcherWorker started for {self.project_root}")

        # Create event loop for async client calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Create a FRESH BackendClient for this worker's event loop.
        # httpx.AsyncClient cannot be shared across event loops, so each
        # worker thread must create its own client instance.
        from simplemem_mcp.client import BackendClient
        self._worker_client = BackendClient()
        log.info(f"Worker BackendClient created with base_url={self._worker_client.base_url}")

        try:
            while not self._stop_event.is_set():
                # Check for ready events (debounce period elapsed)
                events = self.queue.get_ready_events()

                if events:
                    self._process_events(events)

                # Sleep before next check
                time.sleep(self.poll_interval)
        finally:
            # Clean up the worker's BackendClient before closing the loop
            if hasattr(self, '_worker_client') and self._worker_client:
                try:
                    self._loop.run_until_complete(self._worker_client.close())
                except Exception:
                    pass  # Best effort cleanup
            self._loop.close()
            log.info(f"CloudWatcherWorker stopped for {self.project_root}")

    def stop(self) -> None:
        """Signal worker to stop."""
        self._stop_event.set()

    def _process_events(self, events: list[FileEvent]) -> None:
        """Process a batch of file events."""
        log.info(f"Processing {len(events)} file events")

        updates = []
        for event in events:
            try:
                relative_path = str(event.path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(event.path)

            if event.event_type == "deleted":
                updates.append({
                    "path": relative_path,
                    "action": "delete",
                    "content": None,
                })
            elif event.event_type in ("created", "modified"):
                # Read file content
                try:
                    content = self.reader.read_single_file(event.path)
                    if content:
                        action = "add" if event.event_type == "created" else "modify"
                        updates.append({
                            "path": relative_path,
                            "action": action,
                            "content": content,
                        })
                except Exception as e:
                    log.error(f"Failed to read {event.path}: {e}")

        if updates:
            self._send_updates(updates)

    def _send_updates(self, updates: list[dict]) -> None:
        """Send updates to cloud backend."""
        if not self._loop:
            log.error("Event loop not available")
            return

        try:
            # Use run_until_complete since we're in the worker thread and can block.
            # This actually executes the coroutine (unlike run_coroutine_threadsafe
            # which requires the loop to be running separately).
            # Use the worker's own BackendClient to avoid cross-loop issues.
            # Use explicit project_id if provided, otherwise generate from path
            project_id = self.project_id or generate_project_id(self.project_root)
            log.debug(f"Sending {len(updates)} updates for project_id={project_id}")
            result = self._loop.run_until_complete(
                self._worker_client.update_code(
                    project_id=project_id,
                    updates=updates,
                )
            )
            log.info(
                f"Cloud update: {result.get('files_updated', 0)} files, "
                f"+{result.get('chunks_created', 0)}/-{result.get('chunks_deleted', 0)} chunks"
            )
        except Exception as e:
            log.error(f"Failed to send updates to cloud: {e}")


class CloudWatcherManager:
    """Manages file watchers for multiple projects, sending updates to cloud."""

    def __init__(
        self,
        client: "BackendClient",
        reader: "LocalReader",
        trace_watcher_getter: Callable[[], Any] | None = None,
    ):
        """Initialize the manager.

        Args:
            client: BackendClient for cloud API calls
            reader: LocalReader for reading file contents
            trace_watcher_getter: Callback to get TraceWatcherManager for state persistence
        """
        self.client = client
        self.reader = reader
        self._watchers: dict[str, dict] = {}  # project_root -> {observer, worker, queue}
        self._lock = threading.Lock()
        # Cache credentials from client for sync status reporting
        self._base_url = client.base_url
        self._api_key = client.api_key
        self._trace_watcher_getter = trace_watcher_getter

    def _get_watched_paths(self) -> list[str]:
        """Get list of all watched project paths."""
        with self._lock:
            return list(self._watchers.keys())

    def _save_state(self) -> None:
        """Save current watcher state to disk for auto-resume."""
        try:
            code_watchers = self._get_watched_paths()
            trace_watching = False
            if self._trace_watcher_getter:
                trace_mgr = self._trace_watcher_getter()
                if trace_mgr:
                    trace_watching = trace_mgr.get_status().get("is_watching", False)
            _save_watcher_state(code_watchers, trace_watching)
        except Exception as e:
            log.warning(f"Failed to save watcher state: {e}")

    def _report_status(self) -> None:
        """Report watcher status to backend for statusline.

        Uses synchronous HTTP to avoid async/threading issues when called
        from watchdog callbacks.
        """
        try:
            with self._lock:
                watcher_count = len(self._watchers)
                projects = list(self._watchers.keys())

            # Use cached credentials from client (set at init time)
            base_url = self._base_url
            api_key = self._api_key

            url = f"{base_url}/api/v1/code/status"
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            response = requests.post(
                url,
                json={
                    "watchers": watcher_count,
                    "projects_watching": projects,
                },
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()
            log.debug(f"Reported watcher status: {watcher_count} watchers")
        except Exception as e:
            log.warning(f"Failed to report watcher status: {e}")

    def start_watching(
        self,
        project_root: str | Path,
        patterns: list[str] | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Start watching a project directory.

        Args:
            project_root: Directory to watch
            patterns: Glob patterns for files to watch (default: Python/JS/TS)
            project_id: Explicit project ID (config:xxx). If None, generates from path.

        Returns:
            Status dict with watching state
        """
        project_root = Path(project_root).resolve()
        project_key = str(project_root)
        patterns = patterns or DEFAULT_PATTERNS

        with self._lock:
            if project_key in self._watchers:
                return {
                    "status": "already_watching",
                    "project_root": project_key,
                    "patterns": patterns,
                    "project_id": self._watchers[project_key].get("project_id"),
                }

            log.info(f"Starting watcher for {project_root} with patterns {patterns}, project_id={project_id}")

            # Create debounced queue
            queue = DebouncedQueue(quiet_period=1.5)

            # Create handler
            handler = CodeWatchHandler(
                queue=queue,
                patterns=patterns,
                project_root=project_root,
            )

            # Create observer
            observer = Observer()
            observer.schedule(handler, str(project_root), recursive=True)
            observer.start()

            # Create worker with explicit project_id
            worker = CloudWatcherWorker(
                queue=queue,
                client=self.client,
                reader=self.reader,
                project_root=project_root,
                project_id=project_id,
            )
            worker.start()

            self._watchers[project_key] = {
                "observer": observer,
                "worker": worker,
                "queue": queue,
                "patterns": patterns,
                "project_id": project_id,
                "started_at": time.time(),
            }

        # Report status update outside lock
        self._report_status()
        # Persist state for auto-resume
        self._save_state()

        return {
            "status": "started",
            "project_root": project_key,
            "patterns": patterns,
        }

    def stop_watching(self, project_root: str | Path) -> dict:
        """Stop watching a project directory.

        Args:
            project_root: Directory to stop watching

        Returns:
            Status dict
        """
        project_root = Path(project_root).resolve()
        project_key = str(project_root)

        # Pop watcher from dict while holding lock (fast operation)
        with self._lock:
            if project_key not in self._watchers:
                return {
                    "status": "not_watching",
                    "project_root": project_key,
                }

            log.info(f"Stopping watcher for {project_root}")
            watcher = self._watchers.pop(project_key)

        # Stop observer and worker outside lock to avoid deadlock
        # (join() can block for up to 5s each)
        watcher["observer"].stop()
        watcher["observer"].join(timeout=5)

        watcher["worker"].stop()
        watcher["worker"].join(timeout=5)

        # Report status update after stopping
        self._report_status()
        # Persist state for auto-resume
        self._save_state()

        return {
            "status": "stopped",
            "project_root": project_key,
        }

    def get_status(self, project_root: str | Path | None = None) -> dict:
        """Get watcher status.

        Args:
            project_root: Optional specific project to check

        Returns:
            Status dict with watching state
        """
        with self._lock:
            if project_root:
                project_root = Path(project_root).resolve()
                project_key = str(project_root)

                if project_key in self._watchers:
                    watcher = self._watchers[project_key]
                    return {
                        "is_watching": True,
                        "project_root": project_key,
                        "patterns": watcher["patterns"],
                        "started_at": watcher["started_at"],
                    }
                else:
                    return {
                        "is_watching": False,
                        "project_root": project_key,
                    }
            else:
                # Return all watchers
                return {
                    "watching_count": len(self._watchers),
                    "projects": [
                        {
                            "project_root": key,
                            "patterns": w["patterns"],
                            "started_at": w["started_at"],
                        }
                        for key, w in self._watchers.items()
                    ],
                }

    def stop_all(self) -> None:
        """Stop all watchers (for graceful shutdown)."""
        # Collect keys while holding lock, then stop outside lock
        # to avoid deadlock (stop_watching also acquires _lock)
        with self._lock:
            keys = list(self._watchers.keys())
        for project_key in keys:
            self.stop_watching(project_key)


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE WATCHER (Auto-process Claude Code sessions)
# ═══════════════════════════════════════════════════════════════════════════════


class TraceWatchHandler(FileSystemEventHandler):
    """Watchdog event handler for Claude Code session trace files."""

    def __init__(
        self,
        queue: DebouncedQueue,
        traces_dir: Path,
    ):
        """Initialize the handler.

        Args:
            queue: DebouncedQueue to add events to
            traces_dir: Root traces directory (e.g., ~/.claude/projects)
        """
        super().__init__()
        self.queue = queue
        self.traces_dir = traces_dir

    def _is_trace_file(self, path: Path) -> bool:
        """Check if path looks like a trace file (syntactic check only).

        Avoids filesystem I/O in watchdog callbacks to prevent crashes.
        Existence is verified later when actually processing the file.
        """
        return path.suffix == ".jsonl"

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        try:
            path = _event_path_to_path(event.src_path)
            if self._is_trace_file(path):
                log.debug(f"Trace file created: {path}")
                self.queue.put(FileEvent(path=path, event_type="created"))
        except Exception as e:
            log.warning(f"TraceWatchHandler.on_created failed for {event.src_path}: {e}")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        try:
            path = _event_path_to_path(event.src_path)
            if self._is_trace_file(path):
                log.debug(f"Trace file modified: {path}")
                self.queue.put(FileEvent(path=path, event_type="modified"))
        except Exception as e:
            log.warning(f"TraceWatchHandler.on_modified failed for {event.src_path}: {e}")


class TraceWatcherWorker(threading.Thread):
    """Background thread that processes trace file events."""

    def __init__(
        self,
        queue: DebouncedQueue,
        traces_dir: Path,
        poll_interval: float = 1.0,
    ):
        """Initialize the worker.

        Args:
            queue: DebouncedQueue with trace events
            traces_dir: Root traces directory for project resolution
            poll_interval: How often to check for events (seconds)
        """
        super().__init__(daemon=True, name="trace-watcher")
        self.queue = queue
        self.traces_dir = traces_dir
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Per-project indexed sessions cache: project_id -> (timestamp, session_ids)
        self._indexed_by_project: dict[str | None, tuple[float, set[str]]] = {}

    def run(self) -> None:
        """Main worker loop - process events from queue."""
        log.info(f"TraceWatcherWorker started for {self.traces_dir}")

        # Create event loop for async client calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Create a FRESH BackendClient for this worker's event loop
        from simplemem_mcp.client import BackendClient
        self._worker_client = BackendClient()
        log.info(f"TraceWatcher BackendClient created with base_url={self._worker_client.base_url}")

        try:
            while not self._stop_event.is_set():
                # Check for ready events (quiet period elapsed)
                events = self.queue.get_ready_events()

                if events:
                    self._process_events(events)

                # Sleep before next check
                time.sleep(self.poll_interval)
        finally:
            # Clean up
            if hasattr(self, '_worker_client') and self._worker_client:
                try:
                    self._loop.run_until_complete(self._worker_client.close())
                except Exception:
                    pass
            self._loop.close()
            log.info(f"TraceWatcherWorker stopped")

    def stop(self) -> None:
        """Signal worker to stop."""
        self._stop_event.set()

    def _purge_indexed_cache(self) -> None:
        """Purge stale and excess entries from the indexed sessions cache.

        Enforces limits:
        - MAX_INDEXED_CACHE_ENTRIES: Max projects in cache (LRU eviction)
        - MAX_INDEXED_CACHE_SESSIONS: Max sessions per project
        - 5-minute TTL for stale entries
        """
        now = time.time()
        stale_cutoff = now - 300  # 5-minute TTL

        # Remove stale entries
        stale_keys = [
            key for key, (ts, _) in self._indexed_by_project.items()
            if ts < stale_cutoff
        ]
        for key in stale_keys:
            del self._indexed_by_project[key]

        # Trim per-project session counts
        for key, (_, sessions) in self._indexed_by_project.items():
            if len(sessions) > MAX_INDEXED_CACHE_SESSIONS:
                # Keep most recent (arbitrary selection since sets aren't ordered)
                # In practice, sessions are UUIDs and we just need to cap size
                excess = len(sessions) - MAX_INDEXED_CACHE_SESSIONS
                for _ in range(excess):
                    sessions.pop()

        # LRU eviction if still over project limit
        if len(self._indexed_by_project) > MAX_INDEXED_CACHE_ENTRIES:
            # Sort by timestamp (oldest first) and remove excess
            sorted_entries = sorted(
                self._indexed_by_project.items(),
                key=lambda x: x[1][0]
            )
            excess_count = len(self._indexed_by_project) - MAX_INDEXED_CACHE_ENTRIES
            for key, _ in sorted_entries[:excess_count]:
                del self._indexed_by_project[key]
            log.debug(f"Evicted {excess_count} stale cache entries")

    def _get_indexed_sessions(self, project_id: str | None) -> set[str]:
        """Get indexed sessions for a project (with caching).

        Args:
            project_id: Project ID to filter by (or None for global)

        Returns:
            Set of indexed session IDs for this project
        """
        assert self._loop is not None, "Event loop must be initialized"
        now = time.time()
        cached = self._indexed_by_project.get(project_id)

        # Return cached value if fresh (5 min TTL)
        if cached and (now - cached[0]) < 300:
            return cached[1]

        try:
            result = self._loop.run_until_complete(
                self._worker_client.get_indexed_sessions(project_id=project_id, limit=1000)
            )
            sessions = set(result.get("indexed_sessions", []))
            self._indexed_by_project[project_id] = (now, sessions)
            self._purge_indexed_cache()  # Bound cache size
            log.debug(f"Refreshed indexed cache for project={project_id}: {len(sessions)} sessions")
            return sessions
        except Exception as e:
            log.warning(f"Failed to get indexed sessions for {project_id}: {e}")
            return cached[1] if cached else set()

    def _resolve_project_id(self, trace_path: Path) -> str | None:
        """Resolve project ID from trace file path.

        Uses the proper project registry for lossy path encoding handling.
        Returns project_id with "config:" prefix, or None if not resolvable.
        """
        # Use the registry-aware function from projects_utils
        # This handles lossy encoding (e.g., dots in usernames like shimon.vainer)
        return infer_project_from_session_path(trace_path)

    def _process_events(self, events: list[FileEvent]) -> None:
        """Process a batch of trace file events."""
        log.info(f"Processing {len(events)} trace events")

        for event in events:
            if event.event_type == "deleted":
                continue  # Ignore deletes for traces

            trace_path = event.path
            session_id = trace_path.stem

            # Resolve project_id first to check against correct cache
            project_id = self._resolve_project_id(trace_path)
            indexed_sessions = self._get_indexed_sessions(project_id)

            # Skip already-indexed sessions
            if session_id in indexed_sessions:
                log.debug(f"Skipping already-indexed session: {session_id}")
                continue

            self._process_trace(trace_path, session_id, project_id)

    def _process_trace(self, trace_path: Path, session_id: str, project_id: str | None) -> None:
        """Process a single trace file.

        Args:
            trace_path: Path to the trace file
            session_id: Session ID (filename stem)
            project_id: Resolved project ID for isolation
        """
        assert self._loop is not None, "Event loop must be initialized"
        try:
            # Verify file exists (may have been deleted since event)
            if not trace_path.exists():
                log.debug(f"Trace file no longer exists: {trace_path}")
                return

            # Read trace content
            with open(trace_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                log.debug(f"Empty trace file: {trace_path}")
                return

            log.info(f"Auto-processing trace: session={session_id[:8]}, project={project_id}")

            # Send to backend for processing
            result = self._loop.run_until_complete(
                self._worker_client.process_trace(
                    session_id=session_id,
                    trace_content=content,
                    project_id=project_id,
                    background=True,
                )
            )

            # Add to local indexed cache for this project
            cached = self._indexed_by_project.get(project_id)
            if cached:
                cached[1].add(session_id)
            else:
                self._indexed_by_project[project_id] = (time.time(), {session_id})
            self._purge_indexed_cache()  # Bound cache size

            log.info(f"Trace submitted: job_id={result.get('job_id', 'N/A')}")

        except FileNotFoundError:
            log.debug(f"Trace file deleted before processing: {trace_path}")
        except Exception:
            log.exception(f"Failed to process trace {session_id}")


class TraceWatcherManager:
    """Manages the trace file watcher for auto-processing sessions."""

    def __init__(
        self,
        client: "BackendClient",
        code_watcher_getter: Callable[[], Any] | None = None,
    ):
        """Initialize the manager.

        Args:
            client: BackendClient for cloud API calls
            code_watcher_getter: Callback to get CloudWatcherManager for state persistence
        """
        self.client = client
        self._watcher: dict | None = None
        self._lock = threading.Lock()
        self._code_watcher_getter = code_watcher_getter

    def _save_state(self) -> None:
        """Save current watcher state to disk for auto-resume."""
        try:
            trace_watching = self.get_status().get("is_watching", False)
            code_watchers: list[str] = []
            if self._code_watcher_getter:
                code_mgr = self._code_watcher_getter()
                if code_mgr:
                    code_watchers = code_mgr._get_watched_paths()
            _save_watcher_state(code_watchers, trace_watching)
        except Exception as e:
            log.warning(f"Failed to save trace watcher state: {e}")

    def start_watching(
        self,
        traces_dir: str | Path | None = None,
        quiet_period: float = 300.0,  # 5 minutes default
    ) -> dict:
        """Start watching for trace file changes.

        Args:
            traces_dir: Directory to watch (default: ~/.claude/projects)
            quiet_period: Seconds to wait after last activity before processing

        Returns:
            Status dict
        """
        if traces_dir is None:
            traces_dir = Path.home() / ".claude" / "projects"
        else:
            traces_dir = Path(traces_dir).resolve()

        with self._lock:
            if self._watcher is not None:
                return {
                    "status": "already_watching",
                    "traces_dir": str(traces_dir),
                    "quiet_period": quiet_period,
                }

            if not traces_dir.exists():
                return {
                    "status": "error",
                    "error": f"Traces directory does not exist: {traces_dir}",
                }

            log.info(f"Starting trace watcher for {traces_dir} (quiet_period={quiet_period}s)")

            # Create debounced queue with longer quiet period for sessions
            queue = DebouncedQueue(quiet_period=quiet_period)

            # Create handler
            handler = TraceWatchHandler(
                queue=queue,
                traces_dir=traces_dir,
            )

            # Create observer
            observer = Observer()
            observer.schedule(handler, str(traces_dir), recursive=True)
            observer.start()

            # Create worker (creates its own BackendClient to avoid cross-loop issues)
            worker = TraceWatcherWorker(
                queue=queue,
                traces_dir=traces_dir,
            )
            worker.start()

            self._watcher = {
                "observer": observer,
                "worker": worker,
                "queue": queue,
                "traces_dir": str(traces_dir),
                "quiet_period": quiet_period,
                "started_at": time.time(),
            }

        # Persist state for auto-resume
        self._save_state()

        return {
            "status": "started",
            "traces_dir": str(traces_dir),
            "quiet_period": quiet_period,
        }

    def stop_watching(self) -> dict:
        """Stop the trace watcher.

        Returns:
            Status dict
        """
        with self._lock:
            if self._watcher is None:
                return {"status": "not_watching"}

            log.info("Stopping trace watcher")
            watcher = self._watcher
            self._watcher = None

        # Stop outside lock to avoid deadlock
        watcher["observer"].stop()
        watcher["observer"].join(timeout=5)

        watcher["worker"].stop()
        watcher["worker"].join(timeout=5)

        # Persist state for auto-resume
        self._save_state()

        return {
            "status": "stopped",
            "traces_dir": watcher["traces_dir"],
        }

    def get_status(self) -> dict:
        """Get trace watcher status.

        Returns:
            Status dict
        """
        with self._lock:
            if self._watcher is None:
                return {"is_watching": False}

            return {
                "is_watching": True,
                "traces_dir": self._watcher["traces_dir"],
                "quiet_period": self._watcher["quiet_period"],
                "started_at": self._watcher["started_at"],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RESUME FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def auto_resume_watchers(
    code_watcher_manager: CloudWatcherManager,
    trace_watcher_manager: TraceWatcherManager,
) -> dict:
    """Resume watchers from persisted state on startup.

    Should be called once during server initialization.

    Args:
        code_watcher_manager: CloudWatcherManager instance
        trace_watcher_manager: TraceWatcherManager instance

    Returns:
        Dict with resume status and results
    """
    state = _load_watcher_state()
    results = {
        "code_watchers_resumed": 0,
        "code_watchers_failed": 0,
        "trace_watcher_resumed": False,
        "errors": [],
    }

    # Resume code watchers
    for project_path in state.get("code_watchers", []):
        try:
            path = Path(project_path)
            if path.exists():
                result = code_watcher_manager.start_watching(path)
                if result.get("status") in ("started", "already_watching"):
                    results["code_watchers_resumed"] += 1
                    log.info(f"Resumed code watcher for {project_path}")
                else:
                    results["code_watchers_failed"] += 1
                    results["errors"].append(f"Failed to resume {project_path}: {result}")
            else:
                log.warning(f"Skipping stale watcher path: {project_path}")
                results["errors"].append(f"Path no longer exists: {project_path}")
        except Exception as e:
            results["code_watchers_failed"] += 1
            results["errors"].append(f"Error resuming {project_path}: {e}")
            log.warning(f"Failed to resume code watcher for {project_path}: {e}")

    # Resume trace watcher
    if state.get("trace_watcher", False):
        try:
            result = trace_watcher_manager.start_watching()
            if result.get("status") in ("started", "already_watching"):
                results["trace_watcher_resumed"] = True
                log.info("Resumed trace watcher")
            else:
                results["errors"].append(f"Failed to resume trace watcher: {result}")
        except Exception as e:
            results["errors"].append(f"Error resuming trace watcher: {e}")
            log.warning(f"Failed to resume trace watcher: {e}")

    log.info(
        f"Auto-resume complete: {results['code_watchers_resumed']} code watchers, "
        f"trace={results['trace_watcher_resumed']}"
    )
    return results
