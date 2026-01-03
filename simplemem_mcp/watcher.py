"""Cloud-integrated file watcher for automatic code index updates.

Monitors file changes and sends updates through the BackendClient to the cloud API.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from simplemem_mcp.client import BackendClient
    from simplemem_mcp.local_reader import LocalReader

log = logging.getLogger("simplemem_mcp.watcher")

# Default file patterns to watch
DEFAULT_PATTERNS = ["*.py", "*.ts", "*.js", "*.tsx", "*.jsx"]


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
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File created: {path}")
            self.queue.put(FileEvent(path=path, event_type="created"))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File modified: {path}")
            self.queue.put(FileEvent(path=path, event_type="modified"))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        if self._matches_patterns(path):
            log.debug(f"File deleted: {path}")
            self.queue.put(FileEvent(path=path, event_type="deleted"))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
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
        poll_interval: float = 0.5,
    ):
        """Initialize the worker.

        Args:
            queue: DebouncedQueue with file events
            client: BackendClient for sending updates to cloud
            reader: LocalReader for reading file contents
            project_root: Project root directory
            poll_interval: How often to check for events (seconds)
        """
        super().__init__(daemon=True, name=f"cloud-watcher-{project_root.name}")
        self.queue = queue
        self.client = client
        self.reader = reader
        self.project_root = project_root
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
            log.debug(f"Sending {len(updates)} updates to {self._worker_client.base_url}/api/v1/code/update")
            result = self._loop.run_until_complete(
                self._worker_client.update_code(
                    project_root=str(self.project_root),
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
    ):
        """Initialize the manager.

        Args:
            client: BackendClient for cloud API calls
            reader: LocalReader for reading file contents
        """
        self.client = client
        self.reader = reader
        self._watchers: dict[str, dict] = {}  # project_root -> {observer, worker, queue}
        self._lock = threading.Lock()

    def start_watching(
        self,
        project_root: str | Path,
        patterns: list[str] | None = None,
    ) -> dict:
        """Start watching a project directory.

        Args:
            project_root: Directory to watch
            patterns: Glob patterns for files to watch (default: Python/JS/TS)

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
                }

            log.info(f"Starting watcher for {project_root} with patterns {patterns}")

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

            # Create worker
            worker = CloudWatcherWorker(
                queue=queue,
                client=self.client,
                reader=self.reader,
                project_root=project_root,
            )
            worker.start()

            self._watchers[project_key] = {
                "observer": observer,
                "worker": worker,
                "queue": queue,
                "patterns": patterns,
                "started_at": time.time(),
            }

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
