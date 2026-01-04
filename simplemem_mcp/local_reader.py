"""Local file reader for the thin MCP layer.

Reads files and traces from the local filesystem,
preparing them for compression and transport to the backend API.
"""

import fnmatch
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger("simplemem_mcp.local_reader")

# UUID pattern for session ID validation (prevents path traversal)
UUID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)

# Default patterns for code files
DEFAULT_CODE_PATTERNS = [
    "**/*.py",
    "**/*.ts",
    "**/*.js",
    "**/*.tsx",
    "**/*.jsx",
]

# Directories to skip when scanning
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "coverage",
    ".tox",
    "eggs",
    "*.egg-info",
}


class LocalReader:
    """Reads files and traces from local filesystem.

    Used by the thin MCP layer to:
    - Read trace files from ~/.claude/projects/
    - Read code files for indexing
    - Discover sessions for batch processing

    All content is returned as dicts/lists ready for JSON serialization
    and compression before sending to the backend.
    """

    def __init__(
        self,
        traces_dir: Path | None = None,
        code_patterns: list[str] | None = None,
    ):
        """Initialize the local reader.

        Args:
            traces_dir: Directory containing Claude traces (default: ~/.claude/projects)
            code_patterns: Glob patterns for code files (default: Python, JS, TS)
        """
        self.traces_dir = traces_dir or Path.home() / ".claude" / "projects"
        self.code_patterns = code_patterns or DEFAULT_CODE_PATTERNS

    # ═══════════════════════════════════════════════════════════════════════════════
    # TRACE FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    def discover_sessions(
        self,
        days_back: int = 30,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Discover available Claude Code sessions.

        Args:
            days_back: Only include sessions modified within this many days
            limit: Maximum number of sessions to return (default: 20)
            offset: Number of sessions to skip for pagination (default: 0)

        Returns:
            Dict with sessions list, total count, and pagination info
        """
        sessions = []
        cutoff_time = datetime.now(timezone.utc).timestamp() - (days_back * 86400)

        if not self.traces_dir.exists():
            log.warning(f"Traces directory does not exist: {self.traces_dir}")
            return {"sessions": [], "total": 0, "has_more": False}

        for project_dir in self.traces_dir.iterdir():
            if not project_dir.is_dir():
                continue

            for trace_file in project_dir.glob("*.jsonl"):
                try:
                    stat = trace_file.stat()
                    if stat.st_mtime < cutoff_time:
                        continue

                    # Extract human-readable project name from encoded path
                    project_name = project_dir.name
                    if project_name.startswith("-"):
                        # Decode: "-Users-foo-repo-bar" -> "bar"
                        parts = project_name.split("-")
                        project_name = parts[-1] if parts else project_name

                    sessions.append({
                        "id": trace_file.stem[:8],  # Truncated UUID for display
                        "session_id": trace_file.stem,  # Full UUID for operations
                        "project": project_name,
                        "size_kb": stat.st_size // 1024,
                        "modified": stat.st_mtime,
                    })
                except (OSError, PermissionError) as e:
                    log.warning(f"Could not stat {trace_file}: {e}")
                    continue

        # Sort by most recent first
        sessions = sorted(sessions, key=lambda x: x["modified"], reverse=True)
        total = len(sessions)

        # Apply pagination
        paginated = sessions[offset:offset + limit]

        return {
            "sessions": paginated,
            "total": total,
            "has_more": offset + limit < total,
        }

    def find_session_path(self, session_id: str) -> Path | None:
        """Find a session trace file by ID.

        Args:
            session_id: Session UUID

        Returns:
            Path to trace file if found

        Security:
            - Validates session_id is a valid UUID format (prevents path traversal)
            - Ensures resolved path stays within traces_dir (defense in depth)
        """
        # Validate session_id format to prevent path traversal attacks
        # e.g., blocking "../../etc/passwd" or "../../../sensitive"
        if not UUID_PATTERN.match(session_id):
            log.warning(
                f"Invalid session_id format (expected UUID): {session_id[:50]!r}"
            )
            return None

        if not self.traces_dir.exists():
            return None

        # Resolve traces_dir once for path containment check
        try:
            traces_dir_resolved = self.traces_dir.resolve()
        except (OSError, ValueError):
            log.error(f"Could not resolve traces directory: {self.traces_dir}")
            return None

        for project_dir in self.traces_dir.iterdir():
            if not project_dir.is_dir():
                continue

            trace_file = project_dir / f"{session_id}.jsonl"

            # Defense in depth: ensure resolved path is within traces_dir
            try:
                resolved = trace_file.resolve()
                if not str(resolved).startswith(str(traces_dir_resolved)):
                    log.warning(f"Path escape attempt blocked: {trace_file}")
                    continue
            except (OSError, ValueError):
                continue

            if trace_file.exists():
                return trace_file

        return None

    def read_trace_file(self, session_id: str) -> list[dict] | None:
        """Read and parse a trace file.

        Args:
            session_id: Session UUID to read

        Returns:
            List of trace entries (raw JSONL parsed), or None if not found
        """
        trace_path = self.find_session_path(session_id)
        if trace_path is None:
            log.warning(f"Session {session_id} not found")
            return None

        return self.read_trace_file_by_path(trace_path)

    def read_trace_file_by_path(self, trace_path: Path) -> list[dict] | None:
        """Read and parse a trace file by path.

        Args:
            trace_path: Path to trace file

        Returns:
            List of trace entries (raw JSONL parsed), or None if error
        """
        if not trace_path.exists():
            log.warning(f"Trace file not found: {trace_path}")
            return None

        entries = []
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            log.debug(f"Read {len(entries)} entries from {trace_path.name}")
            return entries

        except Exception as e:
            log.error(f"Failed to read trace file {trace_path}: {e}")
            return None

    def get_session_metadata(self, session_id: str) -> dict | None:
        """Get metadata for a session without reading full content.

        Args:
            session_id: Session UUID

        Returns:
            Session metadata dict or None if not found
        """
        trace_path = self.find_session_path(session_id)
        if trace_path is None:
            return None

        try:
            stat = trace_path.stat()
            project_name = trace_path.parent.name

            # Count lines without loading all content
            with open(trace_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)

            return {
                "session_id": session_id,
                "project": project_name,
                "path": str(trace_path),
                "size_kb": stat.st_size // 1024,
                "line_count": line_count,
                "modified": stat.st_mtime,
                "modified_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        except (OSError, PermissionError) as e:
            log.warning(f"Could not get metadata for {session_id}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _is_excluded_path(self, file_path: Path) -> bool:
        """Check if a path should be excluded based on SKIP_DIRS patterns.

        Args:
            file_path: Path to check

        Returns:
            True if any path part matches a skip pattern
        """
        for part in file_path.parts:
            # Skip hidden directories/files
            if part.startswith("."):
                return True
            # Check against skip patterns (supports glob patterns like *.egg-info)
            for skip_pattern in SKIP_DIRS:
                if fnmatch.fnmatch(part, skip_pattern):
                    return True
        return False

    def scan_code_files(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
    ) -> Iterator[dict]:
        """Scan a directory for code files.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match (default: code_patterns)
            max_files: Maximum files to return
            max_file_size_kb: Skip files larger than this

        Yields:
            Dicts with path and content for each file
        """
        patterns = patterns or self.code_patterns
        directory = Path(directory)

        if not directory.exists():
            log.warning(f"Directory does not exist: {directory}")
            return

        # Collect unique files from all patterns to avoid duplicates
        found_paths: set[Path] = set()
        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    found_paths.add(file_path)

        file_count = 0
        max_size_bytes = max_file_size_kb * 1024

        # Sort for deterministic order
        for file_path in sorted(found_paths):
            # Skip hidden and excluded directories using fnmatch
            if self._is_excluded_path(file_path):
                continue

            # Check file size
            try:
                stat = file_path.stat()
                if stat.st_size > max_size_bytes:
                    log.debug(f"Skipping large file: {file_path} ({stat.st_size // 1024}KB)")
                    continue
            except OSError:
                continue

            # Read content
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                log.debug(f"Skipping unreadable file {file_path}: {e}")
                continue

            # Return relative path from directory
            rel_path = file_path.relative_to(directory)
            yield {
                "path": str(rel_path),
                "content": content,
            }

            file_count += 1
            if file_count >= max_files:
                log.warning(f"Reached max_files limit ({max_files})")
                return

    def read_code_files(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
    ) -> list[dict]:
        """Read all matching code files from a directory.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match
            max_files: Maximum files to return
            max_file_size_kb: Skip files larger than this

        Returns:
            List of {"path": str, "content": str} dicts
        """
        return list(self.scan_code_files(
            directory=directory,
            patterns=patterns,
            max_files=max_files,
            max_file_size_kb=max_file_size_kb,
        ))

    def read_single_file(self, file_path: Path) -> str | None:
        """Read a single file's content.

        Args:
            file_path: Path to file

        Returns:
            File content or None if unreadable
        """
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, FileNotFoundError) as e:
            log.warning(f"Could not read {file_path}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════════

    def check_directory_exists(self, directory: Path) -> bool:
        """Check if a directory exists and is accessible."""
        return Path(directory).is_dir()

    def get_directory_info(self, directory: Path) -> dict | None:
        """Get info about a directory.

        Args:
            directory: Directory path

        Returns:
            Dict with exists, is_git, file_count info, or None if error
        """
        directory = Path(directory)

        if not directory.exists():
            return {"exists": False}

        try:
            is_git = (directory / ".git").is_dir()

            # Count files matching patterns
            file_count = 0
            for pattern in self.code_patterns:
                for _ in directory.glob(pattern):
                    file_count += 1
                    if file_count > 10000:  # Limit counting
                        break
                if file_count > 10000:
                    break

            return {
                "exists": True,
                "is_git": is_git,
                "file_count": file_count if file_count <= 10000 else f">{10000}",
                "path": str(directory.absolute()),
            }
        except (OSError, PermissionError) as e:
            log.warning(f"Could not get info for {directory}: {e}")
            return None
