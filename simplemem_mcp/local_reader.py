"""Local file reader for the thin MCP layer.

Reads files and traces from the local filesystem,
preparing them for compression and transport to the backend API.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pathspec

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

# Directories to skip when scanning (gitignore-style patterns)
SKIP_DIRS = [
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    "dist/",
    "build/",
    ".next/",
    "coverage/",
    ".tox/",
    "eggs/",
    "*.egg-info/",
]

# Pre-compiled pathspec for built-in exclusions (built once at module load)
_BUILTIN_SPEC = pathspec.PathSpec.from_lines("gitwildmatch", SKIP_DIRS)


def _aggregate_exclusions(excluded: list[dict]) -> dict:
    """Aggregate file exclusions into category breakdown.

    Args:
        excluded: List of {"path": str, "reason": str} dicts

    Returns:
        Dict with counts and examples per exclusion category
    """
    categories: dict[str, dict] = {}

    for item in excluded:
        reason = item["reason"]
        path = item["path"]

        # Extract category (before colon)
        category = reason.split(":")[0].strip()

        if category not in categories:
            categories[category] = {
                "count": 0,
                "top_folders": set(),
                "examples": set(),
            }

        categories[category]["count"] += 1

        # Extract top-level folder for built-in exclusions
        parts = path.split("/")
        if len(parts) > 0 and parts[0]:
            top_folder = parts[0] + "/"
            categories[category]["top_folders"].add(top_folder)

        # Keep a few examples of the actual patterns/reasons
        if len(categories[category]["examples"]) < 3:
            # Extract the specific pattern from the reason
            if ":" in reason:
                specific = reason.split(":", 1)[1].strip()
                categories[category]["examples"].add(specific)

    # Convert sets to sorted lists for JSON serialization
    result = {}
    for cat, data in categories.items():
        result[cat] = {"count": data["count"]}
        if data["top_folders"]:
            result[cat]["top_folders"] = sorted(data["top_folders"])[:5]
        if data["examples"]:
            result[cat]["examples"] = sorted(data["examples"])[:3]

    return result


def _aggregate_folders(excluded: list[dict]) -> list[dict]:
    """Aggregate exclusions by top-level folder.

    Args:
        excluded: List of {"path": str, "reason": str} dicts

    Returns:
        List of folder summaries sorted by file count (descending)
    """
    folders: dict[str, dict] = {}

    for item in excluded:
        path = item["path"]
        reason = item["reason"]

        # Get top-level folder
        parts = path.split("/")
        if not parts[0]:
            continue

        folder = parts[0] + "/"
        category = reason.split(":")[0].strip()

        if folder not in folders:
            folders[folder] = {
                "folder": folder,
                "reason": category,
                "file_count": 0,
            }
        folders[folder]["file_count"] += 1

    # Sort by file count descending
    return sorted(folders.values(), key=lambda x: x["file_count"], reverse=True)


def _save_full_report(
    would_index: list[dict],
    excluded: list[dict],
    directory: Path,
    summary: dict,
) -> tuple[str, float]:
    """Save complete dry_run report to /tmp/.

    Args:
        would_index: Files that would be indexed
        excluded: Files that were excluded
        directory: The scanned directory
        summary: Summary statistics

    Returns:
        Tuple of (report_path, size_mb)
    """
    import time

    timestamp = int(time.time())
    project_name = Path(directory).name
    report_path = Path("/tmp") / f"simplemem-dry-run-{project_name}-{timestamp}.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory),
        "summary": summary,
        "files_to_index": would_index,
        "files_excluded": excluded,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    size_mb = round(report_path.stat().st_size / (1024 * 1024), 2)
    return str(report_path), size_mb


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
        # Cache for user-provided ignore pattern specs (avoids rebuilding)
        self._ignore_spec_cache: dict[tuple, pathspec.PathSpec] = {}

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
                        "id": trace_file.stem[:8],  # Truncated ID for display
                        "session_id": trace_file.stem,  # Full ID for operations
                        "path": str(trace_file),  # Full path for direct access
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

    def _get_ignore_spec(
        self, ignore_patterns: list[str] | None
    ) -> pathspec.PathSpec | None:
        """Get or create a cached pathspec for user ignore patterns."""
        if not ignore_patterns:
            return None
        key = tuple(sorted(ignore_patterns))
        if key not in self._ignore_spec_cache:
            self._ignore_spec_cache[key] = pathspec.PathSpec.from_lines(
                "gitwildmatch", ignore_patterns
            )
        return self._ignore_spec_cache[key]

    def _is_excluded_path(
        self,
        file_path: Path,
        ignore_patterns: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        """Check if a path should be excluded based on SKIP_DIRS and ignore_patterns.

        Uses pathspec with gitwildmatch semantics for proper gitignore-style matching.
        Supports ** glob patterns and ! negation.

        Args:
            file_path: Path to check (should be relative to scan root)
            ignore_patterns: Additional gitignore-style patterns to exclude

        Returns:
            Tuple of (is_excluded, reason) where reason explains why if excluded
        """
        path_str = str(file_path)

        # Check built-in exclusions first (pre-compiled pathspec)
        if _BUILTIN_SPEC.match_file(path_str):
            # Use simple heuristic to identify matching pattern (avoids PathSpec creation)
            for pattern in SKIP_DIRS:
                pattern_base = pattern.rstrip("/")
                if pattern_base in path_str or (
                    pattern_base.startswith("*") and path_str.endswith(pattern_base[1:])
                ):
                    return True, f"built-in: {pattern_base}"
            return True, "built-in"

        # Check hidden directories (not files - allow .github/, .storybook/, etc.)
        for part in file_path.parts[:-1]:  # Exclude the filename itself
            if part.startswith(".") and part not in {".github", ".storybook", ".circleci"}:
                return True, f"hidden_dir: {part}"

        # Check user-provided ignore patterns with caching
        if ignore_patterns:
            user_spec = self._get_ignore_spec(ignore_patterns)
            if user_spec and user_spec.match_file(path_str):
                # Use simple heuristic for reason (avoids PathSpec creation per pattern)
                for pattern in ignore_patterns:
                    pattern_base = pattern.rstrip("/").lstrip("**/")
                    if pattern_base in path_str or file_path.name == pattern_base:
                        return True, f"ignore_patterns: {pattern}"
                return True, "ignore_patterns"

        return False, None

    def _should_skip_dir(self, dir_name: str, rel_dir_path: Path, ignore_patterns: list[str] | None) -> tuple[bool, str | None]:
        """Check if a directory should be skipped during traversal.

        This is called BEFORE descending into a directory, enabling early pruning.

        Args:
            dir_name: Name of the directory (e.g., "node_modules")
            rel_dir_path: Relative path to directory from scan root
            ignore_patterns: User-provided gitignore-style patterns

        Returns:
            Tuple of (should_skip, reason)
        """
        # Check built-in exclusions (node_modules, .venv, etc.)
        dir_with_slash = dir_name + "/"
        if _BUILTIN_SPEC.match_file(dir_with_slash):
            return True, f"built-in: {dir_name}"

        # Check hidden directories (but allow .github, .storybook, etc.)
        if dir_name.startswith(".") and dir_name not in {".github", ".storybook", ".circleci"}:
            return True, f"hidden_dir: {dir_name}"

        # Check user-provided ignore patterns against the full relative path
        if ignore_patterns:
            user_spec = self._get_ignore_spec(ignore_patterns)
            rel_path_str = str(rel_dir_path / dir_name)
            if user_spec and user_spec.match_file(rel_path_str + "/"):
                return True, f"ignore_patterns: {dir_name}"

        return False, None

    def _scan_files_internal(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
        read_content: bool = True,
    ) -> Iterator[tuple[Path, Path, dict | None]]:
        """Internal generator for file scanning (shared by scan_code_files and dry_run_scan).

        Uses os.walk() with directory pruning for O(matched_files) performance
        instead of O(tree_size) from glob traversal.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match (default: code_patterns)
            ignore_patterns: Gitignore-style patterns to exclude
            max_files: Maximum files to yield
            max_file_size_kb: Skip files larger than this
            read_content: If True, read file content; if False, just validate

        Yields:
            Tuples of (abs_path, rel_path, exclusion_info) where:
            - exclusion_info is None if file should be indexed
            - exclusion_info is {"reason": str} if file should be excluded
        """
        patterns = patterns or self.code_patterns
        directory = Path(directory).resolve()

        if not directory.exists():
            log.warning(f"Directory does not exist: {directory}")
            return

        # Build pathspec for include patterns (converts glob to pathspec)
        include_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        file_count = 0
        max_size_bytes = max_file_size_kb * 1024

        # Collect files to sort them (for deterministic output)
        found_files: list[tuple[Path, Path]] = []

        # Walk directory tree with pruning
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            rel_root = root_path.relative_to(directory)

            # PRUNE: Remove excluded directories IN-PLACE (prevents descent)
            # This is the key optimization - we never enter node_modules, .venv, etc.
            original_dirs = dirs[:]
            dirs[:] = []
            for d in original_dirs:
                should_skip, reason = self._should_skip_dir(d, rel_root, ignore_patterns)
                if not should_skip:
                    dirs.append(d)

            # Check each file in current directory
            for filename in files:
                file_path = root_path / filename
                rel_path = rel_root / filename

                # Check if file matches include patterns
                if not include_spec.match_file(str(rel_path)):
                    continue

                found_files.append((file_path, rel_path))

        # Sort for deterministic output and process
        for file_path, rel_path in sorted(found_files, key=lambda x: x[1]):
            # Check file-level exclusions (patterns that match files, not just dirs)
            is_excluded, reason = self._is_excluded_path(rel_path, ignore_patterns)
            if is_excluded:
                yield file_path, rel_path, {"reason": reason}
                continue

            # Check file size
            try:
                stat = file_path.stat()
                size_kb = stat.st_size / 1024
                if stat.st_size > max_size_bytes:
                    yield file_path, rel_path, {"reason": f"too_large: {size_kb:.1f}KB > {max_file_size_kb}KB"}
                    continue
            except OSError as e:
                yield file_path, rel_path, {"reason": f"stat_error: {e}"}
                continue

            # Check readability (read content if requested)
            if read_content:
                try:
                    file_path.read_text(encoding="utf-8")
                    yield file_path, rel_path, None  # Success - caller will read content
                except (UnicodeDecodeError, OSError) as e:
                    yield file_path, rel_path, {"reason": f"unreadable: {e}"}
                    continue
            else:
                # Just check if readable without reading full content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.read(1)
                    yield file_path, rel_path, None  # Success
                except (UnicodeDecodeError, OSError):
                    yield file_path, rel_path, {"reason": "unreadable: encoding or permission error"}
                    continue

            file_count += 1
            if file_count >= max_files:
                log.info(f"Reached max_files limit ({max_files})")
                return

    def scan_code_files(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
    ) -> Iterator[dict]:
        """Scan a directory for code files.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match (default: code_patterns)
            ignore_patterns: Gitignore-style patterns to exclude
            max_files: Maximum files to return
            max_file_size_kb: Skip files larger than this

        Yields:
            Dicts with path and content for each file
        """
        for file_path, rel_path, exclusion in self._scan_files_internal(
            directory, patterns, ignore_patterns, max_files, max_file_size_kb, read_content=True
        ):
            if exclusion is None:
                # File passed all checks - read and yield content
                try:
                    content = file_path.read_text(encoding="utf-8")
                    yield {"path": str(rel_path), "content": content}
                except (UnicodeDecodeError, OSError):
                    continue

    def read_code_files(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
    ) -> list[dict]:
        """Read all matching code files from a directory.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match
            ignore_patterns: Gitignore-style patterns to exclude
            max_files: Maximum files to return
            max_file_size_kb: Skip files larger than this

        Returns:
            List of {"path": str, "content": str} dicts
        """
        return list(self.scan_code_files(
            directory=directory,
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            max_files=max_files,
            max_file_size_kb=max_file_size_kb,
        ))

    def dry_run_scan(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_files: int = 1000,
        max_file_size_kb: int = 500,
        verbosity: str = "minimal",
    ) -> dict:
        """Preview what files would be indexed without reading content.

        Args:
            directory: Directory to scan
            patterns: Glob patterns to match (default: code_patterns)
            ignore_patterns: Gitignore-style patterns to exclude
            max_files: Maximum files to return
            max_file_size_kb: Skip files larger than this
            verbosity: Output detail level:
                "minimal" (default): ~2KB - summary + exclusion breakdown
                "folders": ~5KB - folder-level aggregation
                "full": Saves complete report to /tmp/, returns path

        Returns:
            Dict with summary and verbosity-appropriate details
        """
        patterns = patterns or self.code_patterns
        directory = Path(directory)

        if not directory.exists():
            return {
                "dry_run": True,
                "verbosity": verbosity,
                "error": f"Directory does not exist: {directory}",
                "summary": {
                    "files_to_index": 0,
                    "files_excluded": 0,
                    "total_size_kb": 0,
                },
            }

        would_index: list[dict] = []
        excluded: list[dict] = []
        total_size_bytes = 0

        for file_path, rel_path, exclusion in self._scan_files_internal(
            directory, patterns, ignore_patterns, max_files, max_file_size_kb, read_content=False
        ):
            if exclusion is not None:
                excluded.append({"path": str(rel_path), "reason": exclusion["reason"]})
            else:
                # File passed all checks
                try:
                    size_bytes = file_path.stat().st_size
                    would_index.append({
                        "path": str(rel_path),
                        "size_kb": round(size_bytes / 1024, 1),
                    })
                    total_size_bytes += size_bytes
                except OSError:
                    pass  # Already checked in _scan_files_internal

        summary = {
            "files_to_index": len(would_index),
            "files_excluded": len(excluded),
            "total_size_kb": round(total_size_bytes / 1024, 1),
        }

        # Return different structures based on verbosity
        if verbosity == "minimal":
            return {
                "dry_run": True,
                "verbosity": "minimal",
                "summary": summary,
                "exclusion_breakdown": _aggregate_exclusions(excluded),
            }

        elif verbosity == "folders":
            return {
                "dry_run": True,
                "verbosity": "folders",
                "summary": summary,
                "excluded_folders": _aggregate_folders(excluded),
            }

        else:  # "full"
            report_path, size_mb = _save_full_report(
                would_index, excluded, directory, summary
            )
            return {
                "dry_run": True,
                "verbosity": "full",
                "summary": summary,
                "report_path": report_path,
                "report_size_mb": size_mb,
            }

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
