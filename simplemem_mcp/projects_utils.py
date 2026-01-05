"""Project ID utilities with deterministic identification strategies.

Hierarchical project identification (most stable first):
1. Git remote URL - stable across machines and paths
2. Config file (.simplemem.yaml) - explicit user control

IMPORTANT: Hash-based and path-based project IDs are DEPRECATED.
Projects MUST have either:
- A git remote (preferred)
- A .simplemem.yaml config file with explicit project_id

ID Format Prefixes:
- git:github.com/user/repo - Git remote based
- config:mycompany/myproject - Config file based
- hash:a1b2c3d4e5f6... - DEPRECATED (legacy, will be removed)
- path:/Users/dev/project - DEPRECATED (legacy, will be removed)
"""

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

log = logging.getLogger("simplemem_mcp.projects_utils")

# Maximum directory levels to search upward for config
CONFIG_SEARCH_MAX_DEPTH = 10

# Config file names in order of preference
CONFIG_FILES = [".simplemem.yaml", ".simplemem.yml", ".simplemem.json"]

# Project marker files in priority order
PROJECT_MARKERS = [
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "CMakeLists.txt",
    "Makefile",
]

# Git command timeout in seconds
GIT_TIMEOUT = 2


def normalize_git_url(url: str) -> str:
    """Normalize git URL to consistent format.

    Handles various git URL formats and normalizes to: host/owner/repo

    Args:
        url: Git remote URL in any format

    Returns:
        Normalized URL without protocol, auth, or .git suffix

    Examples:
        >>> normalize_git_url("git@github.com:user/repo.git")
        'github.com/user/repo'
        >>> normalize_git_url("https://github.com/user/repo.git")
        'github.com/user/repo'
        >>> normalize_git_url("ssh://git@bitbucket.org/team/repo")
        'bitbucket.org/team/repo'
        >>> normalize_git_url("https://gitlab.company.com:8443/group/repo.git")
        'gitlab.company.com:8443/group/repo'
    """
    if not url:
        return ""

    # Remove .git suffix
    url = re.sub(r"\.git$", "", url.strip())

    # HTTPS format: https://github.com/user/repo (check FIRST - more specific)
    # Supports optional port: https://gitlab.company.com:8443/group/repo
    https_match = re.match(r"^https?://(?:[\w.-]+@)?([\w.-]+(?::\d+)?)/(.+)$", url)
    if https_match:
        host, path = https_match.groups()
        return f"{host}/{path}"

    # SSH with explicit protocol: ssh://git@bitbucket.org/team/repo
    ssh_protocol_match = re.match(r"^ssh://(?:[\w.-]+@)?([\w.-]+(?::\d+)?)/(.+)$", url)
    if ssh_protocol_match:
        host, path = ssh_protocol_match.groups()
        return f"{host}/{path}"

    # SSH shorthand format: git@github.com:user/repo (colon separator, no protocol)
    ssh_shorthand_match = re.match(r"^(?:[\w.-]+@)?([\w.-]+):(.+)$", url)
    if ssh_shorthand_match:
        host, path = ssh_shorthand_match.groups()
        # Don't match if path looks like a port number followed by path (would be ssh://)
        if not re.match(r"^\d+/", path):
            return f"{host}/{path}"

    # Already normalized or unknown format
    return url


def get_git_remote_url(path: Path) -> str | None:
    """Extract normalized git remote origin URL.

    Args:
        path: Directory to check for git repo

    Returns:
        Normalized git remote URL or None if not a git repo
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw_url = result.stdout.strip()
            normalized = normalize_git_url(raw_url)
            log.debug(f"Git remote: {raw_url} -> {normalized}")
            return normalized
    except subprocess.TimeoutExpired:
        log.warning(f"Git command timed out for {path}")
    except FileNotFoundError:
        log.debug("Git not installed")
    except Exception as e:
        log.debug(f"Git remote detection failed: {e}")

    return None


def load_simplemem_config(path: Path) -> dict | None:
    """Load .simplemem.json config file if it exists.

    Args:
        path: Project root directory

    Returns:
        Parsed config dict or None if not found/invalid

    Config Schema:
        {
            "version": 1,
            "project_id": "uuid:550e8400-e29b-41d4-a716-446655440000",
            "name": "my-project",
            "created": "2025-01-05T00:00:00Z"
        }
    """
    config_path = path / ".simplemem.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
            if isinstance(config, dict) and "project_id" in config:
                log.debug(f"Loaded config from {config_path}")
                return config
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"Failed to load config {config_path}: {e}")

    return None


def load_simplemem_yaml(path: Path) -> dict | None:
    """Load .simplemem.yaml config file if it exists.

    Args:
        path: Directory containing the config file (not the file path itself)

    Returns:
        Parsed config dict or None if not found/invalid
    """
    for config_name in [".simplemem.yaml", ".simplemem.yml"]:
        config_path = path / config_name
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                    # Validate structure
                    if not isinstance(config, dict):
                        log.warning(f"Config {config_path} must be a YAML mapping")
                        continue

                    # Validate project_id is a non-empty string
                    project_id = config.get("project_id")
                    if not isinstance(project_id, str) or not project_id.strip():
                        log.warning(f"Config {config_path} requires 'project_id' to be a non-empty string")
                        continue

                    log.debug(f"Loaded YAML config from {config_path}")
                    return config

            except yaml.YAMLError as e:
                log.warning(f"Failed to parse YAML config {config_path}: {e}")
            except OSError as e:
                log.warning(f"Failed to read config {config_path}: {e}")

    return None


def find_config_file(start_path: Path) -> tuple[Path, Mapping[str, Any]] | None:
    """Walk up directory tree to find .simplemem.yaml config.

    Searches from start_path upward (max CONFIG_SEARCH_MAX_DEPTH levels)
    looking for .simplemem.yaml, .simplemem.yml, or .simplemem.json.

    Args:
        start_path: Starting directory for search

    Returns:
        Tuple of (config_directory, config_dict) or None if not found
    """
    current = start_path.resolve()
    depth = 0

    while depth < CONFIG_SEARCH_MAX_DEPTH:
        # Try YAML config first (preferred)
        yaml_config = load_simplemem_yaml(current)
        if yaml_config:
            log.debug(f"Found YAML config at {current} (depth={depth})")
            return current, yaml_config

        # Fall back to JSON config (legacy)
        json_config = load_simplemem_config(current)
        if json_config:
            log.debug(f"Found JSON config at {current} (depth={depth})")
            return current, json_config

        # Move up one level
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
        depth += 1

    log.debug(f"No config found searching from {start_path} (searched {depth} levels)")
    return None


def _extract_stable_identity(marker: str, content: bytes) -> str | None:
    """Extract stable identity fields from marker file content.

    Parses marker files to extract only stable fields (name, repository)
    that won't change with version bumps or dependency updates.

    Args:
        marker: Marker filename (e.g., "package.json")
        content: Raw file content

    Returns:
        Stable identity string or None if parsing fails
    """
    try:
        text = content.decode("utf-8")

        if marker == "package.json":
            data = json.loads(text)
            # Use name + repository for stable identity
            name = data.get("name", "")
            repo = data.get("repository", {})
            if isinstance(repo, dict):
                repo = repo.get("url", "")
            return f"npm:{name}:{repo}" if name else None

        elif marker == "pyproject.toml":
            # Simple TOML parsing for name field
            import tomllib
            data = tomllib.loads(text)
            name = data.get("project", {}).get("name") or data.get("tool", {}).get("poetry", {}).get("name")
            return f"pypi:{name}" if name else None

        elif marker == "Cargo.toml":
            # Simple TOML parsing for package name
            import tomllib
            data = tomllib.loads(text)
            name = data.get("package", {}).get("name")
            return f"cargo:{name}" if name else None

        elif marker == "go.mod":
            # Extract module path from first line
            for line in text.split("\n"):
                if line.startswith("module "):
                    module = line.split(" ", 1)[1].strip()
                    return f"go:{module}"
            return None

        elif marker == "pom.xml":
            # Extract groupId:artifactId from XML
            group = re.search(r"<groupId>([^<]+)</groupId>", text)
            artifact = re.search(r"<artifactId>([^<]+)</artifactId>", text)
            if group and artifact:
                return f"maven:{group.group(1)}:{artifact.group(1)}"
            return None

    except Exception as e:
        log.debug(f"Failed to parse {marker} for stable identity: {e}")

    return None


def hash_project_markers(path: Path) -> str | None:
    """Generate hash from stable identity fields in project marker files.

    Extracts only stable fields (name, repository) from marker files,
    ignoring version numbers and dependencies that change frequently.
    Falls back to full content hash if parsing fails.

    Args:
        path: Project root directory

    Returns:
        SHA256 hash prefix (16 chars) or None if no markers found
    """
    for marker in PROJECT_MARKERS:
        marker_path = path / marker
        if marker_path.exists():
            try:
                content = marker_path.read_bytes()

                # Try to extract stable identity first
                stable_identity = _extract_stable_identity(marker, content)
                if stable_identity:
                    hash_digest = hashlib.sha256(stable_identity.encode()).hexdigest()[:16]
                    log.debug(f"Hashed stable identity from {marker}: {stable_identity} -> {hash_digest}")
                    return hash_digest

                # Fallback to full content hash (less stable but still works)
                hash_digest = hashlib.sha256(content).hexdigest()[:16]
                log.debug(f"Hashed full content of {marker}: {hash_digest}")
                return hash_digest

            except IOError as e:
                log.debug(f"Failed to read {marker_path}: {e}")
                continue

    return None


class ProjectIdError(Exception):
    """Raised when project ID cannot be determined."""

    def __init__(self, path: Path, message: str, suggestion: str):
        self.path = path
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message}\n\nSuggestion: {suggestion}")


def get_project_id(path: str | Path, strict: bool = False) -> str:
    """Generate hierarchical project ID using deterministic strategies.

    Tries strategies in order of stability:
    1. Git remote URL (most stable, cross-machine) - PREFERRED
    2. Config file (.simplemem.yaml) - walk up directories
    3. Content hash of project markers - DEPRECATED, logs warning
    4. Resolved absolute path - DEPRECATED, logs warning

    Args:
        path: Project root path (absolute or relative)
        strict: If True, raise ProjectIdError instead of falling back to
                deprecated hash/path strategies. Use strict=True for new code.

    Returns:
        Project ID with format prefix (git:, config:, hash:, path:)

    Raises:
        ProjectIdError: If strict=True and no git remote or config found

    Examples:
        >>> get_project_id("/repo/myproject")  # with git remote
        'git:github.com/user/myproject'
        >>> get_project_id("/repo/myproject")  # with .simplemem.yaml
        'config:mycompany/myproject'
        >>> get_project_id("/repo/myproject", strict=True)  # no git/config
        ProjectIdError: No project ID found. Create .simplemem.yaml...
    """
    resolved_path = Path(path).expanduser().resolve()

    # Strategy 1: Git remote URL (most stable)
    git_url = get_git_remote_url(resolved_path)
    if git_url:
        return f"git:{git_url}"

    # Strategy 2: Walk up directories to find config file (.simplemem.yaml or .simplemem.json)
    config_result = find_config_file(resolved_path)
    if config_result:
        config_dir, config = config_result
        project_id = config["project_id"]
        # Add config: prefix if not already prefixed
        if not any(project_id.startswith(p) for p in ["git:", "uuid:", "config:", "hash:", "path:"]):
            project_id = f"config:{project_id}"
        return project_id

    # --- DEPRECATED FALLBACKS ---
    # These are kept for backwards compatibility but will be removed.
    # New code should use strict=True.

    if strict:
        raise ProjectIdError(
            path=resolved_path,
            message=f"No project ID found for: {resolved_path}",
            suggestion=(
                "Create a .simplemem.yaml file with:\n\n"
                "  version: 1\n"
                f"  project_id: \"{resolved_path.name}\"\n\n"
                "Or initialize a git repository with a remote."
            ),
        )

    # Strategy 3: Content hash (DEPRECATED)
    content_hash = hash_project_markers(resolved_path)
    if content_hash:
        log.warning(
            f"Using DEPRECATED hash-based project ID for {resolved_path}. "
            "Hash IDs are brittle and will be removed in a future version. "
            "Create .simplemem.yaml or add a git remote."
        )
        return f"hash:{content_hash}"

    # Strategy 4: Fallback to resolved path (DEPRECATED)
    log.warning(
        f"Using DEPRECATED path-based project ID for {resolved_path}. "
        "Path IDs are not portable. "
        "Create .simplemem.yaml or add a git remote."
    )
    return f"path:{resolved_path}"


def parse_project_id(project_id: str) -> tuple[str, str]:
    """Parse project ID into type and value.

    Args:
        project_id: Full project ID with optional prefix

    Returns:
        Tuple of (type, value) where type is git/config/uuid/hash/path

    Examples:
        >>> parse_project_id("git:github.com/user/repo")
        ('git', 'github.com/user/repo')
        >>> parse_project_id("config:mycompany/myproject")
        ('config', 'mycompany/myproject')
        >>> parse_project_id("/Users/dev/project")
        ('path', '/Users/dev/project')
    """
    for prefix in ["git:", "config:", "uuid:", "hash:", "path:"]:
        if project_id.startswith(prefix):
            return prefix[:-1], project_id[len(prefix):]

    # No prefix = legacy path format
    return "path", project_id


def extract_project_name(project_id: str) -> str:
    """Extract a human-readable project name from project_id.

    Args:
        project_id: Project ID with optional prefix

    Returns:
        Human-readable project name

    Examples:
        >>> extract_project_name("git:github.com/user/myproject")
        'myproject'
        >>> extract_project_name("path:/Users/shimon/repo/3dtex")
        '3dtex'
    """
    _, value = parse_project_id(project_id)

    # For git URLs, get the repo name
    if "/" in value:
        return value.split("/")[-1]

    # For paths and UUIDs, get the last component
    return Path(value).name if value else "unknown"


def decode_claude_path(encoded_path: str) -> str | None:
    """Decode Claude's encoded project path format.

    Claude Code stores traces in ~/.claude/projects/{encoded-path}/
    where the path is encoded by replacing '/' with '-'.

    Examples:
        -Users-shimon-repo-project -> /Users/shimon/repo/project
        -home-user-code-app -> /home/user/code/app

    Args:
        encoded_path: The encoded directory name from Claude's trace storage

    Returns:
        Decoded absolute path, or None if invalid
    """
    if not encoded_path or encoded_path == "-":
        return "/"

    if encoded_path.startswith("-"):
        decoded = "/" + encoded_path[1:].replace("-", "/")
        return decoded

    return encoded_path


def infer_project_from_session_path(session_path: Path) -> str | None:
    """Infer project_id from a Claude trace file path.

    Args:
        session_path: Path to the session trace file

    Returns:
        Project ID with proper prefix, or None if inference fails
    """
    try:
        # The parent directory name is the encoded project path
        encoded_name = session_path.parent.name
        decoded_path = decode_claude_path(encoded_name)

        if decoded_path:
            resolved = Path(decoded_path).resolve()
            if resolved.exists():
                # Path exists locally - use full hierarchical ID generation
                return get_project_id(resolved)
            else:
                # Path doesn't exist locally - return as path: prefix
                log.debug(f"Decoded path doesn't exist locally: {decoded_path}")
                return f"path:{decoded_path}"

        return None
    except Exception as e:
        log.warning(f"Failed to infer project from session path: {e}")
        return None
