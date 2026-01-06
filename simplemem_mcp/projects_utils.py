"""Project ID utilities with mandatory bootstrap and multi-folder support.

ARCHITECTURE:
- Projects MUST be bootstrapped with .simplemem.yaml before use
- Multiple folders can share the same project_id (memories/code merge)
- 1 folder = 1 project (no overlapping membership)
- Nearest config wins for nested configs (rare/discouraged)

ID Format:
- config:mycompany/myproject - ONLY valid format (from .simplemem.yaml)

DEPRECATED (will error, not fallback):
- git:, hash:, path: prefixes are no longer valid
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

log = logging.getLogger("simplemem_mcp.projects_utils")


# =============================================================================
# DATACLASSES AND MODELS
# =============================================================================


@dataclass
class ProjectNameSuggestion:
    """Suggested project name with source and confidence.

    Used during bootstrap to provide meaningful suggestions based on
    available project markers (git remote, package.json, pyproject.toml, etc.)
    """

    name: str  # Proposed slug (e.g., "simplemem-mcp")
    source: str  # "git_remote" | "pyproject" | "package_json" | "directory"
    confidence: int  # 0-100, higher = more reliable


@dataclass
class NotBootstrappedError(Exception):
    """Raised when a project requires bootstrap but .simplemem.yaml is missing.

    This exception provides helpful context for the user including:
    - The cwd where they attempted to use SimpleMem
    - Paths that were searched for config
    - Suggested project names based on available markers

    Attributes:
        message: Human-readable error description
        cwd: Current working directory where the error occurred
        searched_paths: List of paths that were searched for .simplemem.yaml
        suggested_names: List of ProjectNameSuggestion for bootstrap
        error_code: Machine-readable error code for programmatic handling
    """

    message: str
    cwd: str
    searched_paths: list[str] = field(default_factory=list)
    suggested_names: list[ProjectNameSuggestion] = field(default_factory=list)
    error_code: str = "SIMPLEMEM_NOT_BOOTSTRAPPED"

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization in MCP responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "cwd": self.cwd,
            "searched_paths": self.searched_paths,
            "suggested_names": [
                {"name": s.name, "source": s.source, "confidence": s.confidence}
                for s in self.suggested_names
            ],
            "action_required": "bootstrap",
            "help": (
                "Run bootstrap_project() or attach_to_project() to initialize. "
                "See suggested_names for recommended project identifiers."
            ),
        }


class SimpleMemConfig(BaseModel):
    """Pydantic model for .simplemem.yaml config validation.

    Config file format:
        version: 1
        project_id: "config:simplemem"
        project_name: "SimpleMem"  # optional display name
        folder_role: "source"  # optional: source, tests, docs
        exclude_patterns: []  # optional gitignore-style patterns

    Multiple folders can share the same project_id to merge their
    memories and code indexes into a single project.
    """

    version: int = 1
    project_id: str  # Must start with "config:"
    project_name: str | None = None  # Human-readable display name
    folder_role: str | None = None  # "source" | "tests" | "docs" | None
    exclude_patterns: list[str] = []

    @field_validator("project_id")
    @classmethod
    def must_have_config_prefix(cls, v: str) -> str:
        """Ensure project_id starts with 'config:' prefix."""
        if not v.startswith("config:"):
            raise ValueError(
                f"project_id must start with 'config:' prefix, got: {v!r}. "
                "Example: 'config:mycompany/myproject'"
            )
        # Validate the slug part
        slug = v[7:]  # Remove 'config:' prefix
        if not slug or slug.isspace():
            raise ValueError("project_id slug cannot be empty after 'config:' prefix")
        return v

    @field_validator("folder_role")
    @classmethod
    def validate_folder_role(cls, v: str | None) -> str | None:
        """Validate folder_role is one of allowed values."""
        if v is None:
            return None
        allowed = {"source", "tests", "docs", "config", "scripts"}
        if v not in allowed:
            raise ValueError(f"folder_role must be one of {allowed}, got: {v!r}")
        return v


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum directory levels to search upward for config
CONFIG_SEARCH_MAX_DEPTH = 10

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


# =============================================================================
# BOOTSTRAP FUNCTIONS
# =============================================================================


def find_project_root(path: str | Path) -> tuple[Path, SimpleMemConfig]:
    """Find the nearest .simplemem.yaml and return parsed config.

    Walks up directory tree from path to find the nearest config file.
    Validates through Pydantic and raises NotBootstrappedError if not found.

    Args:
        path: Starting directory for search

    Returns:
        Tuple of (config_directory, SimpleMemConfig)

    Raises:
        NotBootstrappedError: If no config found in any parent directory
    """
    resolved_path = Path(path).expanduser().resolve()
    searched: list[str] = []

    current = resolved_path
    depth = 0

    while depth < CONFIG_SEARCH_MAX_DEPTH:
        searched.append(str(current))

        # Try YAML config files
        for config_name in [".simplemem.yaml", ".simplemem.yml"]:
            config_path = current / config_name
            if config_path.exists():
                try:
                    with open(config_path, encoding="utf-8") as f:
                        raw_config = yaml.safe_load(f)

                    if not isinstance(raw_config, dict):
                        log.warning(f"Config {config_path} must be a YAML mapping")
                        continue

                    # Validate through Pydantic
                    config = SimpleMemConfig(**raw_config)
                    log.debug(f"Found valid config at {config_path}")
                    return current, config

                except yaml.YAMLError as e:
                    log.warning(f"Failed to parse YAML config {config_path}: {e}")
                except ValueError as e:
                    # Pydantic validation error
                    log.warning(f"Invalid config at {config_path}: {e}")
                except OSError as e:
                    log.warning(f"Failed to read config {config_path}: {e}")

        # Move up one level
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
        depth += 1

    # No config found - generate suggestions and raise error
    suggestions = suggest_project_names(resolved_path)

    raise NotBootstrappedError(
        message=(
            f"Project not bootstrapped. No .simplemem.yaml found.\n"
            f"Searched {len(searched)} directories from: {resolved_path}"
        ),
        cwd=str(resolved_path),
        searched_paths=searched,
        suggested_names=suggestions,
    )


def suggest_project_names(path: str | Path) -> list[ProjectNameSuggestion]:
    """Generate suggested project names from available markers.

    Extracts project name suggestions from (in priority order):
    1. Git remote URL (highest confidence)
    2. pyproject.toml [project.name]
    3. package.json name
    4. Cargo.toml [package.name]
    5. go.mod module path
    6. Directory name (lowest confidence)

    Args:
        path: Directory to analyze for project markers

    Returns:
        List of ProjectNameSuggestion sorted by confidence (highest first)
    """
    resolved_path = Path(path).expanduser().resolve()
    suggestions: list[ProjectNameSuggestion] = []

    # 1. Git remote (highest priority)
    git_url = get_git_remote_url(resolved_path)
    if git_url:
        # Extract repo name from URL (e.g., "github.com/user/repo" -> "repo")
        repo_name = git_url.split("/")[-1] if "/" in git_url else git_url
        suggestions.append(
            ProjectNameSuggestion(name=repo_name, source="git_remote", confidence=95)
        )

    # 2. pyproject.toml
    pyproject_path = resolved_path / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomllib

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            name = data.get("project", {}).get("name")
            if not name:
                name = data.get("tool", {}).get("poetry", {}).get("name")
            if name:
                suggestions.append(
                    ProjectNameSuggestion(name=name, source="pyproject", confidence=85)
                )
        except Exception as e:
            log.debug(f"Failed to parse pyproject.toml: {e}")

    # 3. package.json
    package_json_path = resolved_path / "package.json"
    if package_json_path.exists():
        try:
            with open(package_json_path, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name")
            if name:
                # Strip org prefix if present (e.g., "@org/package" -> "package")
                if name.startswith("@") and "/" in name:
                    name = name.split("/")[-1]
                suggestions.append(
                    ProjectNameSuggestion(name=name, source="package_json", confidence=80)
                )
        except Exception as e:
            log.debug(f"Failed to parse package.json: {e}")

    # 4. Cargo.toml
    cargo_path = resolved_path / "Cargo.toml"
    if cargo_path.exists():
        try:
            import tomllib

            with open(cargo_path, "rb") as f:
                data = tomllib.load(f)
            name = data.get("package", {}).get("name")
            if name:
                suggestions.append(
                    ProjectNameSuggestion(name=name, source="cargo", confidence=80)
                )
        except Exception as e:
            log.debug(f"Failed to parse Cargo.toml: {e}")

    # 5. go.mod
    gomod_path = resolved_path / "go.mod"
    if gomod_path.exists():
        try:
            content = gomod_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                if line.startswith("module "):
                    module_path = line.split(" ", 1)[1].strip()
                    # Extract last component (e.g., "github.com/user/repo" -> "repo")
                    name = module_path.split("/")[-1]
                    suggestions.append(
                        ProjectNameSuggestion(name=name, source="go_mod", confidence=80)
                    )
                    break
        except Exception as e:
            log.debug(f"Failed to parse go.mod: {e}")

    # 6. Directory name (fallback, lowest confidence)
    dir_name = resolved_path.name
    if dir_name and dir_name not in [".", "..", ""]:
        suggestions.append(
            ProjectNameSuggestion(name=dir_name, source="directory", confidence=50)
        )

    # Sort by confidence (highest first) and dedupe by name
    seen_names: set[str] = set()
    unique_suggestions: list[ProjectNameSuggestion] = []
    for s in sorted(suggestions, key=lambda x: -x.confidence):
        if s.name not in seen_names:
            seen_names.add(s.name)
            unique_suggestions.append(s)

    return unique_suggestions


def create_bootstrap_config(
    path: str | Path,
    project_name: str,
    project_id: str | None = None,
    folder_role: str | None = None,
    force: bool = False,
) -> tuple[Path, SimpleMemConfig]:
    """Create .simplemem.yaml config file for a project.

    Performs atomic write (tmp + rename) to avoid partial writes.

    Args:
        path: Directory where to create the config
        project_name: Human-readable project name
        project_id: Explicit project_id (default: "config:{project_name}")
        folder_role: Optional role ("source", "tests", "docs", etc.)
        force: Overwrite existing config if True

    Returns:
        Tuple of (config_path, SimpleMemConfig)

    Raises:
        FileExistsError: If config exists and force=False
        ValueError: If project_name/project_id validation fails
    """
    resolved_path = Path(path).expanduser().resolve()

    # Generate project_id if not provided
    if project_id is None:
        # Sanitize project name for ID (lowercase, replace spaces/special chars)
        slug = project_name.lower()
        slug = re.sub(r"[^a-z0-9/_-]", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        project_id = f"config:{slug}"

    # Validate through Pydantic
    config = SimpleMemConfig(
        version=1,
        project_id=project_id,
        project_name=project_name,
        folder_role=folder_role,
    )

    config_path = resolved_path / ".simplemem.yaml"

    # Check for existing config
    if config_path.exists() and not force:
        raise FileExistsError(
            f"Config already exists at {config_path}. Use force=True to overwrite."
        )

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".yaml", prefix=".simplemem-", dir=resolved_path
    )
    try:
        # Build YAML content (manual to preserve formatting)
        yaml_content = f"""# SimpleMem project configuration
# See: https://github.com/shimonvainer/simplemem

version: {config.version}
project_id: "{config.project_id}"
"""
        if config.project_name:
            yaml_content += f'project_name: "{config.project_name}"\n'
        if config.folder_role:
            yaml_content += f'folder_role: "{config.folder_role}"\n'

        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        # Atomic rename
        import os

        os.replace(tmp_path, config_path)
        log.info(f"Created bootstrap config at {config_path}")

    except Exception:
        # Clean up temp file on error
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
        raise

    return config_path, config


def get_project_id(path: str | Path) -> str:
    """Get project ID from .simplemem.yaml config (STRICT MODE).

    IMPORTANT: This function ONLY returns config-based project IDs.
    If no .simplemem.yaml is found, NotBootstrappedError is raised.

    Projects MUST be bootstrapped before use. There are no fallbacks
    to git:, hash:, or path: based IDs.

    Args:
        path: Project root path (absolute or relative)

    Returns:
        Project ID with "config:" prefix (e.g., "config:simplemem")

    Raises:
        NotBootstrappedError: If no .simplemem.yaml found in path or parents

    Examples:
        >>> get_project_id("/repo/myproject")  # with .simplemem.yaml
        'config:mycompany/myproject'
        >>> get_project_id("/repo/not-bootstrapped")
        NotBootstrappedError: Project not bootstrapped...
    """
    # find_project_root() raises NotBootstrappedError if no config found
    _, config = find_project_root(path)
    return config.project_id


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

    IMPORTANT: This function only returns project_id if the decoded path
    has a valid .simplemem.yaml config. Sessions from non-bootstrapped
    projects are skipped.

    Args:
        session_path: Path to the session trace file

    Returns:
        Project ID with "config:" prefix, or None if:
        - Path cannot be decoded
        - Decoded path doesn't exist locally
        - Project is not bootstrapped (no .simplemem.yaml)
    """
    try:
        # The parent directory name is the encoded project path
        encoded_name = session_path.parent.name
        decoded_path = decode_claude_path(encoded_name)

        if decoded_path:
            resolved = Path(decoded_path).resolve()
            if resolved.exists():
                try:
                    # Path exists - try to get config-based project_id
                    return get_project_id(resolved)
                except NotBootstrappedError:
                    log.debug(f"Session path {decoded_path} is not bootstrapped, skipping")
                    return None
            else:
                # Path doesn't exist locally - can't process
                log.debug(f"Decoded path doesn't exist locally: {decoded_path}")
                return None

        return None
    except Exception as e:
        log.warning(f"Failed to infer project from session path: {e}")
        return None
