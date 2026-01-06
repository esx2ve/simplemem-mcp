"""Tests for mandatory bootstrap functionality.

Tests cover:
1. SimpleMemConfig Pydantic model validation
2. NotBootstrappedError exception
3. find_project_root() path traversal
4. suggest_project_names() marker file parsing
5. create_bootstrap_config() atomic writes
6. get_project_id() strict config-only behavior
7. infer_project_from_session_path() bootstrap enforcement
8. MCP tools bootstrap enforcement
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
import yaml
from pydantic import ValidationError

from simplemem_mcp.projects_utils import (
    SimpleMemConfig,
    NotBootstrappedError,
    ProjectNameSuggestion,
    find_project_root,
    suggest_project_names,
    create_bootstrap_config,
    get_project_id,
    infer_project_from_session_path,
    normalize_git_url,
    decode_claude_path,
    CONFIG_SEARCH_MAX_DEPTH,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bootstrapped_project(temp_project_dir):
    """Create a bootstrapped project with .simplemem.yaml."""
    config_path = temp_project_dir / ".simplemem.yaml"
    config_path.write_text(
        """version: 1
project_id: "config:test-project"
project_name: "Test Project"
"""
    )
    return temp_project_dir


@pytest.fixture
def nested_project_dirs(temp_project_dir):
    """Create nested directory structure for testing path traversal."""
    # Create: root/parent/child/grandchild
    grandchild = temp_project_dir / "parent" / "child" / "grandchild"
    grandchild.mkdir(parents=True)

    # Put config at parent level
    config_path = temp_project_dir / "parent" / ".simplemem.yaml"
    config_path.write_text(
        """version: 1
project_id: "config:parent-project"
"""
    )

    return {
        "root": temp_project_dir,
        "parent": temp_project_dir / "parent",
        "child": temp_project_dir / "parent" / "child",
        "grandchild": grandchild,
    }


# =============================================================================
# SIMPLEMEM CONFIG MODEL TESTS
# =============================================================================


class TestSimpleMemConfig:
    """Tests for SimpleMemConfig Pydantic model."""

    def test_valid_config_minimal(self):
        """Minimal valid config with just project_id."""
        config = SimpleMemConfig(project_id="config:myproject")
        assert config.version == 1
        assert config.project_id == "config:myproject"
        assert config.project_name is None
        assert config.folder_role is None
        assert config.exclude_patterns == []

    def test_valid_config_full(self):
        """Full valid config with all fields."""
        config = SimpleMemConfig(
            version=1,
            project_id="config:company/project",
            project_name="My Project",
            folder_role="source",
            exclude_patterns=["*.test.py", "__pycache__/"],
        )
        assert config.project_id == "config:company/project"
        assert config.project_name == "My Project"
        assert config.folder_role == "source"
        assert config.exclude_patterns == ["*.test.py", "__pycache__/"]

    def test_project_id_requires_config_prefix(self):
        """project_id must start with 'config:' prefix."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="myproject")
        assert "must start with 'config:'" in str(exc_info.value)

    def test_project_id_rejects_git_prefix(self):
        """Reject legacy git: prefix."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="git:github.com/user/repo")
        assert "must start with 'config:'" in str(exc_info.value)

    def test_project_id_rejects_hash_prefix(self):
        """Reject legacy hash: prefix."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="hash:abc123")
        assert "must start with 'config:'" in str(exc_info.value)

    def test_project_id_rejects_path_prefix(self):
        """Reject legacy path: prefix."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="path:/users/project")
        assert "must start with 'config:'" in str(exc_info.value)

    def test_project_id_empty_slug_rejected(self):
        """project_id cannot have empty slug after prefix."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="config:")
        assert "cannot be empty" in str(exc_info.value)

    def test_project_id_whitespace_only_slug_rejected(self):
        """project_id cannot have whitespace-only slug."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="config:   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_folder_role_valid_values(self):
        """Valid folder_role values should be accepted."""
        for role in ["source", "tests", "docs", "config", "scripts"]:
            config = SimpleMemConfig(project_id="config:test", folder_role=role)
            assert config.folder_role == role

    def test_folder_role_invalid_value(self):
        """Invalid folder_role should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMemConfig(project_id="config:test", folder_role="invalid")
        assert "must be one of" in str(exc_info.value)

    def test_folder_role_none_allowed(self):
        """folder_role can be None."""
        config = SimpleMemConfig(project_id="config:test", folder_role=None)
        assert config.folder_role is None

    def test_config_with_special_characters_in_id(self):
        """project_id can contain slashes and dashes."""
        config = SimpleMemConfig(project_id="config:org/team-project/sub-module")
        assert config.project_id == "config:org/team-project/sub-module"

    def test_unicode_project_name(self):
        """project_name can contain unicode."""
        config = SimpleMemConfig(
            project_id="config:test",
            project_name="Проект 日本語 🚀",
        )
        assert config.project_name == "Проект 日本語 🚀"


# =============================================================================
# NOT BOOTSTRAPPED ERROR TESTS
# =============================================================================


class TestNotBootstrappedError:
    """Tests for NotBootstrappedError exception."""

    def test_basic_error(self):
        """Basic NotBootstrappedError construction."""
        error = NotBootstrappedError(
            message="Project not bootstrapped",
            cwd="/path/to/project",
        )
        assert error.message == "Project not bootstrapped"
        assert error.cwd == "/path/to/project"
        assert error.error_code == "SIMPLEMEM_NOT_BOOTSTRAPPED"
        assert error.searched_paths == []
        assert error.suggested_names == []

    def test_error_with_suggestions(self):
        """NotBootstrappedError with project name suggestions."""
        suggestions = [
            ProjectNameSuggestion(name="myproject", source="git_remote", confidence=95),
            ProjectNameSuggestion(name="myproject", source="package_json", confidence=80),
        ]
        error = NotBootstrappedError(
            message="Not bootstrapped",
            cwd="/repo/myproject",
            searched_paths=["/repo/myproject", "/repo"],
            suggested_names=suggestions,
        )
        assert len(error.suggested_names) == 2
        assert error.suggested_names[0].confidence == 95

    def test_to_dict_format(self):
        """to_dict() returns proper structure for MCP JSON response."""
        suggestions = [
            ProjectNameSuggestion(name="test", source="directory", confidence=50),
        ]
        error = NotBootstrappedError(
            message="Test error",
            cwd="/test/path",
            searched_paths=["/test/path", "/test"],
            suggested_names=suggestions,
        )

        result = error.to_dict()

        assert result["error"] == "SIMPLEMEM_NOT_BOOTSTRAPPED"
        assert result["message"] == "Test error"
        assert result["cwd"] == "/test/path"
        assert result["searched_paths"] == ["/test/path", "/test"]
        assert result["action_required"] == "bootstrap"
        assert "help" in result
        assert len(result["suggested_names"]) == 1
        assert result["suggested_names"][0] == {
            "name": "test",
            "source": "directory",
            "confidence": 50,
        }

    def test_exception_message(self):
        """Error message is accessible via str()."""
        error = NotBootstrappedError(
            message="Custom error message",
            cwd="/path",
        )
        assert "Custom error message" in str(error)

    def test_exception_is_catchable(self):
        """NotBootstrappedError is a proper Exception."""
        with pytest.raises(NotBootstrappedError) as exc_info:
            raise NotBootstrappedError(message="Test", cwd="/test")
        assert exc_info.value.cwd == "/test"

    def test_exception_is_catchable_as_generic_exception(self):
        """NotBootstrappedError can be caught as Exception."""
        with pytest.raises(Exception):
            raise NotBootstrappedError(message="Test", cwd="/test")


# =============================================================================
# FIND PROJECT ROOT TESTS
# =============================================================================


class TestFindProjectRoot:
    """Tests for find_project_root() function."""

    def test_finds_config_in_current_dir(self, bootstrapped_project):
        """Finds config in current directory."""
        root, config = find_project_root(bootstrapped_project)
        # Use resolve() on both sides to handle /var -> /private/var symlinks on macOS
        assert root.resolve() == bootstrapped_project.resolve()
        assert config.project_id == "config:test-project"

    def test_finds_config_in_parent_dir(self, nested_project_dirs):
        """Finds config in parent directory when starting from child."""
        root, config = find_project_root(nested_project_dirs["grandchild"])
        # Use resolve() on both sides to handle /var -> /private/var symlinks on macOS
        assert root.resolve() == nested_project_dirs["parent"].resolve()
        assert config.project_id == "config:parent-project"

    def test_raises_error_when_no_config(self, temp_project_dir):
        """Raises NotBootstrappedError when no config found."""
        with pytest.raises(NotBootstrappedError) as exc_info:
            find_project_root(temp_project_dir)

        error = exc_info.value
        assert str(temp_project_dir) in error.cwd
        assert len(error.searched_paths) > 0

    def test_includes_suggested_names_in_error(self, temp_project_dir):
        """NotBootstrappedError includes project name suggestions."""
        # Create a package.json for suggestions
        (temp_project_dir / "package.json").write_text('{"name": "test-pkg"}')

        with pytest.raises(NotBootstrappedError) as exc_info:
            find_project_root(temp_project_dir)

        error = exc_info.value
        names = [s.name for s in error.suggested_names]
        assert "test-pkg" in names

    def test_supports_yml_extension(self, temp_project_dir):
        """Supports .simplemem.yml extension."""
        config_path = temp_project_dir / ".simplemem.yml"
        config_path.write_text('version: 1\nproject_id: "config:yml-test"')

        root, config = find_project_root(temp_project_dir)
        assert config.project_id == "config:yml-test"

    def test_yaml_extension_takes_precedence(self, temp_project_dir):
        """If both .yaml and .yml exist, .yaml is checked first."""
        (temp_project_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:yaml-version"'
        )
        (temp_project_dir / ".simplemem.yml").write_text(
            'version: 1\nproject_id: "config:yml-version"'
        )

        root, config = find_project_root(temp_project_dir)
        assert config.project_id == "config:yaml-version"

    def test_stops_at_max_depth(self, temp_project_dir):
        """Stops searching after CONFIG_SEARCH_MAX_DEPTH levels."""
        # Create deeply nested directory (deeper than MAX_DEPTH)
        deep_path = temp_project_dir
        for i in range(CONFIG_SEARCH_MAX_DEPTH + 5):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)

        # Put config at root (unreachable from deep path)
        (temp_project_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:root"'
        )

        # Should fail because config is too far up
        with pytest.raises(NotBootstrappedError):
            find_project_root(deep_path)

    def test_handles_invalid_yaml(self, temp_project_dir):
        """Skips invalid YAML and continues searching."""
        # Create invalid YAML
        (temp_project_dir / ".simplemem.yaml").write_text("invalid: yaml: content: ][")

        with pytest.raises(NotBootstrappedError):
            find_project_root(temp_project_dir)

    def test_handles_yaml_with_invalid_config(self, temp_project_dir):
        """Skips YAML with invalid config schema."""
        # Valid YAML but invalid config (missing prefix)
        (temp_project_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "invalid-no-prefix"'
        )

        with pytest.raises(NotBootstrappedError):
            find_project_root(temp_project_dir)

    def test_handles_non_dict_yaml(self, temp_project_dir):
        """Skips YAML files that are not mappings."""
        (temp_project_dir / ".simplemem.yaml").write_text("- just\n- a\n- list")

        with pytest.raises(NotBootstrappedError):
            find_project_root(temp_project_dir)

    def test_nearest_config_wins(self, temp_project_dir):
        """When nested configs exist, nearest wins."""
        # Create nested structure with configs at multiple levels
        child = temp_project_dir / "child"
        child.mkdir()

        (temp_project_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:parent"'
        )
        (child / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:child"'
        )

        # Starting from child should find child config
        root, config = find_project_root(child)
        assert config.project_id == "config:child"

        # Starting from parent should find parent config
        root, config = find_project_root(temp_project_dir)
        assert config.project_id == "config:parent"

    def test_expands_user_home(self):
        """Expands ~ in paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config in temp dir
            config_path = Path(tmpdir) / ".simplemem.yaml"
            config_path.write_text('version: 1\nproject_id: "config:home-test"')

            # Mock expanduser to return our temp dir
            with patch.object(Path, "expanduser", return_value=Path(tmpdir)):
                with patch.object(Path, "resolve", return_value=Path(tmpdir)):
                    # This should work after expansion
                    root, config = find_project_root(tmpdir)
                    assert config.project_id == "config:home-test"

    def test_resolves_symlinks(self, temp_project_dir):
        """Resolves symlinks to canonical paths."""
        real_dir = temp_project_dir / "real"
        real_dir.mkdir()
        (real_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:real"'
        )

        link_dir = temp_project_dir / "link"
        link_dir.symlink_to(real_dir)

        root, config = find_project_root(link_dir)
        # Root should be resolved real path
        assert root == real_dir.resolve()
        assert config.project_id == "config:real"


# =============================================================================
# SUGGEST PROJECT NAMES TESTS
# =============================================================================


class TestSuggestProjectNames:
    """Tests for suggest_project_names() function."""

    def test_suggests_directory_name(self, temp_project_dir):
        """Suggests directory name as fallback."""
        suggestions = suggest_project_names(temp_project_dir)
        assert any(s.source == "directory" for s in suggestions)

    def test_parses_package_json(self, temp_project_dir):
        """Parses package.json for project name."""
        (temp_project_dir / "package.json").write_text(
            '{"name": "@org/my-package", "version": "1.0.0"}'
        )

        suggestions = suggest_project_names(temp_project_dir)
        pkg_suggestion = next(s for s in suggestions if s.source == "package_json")

        # Should strip org prefix
        assert pkg_suggestion.name == "my-package"
        assert pkg_suggestion.confidence == 80

    def test_parses_pyproject_toml(self, temp_project_dir):
        """Parses pyproject.toml for project name."""
        (temp_project_dir / "pyproject.toml").write_text(
            '[project]\nname = "my-python-pkg"'
        )

        suggestions = suggest_project_names(temp_project_dir)
        py_suggestion = next(s for s in suggestions if s.source == "pyproject")

        assert py_suggestion.name == "my-python-pkg"
        assert py_suggestion.confidence == 85

    def test_parses_poetry_pyproject(self, temp_project_dir):
        """Parses Poetry-style pyproject.toml."""
        (temp_project_dir / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "poetry-project"'
        )

        suggestions = suggest_project_names(temp_project_dir)
        py_suggestion = next(s for s in suggestions if s.source == "pyproject")

        assert py_suggestion.name == "poetry-project"

    def test_parses_cargo_toml(self, temp_project_dir):
        """Parses Cargo.toml for Rust project name."""
        (temp_project_dir / "Cargo.toml").write_text(
            '[package]\nname = "rust-project"'
        )

        suggestions = suggest_project_names(temp_project_dir)
        cargo_suggestion = next(s for s in suggestions if s.source == "cargo")

        assert cargo_suggestion.name == "rust-project"
        assert cargo_suggestion.confidence == 80

    def test_parses_go_mod(self, temp_project_dir):
        """Parses go.mod for Go module name."""
        (temp_project_dir / "go.mod").write_text(
            "module github.com/user/go-project\n\ngo 1.21"
        )

        suggestions = suggest_project_names(temp_project_dir)
        go_suggestion = next(s for s in suggestions if s.source == "go_mod")

        assert go_suggestion.name == "go-project"
        assert go_suggestion.confidence == 80

    def test_git_remote_highest_priority(self, temp_project_dir):
        """Git remote suggestion has highest confidence."""
        with patch(
            "simplemem_mcp.projects_utils.get_git_remote_url",
            return_value="github.com/user/git-repo",
        ):
            suggestions = suggest_project_names(temp_project_dir)

        git_suggestion = next(s for s in suggestions if s.source == "git_remote")
        assert git_suggestion.name == "git-repo"
        assert git_suggestion.confidence == 95

        # Should be first after sorting
        assert suggestions[0].source == "git_remote"

    def test_deduplicates_by_name(self, temp_project_dir):
        """Deduplicates suggestions with same name (keeps highest confidence)."""
        # Create both package.json and directory with same name
        dir_name = temp_project_dir.name
        (temp_project_dir / "package.json").write_text(f'{{"name": "{dir_name}"}}')

        suggestions = suggest_project_names(temp_project_dir)
        names = [s.name for s in suggestions]

        # Name should only appear once
        assert names.count(dir_name) == 1

    def test_handles_invalid_package_json(self, temp_project_dir):
        """Gracefully handles invalid package.json."""
        (temp_project_dir / "package.json").write_text("not valid json {")

        # Should not raise, just skip package.json
        suggestions = suggest_project_names(temp_project_dir)
        assert not any(s.source == "package_json" for s in suggestions)

    def test_handles_invalid_pyproject_toml(self, temp_project_dir):
        """Gracefully handles invalid pyproject.toml."""
        (temp_project_dir / "pyproject.toml").write_text("[[invalid toml")

        suggestions = suggest_project_names(temp_project_dir)
        assert not any(s.source == "pyproject" for s in suggestions)

    def test_handles_missing_name_in_markers(self, temp_project_dir):
        """Handles marker files without name field."""
        (temp_project_dir / "package.json").write_text('{"version": "1.0.0"}')

        suggestions = suggest_project_names(temp_project_dir)
        assert not any(s.source == "package_json" for s in suggestions)

    def test_sorted_by_confidence_descending(self, temp_project_dir):
        """Results sorted by confidence, highest first."""
        (temp_project_dir / "package.json").write_text('{"name": "pkg"}')
        (temp_project_dir / "pyproject.toml").write_text('[project]\nname = "pyproj"')

        with patch(
            "simplemem_mcp.projects_utils.get_git_remote_url",
            return_value="github.com/user/gitrepo",
        ):
            suggestions = suggest_project_names(temp_project_dir)

        confidences = [s.confidence for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)


# =============================================================================
# CREATE BOOTSTRAP CONFIG TESTS
# =============================================================================


class TestCreateBootstrapConfig:
    """Tests for create_bootstrap_config() function."""

    def test_creates_config_file(self, temp_project_dir):
        """Creates .simplemem.yaml config file."""
        config_path, config = create_bootstrap_config(
            temp_project_dir, project_name="Test Project"
        )

        assert config_path.exists()
        assert config_path.name == ".simplemem.yaml"
        assert config.project_id == "config:test-project"

    def test_generates_project_id_from_name(self, temp_project_dir):
        """Generates project_id from project_name."""
        _, config = create_bootstrap_config(
            temp_project_dir, project_name="My Cool Project!"
        )

        assert config.project_id == "config:my-cool-project"

    def test_uses_explicit_project_id(self, temp_project_dir):
        """Uses explicit project_id when provided."""
        _, config = create_bootstrap_config(
            temp_project_dir,
            project_name="Test",
            project_id="config:custom/id",
        )

        assert config.project_id == "config:custom/id"

    def test_refuses_overwrite_without_force(self, temp_project_dir):
        """Raises FileExistsError when config exists."""
        (temp_project_dir / ".simplemem.yaml").write_text("existing: content")

        with pytest.raises(FileExistsError):
            create_bootstrap_config(temp_project_dir, project_name="Test")

    def test_overwrites_with_force(self, temp_project_dir):
        """Overwrites existing config when force=True."""
        (temp_project_dir / ".simplemem.yaml").write_text("existing: content")

        config_path, config = create_bootstrap_config(
            temp_project_dir, project_name="New Project", force=True
        )

        assert config_path.exists()
        assert config.project_name == "New Project"

    def test_includes_folder_role(self, temp_project_dir):
        """Includes folder_role in config."""
        config_path, config = create_bootstrap_config(
            temp_project_dir, project_name="Test", folder_role="tests"
        )

        assert config.folder_role == "tests"

        # Verify file content
        content = config_path.read_text()
        assert 'folder_role: "tests"' in content

    def test_validates_project_id_format(self, temp_project_dir):
        """Raises ValueError for invalid project_id."""
        with pytest.raises(ValueError):
            create_bootstrap_config(
                temp_project_dir,
                project_name="Test",
                project_id="invalid-no-prefix",
            )

    def test_atomic_write(self, temp_project_dir):
        """Uses atomic write (temp + rename)."""
        # Mock os.replace to verify it's called
        original_replace = os.replace
        replace_called = []

        def mock_replace(src, dst):
            replace_called.append((src, dst))
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            create_bootstrap_config(temp_project_dir, project_name="Test")

        assert len(replace_called) == 1
        # Compare resolved paths to handle /var -> /private/var symlinks on macOS
        expected_path = (temp_project_dir / ".simplemem.yaml").resolve()
        actual_path = Path(replace_called[0][1]).resolve()
        assert actual_path == expected_path

    def test_cleanup_on_error(self, temp_project_dir):
        """Cleans up temp file on error."""
        # Make directory read-only to cause write failure
        with patch("builtins.open", side_effect=OSError("Write failed")):
            with pytest.raises(OSError):
                create_bootstrap_config(temp_project_dir, project_name="Test")

        # No temp files should remain
        temp_files = list(temp_project_dir.glob(".simplemem-*.yaml"))
        assert len(temp_files) == 0

    def test_yaml_content_format(self, temp_project_dir):
        """Generated YAML has correct format."""
        config_path, _ = create_bootstrap_config(
            temp_project_dir,
            project_name="Test Project",
            project_id="config:test",
            folder_role="source",
        )

        content = config_path.read_text()

        # Check structure
        assert "version: 1" in content
        assert 'project_id: "config:test"' in content
        assert 'project_name: "Test Project"' in content
        assert 'folder_role: "source"' in content

        # Verify it's valid YAML
        parsed = yaml.safe_load(content)
        assert parsed["version"] == 1
        assert parsed["project_id"] == "config:test"

    def test_sanitizes_special_characters(self, temp_project_dir):
        """Sanitizes special characters in auto-generated ID."""
        _, config = create_bootstrap_config(
            temp_project_dir, project_name="My $pecial Project! @#$"
        )

        # Should only have alphanumeric, dash, underscore, slash
        assert config.project_id == "config:my-pecial-project"

    def test_handles_unicode_project_name(self, temp_project_dir):
        """Handles unicode in project name with explicit project_id."""
        # Unicode project names work when project_id is provided explicitly
        # (auto-generation strips non-ascii, so provide explicit ID)
        config_path, config = create_bootstrap_config(
            temp_project_dir,
            project_name="Проект 日本語",
            project_id="config:unicode-project",
        )

        assert config.project_name == "Проект 日本語"
        content = config_path.read_text()
        assert "Проект 日本語" in content

    def test_handles_unicode_only_name_fallback(self, temp_project_dir):
        """Unicode-only project name requires explicit project_id."""
        # Auto-generated project_id from pure unicode fails - user must provide ID
        with pytest.raises(ValidationError):
            create_bootstrap_config(
                temp_project_dir, project_name="Проект 日本語"
            )


# =============================================================================
# GET PROJECT ID TESTS (STRICT MODE)
# =============================================================================


class TestGetProjectId:
    """Tests for get_project_id() strict config-only function."""

    def test_returns_config_project_id(self, bootstrapped_project):
        """Returns project_id from config."""
        project_id = get_project_id(bootstrapped_project)
        assert project_id == "config:test-project"

    def test_raises_not_bootstrapped_error(self, temp_project_dir):
        """Raises NotBootstrappedError when no config."""
        with pytest.raises(NotBootstrappedError):
            get_project_id(temp_project_dir)

    def test_walks_up_to_find_config(self, nested_project_dirs):
        """Walks up directory tree to find config."""
        project_id = get_project_id(nested_project_dirs["grandchild"])
        assert project_id == "config:parent-project"

    def test_no_fallback_to_git(self, temp_project_dir):
        """Does NOT fall back to git remote."""
        # Create .git directory (simulating git repo)
        git_dir = temp_project_dir / ".git"
        git_dir.mkdir()

        with patch(
            "simplemem_mcp.projects_utils.get_git_remote_url",
            return_value="github.com/user/repo",
        ):
            # Should still fail - no fallback to git
            with pytest.raises(NotBootstrappedError):
                get_project_id(temp_project_dir)

    def test_no_fallback_to_path(self, temp_project_dir):
        """Does NOT fall back to path-based ID."""
        # Should fail - no fallback to path
        with pytest.raises(NotBootstrappedError):
            get_project_id(temp_project_dir)


# =============================================================================
# INFER PROJECT FROM SESSION PATH TESTS
# =============================================================================


class TestInferProjectFromSessionPath:
    """Tests for infer_project_from_session_path() function."""

    def test_returns_none_for_non_existent_path(self):
        """Returns None when decoded path doesn't exist."""
        session_path = Path("/home/user/.claude/projects/-nonexistent-path/abc.jsonl")
        result = infer_project_from_session_path(session_path)
        assert result is None

    def test_returns_none_for_non_bootstrapped_project(self, temp_project_dir):
        """Returns None when project is not bootstrapped."""
        # Create a mock session path pointing to our temp dir
        encoded = temp_project_dir.as_posix().replace("/", "-")
        session_path = Path(f"/home/user/.claude/projects/{encoded}/session.jsonl")

        with patch.object(Path, "resolve", return_value=temp_project_dir):
            with patch.object(Path, "exists", return_value=True):
                result = infer_project_from_session_path(session_path)

        assert result is None

    def test_returns_project_id_for_bootstrapped_project(self, bootstrapped_project):
        """Returns project_id when project is bootstrapped."""
        # Encode the bootstrapped project path
        encoded = bootstrapped_project.as_posix().replace("/", "-")
        session_path = Path(f"/home/user/.claude/projects/{encoded}/session.jsonl")

        # Patch to return our real bootstrapped project
        with patch(
            "simplemem_mcp.projects_utils.decode_claude_path",
            return_value=str(bootstrapped_project),
        ):
            result = infer_project_from_session_path(session_path)

        assert result == "config:test-project"

    def test_handles_decode_failure(self):
        """Handles exceptions during path decoding."""
        session_path = Path("/invalid/path")
        result = infer_project_from_session_path(session_path)
        # Should return the path as-is (not a valid encoded path)
        # or None if it can't be processed
        assert result is None or isinstance(result, str)


# =============================================================================
# DECODE CLAUDE PATH TESTS
# =============================================================================


class TestDecodeClaudePath:
    """Tests for decode_claude_path() function."""

    def test_decodes_typical_path(self):
        """Decodes typical encoded path."""
        result = decode_claude_path("-Users-shimon-repo-project")
        assert result == "/Users/shimon/repo/project"

    def test_decodes_root(self):
        """Decodes root path."""
        assert decode_claude_path("-") == "/"

    def test_handles_empty_string(self):
        """Handles empty string."""
        assert decode_claude_path("") == "/"

    def test_handles_non_encoded_path(self):
        """Returns non-encoded paths as-is."""
        result = decode_claude_path("regular-string")
        assert result == "regular-string"


# =============================================================================
# NORMALIZE GIT URL TESTS
# =============================================================================


class TestNormalizeGitUrl:
    """Tests for normalize_git_url() function."""

    def test_normalizes_https_url(self):
        """Normalizes HTTPS URLs."""
        result = normalize_git_url("https://github.com/user/repo.git")
        assert result == "github.com/user/repo"

    def test_normalizes_ssh_url(self):
        """Normalizes SSH URLs."""
        result = normalize_git_url("git@github.com:user/repo.git")
        assert result == "github.com/user/repo"

    def test_normalizes_ssh_protocol_url(self):
        """Normalizes SSH protocol URLs."""
        result = normalize_git_url("ssh://git@bitbucket.org/team/repo.git")
        assert result == "bitbucket.org/team/repo"

    def test_handles_urls_without_git_suffix(self):
        """Handles URLs without .git suffix."""
        result = normalize_git_url("https://github.com/user/repo")
        assert result == "github.com/user/repo"

    def test_handles_empty_string(self):
        """Returns empty string for empty input."""
        assert normalize_git_url("") == ""

    def test_handles_url_with_port(self):
        """Handles URLs with custom port."""
        result = normalize_git_url("https://gitlab.company.com:8443/group/repo.git")
        assert result == "gitlab.company.com:8443/group/repo"


# =============================================================================
# MCP TOOLS BOOTSTRAP ENFORCEMENT TESTS
# =============================================================================


class TestMcpToolsBootstrapEnforcement:
    """Tests that MCP tools enforce bootstrap requirement."""

    @pytest.mark.asyncio
    async def test_store_memory_requires_bootstrap(self, temp_project_dir):
        """store_memory returns error dict when not bootstrapped."""
        import simplemem_mcp.server as server_module

        # Mock get_project_id to raise NotBootstrappedError
        error = NotBootstrappedError(
            message="Not bootstrapped",
            cwd=str(temp_project_dir),
        )

        with patch.object(
            server_module,
            "_resolve_project_id",
            side_effect=error,
        ):
            result = await server_module.store_memory(
                text="Test memory",
                type="fact",
            )

        assert result.get("error") == "SIMPLEMEM_NOT_BOOTSTRAPPED"
        assert result.get("action_required") == "bootstrap"

    @pytest.mark.asyncio
    async def test_search_memories_requires_bootstrap(self, temp_project_dir):
        """search_memories returns error when not bootstrapped."""
        import simplemem_mcp.server as server_module

        error = NotBootstrappedError(
            message="Not bootstrapped",
            cwd=str(temp_project_dir),
        )

        with patch.object(
            server_module,
            "_resolve_project_id",
            side_effect=error,
        ):
            result = await server_module.search_memories(query="test query")

        assert result.get("error") == "SIMPLEMEM_NOT_BOOTSTRAPPED"

    @pytest.mark.asyncio
    async def test_index_directory_requires_bootstrap(self, temp_project_dir):
        """index_directory returns error when not bootstrapped."""
        import simplemem_mcp.server as server_module

        error = NotBootstrappedError(
            message="Not bootstrapped",
            cwd=str(temp_project_dir),
        )

        with patch.object(
            server_module,
            "_resolve_project_id",
            side_effect=error,
        ):
            result = await server_module.index_directory(path=str(temp_project_dir))

        assert result.get("error") == "SIMPLEMEM_NOT_BOOTSTRAPPED"

    @pytest.mark.asyncio
    async def test_get_project_status_works_without_bootstrap(self, temp_project_dir):
        """get_project_status works without bootstrap (exempt tool)."""
        import simplemem_mcp.server as server_module

        result = await server_module.get_project_status(
            project_root=str(temp_project_dir)
        )

        # Should return status without error
        assert result.get("is_bootstrapped") is False
        assert "error" not in result or result.get("error") != "SIMPLEMEM_NOT_BOOTSTRAPPED"

    @pytest.mark.asyncio
    async def test_suggest_bootstrap_works_without_bootstrap(self, temp_project_dir):
        """suggest_bootstrap works without bootstrap (exempt tool)."""
        import simplemem_mcp.server as server_module

        result = await server_module.suggest_bootstrap(path=str(temp_project_dir))

        # Should return suggestions without error
        assert "error" not in result or result.get("error") != "SIMPLEMEM_NOT_BOOTSTRAPPED"
        assert "suggested_names" in result

    @pytest.mark.asyncio
    async def test_bootstrap_project_creates_config(self, temp_project_dir):
        """bootstrap_project creates config file."""
        import simplemem_mcp.server as server_module

        result = await server_module.bootstrap_project(
            project_name="Test Project",
            path=str(temp_project_dir),
        )

        assert result.get("success") is True
        assert (temp_project_dir / ".simplemem.yaml").exists()


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_permission_denied(self, temp_project_dir):
        """Handles permission denied gracefully."""
        # Create unreadable config
        config_path = temp_project_dir / ".simplemem.yaml"
        config_path.write_text('version: 1\nproject_id: "config:test"')

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(NotBootstrappedError):
                find_project_root(temp_project_dir)

    def test_handles_broken_symlink(self, temp_project_dir):
        """Handles broken symlinks gracefully."""
        broken_link = temp_project_dir / "broken"
        broken_link.symlink_to("/nonexistent/path")

        # Should not crash
        with pytest.raises(NotBootstrappedError):
            find_project_root(broken_link)

    def test_handles_circular_symlinks(self, temp_project_dir):
        """Handles circular symlinks gracefully."""
        link_a = temp_project_dir / "link_a"
        link_b = temp_project_dir / "link_b"

        # Create circular symlinks
        link_a.symlink_to(link_b)
        try:
            link_b.symlink_to(link_a)
        except (OSError, FileExistsError):
            # On some systems this fails immediately
            pass

        # Should not hang or crash - may raise RuntimeError, OSError, or NotBootstrappedError
        with pytest.raises((NotBootstrappedError, OSError, RuntimeError)):
            find_project_root(link_a)

    def test_handles_very_long_path(self, temp_project_dir):
        """Handles very long paths."""
        # Create deeply nested path
        long_path = temp_project_dir
        for i in range(50):
            long_path = long_path / f"dir_{i:04d}"

        try:
            long_path.mkdir(parents=True)
        except OSError:
            pytest.skip("Cannot create very deep directory")

        # Should not crash
        with pytest.raises(NotBootstrappedError):
            find_project_root(long_path)

    def test_handles_special_characters_in_path(self, temp_project_dir):
        """Handles special characters in path names."""
        special_dir = temp_project_dir / "project with spaces & symbols!"
        try:
            special_dir.mkdir()
        except OSError:
            pytest.skip("Cannot create directory with special characters")

        (special_dir / ".simplemem.yaml").write_text(
            'version: 1\nproject_id: "config:special"'
        )

        root, config = find_project_root(special_dir)
        assert config.project_id == "config:special"

    def test_config_with_extra_fields(self, temp_project_dir):
        """Config with extra fields should still work."""
        (temp_project_dir / ".simplemem.yaml").write_text(
            """version: 1
project_id: "config:test"
extra_field: "should be ignored"
another_field: 123
"""
        )

        root, config = find_project_root(temp_project_dir)
        assert config.project_id == "config:test"

    def test_empty_yaml_file(self, temp_project_dir):
        """Empty YAML file should fail validation."""
        (temp_project_dir / ".simplemem.yaml").write_text("")

        with pytest.raises(NotBootstrappedError):
            find_project_root(temp_project_dir)

    def test_yaml_with_only_comments(self, temp_project_dir):
        """YAML with only comments should fail."""
        (temp_project_dir / ".simplemem.yaml").write_text(
            "# Just a comment\n# Another comment"
        )

        with pytest.raises(NotBootstrappedError):
            find_project_root(temp_project_dir)
