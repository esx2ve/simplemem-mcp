"""Tests for project registry functionality.

The registry handles Claude's lossy path encoding where dots in paths
(e.g., shimon.vainer) become indistinguishable from path separators.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from simplemem_mcp import projects_utils
from simplemem_mcp.projects_utils import (
    REGISTRY_PATH,
    encode_path_for_claude,
    load_project_registry,
    register_project,
    save_project_registry,
)


class TestEncodePathForClaude:
    """Tests for encode_path_for_claude function."""

    def test_encode_simple_path(self):
        """Simple path encodes correctly."""
        assert encode_path_for_claude("/Users/test/repo") == "-Users-test-repo"

    def test_encode_path_with_dots(self):
        """Path with dots preserves dots (they become ambiguous)."""
        result = encode_path_for_claude("/Users/shimon.vainer/repo")
        assert result == "-Users-shimon.vainer-repo"

    def test_encode_deep_path(self):
        """Deep nested path encodes all separators."""
        result = encode_path_for_claude("/Users/dev/projects/company/team/service/src")
        assert result == "-Users-dev-projects-company-team-service-src"

    def test_encode_root_path(self):
        """Root path encodes to single dash."""
        assert encode_path_for_claude("/") == "-"

    def test_encode_path_with_dashes(self):
        """Path with dashes preserves them (lossy encoding)."""
        result = encode_path_for_claude("/Users/dev/my-project")
        assert result == "-Users-dev-my-project"


class TestLoadProjectRegistry:
    """Tests for load_project_registry function."""

    def test_load_empty_registry_when_not_exists(self, tmp_path, monkeypatch):
        """Returns empty dict when registry file doesn't exist."""
        reg_path = tmp_path / "nonexistent" / "registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        result = load_project_registry()
        assert result == {}

    def test_load_existing_registry(self, tmp_path, monkeypatch):
        """Loads registry from existing file."""
        reg_path = tmp_path / "project_registry.json"
        data = {
            "-Users-test-repo": {
                "canonical_path": "/Users/test/repo",
                "project_id": "config:myproject",
            }
        }
        reg_path.write_text(json.dumps(data))
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        result = load_project_registry()
        assert result == data

    def test_load_corrupted_registry_returns_empty(self, tmp_path, monkeypatch):
        """Returns empty dict when registry is corrupted JSON."""
        reg_path = tmp_path / "project_registry.json"
        reg_path.write_text("not valid json{{{")
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        result = load_project_registry()
        assert result == {}

    def test_load_empty_file_returns_empty(self, tmp_path, monkeypatch):
        """Returns empty dict when file is empty."""
        reg_path = tmp_path / "project_registry.json"
        reg_path.write_text("")
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        result = load_project_registry()
        assert result == {}


class TestSaveProjectRegistry:
    """Tests for save_project_registry function."""

    def test_save_creates_parent_directories(self, tmp_path, monkeypatch):
        """Creates parent directories if they don't exist."""
        reg_path = tmp_path / ".simplemem" / "nested" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        data = {"key": {"value": "test"}}
        save_project_registry(data)

        assert reg_path.exists()
        assert json.loads(reg_path.read_text()) == data

    def test_save_overwrites_existing(self, tmp_path, monkeypatch):
        """Overwrites existing registry file."""
        reg_path = tmp_path / "project_registry.json"
        reg_path.write_text('{"old": "data"}')
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        new_data = {"new": {"data": "here"}}
        save_project_registry(new_data)

        assert json.loads(reg_path.read_text()) == new_data

    def test_save_atomic_no_temp_file_left(self, tmp_path, monkeypatch):
        """Atomic save doesn't leave temp file behind."""
        reg_path = tmp_path / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        save_project_registry({"test": {"data": "value"}})

        # Check no .tmp file remains
        tmp_file = reg_path.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert reg_path.exists()


class TestRegisterProject:
    """Tests for register_project function."""

    def test_register_new_project(self, tmp_path, monkeypatch):
        """Registers a new project in empty registry."""
        reg_path = tmp_path / ".simplemem" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        register_project(project_dir, "config:myproject")

        registry = load_project_registry()
        encoded = encode_path_for_claude(str(project_dir.resolve()))
        assert encoded in registry
        assert registry[encoded]["project_id"] == "config:myproject"
        assert registry[encoded]["canonical_path"] == str(project_dir.resolve())

    def test_register_project_with_dotted_path(self, tmp_path, monkeypatch):
        """Registers project with dots in path (the critical case)."""
        reg_path = tmp_path / ".simplemem" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        # Simulate a path with dots like shimon.vainer
        project_dir = tmp_path / "user.name" / "repo"
        project_dir.mkdir(parents=True)

        register_project(project_dir, "config:testrepo")

        registry = load_project_registry()
        encoded = encode_path_for_claude(str(project_dir.resolve()))
        assert encoded in registry
        assert registry[encoded]["project_id"] == "config:testrepo"

    def test_register_updates_existing_entry(self, tmp_path, monkeypatch):
        """Re-registering updates the existing entry."""
        reg_path = tmp_path / ".simplemem" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        # Register with first project_id
        register_project(project_dir, "config:old-id")
        # Re-register with new project_id
        register_project(project_dir, "config:new-id")

        registry = load_project_registry()
        encoded = encode_path_for_claude(str(project_dir.resolve()))
        # Should have the new ID
        assert registry[encoded]["project_id"] == "config:new-id"

    def test_register_multiple_projects(self, tmp_path, monkeypatch):
        """Can register multiple projects."""
        reg_path = tmp_path / ".simplemem" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        register_project(project1, "config:proj1")
        register_project(project2, "config:proj2")

        registry = load_project_registry()
        assert len(registry) == 2
        assert registry[encode_path_for_claude(str(project1.resolve()))]["project_id"] == "config:proj1"
        assert registry[encode_path_for_claude(str(project2.resolve()))]["project_id"] == "config:proj2"

    def test_register_resolves_symlinks(self, tmp_path, monkeypatch):
        """Register resolves symlinks to canonical path."""
        reg_path = tmp_path / ".simplemem" / "project_registry.json"
        monkeypatch.setattr(projects_utils, "REGISTRY_PATH", reg_path)

        # Create actual dir and symlink
        actual_dir = tmp_path / "actual"
        actual_dir.mkdir()
        symlink = tmp_path / "symlink"
        symlink.symlink_to(actual_dir)

        register_project(symlink, "config:myproject")

        registry = load_project_registry()
        # Should be registered under the resolved (actual) path
        encoded = encode_path_for_claude(str(actual_dir.resolve()))
        assert encoded in registry
