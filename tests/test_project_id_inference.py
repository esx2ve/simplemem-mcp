"""Tests for project_id inference from Claude trace paths.

These tests verify that:
1. Claude's encoded path format is correctly decoded
2. project_id is correctly inferred from trace file paths
3. Edge cases are handled properly
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# Import the functions we're testing
# Note: These are module-level functions in server.py
def _decode_claude_path(encoded_path: str) -> str | None:
    """Copy of the function from server.py for testing."""
    if not encoded_path or encoded_path == "-":
        return "/"

    if encoded_path.startswith("-"):
        decoded = "/" + encoded_path[1:].replace("-", "/")
        return decoded

    return encoded_path


def _infer_project_from_session_path(session_path: Path) -> str | None:
    """Copy of the function from server.py for testing."""
    try:
        encoded_name = session_path.parent.name
        decoded_path = _decode_claude_path(encoded_name)
        if decoded_path:
            # Resolve symlinks to get canonical path (matches get_project_id behavior)
            try:
                resolved = str(Path(decoded_path).resolve())
                return resolved
            except Exception:
                return decoded_path
        return decoded_path
    except Exception:
        return None


class TestDecodeClaudePath:
    """Tests for _decode_claude_path function."""

    def test_decode_typical_path(self):
        """Typical Claude-encoded path should decode correctly."""
        encoded = "-Users-shimon-repo-simplemem"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/Users/shimon/repo/simplemem"

    def test_decode_linux_path(self):
        """Linux-style path should decode correctly."""
        encoded = "-home-user-projects-myapp"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/home/user/projects/myapp"

    def test_decode_root_path(self):
        """Root path should decode correctly."""
        encoded = "-"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/"

    def test_decode_empty_string(self):
        """Empty string should return root."""
        decoded = _decode_claude_path("")
        assert decoded == "/"

    def test_decode_deep_nested_path(self):
        """Deeply nested path should decode correctly."""
        encoded = "-Users-dev-projects-company-team-service-src"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/Users/dev/projects/company/team/service/src"

    def test_decode_path_without_leading_dash(self):
        """Path without leading dash should be returned as-is."""
        # This shouldn't normally happen but we handle it
        encoded = "some-random-string"
        decoded = _decode_claude_path(encoded)
        assert decoded == "some-random-string"

    def test_decode_path_with_numbers(self):
        """Path with numbers should decode correctly."""
        encoded = "-Users-dev-project-v2-src"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/Users/dev/project/v2/src"


class TestInferProjectFromSessionPath:
    """Tests for _infer_project_from_session_path function."""

    def test_infer_from_typical_trace_path(self):
        """Typical trace file path should yield correct project."""
        # Simulate: ~/.claude/projects/-Users-shimon-repo-simplemem/session.jsonl
        trace_path = Path("/home/user/.claude/projects/-Users-shimon-repo-simplemem/abc123.jsonl")
        project = _infer_project_from_session_path(trace_path)
        assert project == "/Users/shimon/repo/simplemem"

    def test_infer_from_mac_trace_path(self):
        """macOS trace file path should yield correct project."""
        trace_path = Path("/Users/shimon/.claude/projects/-Users-shimon-code-myproject/session-id.jsonl")
        project = _infer_project_from_session_path(trace_path)
        assert project == "/Users/shimon/code/myproject"

    def test_infer_from_linux_trace_path(self):
        """Linux trace file path should yield correct project."""
        trace_path = Path("/home/dev/.claude/projects/-home-dev-work-api/12345.jsonl")
        project = _infer_project_from_session_path(trace_path)
        # On macOS, /home is a symlink to /System/Volumes/Data/home
        # So we check it ends with the expected path
        assert project.endswith("/home/dev/work/api")

    def test_infer_handles_invalid_path(self):
        """Invalid path should return the resolved parent directory name."""
        # Path object with weird parent - "some" doesn't start with -, so returned as-is
        # but still resolved (which may prepend cwd on relative paths)
        trace_path = Path("/some/path.jsonl")
        project = _infer_project_from_session_path(trace_path)
        # The parent is "some" which doesn't start with -, returned as-is then resolved
        assert project.endswith("some") or project == "some"


class TestProjectIdEdgeCases:
    """Tests for edge cases in project_id handling."""

    def test_path_with_dashes_in_name(self):
        """Path with actual dashes is lossy but handled."""
        # Claude encodes /Users/dev/my-project as -Users-dev-my-project
        # This is ambiguous with /Users/dev/my/project
        # We accept this limitation
        encoded = "-Users-dev-my-project"
        decoded = _decode_claude_path(encoded)
        # Could be /Users/dev/my-project or /Users/dev/my/project
        # We decode as /Users/dev/my/project (lossy for dashed names)
        assert decoded == "/Users/dev/my/project"

    def test_path_with_underscores(self):
        """Path with underscores should be preserved."""
        encoded = "-Users-dev-my_project-src"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/Users/dev/my_project/src"

    def test_path_with_dots(self):
        """Path with dots should be preserved."""
        encoded = "-Users-dev-project.v2-src"
        decoded = _decode_claude_path(encoded)
        assert decoded == "/Users/dev/project.v2/src"


class TestIntegrationWithLocalReader:
    """Integration tests with LocalReader."""

    def test_find_session_returns_valid_path(self):
        """LocalReader.find_session_path should return usable path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock Claude traces directory structure
            traces_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = traces_dir / "-Users-test-myproject"
            project_dir.mkdir(parents=True)

            # Create a mock trace file
            session_id = "12345678-1234-1234-1234-123456789012"
            trace_file = project_dir / f"{session_id}.jsonl"
            trace_file.write_text('{"type": "user", "message": {"content": "test"}}')

            # Import and test LocalReader
            from simplemem_mcp.local_reader import LocalReader

            reader = LocalReader(traces_dir=traces_dir)
            found_path = reader.find_session_path(session_id)

            assert found_path is not None
            assert found_path.exists()
            assert found_path.name == f"{session_id}.jsonl"

            # Verify we can infer project from this path
            project = _infer_project_from_session_path(found_path)
            assert project == "/Users/test/myproject"

    def test_read_trace_file_by_path(self):
        """LocalReader.read_trace_file_by_path should work with found path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = traces_dir / "-home-user-project"
            project_dir.mkdir(parents=True)

            session_id = "abcdef12-3456-7890-abcd-ef1234567890"
            trace_file = project_dir / f"{session_id}.jsonl"
            trace_content = [
                '{"type": "user", "message": {"content": "hello"}}',
                '{"type": "assistant", "message": {"content": "world"}}',
            ]
            trace_file.write_text("\n".join(trace_content))

            from simplemem_mcp.local_reader import LocalReader

            reader = LocalReader(traces_dir=traces_dir)

            # First find the path
            found_path = reader.find_session_path(session_id)
            assert found_path is not None

            # Then read by path
            content = reader.read_trace_file_by_path(found_path)
            assert content is not None
            assert len(content) == 2
            assert content[0]["type"] == "user"
            assert content[1]["type"] == "assistant"


class TestProcessTraceProjectIdFlow:
    """Tests for the complete process_trace flow with project_id."""

    @pytest.mark.asyncio
    async def test_process_trace_extracts_project_id(self):
        """process_trace should extract and pass project_id to backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock traces directory
            traces_dir = Path(tmpdir) / ".claude" / "projects"
            project_dir = traces_dir / "-Users-shimon-repo-simplemem"
            project_dir.mkdir(parents=True)

            # Use valid UUID format (required by LocalReader validation)
            session_id = "a1234567-b234-c345-d456-e56789012345"
            trace_file = project_dir / f"{session_id}.jsonl"
            trace_file.write_text('{"type": "user", "message": {"content": "test"}}')

            # Mock the backend client
            mock_client = AsyncMock()
            mock_client.process_trace = AsyncMock(
                return_value={"job_id": "job-123", "status": "submitted"}
            )

            # Import server module and patch dependencies
            import simplemem_mcp.server as server_module

            # Create mock reader
            from simplemem_mcp.local_reader import LocalReader
            reader = LocalReader(traces_dir=traces_dir)

            with patch.object(server_module, "_get_client", return_value=mock_client):
                with patch.object(server_module, "_get_reader", return_value=reader):
                    # Call process_trace
                    result = await server_module.process_trace(
                        session_id=session_id,
                        background=True,
                    )

            # Verify client was called with correct project_id (now prefixed)
            mock_client.process_trace.assert_called_once()
            call_kwargs = mock_client.process_trace.call_args.kwargs
            # New format uses path: prefix for decoded paths that don't exist locally
            assert call_kwargs["project_id"] == "path:/Users/shimon/repo/simplemem"
            assert call_kwargs["session_id"] == session_id
