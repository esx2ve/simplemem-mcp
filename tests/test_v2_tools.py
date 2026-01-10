"""Tests for V2 Unified MCP tools.

Tests cover:
1. remember - Store memories
2. recall - Find memories (fast, deep, ask modes)
3. index_v2 - Index code files and traces
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_backend_client():
    """Mock backend client for testing."""
    client = MagicMock()
    client.remember = AsyncMock(return_value={"uuid": "test-uuid-123", "relations_created": 0})
    client.recall = AsyncMock(return_value={"results": []})
    client.index_v2 = AsyncMock(return_value={"job_id": "job-123", "status": "submitted"})
    return client


# =============================================================================
# REMEMBER TOOL TESTS
# =============================================================================


class TestRememberTool:
    """Tests for remember MCP tool."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, mock_backend_client):
        """Test basic memory storage."""
        from simplemem_mcp.server import remember

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await remember(
                    content="Test content",
                    project="test",
                    type="fact",
                )

                assert "uuid" in result
                mock_backend_client.remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_with_relations(self, mock_backend_client):
        """Test memory storage with relations."""
        mock_backend_client.remember = AsyncMock(return_value={"uuid": "test-uuid", "relations_created": 2})

        from simplemem_mcp.server import remember

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await remember(
                    content="Test decision",
                    project="test",
                    type="decision",
                    relations=["related-1", "related-2"],
                )

                assert result["relations_created"] == 2
                mock_backend_client.remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_not_bootstrapped(self):
        """Test remember returns error when not bootstrapped."""
        from simplemem_mcp.server import remember
        from simplemem_mcp.projects_utils import NotBootstrappedError

        with patch("simplemem_mcp.server._resolve_project_id") as mock_resolve:
            mock_resolve.side_effect = NotBootstrappedError(
                message="Project not bootstrapped",
                cwd="/test/path"
            )

            result = await remember(content="Test")

            assert "error" in result
            assert "SIMPLEMEM_NOT_BOOTSTRAPPED" in result.get("error", "")


# =============================================================================
# RECALL TOOL TESTS
# =============================================================================


class TestRecallTool:
    """Tests for recall MCP tool."""

    @pytest.mark.asyncio
    async def test_recall_requires_query_or_id(self):
        """Test that either query or id is required."""
        from simplemem_mcp.server import recall

        result = await recall()

        assert "error" in result
        assert "query" in result["error"].lower() or "id" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_recall_by_query(self, mock_backend_client):
        """Test recall by query."""
        mock_backend_client.recall = AsyncMock(return_value={
            "results": [{"uuid": "result-1", "content": "Test", "score": 0.9}]
        })

        from simplemem_mcp.server import recall

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await recall(query="test query", project="test")

                assert "results" in result
                mock_backend_client.recall.assert_called_once()
                call_args = mock_backend_client.recall.call_args
                assert call_args.kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_recall_by_id(self, mock_backend_client):
        """Test recall by exact ID."""
        mock_backend_client.recall = AsyncMock(return_value={
            "results": [{"uuid": "exact-uuid", "content": "Test"}]
        })

        from simplemem_mcp.server import recall

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await recall(id="exact-uuid", project="test")

                assert "results" in result
                call_args = mock_backend_client.recall.call_args
                assert call_args.kwargs["id"] == "exact-uuid"

    @pytest.mark.asyncio
    async def test_recall_ask_mode(self, mock_backend_client):
        """Test recall in ask mode."""
        mock_backend_client.recall = AsyncMock(return_value={
            "answer": "Synthesized answer",
            "confidence": "high",
            "sources": [],
        })

        from simplemem_mcp.server import recall

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await recall(
                    query="How did we fix the bug?",
                    project="test",
                    mode="ask",
                )

                assert "answer" in result
                call_args = mock_backend_client.recall.call_args
                assert call_args.kwargs["mode"] == "ask"


# =============================================================================
# INDEX_V2 TOOL TESTS
# =============================================================================


class TestIndexV2Tool:
    """Tests for index_v2 MCP tool."""

    @pytest.mark.asyncio
    async def test_index_requires_files_or_traces(self):
        """Test that either files or traces is required."""
        from simplemem_mcp.server import index_v2

        result = await index_v2(project="test")

        assert "error" in result
        assert "files" in result["error"].lower() or "traces" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_cannot_mix_files_and_traces(self):
        """Test that files and traces cannot be mixed."""
        from simplemem_mcp.server import index_v2

        result = await index_v2(
            project="test",
            files=[{"path": "test.py", "content": "print('hi')"}],
            traces=[{"session_id": "abc", "content": {}}],
        )

        assert "error" in result
        assert "Cannot index both" in result["error"]

    @pytest.mark.asyncio
    async def test_index_files(self, mock_backend_client):
        """Test code file indexing."""
        from simplemem_mcp.server import index_v2

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await index_v2(
                    project="test",
                    files=[{"path": "test.py", "content": "print('hello')"}],
                )

                assert result["status"] == "submitted"
                mock_backend_client.index_v2.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_traces(self, mock_backend_client):
        """Test trace indexing."""
        from simplemem_mcp.server import index_v2

        with patch("simplemem_mcp.server._get_client", return_value=mock_backend_client):
            with patch("simplemem_mcp.server._resolve_project_id", return_value="config:test"):
                result = await index_v2(
                    project="test",
                    traces=[{"session_id": "abc-123", "content": {"messages": []}}],
                )

                assert result["status"] == "submitted"
                call_args = mock_backend_client.index_v2.call_args
                assert call_args.kwargs["traces"] is not None

    @pytest.mark.asyncio
    async def test_index_requires_project(self):
        """Test that project is required for indexing."""
        from simplemem_mcp.server import index_v2
        from simplemem_mcp.projects_utils import NotBootstrappedError

        # When project is None and bootstrap fails
        with patch("simplemem_mcp.server._resolve_project_id") as mock_resolve:
            mock_resolve.side_effect = NotBootstrappedError(
                message="Project not bootstrapped",
                cwd="/test/path"
            )

            result = await index_v2(
                project=None,
                files=[{"path": "test.py", "content": "print('hi')"}],
            )

            # Should return not bootstrapped error
            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
