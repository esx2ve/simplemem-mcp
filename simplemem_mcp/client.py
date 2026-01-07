"""Backend API client for SimpleMem MCP.

Makes HTTP requests to the SimpleMem backend API (cloud or local),
handling compression, authentication, and error mapping.
"""

import logging
import os
import urllib.parse
from typing import Any

import httpx

from simplemem_mcp import DEFAULT_BACKEND_URL
from simplemem_mcp.compression import compress_if_large

log = logging.getLogger("simplemem_mcp.client")

# Local hosts that don't require HTTPS
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

COMPRESSION_THRESHOLD = 4096  # 4KB - compress payloads larger than this


class BackendError(Exception):
    """Error from backend API."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Backend error {status_code}: {detail}")


class BackendClient:
    """HTTP client for SimpleMem backend API.

    Handles:
    - Async HTTP requests to backend
    - Automatic compression of large payloads
    - API key authentication
    - Error handling and retries
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize the backend client.

        Args:
            base_url: Backend API URL (defaults to SIMPLEMEM_BACKEND_URL env or cloud)
            api_key: API key for authentication (defaults to SIMPLEMEM_API_KEY env)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is used with non-HTTPS remote backend

        Security:
            When an API key is provided, HTTPS is required for remote backends
            to prevent credential leakage. Local backends (localhost, 127.0.0.1)
            are exempt for development convenience.
        """
        self.base_url = (
            base_url
            or os.environ.get("SIMPLEMEM_BACKEND_URL")
            or DEFAULT_BACKEND_URL
        )
        self.api_key = api_key or os.environ.get("SIMPLEMEM_API_KEY")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

        # Enforce HTTPS for remote backends when using API key
        if self.api_key:
            self._validate_https_for_api_key()

    def _validate_https_for_api_key(self) -> None:
        """Validate that HTTPS is used when API key is present.

        Raises:
            ValueError: If using HTTP with API key on non-local backend
        """
        parsed = urllib.parse.urlparse(self.base_url)
        hostname = parsed.hostname or ""

        # Allow HTTP for local development
        is_local = hostname.lower() in LOCAL_HOSTS

        if not is_local and parsed.scheme != "https":
            raise ValueError(
                f"HTTPS required when using API key with remote backend. "
                f"Got: {self.base_url}. "
                f"Either use https:// URL, remove API key for insecure local dev, "
                f"or use localhost/127.0.0.1 for local development."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict | None = None,
        params: dict | None = None,
    ) -> dict | str:
        """Make an HTTP request to the backend.

        Returns:
            dict for JSON responses, str for plain text (TOON format)
        """
        client = await self._get_client()

        try:
            log.debug(f"HTTP {method} {self.base_url}{path}")
            response = await client.request(
                method=method,
                url=path,
                json=json_data,
                params=params,
            )
            log.debug(f"Response: {response.status_code}")

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", response.text)
                except Exception:
                    detail = response.text
                raise BackendError(response.status_code, detail)

            # Handle plain text responses (TOON format)
            content_type = response.headers.get("content-type", "")
            if "text/plain" in content_type:
                return response.text

            return response.json()

        except httpx.RequestError as e:
            log.error(f"Request to {path} failed: {e}")
            raise BackendError(0, str(e)) from e

    # ═══════════════════════════════════════════════════════════════════════════════
    # MEMORIES API
    # ═══════════════════════════════════════════════════════════════════════════════

    async def store_memory(
        self,
        text: str,
        type: str = "fact",
        source: str = "user",
        relations: list[dict] | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Store a memory via backend API."""
        data = {"text": text, "type": type, "source": source}
        if relations:
            data["relations"] = relations
        if project_id:
            data["project_id"] = project_id
        return await self._request("POST", "/api/v1/memories/store", json_data=data)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        use_graph: bool = True,
        type_filter: str | None = None,
        project_id: str | None = None,
        output_format: str | None = None,
    ) -> dict | str:
        """Search memories via backend API.

        Returns:
            dict with results for JSON format, str for TOON format
        """
        data = {"query": query, "limit": limit, "use_graph": use_graph}
        if type_filter:
            data["type_filter"] = type_filter
        if project_id:
            data["project_id"] = project_id
        if output_format:
            data["output_format"] = output_format
        return await self._request("POST", "/api/v1/memories/search", json_data=data)

    async def relate_memories(
        self, from_id: str, to_id: str, relation_type: str = "relates"
    ) -> dict:
        """Create a relationship between memories."""
        data = {"from_id": from_id, "to_id": to_id, "relation_type": relation_type}
        return await self._request("POST", "/api/v1/memories/relate", json_data=data)

    async def ask_memories(
        self,
        query: str,
        max_memories: int = 8,
        max_hops: int = 2,
        project_id: str | None = None,
    ) -> dict:
        """Ask a question with LLM-synthesized answer from memory graph."""
        data = {
            "query": query,
            "max_memories": max_memories,
            "max_hops": max_hops,
            "project_id": project_id,
        }
        return await self._request("POST", "/api/v1/memories/ask", json_data=data)

    async def reason_memories(
        self,
        query: str,
        max_hops: int = 2,
        min_score: float = 0.1,
        project_id: str | None = None,
    ) -> dict:
        """Multi-hop reasoning over memory graph."""
        data = {
            "query": query,
            "max_hops": max_hops,
            "min_score": min_score,
            "project_id": project_id,
        }
        return await self._request("POST", "/api/v1/memories/reason", json_data=data)

    async def get_stats(self) -> dict:
        """Get memory store statistics."""
        return await self._request("GET", "/api/v1/memories/stats")

    async def search_memories_deep(
        self,
        query: str,
        limit: int = 10,
        project_id: str | None = None,
        rerank_pool: int = 20,
    ) -> dict:
        """LLM-reranked semantic search with conflict detection."""
        data = {
            "query": query,
            "limit": limit,
            "rerank_pool": rerank_pool,
        }
        if project_id:
            data["project_id"] = project_id
        return await self._request("POST", "/api/v1/memories/search-deep", json_data=data)

    async def check_contradictions(
        self,
        content: str,
        memory_uuid: str | None = None,
        apply_supersession: bool = False,
        project_id: str | None = None,
    ) -> dict:
        """Check if content contradicts existing memories."""
        data = {
            "content": content,
            "apply_supersession": apply_supersession,
        }
        if memory_uuid:
            data["memory_uuid"] = memory_uuid
        if project_id:
            data["project_id"] = project_id
        return await self._request("POST", "/api/v1/memories/check-contradictions", json_data=data)

    async def get_sync_health(self, project_id: str | None = None) -> dict:
        """Check synchronization health between graph and vector stores."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/api/v1/memories/sync-health", params=params)

    async def repair_sync(
        self,
        project_id: str | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Repair synchronization issues between graph and vector stores."""
        data = {"dry_run": dry_run}
        if project_id:
            data["project_id"] = project_id
        return await self._request("POST", "/api/v1/memories/repair-sync", json_data=data)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TRACES API
    # ═══════════════════════════════════════════════════════════════════════════════

    async def process_trace(
        self,
        session_id: str,
        trace_content: list | dict | str,
        background: bool = True,
        project_id: str | None = None,
    ) -> dict:
        """Process a trace via backend API."""
        compressed, was_compressed = compress_if_large(
            trace_content, threshold_bytes=COMPRESSION_THRESHOLD
        )
        data = {
            "session_id": session_id,
            "trace_content": compressed,
            "compressed": was_compressed,
            "background": background,
        }
        if project_id:
            data["project_id"] = project_id
        return await self._request("POST", "/api/v1/traces/process", json_data=data)

    async def process_trace_batch(
        self,
        traces: list[dict],
        max_concurrent: int = 3,
    ) -> dict:
        """Process multiple traces via backend batch API.

        Args:
            traces: List of dicts with session_id, trace_content, compressed
            max_concurrent: Max concurrent processing (default: 3)

        Returns:
            {queued: [...], errors: [...], job_ids: {...}}
        """
        data = {"traces": traces, "max_concurrent": max_concurrent}
        return await self._request("POST", "/api/v1/traces/process-batch", json_data=data)

    async def get_job_status(self, job_id: str) -> dict:
        """Get status of a background processing job."""
        return await self._request("GET", f"/api/v1/traces/job/{job_id}")

    async def list_jobs(self, include_completed: bool = True, limit: int = 20) -> dict:
        """List background jobs."""
        params = {"include_completed": str(include_completed).lower(), "limit": limit}
        return await self._request("GET", "/api/v1/traces/jobs", params=params)

    async def cancel_job(self, job_id: str) -> dict:
        """Cancel a running background job."""
        return await self._request("POST", f"/api/v1/traces/job/{job_id}/cancel")

    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE API
    # ═══════════════════════════════════════════════════════════════════════════════

    async def index_code(
        self,
        project_id: str,
        files: list[dict],
        clear_existing: bool = True,
        background: bool = True,
    ) -> dict:
        """Index code files via backend API.

        Args:
            project_id: Project identifier for isolation
            files: List of {path, content} dicts
            clear_existing: Clear existing index for this project
            background: Run in background (default: True). Returns job_id immediately.

        Returns:
            If background=True: {job_id, status: "submitted", message: ...}
            If background=False: {files_indexed, chunks_created, ...}
        """
        compressed_files = []
        for file_info in files:
            content, was_compressed = compress_if_large(
                file_info["content"], threshold_bytes=COMPRESSION_THRESHOLD
            )
            compressed_files.append({
                "path": file_info["path"],
                "content": content,
                "compressed": was_compressed,
            })
        data = {
            "project_id": project_id,
            "files": compressed_files,
            "clear_existing": clear_existing,
            "background": background,
        }
        return await self._request("POST", "/api/v1/code/index", json_data=data)

    async def update_code(self, project_id: str, updates: list[dict]) -> dict:
        """Incrementally update code index via backend API."""
        compressed_updates = []
        for update in updates:
            action = update.get("action", "modify")
            content = update.get("content")
            if content and action in ("add", "modify"):
                content, was_compressed = compress_if_large(
                    content, threshold_bytes=COMPRESSION_THRESHOLD
                )
                compressed_updates.append({
                    "path": update["path"],
                    "action": action,
                    "content": content,
                    "compressed": was_compressed,
                })
            else:
                compressed_updates.append({
                    "path": update["path"],
                    "action": action,
                    "content": None,
                    "compressed": False,
                })
        data = {"project_id": project_id, "updates": compressed_updates}
        return await self._request("POST", "/api/v1/code/update", json_data=data)

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        project_id: str | None = None,
        output_format: str | None = None,
    ) -> dict | str:
        """Search code via backend API."""
        data = {"query": query, "limit": limit}
        if project_id:
            data["project_id"] = project_id
        if output_format:
            data["output_format"] = output_format
        return await self._request("POST", "/api/v1/code/search", json_data=data)

    async def code_stats(self, project_id: str | None = None) -> dict:
        """Get code index statistics."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/api/v1/code/stats", params=params)

    async def code_related_memories(
        self, chunk_uuid: str, limit: int = 10
    ) -> dict:
        """Find memories related to a code chunk via shared entities."""
        data = {"chunk_uuid": chunk_uuid, "limit": limit}
        return await self._request("POST", "/api/v1/code/related-memories", json_data=data)

    async def memory_related_code(
        self, memory_uuid: str, limit: int = 10
    ) -> dict:
        """Find code chunks related to a memory via shared entities."""
        data = {"memory_uuid": memory_uuid, "limit": limit}
        return await self._request("POST", "/api/v1/code/related-code", json_data=data)

    async def update_code_index_status(
        self,
        status: str | None = None,
        watchers: int | None = None,
        projects_watching: list[str] | None = None,
        indexing_in_progress: bool | None = None,
        files_done: int | None = None,
        files_total: int | None = None,
        current_file: str | None = None,
        total_files: int | None = None,
        total_chunks: int | None = None,
    ) -> dict:
        """Update code index status on the backend for statusline display.

        Args:
            status: Overall status (idle, indexing, watching)
            watchers: Number of active file watchers
            projects_watching: List of project roots being watched
            indexing_in_progress: Whether indexing is currently running
            files_done: Files indexed so far
            files_total: Total files to index
            current_file: Currently indexing file path
            total_files: Total indexed files (stats)
            total_chunks: Total code chunks (stats)

        Returns:
            Status update confirmation
        """
        data: dict[str, Any] = {}
        if status is not None:
            data["status"] = status
        if watchers is not None:
            data["watchers"] = watchers
        if projects_watching is not None:
            data["projects_watching"] = projects_watching
        if indexing_in_progress is not None:
            data["indexing_in_progress"] = indexing_in_progress
        if files_done is not None:
            data["files_done"] = files_done
        if files_total is not None:
            data["files_total"] = files_total
        if current_file is not None:
            data["current_file"] = current_file
        if total_files is not None:
            data["total_files"] = total_files
        if total_chunks is not None:
            data["total_chunks"] = total_chunks

        return await self._request("POST", "/api/v1/code/status", json_data=data)

    # ═══════════════════════════════════════════════════════════════════════════════
    # GRAPH API
    # ═══════════════════════════════════════════════════════════════════════════════

    async def get_graph_schema(self) -> dict:
        """Get the graph schema."""
        return await self._request("GET", "/api/v1/graph/schema")

    async def run_cypher_query(
        self, query: str, params: dict | None = None, max_results: int = 100
    ) -> dict:
        """Execute a Cypher query against the graph.

        In PROD mode, if query matches a template name (no spaces, alphanumeric + underscore),
        it's sent as query_name. Otherwise sent as raw query (requires DEV mode).
        """
        data: dict = {"max_results": max_results}

        # Check if query looks like a template name (no spaces, simple identifier)
        # Template names: get_memory, all_goals, session_goals, etc.
        import re
        if re.match(r'^[a-z][a-z0-9_]*$', query):
            # Looks like a template name
            data["query_name"] = query
        else:
            # Raw Cypher query (only works in DEV mode)
            data["query"] = query

        if params:
            data["params"] = params
        return await self._request("POST", "/api/v1/graph/query", json_data=data)

    # ═══════════════════════════════════════════════════════════════════════════════
    # PROJECTS API
    # ═══════════════════════════════════════════════════════════════════════════════

    async def get_project_status(self, project_root: str) -> dict:
        """Get bootstrap and metadata status for a project.

        Args:
            project_root: Absolute path to the project root directory

        Returns:
            Dict with is_known, is_bootstrapped, should_ask, project_name, etc.
        """
        params = {"project_root": project_root}
        return await self._request("GET", "/api/v1/projects/status", params=params)

    async def set_bootstrap_status(
        self,
        project_root: str,
        is_bootstrapped: bool = True,
        never_ask: bool = False,
    ) -> dict:
        """Set the bootstrap status for a project.

        Args:
            project_root: Absolute path to the project root directory
            is_bootstrapped: Whether the project has been bootstrapped
            never_ask: If True, never prompt for bootstrap again

        Returns:
            Updated project state
        """
        data = {
            "project_root": project_root,
            "is_bootstrapped": is_bootstrapped,
            "never_ask": never_ask,
        }
        return await self._request("POST", "/api/v1/projects/bootstrap", json_data=data)

    async def list_projects(self) -> dict:
        """List all tracked projects.

        Returns:
            Dict with projects list and count
        """
        return await self._request("GET", "/api/v1/projects/list")

    # ═══════════════════════════════════════════════════════════════════════════════
    # HEALTH CHECK
    # ═══════════════════════════════════════════════════════════════════════════════

    async def health_check(self) -> dict:
        """Check backend health."""
        return await self._request("GET", "/health")

    async def is_healthy(self) -> bool:
        """Check if backend is healthy and reachable."""
        try:
            result = await self.health_check()
            return result.get("status") == "healthy"
        except Exception:
            return False
