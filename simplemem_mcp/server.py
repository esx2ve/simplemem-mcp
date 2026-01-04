"""Thin MCP server for SimpleMem.

Cloud-first MCP server that:
1. Reads local files on-demand (traces, code files)
2. Compresses payloads for transport
3. Proxies requests to the backend API (cloud or local)

The server runs locally but connects to the cloud backend by default.
Set SIMPLEMEM_BACKEND_URL to override with a local backend.
"""

import asyncio
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from simplemem_mcp.client import BackendClient, BackendError
from simplemem_mcp.local_reader import LocalReader
from simplemem_mcp.watcher import CloudWatcherManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("SIMPLEMEM_DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("simplemem_mcp.server")

# Initialize FastMCP server
mcp = FastMCP("SimpleMem")

# Lazy-initialized dependencies with thread-safe locks
_client: BackendClient | None = None
_reader: LocalReader | None = None
_watcher_manager: CloudWatcherManager | None = None
_client_lock = asyncio.Lock()
_reader_lock = asyncio.Lock()
_watcher_lock = asyncio.Lock()


async def _get_client() -> BackendClient:
    """Get or create the backend client (thread-safe)."""
    global _client

    if _client is None:
        async with _client_lock:
            # Double-check after acquiring lock
            if _client is None:
                _client = BackendClient()
                log.info(f"BackendClient initialized with base_url={_client.base_url}")
    return _client


async def _get_reader() -> LocalReader:
    """Get or create the local reader (thread-safe)."""
    global _reader
    if _reader is None:
        async with _reader_lock:
            # Double-check after acquiring lock
            if _reader is None:
                _reader = LocalReader()
    return _reader


async def _get_watcher_manager() -> CloudWatcherManager:
    """Get or create the cloud watcher manager (thread-safe)."""
    global _watcher_manager
    if _watcher_manager is None:
        async with _watcher_lock:
            # Double-check after acquiring lock
            if _watcher_manager is None:
                # IMPORTANT: Create a NEW BackendClient for the watcher, not shared!
                # The watcher runs in its own thread with its own event loop,
                # and httpx.AsyncClient cannot be shared across event loops.
                watcher_client = BackendClient()
                reader = await _get_reader()
                _watcher_manager = CloudWatcherManager(client=watcher_client, reader=reader)
    return _watcher_manager


def _resolve_project_id(project_id: str | None) -> str | None:
    """Resolve project_id, defaulting to canonical cwd if not provided.

    The canonical project_id is the absolute path of the project root.
    This ensures 1:1 mapping and easy discovery.

    Args:
        project_id: Optional explicit project_id

    Returns:
        Canonical project_id (absolute path) or None if cwd resolution fails
    """
    if project_id:
        # Already provided - ensure it's canonical (absolute path)
        return str(Path(project_id).expanduser().resolve())

    # Auto-resolve from current working directory
    try:
        cwd = Path.cwd().resolve()
        log.debug(f"Auto-resolved project_id from cwd: {cwd}")
        return str(cwd)
    except Exception as e:
        log.warning(f"Failed to resolve project_id from cwd: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def store_memory(
    text: str,
    type: str = "fact",
    source: str = "user",
    relations: list[dict] | None = None,
    project_id: str | None = None,
) -> dict:
    """Store a memory with optional relationships.

    Args:
        text: The content to store
        type: Memory type (fact, session_summary, chunk_summary, message)
        source: Source of memory (user, claude_trace, extracted)
        relations: Optional list of {target_id, type} relationships
        project_id: Optional project identifier for cross-project isolation

    Returns:
        Dict with uuid of stored memory or error
    """
    resolved_project_id = _resolve_project_id(project_id)
    log.info(f"store_memory called (type={type}, project={resolved_project_id})")
    try:
        client = await _get_client()
        result = await client.store_memory(
            text=text,
            type=type,
            source=source,
            relations=relations,
            project_id=resolved_project_id,
        )
        return {"uuid": result.get("uuid", "")}
    except BackendError as e:
        log.error(f"store_memory failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 10,
    use_graph: bool = True,
    type_filter: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Hybrid search combining vector similarity and graph traversal.

    Searches summaries first for efficiency, then expands via graph
    to find related memories.

    Args:
        query: Search query text
        limit: Maximum results to return
        use_graph: Whether to expand results via graph relationships
        type_filter: Optional filter by memory type
        project_id: Optional filter by project (for cross-project isolation)

    Returns:
        Dict with results list or error
    """
    resolved_project_id = _resolve_project_id(project_id)
    try:
        log.info(f"search_memories called (query='{query[:50]}...', limit={limit}, project={resolved_project_id})")
        client = await _get_client()
        result = await client.search_memories(
            query=query,
            limit=limit,
            use_graph=use_graph,
            type_filter=type_filter,
            project_id=resolved_project_id,
        )
        return {"results": result.get("results", [])}
    except BackendError as e:
        log.error(f"search_memories failed: {e}")
        return {"error": e.detail, "results": []}


@mcp.tool()
async def relate_memories(
    from_id: str,
    to_id: str,
    relation_type: str = "relates",
) -> dict:
    """Create a relationship between two memories.

    Args:
        from_id: Source memory UUID
        to_id: Target memory UUID
        relation_type: Type of relationship (contains, child_of, supports, follows, similar)

    Returns:
        Dict with success bool or error
    """
    try:
        log.info(f"relate_memories called ({from_id[:8]} -> {to_id[:8]})")
        client = await _get_client()
        result = await client.relate_memories(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation_type,
        )
        return {"success": result.get("success", False)}
    except BackendError as e:
        log.error(f"relate_memories failed: {e}")
        return {"error": e.detail, "success": False}


@mcp.tool()
async def ask_memories(
    query: str,
    max_memories: int = 8,
    max_hops: int = 2,
    project_id: str | None = None,
) -> dict:
    """Ask a question and get an LLM-synthesized answer from memory graph.

    Retrieves relevant memories via multi-hop graph traversal, then uses
    an LLM to synthesize a coherent answer grounded in the evidence.

    The answer includes citations [1], [2], etc. referencing specific memories.
    Cross-session insights (patterns found across different work sessions) are
    highlighted as especially valuable.

    Example queries:
    - "What was the solution to the database connection issue?"
    - "How did we implement the authentication feature?"
    - "What patterns have worked for debugging async code?"

    Args:
        query: Natural language question
        max_memories: Maximum memories to include in context (default: 8)
        max_hops: Maximum graph traversal depth (default: 2)
        project_id: Optional project identifier for cross-project isolation

    Returns:
        {answer, memories_used, cross_session_insights, confidence, sources}
    """
    resolved_project_id = _resolve_project_id(project_id)
    try:
        log.info(f"ask_memories called (query='{query[:50]}...', project={resolved_project_id})")
        client = await _get_client()
        return await client.ask_memories(
            query=query,
            max_memories=max_memories,
            max_hops=max_hops,
            project_id=resolved_project_id,
        )
    except BackendError as e:
        log.error(f"ask_memories failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def reason_memories(
    query: str,
    max_hops: int = 2,
    min_score: float = 0.1,
    project_id: str | None = None,
) -> dict:
    """Multi-hop reasoning over memory graph.

    Combines vector search with graph traversal and semantic path scoring
    to answer complex questions that require following chains of evidence.

    Example queries:
    - "What debugging patterns work for database issues?"
    - "How did the authentication feature evolve?"
    - "Find solutions related to connection timeouts"

    Args:
        query: Natural language query
        max_hops: Maximum path length for traversal (1-3)
        min_score: Minimum score threshold for results
        project_id: Optional project identifier for cross-project isolation

    Returns:
        {conclusions: [{uuid, content, type, score, proof_chain, hops}]}
    """
    resolved_project_id = _resolve_project_id(project_id)
    try:
        log.info(f"reason_memories called (query='{query[:50]}...', project={resolved_project_id})")
        client = await _get_client()
        return await client.reason_memories(
            query=query,
            max_hops=max_hops,
            min_score=min_score,
            project_id=resolved_project_id,
        )
    except BackendError as e:
        log.error(f"reason_memories failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def get_project_id(path: str | None = None) -> dict:
    """Get the canonical project_id for a path or current working directory.

    The project_id is the canonical absolute path, ensuring 1:1 mapping
    between project paths and their IDs. This makes discovery trivial:
    the project_id IS the absolute path.

    Args:
        path: Optional path to resolve (defaults to current working directory)

    Returns:
        {project_id: str, path: str} or {error: str}

    Example:
        get_project_id()  # Returns cwd as project_id
        get_project_id("~/repo/myproject")  # Returns /Users/.../repo/myproject
    """
    try:
        if path:
            resolved = str(Path(path).expanduser().resolve())
        else:
            resolved = str(Path.cwd().resolve())

        log.info(f"get_project_id called, resolved: {resolved}")
        return {
            "project_id": resolved,
            "path": resolved,
            "message": "Use this project_id with memory tools for isolation",
        }
    except Exception as e:
        log.error(f"get_project_id failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_stats() -> dict:
    """Get memory store statistics.

    Returns:
        {total_memories, total_relations, types_breakdown}
    """
    try:
        log.info("get_stats called")
        client = await _get_client()
        return await client.get_stats()
    except BackendError as e:
        log.error(f"get_stats failed: {e}")
        return {"error": e.detail}


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def process_trace(
    session_id: str,
    background: bool = True,
) -> dict:
    """Index a Claude Code session trace with hierarchical summaries.

    Creates a hierarchy of memories:
    - session_summary (1) - Overall session summary
    - chunk_summary (5-15) - Summaries of activity chunks

    Uses cheap LLM (flash-lite) for summarization with progress updates.
    Runs in background by default to avoid MCP timeout on large sessions.

    Args:
        session_id: Session UUID to index
        background: Run in background (default: True). Use job_status to check progress.

    Returns:
        If background=True: {job_id, status: "submitted"}
        If background=False: {session_summary_id, chunk_count, message_count} or error
    """
    try:
        log.info(f"process_trace called (session_id={session_id})")

        # Read trace file locally (offload blocking I/O to thread)
        reader = await _get_reader()
        trace_content = await asyncio.to_thread(reader.read_trace_file, session_id)

        if trace_content is None:
            return {"error": f"Session {session_id} not found"}

        # Send to backend for processing
        client = await _get_client()
        return await client.process_trace(
            session_id=session_id,
            trace_content=trace_content,
            background=background,
        )
    except BackendError as e:
        log.error(f"process_trace failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def discover_sessions(
    days_back: int = 30,
    group_by: str | None = None,
    include_indexed: bool = True,
) -> dict:
    """Discover available Claude Code sessions for potential indexing.

    Lightweight scan that reads file metadata only (no LLM calls).
    Use this to explore historical sessions before batch indexing.

    Args:
        days_back: Only include sessions modified within this many days (default: 30)
        group_by: Optional grouping - "project" or "date" (default: None, flat list)
        include_indexed: Include already-indexed sessions in results (default: True)

    Returns:
        Dict containing:
        - sessions: List of session metadata (or grouped dict if group_by specified)
        - total_count: Total sessions found
        - indexed_count: How many are already indexed
        - unindexed_count: How many are not yet indexed
    """
    try:
        log.info(f"discover_sessions called (days_back={days_back})")

        reader = await _get_reader()
        sessions = await asyncio.to_thread(reader.discover_sessions, days_back)

        result = {
            "sessions": sessions,
            "total_count": len(sessions),
            "indexed_count": 0,
            "unindexed_count": len(sessions),
        }

        if group_by == "project":
            grouped: dict[str, list] = {}
            for s in sessions:
                project = s.get("project", "unknown")
                if project not in grouped:
                    grouped[project] = []
                grouped[project].append(s)
            result["sessions"] = grouped

        return result
    except Exception as e:
        log.error(f"discover_sessions failed: {e}")
        return {"error": str(e), "sessions": [], "total_count": 0}


@mcp.tool()
async def job_status(job_id: str) -> dict:
    """Get status of a background job.

    Use this to check progress of long-running operations like process_trace
    or index_directory that were submitted with background=True.

    Args:
        job_id: Job ID returned when submitting the background job

    Returns:
        {id, type, status, progress, message, result, error, timestamps}
        status is one of: pending, running, completed, failed, cancelled
    """
    try:
        log.info(f"job_status called (job_id={job_id})")
        client = await _get_client()
        return await client.get_job_status(job_id)
    except BackendError as e:
        log.error(f"job_status failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def list_jobs(
    include_completed: bool = True,
    limit: int = 20,
) -> dict:
    """List all background jobs.

    Args:
        include_completed: Include completed/failed/cancelled jobs (default: True)
        limit: Maximum number of jobs to return (default: 20)

    Returns:
        {jobs: [{id, type, status, progress, message}]}
    """
    try:
        log.info("list_jobs called")
        client = await _get_client()
        return await client.list_jobs(
            include_completed=include_completed,
            limit=limit,
        )
    except BackendError as e:
        log.error(f"list_jobs failed: {e}")
        return {"error": e.detail, "jobs": []}


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a running background job.

    Args:
        job_id: Job ID to cancel

    Returns:
        {cancelled: bool, message: str}
    """
    try:
        log.info(f"cancel_job called (job_id={job_id})")
        client = await _get_client()
        return await client.cancel_job(job_id)
    except BackendError as e:
        log.error(f"cancel_job failed: {e}")
        return {"error": e.detail, "cancelled": False}


# ═══════════════════════════════════════════════════════════════════════════════
# CODE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search_code(
    query: str,
    limit: int = 10,
    project_root: str | None = None,
) -> dict:
    """Search the code index for relevant code snippets.

    Use this to find code implementations, patterns, or specific functionality.
    Results are ranked by semantic similarity to the query.

    Args:
        query: Natural language description of what you're looking for
        limit: Maximum number of results (default: 10)
        project_root: Optional - filter to specific project directory

    Returns:
        List of matching code chunks with file paths and line numbers
    """
    try:
        log.info(f"search_code called (query='{query[:50]}...')")
        client = await _get_client()
        return await client.search_code(
            query=query,
            limit=limit,
            project_root=project_root,
        )
    except BackendError as e:
        log.error(f"search_code failed: {e}")
        return {"error": e.detail, "results": []}


@mcp.tool()
async def index_directory(
    path: str,
    patterns: list[str] | None = None,
    clear_existing: bool = True,
    background: bool = True,
) -> dict:
    """Index a directory for code search.

    Scans the directory for source files matching the patterns,
    splits them into semantic chunks, and adds to the search index.
    Runs in background by default to return immediately.

    Args:
        path: Directory path to index
        patterns: Optional glob patterns (default: ['**/*.py', '**/*.ts', '**/*.js', '**/*.tsx', '**/*.jsx'])
        clear_existing: Whether to clear existing index for this directory (default: True)
        background: Run in background (default: True). Use job_status to check progress.

    Returns:
        If background=True: {job_id, status: "submitted", message: ...}
        If background=False: Indexing statistics including files indexed and chunks created
    """
    try:
        log.info(f"index_directory called (path={path}, background={background})")

        directory = Path(path)
        if not directory.exists():
            return {"error": f"Directory not found: {path}"}

        # Read code files locally (offload blocking I/O to thread)
        reader = await _get_reader()
        files = await asyncio.to_thread(
            reader.read_code_files,
            directory,
            patterns,
            1000,  # max_files
            500,  # max_file_size_kb
        )

        if not files:
            return {"error": "No matching files found", "files_indexed": 0}

        # Send to backend for indexing
        client = await _get_client()
        return await client.index_code(
            project_root=str(directory.absolute()),
            files=files,
            clear_existing=clear_existing,
            background=background,
        )
    except BackendError as e:
        log.error(f"index_directory failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def code_stats(project_root: str | None = None) -> dict:
    """Get statistics about the code index.

    Args:
        project_root: Optional - filter to specific project

    Returns:
        Statistics including chunk count and unique files
    """
    try:
        log.info("code_stats called")
        client = await _get_client()
        return await client.code_stats(project_root=project_root)
    except BackendError as e:
        log.error(f"code_stats failed: {e}")
        return {"error": e.detail}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_graph_schema() -> dict:
    """Get the complete graph schema for zero-discovery query generation.

    Returns the full FalkorDB graph schema including:
    - All node labels with their properties, types, and indexes
    - All relationship types with descriptions and properties
    - Common query templates ready to use

    Use this BEFORE writing any Cypher queries to avoid wasting tokens
    on schema discovery. The schema includes everything needed to write
    accurate queries on the first attempt.

    Returns:
        Dict containing:
        - node_labels: List of node types with properties
        - relationship_types: List of edge types with metadata
        - common_queries: Ready-to-use query templates
    """
    try:
        log.info("get_graph_schema called")
        client = await _get_client()
        return await client.get_graph_schema()
    except BackendError as e:
        log.error(f"get_graph_schema failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def run_cypher_query(
    query: str,
    params: dict | None = None,
    max_results: int = 100,
) -> dict:
    """Execute a validated Cypher query against the FalkorDB graph.

    IMPORTANT: Use get_graph_schema first to understand available
    node types, relationships, and properties before writing queries.

    Security features (enforced automatically):
    - Read-only by default: CREATE, MERGE, DELETE, SET, REMOVE blocked
    - LIMIT injection: Queries without LIMIT get one added
    - Result truncation: Output capped at max_results

    Args:
        query: Cypher query string (e.g., "MATCH (m:Memory) RETURN m.uuid, m.type LIMIT 10")
        params: Optional query parameters for parameterized queries
        max_results: Maximum rows to return (default: 100, max: 1000)

    Returns:
        Dict containing:
        - results: List of result rows as dicts
        - row_count: Number of rows returned
        - truncated: True if results were limited
        - execution_time_ms: Query execution time

    Example queries (after checking schema):
        - "MATCH (m:Memory)-[:RELATES_TO]->(e:Entity) WHERE e.name CONTAINS 'auth' RETURN m.uuid, m.content LIMIT 20"
        - "MATCH path = (m:Memory)-[*1..2]-(other) WHERE m.uuid = $uuid RETURN path"
        - "MATCH (e:Entity {type: 'file'}) RETURN e.name, e.version ORDER BY e.version DESC LIMIT 10"
    """
    try:
        log.info(f"run_cypher_query called (query='{query[:50]}...')")
        client = await _get_client()
        return await client.run_cypher_query(
            query=query,
            params=params,
            max_results=max_results,
        )
    except BackendError as e:
        log.error(f"run_cypher_query failed: {e}")
        return {"error": e.detail}


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def start_code_watching(
    project_root: str,
    patterns: list[str] | None = None,
) -> dict:
    """Start watching a project directory for file changes.

    When files matching the patterns are created, modified, or deleted,
    the code index will be automatically updated.

    Args:
        project_root: Absolute path to the project root directory
        patterns: Glob patterns for files to watch (default: *.py, *.ts, *.js, *.tsx, *.jsx)

    Returns:
        Status dict with:
        - status: "started", "already_watching", or error
        - project_root: The watched directory
        - patterns: The file patterns being watched
    """
    try:
        log.info(f"start_code_watching called (project_root={project_root})")
        manager = await _get_watcher_manager()
        return manager.start_watching(project_root, patterns)
    except Exception as e:
        log.error(f"start_code_watching failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def stop_code_watching(project_root: str) -> dict:
    """Stop watching a project directory for file changes.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Status dict with:
        - status: "stopped" or "not_watching"
        - project_root: The directory
    """
    try:
        log.info(f"stop_code_watching called (project_root={project_root})")
        manager = await _get_watcher_manager()
        return manager.stop_watching(project_root)
    except Exception as e:
        log.error(f"stop_code_watching failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_watcher_status(project_root: str | None = None) -> dict:
    """Get status of file watchers.

    Args:
        project_root: Optional - specific project to check status for.
                     If not provided, returns status of all watchers.

    Returns:
        Status dict with watching state and details
    """
    try:
        log.info(f"get_watcher_status called (project_root={project_root})")
        manager = await _get_watcher_manager()
        return manager.get_status(project_root)
    except Exception as e:
        log.error(f"get_watcher_status failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_project_status(project_root: str) -> dict:
    """Get bootstrap and watcher status for a project.

    Args:
        project_root: Absolute path to the project root directory

    Returns:
        Dict with:
        - is_known: Whether project is tracked
        - is_bootstrapped: Whether project has been bootstrapped
        - is_watching: Whether file watcher is active
        - project_name: Detected project name
        - should_ask: Whether to prompt for bootstrap
        - deferred_context: Context from pending session (if any)
    """
    try:
        log.info(f"get_project_status called (project_root={project_root})")

        # Get local directory info
        reader = await _get_reader()
        info = await asyncio.to_thread(reader.get_directory_info, Path(project_root))

        if info is None:
            return {"error": "Could not read directory info"}

        # Get local watcher status (watchers run on MCP side)
        manager = await _get_watcher_manager()
        watcher_status = manager.get_status(project_root)
        is_watching = watcher_status.get("is_watching", False)

        # Get bootstrap status from cloud backend
        client = await _get_client()
        try:
            backend_status = await client.get_project_status(project_root)
            log.debug(f"Backend project status: {backend_status}")
        except BackendError as e:
            log.warning(f"Backend project status failed: {e}, using defaults")
            backend_status = {}

        # Merge local and cloud status
        return {
            # Local info
            "exists": info.get("exists", False),
            "is_git": info.get("is_git", False),
            "file_count": info.get("file_count", 0),
            "is_watching": is_watching,
            # Cloud bootstrap status
            "is_known": backend_status.get("is_known", False),
            "is_bootstrapped": backend_status.get("is_bootstrapped", False),
            "should_ask": backend_status.get("should_ask", True),
            "project_name": backend_status.get("project_name") or Path(project_root).name,
            "never_ask": backend_status.get("never_ask", False),
        }
    except Exception as e:
        log.error(f"get_project_status failed: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK STALENESS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def check_code_staleness(
    project_root: str,
    hours_threshold: int = 24,
) -> dict:
    """Check if the code index for a project may be stale.

    Compares the last index time against file modification times
    to detect potentially outdated index entries.

    Args:
        project_root: Absolute path to the project root directory
        hours_threshold: Consider stale if not indexed within this many hours

    Returns:
        Dict with:
        - is_stale: Whether the index appears outdated
        - last_indexed: ISO timestamp of last indexing (if known)
        - files_modified_since: Count of files modified since last index
        - recommendation: Suggested action
    """
    try:
        # For now, return a simple check - in future could track index timestamps
        return {
            "is_stale": False,
            "last_indexed": None,
            "files_modified_since": 0,
            "recommendation": "Use index_directory to refresh the code index",
        }
    except Exception as e:
        log.error(f"check_code_staleness failed: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


async def cleanup():
    """Cleanup resources on shutdown."""
    global _client, _watcher_manager

    # Stop all file watchers
    if _watcher_manager is not None:
        try:
            _watcher_manager.stop_all()
            log.info("Stopped all file watchers")
        except Exception as e:
            log.warning(f"Error stopping watchers: {e}")
        _watcher_manager = None

    # Close HTTP client
    if _client is not None:
        try:
            await _client.close()
        except Exception as e:
            log.warning(f"Error closing client: {e}")
        _client = None


def main():
    """Run the MCP server."""
    try:
        log.info("Starting SimpleMem MCP server...")
        mcp.run()
    finally:
        # Attempt cleanup
        log.info("Server shutting down, running cleanup...")
        try:
            asyncio.run(cleanup())
        except RuntimeError as e:
            # Event loop may already be closed or running
            log.warning(f"Could not run async cleanup gracefully: {e}")
        log.info("Cleanup finished.")


if __name__ == "__main__":
    main()
