"""Thin MCP server for SimpleMem.

Cloud-first MCP server that:
1. Reads local files on-demand (traces, code files)
2. Compresses payloads for transport
3. Proxies requests to the backend API (cloud or local)

The server runs locally but connects to the cloud backend by default.
Set SIMPLEMEM_BACKEND_URL to override with a local backend.

CONSOLIDATED API (10 tools):
- remember: Store memories with optional relations
- recall: Search/retrieve memories (fast, deep, ask modes)
- index: Index code files or session traces
- search_code: Semantic code search with related memories
- trace: Manage trace processing and jobs
- project: Bootstrap and manage projects
- graph: Graph schema and queries
- scratchpad: Task context management
- admin: Stats, sync, watchers, bootstrap
"""

import asyncio
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from simplemem_mcp.client import BackendClient, BackendError
from simplemem_mcp.local_reader import LocalReader
from simplemem_mcp.projects_utils import (
    # Core functions
    get_project_id as generate_project_id,
    infer_project_from_session_path,
    parse_project_id,
    extract_project_name,
    # Bootstrap functions
    find_project_root,
    suggest_project_names,
    create_bootstrap_config,
    # Registry functions (for lossy path encoding)
    register_project,
    # Exceptions and models
    NotBootstrappedError,
    SimpleMemConfig,
    ProjectNameSuggestion,
)
from simplemem_mcp.watcher import CloudWatcherManager, TraceWatcherManager, auto_resume_watchers
from simplemem_mcp.compression import compress_if_large

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
_trace_watcher_manager: TraceWatcherManager | None = None
_watchers_auto_resumed: bool = False  # Set once auto_resume_watchers is called
_client_lock = asyncio.Lock()
_reader_lock = asyncio.Lock()
_watcher_lock = asyncio.Lock()
_trace_watcher_lock = asyncio.Lock()
_auto_resume_lock = asyncio.Lock()

# Default output format from environment
OUTPUT_FORMAT = os.environ.get("SIMPLEMEM_OUTPUT_FORMAT", "toon")

# Compression threshold
COMPRESSION_THRESHOLD = 4096  # 4KB


def _to_toon(records: list[dict], headers: list[str]) -> str:
    """Convert list of dicts to TOON (Token-Optimized Object Notation) format.

    TOON uses tab-separated values with headers on the first line.
    This achieves 30-60% token savings compared to JSON.
    """
    if not records:
        return "\t".join(headers)
    lines = ["\t".join(headers)]
    for record in records:
        values = [str(record.get(h, "")).replace("\t", " ").replace("\n", " ") for h in headers]
        lines.append("\t".join(values))
    return "\n".join(lines)


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


def _get_trace_watcher_sync() -> TraceWatcherManager | None:
    """Sync getter for trace watcher (for state persistence callbacks)."""
    return _trace_watcher_manager


def _get_watcher_sync() -> CloudWatcherManager | None:
    """Sync getter for code watcher (for state persistence callbacks)."""
    return _watcher_manager


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
                _watcher_manager = CloudWatcherManager(
                    client=watcher_client,
                    reader=reader,
                    trace_watcher_getter=_get_trace_watcher_sync,
                )
    return _watcher_manager


async def _get_trace_watcher_manager() -> TraceWatcherManager:
    """Get or create the trace watcher manager (thread-safe)."""
    global _trace_watcher_manager
    if _trace_watcher_manager is None:
        async with _trace_watcher_lock:
            # Double-check after acquiring lock
            if _trace_watcher_manager is None:
                # Create a NEW BackendClient for the trace watcher (separate event loop)
                trace_watcher_client = BackendClient()
                _trace_watcher_manager = TraceWatcherManager(
                    client=trace_watcher_client,
                    code_watcher_getter=_get_watcher_sync,
                )
    return _trace_watcher_manager


async def _ensure_watchers_initialized() -> tuple[CloudWatcherManager, TraceWatcherManager]:
    """Initialize both watcher managers and auto-resume from persisted state.

    Should be called when watcher functionality is first needed.
    Auto-resume only runs once (idempotent).

    Returns:
        Tuple of (code_watcher_manager, trace_watcher_manager)
    """
    global _watchers_auto_resumed

    # Get both managers (may create them)
    code_mgr = await _get_watcher_manager()
    trace_mgr = await _get_trace_watcher_manager()

    # Auto-resume from persisted state (only once)
    if not _watchers_auto_resumed:
        async with _auto_resume_lock:
            # Double-check after lock
            if not _watchers_auto_resumed:
                try:
                    result = auto_resume_watchers(code_mgr, trace_mgr)
                    _watchers_auto_resumed = True
                    if result.get("code_watchers_resumed", 0) > 0 or result.get("trace_watcher_resumed"):
                        log.info(
                            f"Auto-resumed watchers: {result['code_watchers_resumed']} code, "
                            f"trace={result['trace_watcher_resumed']}"
                        )
                except Exception as e:
                    log.warning(f"Auto-resume failed: {e}")
                    _watchers_auto_resumed = True  # Don't retry

    return code_mgr, trace_mgr


def _resolve_project_id(
    project_id: str | None = None,
    path: str | None = None,
    require_bootstrap: bool = True,
) -> str | None:
    """Resolve project_id from explicit param or .simplemem.yaml config.

    IMPORTANT: Explicit project_id takes precedence over config file.
    The config is only used as a fallback when no explicit project_id is provided.

    Args:
        project_id: Optional explicit project_id. If provided, this is used
                    directly (with "config:" prefix normalization).
        path: Optional path to resolve from. Defaults to cwd.
        require_bootstrap: If True (default), raise NotBootstrappedError
                          if no config found and no explicit project_id.
                          If False, return None.

    Returns:
        Project ID with "config:" prefix (from explicit param or config)

    Raises:
        NotBootstrappedError: If require_bootstrap=True, no explicit project_id,
                             and no config found
    """
    # If explicit project_id provided, use it directly (with normalization)
    if project_id:
        # Normalize: add config: prefix if missing
        if not project_id.startswith("config:"):
            project_id = f"config:{project_id}"
        log.debug(f"Using explicit project_id: {project_id}")
        return project_id

    # No explicit project_id - try to load from config file
    resolved_path = Path(path).resolve() if path else Path.cwd().resolve()

    try:
        # find_project_root raises NotBootstrappedError if no config
        config_dir, config = find_project_root(resolved_path)
        actual_project_id = config.project_id
        log.debug(f"Resolved project_id from config: {actual_project_id} ({config_dir})")
        return actual_project_id

    except NotBootstrappedError:
        if require_bootstrap:
            raise
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATED API - 10 TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def remember(
    content: str,
    project: str | None = None,
    type: str = "fact",
    relations: list[str] | None = None,
) -> dict:
    """Store a memory with optional relations.

    PURPOSE: Unified entry point for storing memories. Simpler API than store_memory.

    MEMORY TYPES:
    - "fact": Project-specific facts, configurations, conventions (default)
    - "lesson_learned": Debugging insights, gotchas, what worked/didn't work
    - "decision": Architectural choices with rationale and rejected alternatives
    - "pattern": Reusable code patterns, approaches, templates

    EXAMPLES:
        # Store a lesson learned
        remember(content="Fix: Check Docker when DB fails", type="lesson_learned", project="myproject")

        # Store a decision with relations
        remember(
            content="Decision: Use Redis for caching | Reason: Speed",
            type="decision",
            relations=["uuid-of-related-memory"],
            project="myproject"
        )

    Args:
        content: The content to store. Be specific and actionable.
        project: Project ID for isolation (recommended). Auto-inferred from cwd if not specified.
        type: Memory type: fact (default), lesson_learned, decision, pattern
        relations: Optional list of memory UUIDs to relate this memory to

    Returns:
        On success: {"uuid": "<memory-uuid>", "relations_created": N}
        On error: {"error": "<error-message>"}
    """
    try:
        resolved_project_id = _resolve_project_id(project)
    except NotBootstrappedError as e:
        return e.to_dict()

    log.info(f"remember called (type={type}, project={resolved_project_id})")
    try:
        client = await _get_client()
        result = await client.remember(
            content=content,
            project=resolved_project_id,
            type=type,
            relations=relations,
        )
        return result
    except BackendError as e:
        log.error(f"remember failed: {e}")
        return {"error": e.detail}


def _detect_query_intent(query: str) -> str:
    """Detect if query is code-oriented, memory-oriented, or both.

    Used for auto-routing queries to the appropriate search backend:
    - "code": Query should primarily search code index
    - "memory": Query should primarily search memories
    - "both": Query should search both and merge results

    Args:
        query: The search query string

    Returns:
        Intent classification: "code", "memory", or "both"
    """
    # Keywords strongly indicating code search intent
    code_keywords = {
        # Python-specific
        "function", "class", "def", "import", "module", "async",
        # General code terms
        "implementation", "method", "api", "endpoint", "handler",
        "controller", "service", "repository", "schema", "model",
        # File extensions (with and without dots)
        ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs",
        "py file", "ts file", "js file",
        # Code actions
        "where is", "find the", "locate", "source code",
        "how is implemented", "how does the code",
    }

    # Keywords strongly indicating memory/knowledge search
    memory_keywords = {
        "decision", "why did we", "lesson learned", "what happened",
        "remember", "previous", "last time", "history", "debug",
        "fix", "error", "bug", "issue", "problem", "solved",
        "architecture decision", "rationale", "reason for",
    }

    query_lower = query.lower()

    # Count keyword matches
    code_score = sum(1 for kw in code_keywords if kw in query_lower)
    memory_score = sum(1 for kw in memory_keywords if kw in query_lower)

    # Decision logic with thresholds
    if code_score >= 2 and memory_score == 0:
        return "code"
    elif memory_score >= 2 and code_score == 0:
        return "memory"
    elif code_score >= 1 and memory_score >= 1:
        return "both"
    elif code_score >= 1:
        return "both"  # Single code keyword = search both to be safe
    else:
        return "memory"  # Default to memory search


@mcp.tool()
async def recall(
    query: str | None = None,
    id: str | None = None,
    project: str | None = None,
    mode: str = "fast",
    limit: int = 10,
    sort_by: str = "relevance",
    since: str | None = None,
    until: str | None = None,
    max_hops: int = 2,
    min_score: float = 0.1,
    output_format: str | None = None,
    auto_route: bool = False,
) -> dict | str:
    """Find memories by query or exact ID.

    PURPOSE: Unified entry point for all memory retrieval. Replaces search_memories,
    search_memories_deep, ask_memories, and get-by-ID operations.

    MODES:
    - "fast": Vector similarity search (default, quickest)
    - "deep": LLM-reranked results with conflict detection
    - "ask": LLM-synthesized answer with citations
    - "reason": Multi-hop graph reasoning with LLM synthesis (follows relationships)

    AUTO-ROUTING (auto_route=True):
    When enabled, automatically detects query intent and routes to the appropriate
    search backend:
    - Code-oriented queries ("function", "implementation", "class") → code index
    - Memory-oriented queries ("decision", "why", "lesson") → memory search
    - Mixed queries → searches both and merges results

    EXAMPLES:
        # Quick search
        recall(query="database timeout fix", project="myproject")

        # Get specific memory by ID
        recall(id="abc-123-uuid")

        # Deep search with reranking
        recall(query="authentication patterns", mode="deep")

        # Get synthesized answer
        recall(query="How did we fix the memory leak?", mode="ask")

        # Multi-hop reasoning (follows graph relationships)
        recall(query="What led to the database redesign?", mode="reason", max_hops=3)

        # Get memories from last 2 days only (prevent stale context)
        recall(query="auth bug", project="myproject", since="2d")

        # Get newest memories first (browse recent)
        recall(query="", project="myproject", sort_by="newest", limit=20)

        # Auto-routing: finds both code implementations and debugging memories
        recall(query="authentication handler implementation", auto_route=True)

    Args:
        query: Search query (required if no id)
        id: Exact memory UUID to fetch (bypasses search)
        project: Project ID for isolation. Auto-inferred from cwd if not specified.
        mode: Search mode - fast (default), deep (reranked), ask (LLM synthesis), reason (multi-hop graph reasoning)
        limit: Maximum results (default: 10)
        sort_by: Sort order - relevance (default), newest, oldest
        since: Only return memories after this time. Supports relative ("2d", "1w", "30d") or ISO date ("2024-01-15")
        until: Only return memories before this time. Supports relative ("2d", "1w") or ISO date ("2024-01-15")
        max_hops: For mode="reason": max graph hops to traverse (1-3, default: 2)
        min_score: For mode="reason": minimum relevance score threshold (0.0-1.0, default: 0.1)
        output_format: Response format - "toon" (default) or "json"
        auto_route: Enable intelligent query routing to code/memory search (default: False)

    Returns:
        For fast/deep modes: {"results": [...], "conflicts": [...]}
        For ask mode: {"answer": "...", "sources": [...], "confidence": "..."}
        For reason mode: {"reasoning": "...", "conclusions": [...], "sources": [...]}
        For auto_route with "both": {"memories": [...], "code": [...], "routing": "both"}
        On error: {"error": "<error-message>"}
    """
    # Validate: need either query or id
    if not query and not id:
        return {"error": "Either 'query' or 'id' is required"}

    try:
        resolved_project_id = _resolve_project_id(project)
    except NotBootstrappedError as e:
        return e.to_dict()

    log.info(f"recall called (mode={mode}, project={resolved_project_id}, sort_by={sort_by}, since={since}, until={until}, auto_route={auto_route})")
    try:
        client = await _get_client()

        # Auto-routing: detect query intent and route to appropriate search
        if auto_route and query and mode == "fast":
            intent = _detect_query_intent(query)
            log.info(f"Auto-routing detected intent: {intent} for query: '{query[:50]}...'")

            if intent == "code":
                # Route to code search only
                code_result = await client.search_code(
                    query=query,
                    limit=limit,
                    project_id=resolved_project_id,
                    content_mode="preview",
                    output_format="json",  # Need JSON for structured response
                )
                code_chunks = []
                if isinstance(code_result, dict):
                    code_chunks = code_result.get("results", [])
                return {
                    "code": code_chunks,
                    "memories": [],
                    "routing": "code",
                    "intent_detected": intent,
                }

            elif intent == "both":
                # Search both memory and code, merge results
                import asyncio
                memory_task = client.recall(
                    query=query,
                    id=None,
                    project=resolved_project_id,
                    mode="fast",
                    limit=limit // 2 + 1,
                    sort_by=sort_by,
                    since=since,
                    until=until,
                    output_format="json",  # Need JSON for merging
                )
                code_task = client.search_code(
                    query=query,
                    limit=limit // 2 + 1,
                    project_id=resolved_project_id,
                    content_mode="preview",
                    output_format="json",
                )

                memory_result, code_result = await asyncio.gather(
                    memory_task, code_task, return_exceptions=True
                )

                # Handle potential errors
                memories = []
                code_chunks = []
                if isinstance(memory_result, dict):
                    memories = memory_result.get("results", [])
                if isinstance(code_result, dict):
                    code_chunks = code_result.get("results", [])

                return {
                    "memories": memories,
                    "code": code_chunks,
                    "routing": "both",
                    "intent_detected": intent,
                }

            # intent == "memory" falls through to normal memory search

        # Handle "reason" mode separately - uses different API endpoint
        if mode == "reason":
            if not query:
                return {"error": "Query is required for mode='reason'"}
            result = await client.reason_memories(
                query=query,
                max_hops=max_hops,
                min_score=min_score,
                project_id=resolved_project_id,
                output_format=output_format or OUTPUT_FORMAT,
            )
            return result

        # All other modes use the unified recall endpoint
        result = await client.recall(
            query=query,
            id=id,
            project=resolved_project_id,
            mode=mode,
            limit=limit,
            sort_by=sort_by,
            since=since,
            until=until,
            output_format=output_format or OUTPUT_FORMAT,
        )
        return result
    except BackendError as e:
        log.error(f"recall failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def index(
    project: str,
    path: str | None = None,
    patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    files: list[dict] | None = None,
    traces: list[dict] | None = None,
    clear_existing: bool = True,
    wait: bool = False,
    dry_run: bool = False,
) -> dict:
    """Index code files or session traces for semantic search.

    PURPOSE: Unified entry point for all indexing. Can index:
    - A local directory (provide 'path')
    - Explicit file contents (provide 'files')
    - Session traces (provide 'traces')

    DIRECTORY INDEXING:
        index(project="myproject", path=".", dry_run=True)  # Preview first!
        index(project="myproject", path=".")  # Then index

    FILE INDEXING:
        index(project="myproject", files=[{"path": "src/app.py", "content": "..."}])

    TRACE INDEXING:
        index(project="myproject", traces=[{"session_id": "abc-123", "content": {...}}])

    Args:
        project: Project ID for isolation (required)
        path: Directory path to index (reads files locally, sends to backend)
        patterns: Glob patterns for files to include (default: *.py, *.ts, *.js, *.tsx, *.jsx)
        ignore_patterns: Gitignore-style patterns for files to exclude
        files: Files to index (each with "path" and "content" keys)
        traces: Traces to index (each with "session_id" and "content" keys)
        clear_existing: Clear existing code index before indexing (default: True)
        wait: Wait for completion (default: False, runs in background)
        dry_run: Preview mode - returns files that would be indexed without indexing

    Returns:
        If dry_run=True: {"dry_run": True, "summary": {...}, "files": [...]}
        If wait=False: {"job_id": "...", "status": "submitted", "message": "..."}
        If wait=True: {"status": "completed", "files_indexed": N, "chunks_created": N}
        On error: {"error": "<error-message>"}
    """
    try:
        resolved_project_id = _resolve_project_id(project)
    except NotBootstrappedError as e:
        return e.to_dict()

    if resolved_project_id is None:
        return {"error": "Project ID is required for indexing"}

    # If path provided, read files locally first
    if path:
        reader = await _get_reader()
        default_patterns = ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"]
        file_patterns = patterns or default_patterns

        # Resolve path
        resolved_path = Path(path).resolve()

        if dry_run:
            # Use dry_run_scan for preview
            return reader.dry_run_scan(
                directory=resolved_path,
                patterns=file_patterns,
                ignore_patterns=ignore_patterns,
                verbosity="minimal",
            )

        # Read code files using the proper API
        files = reader.read_code_files(
            directory=resolved_path,
            patterns=file_patterns,
            ignore_patterns=ignore_patterns,
        )

    # Validate: need files or traces
    if not files and not traces:
        return {"error": "Either 'path', 'files', or 'traces' is required"}

    if files and traces:
        return {"error": "Cannot index both files and traces in same request"}

    log.info(f"index called (project={resolved_project_id}, files={len(files or [])}, traces={len(traces or [])})")
    try:
        client = await _get_client()
        result = await client.index_v2(
            project=resolved_project_id,
            files=files,
            traces=traces,
            clear_existing=clear_existing,
            wait=wait,
        )
        return result
    except BackendError as e:
        log.error(f"index failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def search_code(
    query: str,
    project: str | None = None,
    mode: str = "search",
    limit: int = 10,
    content_mode: str = "preview",
    chunk_uuid: str | None = None,
    memory_uuid: str | None = None,
    output_format: str | None = None,
) -> dict | str:
    """Semantic code search with related memories.

    PURPOSE: Find code implementations and their related debugging history.

    MODES:
    - "search": Semantic search over indexed code (default)
    - "related_memories": Find memories related to a code chunk
    - "related_code": Find code related to a memory
    - "stats": Get code index statistics

    EXAMPLES:
        # Search for code
        search_code(query="authentication handler", project="myproject")

        # Token-efficient search (signature only)
        search_code(query="authentication", project="myproject", content_mode="signature")

        # Find memories related to a code chunk
        search_code(mode="related_memories", chunk_uuid="abc-123")

        # Find code related to a memory
        search_code(mode="related_code", memory_uuid="def-456")

        # Get stats
        search_code(mode="stats", project="myproject")

    Args:
        query: Search query (required for mode="search")
        project: Project ID for isolation. Auto-inferred from cwd if not specified.
        mode: Operation mode - search, related_memories, related_code, stats
        limit: Maximum results (default: 10)
        content_mode: Content verbosity for mode="search". Default "preview".
            - "signature": Function/class signature only (~15 tokens/result)
            - "preview": Signature + truncated body (~80 tokens/result)
            - "full": Complete content (~400 tokens/result)
        chunk_uuid: Code chunk UUID (required for mode="related_memories")
        memory_uuid: Memory UUID (required for mode="related_code")
        output_format: Response format - "toon" (default) or "json"

    Returns:
        For search: {"results": [...]}
        For related_*: {"related_memories/related_code": [...], "count": N}
        For stats: {"chunk_count": N, "unique_files": N}
        On error: {"error": "<error-message>"}
    """
    try:
        resolved_project_id = _resolve_project_id(project, require_bootstrap=mode != "stats")
    except NotBootstrappedError as e:
        return e.to_dict()

    log.info(f"search_code called (mode={mode}, project={resolved_project_id})")
    try:
        client = await _get_client()

        if mode == "search":
            if not query:
                return {"error": "Query is required for mode='search'"}
            return await client.search_code(
                query=query,
                limit=limit,
                project_id=resolved_project_id,
                content_mode=content_mode,
                output_format=output_format or OUTPUT_FORMAT,
            )

        elif mode == "related_memories":
            if not chunk_uuid:
                return {"error": "chunk_uuid is required for mode='related_memories'"}
            return await client.code_related_memories(
                chunk_uuid=chunk_uuid,
                limit=limit,
                output_format=output_format or OUTPUT_FORMAT,
            )

        elif mode == "related_code":
            if not memory_uuid:
                return {"error": "memory_uuid is required for mode='related_code'"}
            return await client.memory_related_code(
                memory_uuid=memory_uuid,
                limit=limit,
                output_format=output_format or OUTPUT_FORMAT,
            )

        elif mode == "stats":
            return await client.code_stats(project_id=resolved_project_id)

        else:
            return {"error": f"Unknown mode: {mode}. Valid modes: search, related_memories, related_code, stats"}

    except BackendError as e:
        log.error(f"search_code failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def trace(
    action: str,
    session_id: str | None = None,
    job_id: str | None = None,
    sessions: list[dict] | None = None,
    project: str | None = None,
    days_back: int = 30,
    limit: int = 20,
    include_indexed: bool = True,
    include_completed: bool = True,
    output_format: str | None = None,
) -> dict | str:
    """Manage trace processing and background jobs.

    PURPOSE: Process Claude Code session traces into searchable memories.

    ACTIONS:
    - "discover": Find available sessions to index
    - "process": Process a single session trace
    - "batch": Process multiple sessions at once
    - "job_status": Check status of a background job
    - "jobs": List all background jobs
    - "cancel": Cancel a running job

    EXAMPLES:
        # Discover sessions from last 7 days
        trace(action="discover", days_back=7)

        # Process a single session
        trace(action="process", session_id="abc-123")

        # Process multiple sessions
        trace(action="batch", sessions=[...])

        # Check job status
        trace(action="job_status", job_id="job-uuid")

        # List jobs
        trace(action="jobs")

        # Cancel a job
        trace(action="cancel", job_id="job-uuid")

    Args:
        action: Operation to perform (discover, process, batch, job_status, jobs, cancel)
        session_id: Session UUID for action="process"
        job_id: Job UUID for action="job_status" or "cancel"
        sessions: List of session dicts for action="batch"
        project: Project ID (auto-inferred for process)
        days_back: Days to look back for action="discover" (default: 30)
        limit: Max sessions for action="discover" (default: 20)
        include_indexed: Include already indexed sessions (default: True)
        include_completed: Include completed jobs (default: True)
        output_format: Response format - "toon" (default) or "json"

    Returns:
        Action-specific response or {"error": "..."}
    """
    log.info(f"trace called (action={action})")
    try:
        client = await _get_client()
        reader = await _get_reader()

        if action == "discover":
            # Discover sessions locally
            sessions_list = reader.discover_sessions(days_back=days_back, limit=limit)

            # Filter by indexed status if needed
            if not include_indexed:
                try:
                    # Get indexed sessions from backend (scoped to project if specified)
                    resolved_project_id = _resolve_project_id(project, require_bootstrap=False) if project else None
                    indexed_result = await client.get_indexed_sessions(
                        project_id=resolved_project_id, limit=500
                    )
                    indexed_session_ids = set(indexed_result.get("indexed_sessions", []))

                    # Filter out already-indexed sessions
                    sessions_list["sessions"] = [
                        s for s in sessions_list["sessions"]
                        if s.get("session_id") not in indexed_session_ids
                    ]
                    sessions_list["total"] = len(sessions_list["sessions"])
                    log.debug(f"Filtered out {len(indexed_session_ids)} indexed sessions")
                except Exception as e:
                    log.warning(f"Failed to get indexed sessions: {e}")
                    # Continue with unfiltered results

            if output_format == "toon" or (output_format is None and OUTPUT_FORMAT == "toon"):
                return _to_toon(
                    sessions_list["sessions"],
                    ["session_id", "path", "modified", "size_kb"]
                )
            return sessions_list

        elif action == "process":
            if not session_id:
                return {"error": "session_id is required for action='process'"}

            # Read trace content locally
            trace_content = reader.read_trace_file(session_id)
            if trace_content is None:
                return {"error": f"Session not found: {session_id}"}

            # Infer project from session path
            session_path = reader.find_session_path(session_id)
            inferred_project = None
            if session_path:
                inferred_project = infer_project_from_session_path(session_path)

            resolved_project = project or inferred_project
            if resolved_project and not resolved_project.startswith("config:"):
                resolved_project = f"config:{resolved_project}"

            return await client.process_trace(
                session_id=session_id,
                trace_content=trace_content,
                background=True,
                project_id=resolved_project,
            )

        elif action == "batch":
            if not sessions:
                return {"error": "sessions list is required for action='batch'"}

            # Read each trace and compress
            traces_to_send = []
            for s in sessions:
                sid = s.get("session_id")
                if not sid:
                    continue
                content = reader.read_trace_file(sid)
                if content:
                    compressed, was_compressed = compress_if_large(content, threshold_bytes=COMPRESSION_THRESHOLD)
                    traces_to_send.append({
                        "session_id": sid,
                        "trace_content": compressed,
                        "compressed": was_compressed,
                    })

            return await client.process_trace_batch(traces=traces_to_send)

        elif action == "job_status":
            if not job_id:
                return {"error": "job_id is required for action='job_status'"}
            return await client.get_job_status(job_id)

        elif action == "jobs":
            return await client.list_jobs(
                include_completed=include_completed,
                limit=limit,
                output_format=output_format or OUTPUT_FORMAT,
            )

        elif action == "cancel":
            if not job_id:
                return {"error": "job_id is required for action='cancel'"}
            return await client.cancel_job(job_id)

        else:
            return {"error": f"Unknown action: {action}. Valid: discover, process, batch, job_status, jobs, cancel"}

    except BackendError as e:
        log.error(f"trace failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def project(
    action: str = "status",
    path: str | None = None,
    project_name: str | None = None,
    project_id: str | None = None,
    folder_role: str | None = None,
    force: bool = False,
) -> dict:
    """Bootstrap and manage SimpleMem projects.

    PURPOSE: Setup and query project configuration.

    ACTIONS:
    - "status": Get project bootstrap status and suggestions (default)
    - "bootstrap": Create new project with .simplemem.yaml
    - "attach": Attach folder to existing project

    EXAMPLES:
        # Check project status
        project(action="status")

        # Bootstrap new project
        project(action="bootstrap", project_name="My Project")

        # Attach to existing project
        project(action="attach", project_id="config:simplemem")

    Args:
        action: Operation to perform (status, bootstrap, attach)
        path: Directory path (defaults to cwd)
        project_name: Human-readable project name (for bootstrap)
        project_id: Project ID to attach to (for attach) or explicit ID (for bootstrap)
        folder_role: Optional role (source, tests, docs, config, scripts)
        force: Overwrite existing config (default: False)

    Returns:
        For status: {"is_bootstrapped": bool, "project_id": "...", "suggested_names": [...]}
        For bootstrap: {"success": True, "project_id": "...", "config_path": "..."}
        For attach: {"success": True, "attached_path": "..."}
        On error: {"error": "..."}
    """
    resolved_path = Path(path).resolve() if path else Path.cwd().resolve()
    log.info(f"project called (action={action}, path={resolved_path})")

    try:
        if action == "status":
            # Check if bootstrapped
            try:
                config_dir, config = find_project_root(resolved_path)
                return {
                    "is_bootstrapped": True,
                    "project_id": config.project_id,
                    "project_name": config.project_name,
                    "config_path": str(config_dir / ".simplemem.yaml"),
                    "folder_role": config.folder_role,
                }
            except NotBootstrappedError:
                # Get suggestions
                suggestions = suggest_project_names(resolved_path)
                return {
                    "is_bootstrapped": False,
                    "path": str(resolved_path),
                    "suggested_names": [
                        {"name": s.name, "source": s.source, "confidence": s.confidence}
                        for s in suggestions
                    ],
                    "recommended": suggestions[0].name if suggestions else None,
                    "action_required": "bootstrap",
                }

        elif action == "bootstrap":
            if not project_name:
                return {"error": "project_name is required for action='bootstrap'"}

            config_path = resolved_path / ".simplemem.yaml"
            if config_path.exists() and not force:
                return {"error": f"Config already exists: {config_path}. Use force=True to overwrite."}

            # Generate project_id if not provided
            if not project_id:
                # Sanitize name to ID
                import re
                sanitized = re.sub(r'[^a-z0-9-]', '-', project_name.lower())
                sanitized = re.sub(r'-+', '-', sanitized).strip('-')
                project_id = f"config:{sanitized}"

            # Ensure prefix
            if not project_id.startswith("config:"):
                project_id = f"config:{project_id}"

            # Create config
            create_bootstrap_config(
                path=resolved_path,
                project_name=project_name,
                project_id=project_id,
                folder_role=folder_role,
                force=force,
            )

            # Register in local registry
            register_project(resolved_path, project_id)

            # Auto-start watchers (triggers auto-resume from persisted state)
            watcher_result = None
            trace_watcher_result = None
            try:
                watcher_manager, trace_watcher = await _ensure_watchers_initialized()

                # Start code watcher for this project
                watcher_result = watcher_manager.start_watching(
                    project_root=str(resolved_path),
                    patterns=["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"],
                    project_id=project_id,
                )
                log.info(f"Auto-started watcher for bootstrapped project: {watcher_result}")

                # Start trace watcher
                trace_watcher_result = trace_watcher.start_watching()
                log.info(f"Auto-started trace watcher: {trace_watcher_result}")
            except Exception as e:
                log.warning(f"Failed to auto-start watchers: {e}")

            return {
                "success": True,
                "project_id": project_id,
                "project_name": project_name,
                "config_path": str(config_path),
                "watcher_started": watcher_result is not None and watcher_result.get("status") in ("started", "already_watching"),
                "trace_watcher_started": trace_watcher_result is not None and trace_watcher_result.get("status") in ("started", "already_watching"),
                "message": f"Project bootstrapped successfully!",
            }

        elif action == "attach":
            if not project_id:
                return {"error": "project_id is required for action='attach'"}

            # Ensure prefix
            if not project_id.startswith("config:"):
                project_id = f"config:{project_id}"

            config_path = resolved_path / ".simplemem.yaml"
            if config_path.exists() and not force:
                return {"error": f"Config already exists: {config_path}. Use force=True to overwrite."}

            # Use directory name as project_name if not provided
            display_name = project_name or resolved_path.name

            create_bootstrap_config(
                path=resolved_path,
                project_name=display_name,
                project_id=project_id,
                folder_role=folder_role,
                force=force,
            )

            # Register in local registry
            register_project(resolved_path, project_id)

            return {
                "success": True,
                "project_id": project_id,
                "attached_path": str(resolved_path),
                "message": f"Folder attached to project {project_id}",
            }

        else:
            return {"error": f"Unknown action: {action}. Valid: status, bootstrap, attach"}

    except Exception as e:
        log.error(f"project failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def graph(
    action: str = "schema",
    query: str | None = None,
    params: dict | None = None,
    max_results: int = 100,
) -> dict:
    """Execute graph queries or get schema.

    PURPOSE: Direct access to the knowledge graph for advanced queries.

    ACTIONS:
    - "schema": Get the graph schema with node/edge types (default)
    - "query": Execute a Cypher query (read-only in PROD mode)

    EXAMPLES:
        # Get schema
        graph(action="schema")

        # Run a query
        graph(action="query", query="MATCH (m:Memory) RETURN m.uuid LIMIT 10")

        # Query with parameters
        graph(action="query", query="MATCH (m:Memory {type: $type}) RETURN m", params={"type": "decision"})

    Args:
        action: Operation to perform (schema, query)
        query: Cypher query string (for action="query")
        params: Query parameters (for action="query")
        max_results: Maximum rows to return (default: 100, max: 1000)

    Returns:
        For schema: {"node_labels": [...], "relationship_types": [...], "common_queries": [...]}
        For query: {"results": [...], "row_count": N, "execution_time_ms": N}
        On error: {"error": "..."}
    """
    log.info(f"graph called (action={action})")
    try:
        client = await _get_client()

        if action == "schema":
            return await client.get_graph_schema()

        elif action == "query":
            if not query:
                return {"error": "query is required for action='query'"}
            return await client.run_cypher_query(
                query=query,
                params=params,
                max_results=max_results,
            )

        else:
            return {"error": f"Unknown action: {action}. Valid: schema, query"}

    except BackendError as e:
        log.error(f"graph failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def scratchpad(
    action: str,
    task_id: str,
    project: str | None = None,
    data: dict | None = None,
    memory_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    expand_memories: bool = False,
    format: str = "markdown",
) -> dict:
    """Manage task scratchpads for context persistence.

    PURPOSE: Save, load, update task context across conversation turns.

    ACTIONS:
    - "save": Save or replace a scratchpad
    - "load": Load a scratchpad
    - "update": Partially update a scratchpad
    - "attach": Attach memories/sessions to a scratchpad
    - "render": Render scratchpad in human-readable format
    - "delete": Delete a scratchpad

    EXAMPLES:
        # Save a scratchpad
        scratchpad(action="save", task_id="feature-auth", data={"current_focus": "..."})

        # Load a scratchpad
        scratchpad(action="load", task_id="feature-auth")

        # Update fields
        scratchpad(action="update", task_id="feature-auth", data={"current_focus": "new focus"})

        # Attach memories
        scratchpad(action="attach", task_id="feature-auth", memory_ids=["uuid1", "uuid2"])

        # Render for display
        scratchpad(action="render", task_id="feature-auth", format="markdown")

    Args:
        action: Operation (save, load, update, attach, render, delete)
        task_id: Unique task identifier
        project: Project ID for isolation
        data: Scratchpad data (for save/update)
        memory_ids: Memory UUIDs to attach
        session_ids: Session IDs to attach
        reasons: Optional reasons per ID for attach
        expand_memories: Include full memory content on load (default: False)
        format: Output format for render (markdown, json)

    Returns:
        Action-specific response or {"error": "..."}
    """
    try:
        resolved_project_id = _resolve_project_id(project)
    except NotBootstrappedError as e:
        return e.to_dict()

    if resolved_project_id is None:
        return {"error": "Project ID is required"}

    log.info(f"scratchpad called (action={action}, task_id={task_id})")
    try:
        client = await _get_client()

        if action == "save":
            if not data:
                return {"error": "data is required for action='save'"}
            return await client.save_scratchpad(
                task_id=task_id,
                scratchpad=data,
                project_id=resolved_project_id,
            )

        elif action == "load":
            return await client.load_scratchpad(
                task_id=task_id,
                project_id=resolved_project_id,
                expand_memories=expand_memories,
            )

        elif action == "update":
            if not data:
                return {"error": "data (patch) is required for action='update'"}
            return await client.update_scratchpad(
                task_id=task_id,
                patch=data,
                project_id=resolved_project_id,
            )

        elif action == "attach":
            if not memory_ids and not session_ids:
                return {"error": "memory_ids or session_ids required for action='attach'"}
            return await client.attach_to_scratchpad(
                task_id=task_id,
                project_id=resolved_project_id,
                memory_ids=memory_ids,
                session_ids=session_ids,
                reasons=reasons,
            )

        elif action == "render":
            return await client.render_scratchpad(
                task_id=task_id,
                project_id=resolved_project_id,
                format=format,
            )

        elif action == "delete":
            return await client.delete_scratchpad(
                task_id=task_id,
                project_id=resolved_project_id,
            )

        else:
            return {"error": f"Unknown action: {action}. Valid: save, load, update, attach, render, delete"}

    except BackendError as e:
        if e.status_code == 404:
            return {"error": f"Scratchpad not found for task: {task_id}"}
        log.error(f"scratchpad failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def admin(
    action: str = "stats",
    project: str | None = None,
    path: str | None = None,
    dry_run: bool = True,
    patterns: list[str] | None = None,
) -> dict:
    """Administrative operations: stats, sync, watchers, bootstrap.

    PURPOSE: System administration and diagnostics.

    ACTIONS:
    - "stats": Get memory store statistics (default)
    - "health": Check sync health between graph and vector stores
    - "sync": Repair synchronization issues (use dry_run=True first)
    - "bootstrap": Bootstrap Claude Code session (reads CLAUDE.md, recalls context)
    - "watch_start": Start file watcher for a project
    - "watch_stop": Stop file watcher for a project
    - "watch_status": Get status of file watchers
    - "trace_watch_start": Start auto-processing trace watcher
    - "trace_watch_stop": Stop trace watcher
    - "trace_watch_status": Get trace watcher status

    EXAMPLES:
        # Get stats
        admin(action="stats")

        # Check sync health
        admin(action="health", project="myproject")

        # Preview sync repair
        admin(action="sync", project="myproject", dry_run=True)

        # Bootstrap session
        admin(action="bootstrap")

        # Start file watcher
        admin(action="watch_start", path="/path/to/project")

        # Stop file watcher
        admin(action="watch_stop", path="/path/to/project")

        # Start trace watcher (auto-process sessions)
        admin(action="trace_watch_start")

        # Stop trace watcher
        admin(action="trace_watch_stop")

    Args:
        action: Operation (stats, health, sync, bootstrap, watch_start, watch_stop, watch_status, trace_watch_start, trace_watch_stop, trace_watch_status)
        project: Project ID for project-scoped operations
        path: Directory path for watcher operations
        dry_run: Preview changes without applying (for sync, default: True)
        patterns: File patterns for watch_start (default: *.py, *.ts, *.js)

    Returns:
        Action-specific response or {"error": "..."}
    """
    log.info(f"admin called (action={action})")
    try:
        client = await _get_client()

        if action == "stats":
            return await client.get_stats()

        elif action == "health":
            resolved_project_id = None
            if project:
                try:
                    resolved_project_id = _resolve_project_id(project)
                except NotBootstrappedError:
                    pass
            return await client.get_sync_health(project_id=resolved_project_id)

        elif action == "sync":
            resolved_project_id = None
            if project:
                try:
                    resolved_project_id = _resolve_project_id(project)
                except NotBootstrappedError:
                    pass
            return await client.repair_sync(
                project_id=resolved_project_id,
                dry_run=dry_run,
            )

        elif action == "bootstrap":
            # Read CLAUDE.md and recall context
            resolved_path = Path(path).resolve() if path else Path.cwd().resolve()

            # Check for CLAUDE.md
            claude_md_path = resolved_path / "CLAUDE.md"
            claude_md_content = None
            if claude_md_path.exists():
                claude_md_content = claude_md_path.read_text()

            # Get project status
            try:
                _, config = find_project_root(resolved_path)
                project_id = config.project_id

                # Search for relevant context
                try:
                    context = await client.recall(
                        query="project context session start",
                        project=project_id,
                        mode="fast",
                        limit=5,
                    )
                except Exception:
                    context = None

                return {
                    "status": "bootstrapped",
                    "project_id": project_id,
                    "project_name": config.project_name,
                    "claude_md_found": claude_md_content is not None,
                    "context_recalled": context is not None,
                    "emphatic_directive": (
                        "YOU MUST USE SimpleMem at these critical moments:\n"
                        "1. SESSION START: Search before you work\n"
                        "2. ERRORS: Search for past solutions\n"
                        "3. DECISIONS: Store with rationale\n"
                        "4. SESSION END: Capture learnings"
                    ),
                }
            except NotBootstrappedError as e:
                return {
                    "status": "not_bootstrapped",
                    **e.to_dict(),
                    "emphatic_directive": "Run project(action='bootstrap', project_name='...') first!",
                }

        elif action == "watch_start":
            if not path:
                return {"error": "path is required for action='watch_start'"}
            resolved_path = Path(path).resolve()

            # Try to resolve project_id from config
            resolved_project_id = None
            try:
                resolved_project_id = _resolve_project_id(project, path=str(resolved_path), require_bootstrap=False)
            except Exception:
                pass  # Watcher can still work without explicit project_id

            # Initialize watchers (triggers auto-resume on first use)
            watcher_manager, _ = await _ensure_watchers_initialized()
            default_patterns = patterns or ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"]

            result = watcher_manager.start_watching(
                project_root=str(resolved_path),
                patterns=default_patterns,
                project_id=resolved_project_id,
            )

            # Update backend status
            watcher_status = watcher_manager.get_status()
            try:
                await client.update_code_index_status(
                    status="watching",
                    watchers=watcher_status.get("watching_count", 0),
                    projects_watching=[p["project_root"] for p in watcher_status.get("projects", [])],
                )
            except Exception:
                pass

            return {
                "status": result.get("status", "started"),
                "project_root": str(resolved_path),
                "project_id": resolved_project_id,
                "patterns": default_patterns,
            }

        elif action == "watch_stop":
            if not path:
                return {"error": "path is required for action='watch_stop'"}
            resolved_path = Path(path).resolve()

            watcher_manager, _ = await _ensure_watchers_initialized()
            watcher_manager.stop_watching(str(resolved_path))

            # Update backend status
            watcher_status = watcher_manager.get_status()
            watching_count = watcher_status.get("watching_count", 0)
            try:
                await client.update_code_index_status(
                    status="idle" if watching_count == 0 else "watching",
                    watchers=watching_count,
                    projects_watching=[p["project_root"] for p in watcher_status.get("projects", [])],
                )
            except Exception:
                pass

            return {
                "status": "stopped",
                "project_root": str(resolved_path),
            }

        elif action == "watch_status":
            watcher_manager, _ = await _ensure_watchers_initialized()
            watcher_status = watcher_manager.get_status()
            return {
                "active_watchers": watcher_status.get("watching_count", 0),
                "projects": [p["project_root"] for p in watcher_status.get("projects", [])],
            }

        elif action == "trace_watch_start":
            _, trace_watcher = await _ensure_watchers_initialized()
            result = trace_watcher.start_watching()
            return result

        elif action == "trace_watch_stop":
            _, trace_watcher = await _ensure_watchers_initialized()
            result = trace_watcher.stop_watching()
            return result

        elif action == "trace_watch_status":
            _, trace_watcher = await _ensure_watchers_initialized()
            return trace_watcher.get_status()

        else:
            return {"error": f"Unknown action: {action}. Valid: stats, health, sync, bootstrap, watch_start, watch_stop, watch_status, trace_watch_start, trace_watch_stop, trace_watch_status"}

    except BackendError as e:
        log.error(f"admin failed: {e}")
        return {"error": e.detail}


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


async def cleanup():
    """Cleanup resources on shutdown."""
    global _client, _watcher_manager, _trace_watcher_manager

    # Stop all file watchers
    if _watcher_manager is not None:
        try:
            _watcher_manager.stop_all()
            log.info("Stopped all file watchers")
        except Exception as e:
            log.warning(f"Error stopping watchers: {e}")
        _watcher_manager = None

    # Stop trace watcher
    if _trace_watcher_manager is not None:
        try:
            _trace_watcher_manager.stop_watching()
            log.info("Stopped trace watcher")
        except Exception as e:
            log.warning(f"Error stopping trace watcher: {e}")
        _trace_watcher_manager = None

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
        log.info("Starting SimpleMem MCP server (10-tool consolidated API)...")
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
