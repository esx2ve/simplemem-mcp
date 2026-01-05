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
from simplemem_mcp.projects_utils import (
    get_project_id as generate_project_id,
    infer_project_from_session_path,
    parse_project_id,
    extract_project_name,
)
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
    """Resolve project_id using hybrid identification strategies.

    Uses hierarchical ID generation:
    1. Git remote URL (most stable, cross-machine)
    2. Config file UUID (.simplemem.json)
    3. Content hash of project markers
    4. Resolved absolute path (fallback)

    Args:
        project_id: Optional explicit project_id (if prefixed, used as-is)

    Returns:
        Project ID with prefix (git:, uuid:, hash:, path:) or None
    """
    if project_id:
        # Check if already has a prefix (properly formatted ID)
        if any(project_id.startswith(p) for p in ["git:", "uuid:", "hash:", "path:"]):
            return project_id
        # Treat as path and generate proper hierarchical ID
        return generate_project_id(project_id)

    # Auto-resolve from current working directory
    try:
        cwd = Path.cwd().resolve()
        project_id = generate_project_id(cwd)
        log.debug(f"Auto-resolved project_id from cwd: {project_id}")
        return project_id
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

    PURPOSE: Persist insights, decisions, patterns, and learnings to the memory graph
    for retrieval in future sessions. This is the PRIMARY tool for building
    cross-session knowledge that compounds over time.

    WHEN TO USE:
    - After solving a bug: Store the root cause and fix
    - After making architectural decisions: Store the decision with rationale
    - After discovering patterns: Store reusable approaches
    - Before session ends: Store key learnings and context
    - When user says "remember this" or "save for later"

    WHEN NOT TO USE:
    - For transient conversation context (use working memory instead)
    - For large code blocks (use code indexing instead)
    - For raw session logs (use process_trace instead)

    MEMORY TYPES (use the right type for better retrieval):
    - "fact": Project-specific facts, configurations, conventions
    - "lesson_learned": Debugging insights, gotchas, what worked/didn't work
    - "decision": Architectural choices with rationale and rejected alternatives
    - "pattern": Reusable code patterns, approaches, templates
    - "session_summary": End-of-session comprehensive summary (auto-generated)
    - "chunk_summary": Activity chunk summary (auto-generated by process_trace)

    EXAMPLES:
        # After fixing a bug
        store_memory(
            text="Database connection timeout was caused by missing connection pool limits. Fixed by setting max_connections=20 in config.py:45",
            type="lesson_learned",
            project_id="git:github.com/user/myproject"
        )

        # After architectural decision
        store_memory(
            text="Decision: Use Redis for session storage. Reason: Need distributed sessions for horizontal scaling. Rejected: In-memory (not distributed), PostgreSQL (too slow for session lookups)",
            type="decision"
        )

        # Linking memories
        store_memory(
            text="Pattern: Always use context managers for DB connections",
            type="pattern",
            relations=[{"target_id": "<uuid-of-related-memory>", "type": "supports"}]
        )

    Args:
        text: The content to store. Be specific and actionable - include file paths,
              line numbers, error messages, and concrete solutions. Future sessions
              will rely on this text for retrieval.
        type: Memory type. Use "lesson_learned" for debugging, "decision" for
              architectural choices, "pattern" for reusable approaches, "fact" for
              project-specific information.
        source: Origin of the memory. "user" for direct input, "claude_trace" for
                auto-extracted from sessions, "extracted" for LLM-derived insights.
        relations: Link to related memories. Each dict needs {target_id: str, type: str}.
                   Relation types: "contains", "child_of", "supports", "follows", "similar"
        project_id: Project isolation. Auto-inferred from cwd if not specified.
                    ALWAYS use project_id to prevent cross-project data leakage.

    Returns:
        On success: {"uuid": "<memory-uuid>"} - Save this UUID for creating relations
        On error: {"error": "<error-message>"}
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

    PURPOSE: Find relevant memories from past sessions using semantic search.
    This is your PRIMARY retrieval tool - use it BEFORE starting work to
    check if similar problems have been solved before.

    WHEN TO USE (CRITICAL - use at these moments):
    - SESSION START: Always search for context before starting complex tasks
    - ENCOUNTERING ERRORS: Search for past solutions to similar errors
    - BEFORE IMPLEMENTING: Check if patterns exist for this type of work
    - DEBUGGING: Search for related debugging sessions and fixes
    - WHEN STUCK: Past sessions may have encountered the same blockers

    WHEN NOT TO USE:
    - For code search (use search_code instead)
    - When you need synthesized answers (use ask_memories instead)
    - For multi-hop reasoning chains (use reason_memories instead)

    SEARCH STRATEGY:
    1. Uses vector similarity to find semantically similar memories
    2. Expands results via graph relationships (if use_graph=True)
    3. Returns ranked results with relevance scores

    EXAMPLES:
        # At session start - recall project context
        search_memories(
            query="project architecture and key patterns",
            type_filter="decision",
            project_id="git:github.com/user/myproject"
        )

        # When hitting an error
        search_memories(
            query="TypeError NoneType connection pool database",
            type_filter="lesson_learned"
        )

        # Before implementing a feature
        search_memories(query="authentication JWT implementation patterns")

        # Finding debugging history
        search_memories(
            query="memory leak performance issues",
            limit=20,
            use_graph=True
        )

    Args:
        query: Natural language description of what you're looking for.
               Be specific - include error messages, file names, concepts.
               Examples: "database connection timeout fix", "React hooks patterns",
               "authentication implementation decision"
        limit: Maximum results (default: 10). Increase for broad exploration,
               decrease for focused queries.
        use_graph: Expand results via graph relationships (default: True).
                   Set False for faster, more focused results.
        type_filter: Filter by memory type. Values: "fact", "lesson_learned",
                     "decision", "pattern", "session_summary", "chunk_summary"
        project_id: Project isolation. Auto-inferred from cwd if not specified.
                    CRITICAL: Always use to prevent retrieving unrelated memories.

    Returns:
        On success: {"results": [
            {"uuid": "...", "content": "...", "type": "...", "score": 0.85, "created_at": "..."},
            ...
        ]}
        On error: {"error": "...", "results": []}
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
    """Create a relationship between two memories in the knowledge graph.

    PURPOSE: Build connections between memories to enable graph traversal
    and multi-hop reasoning. Relationships make memories discoverable via
    related content, not just semantic similarity.

    WHEN TO USE:
    - After storing related memories (e.g., problem → solution)
    - When creating hierarchical knowledge (e.g., summary → details)
    - To link decisions to supporting evidence
    - To connect debugging sessions to the fixes they produced

    RELATION TYPES:
    - "contains": Parent contains children (session_summary → chunk_summaries)
    - "child_of": Inverse of contains (detail → summary)
    - "supports": Evidence supports a conclusion (facts → decision)
    - "follows": Temporal/logical sequence (problem → investigation → solution)
    - "similar": Related but not hierarchical (alternative approaches)
    - "relates": Generic relationship (default)

    EXAMPLES:
        # Link a lesson learned to the decision it supports
        relate_memories(
            from_id="<lesson-uuid>",
            to_id="<decision-uuid>",
            relation_type="supports"
        )

        # Create a sequence of debugging steps
        relate_memories(
            from_id="<error-memory-uuid>",
            to_id="<fix-memory-uuid>",
            relation_type="follows"
        )

    Args:
        from_id: Source memory UUID (the one doing the relating)
        to_id: Target memory UUID (the one being related to)
        relation_type: Type of relationship. Choose the most specific type
                       that describes the connection.

    Returns:
        On success: {"success": true}
        On error: {"error": "...", "success": false}
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

    PURPOSE: Get synthesized, coherent answers from your memory graph instead
    of raw search results. The LLM reads relevant memories and produces a
    grounded answer with citations. Use this when you need UNDERSTANDING,
    not just retrieval.

    WHEN TO USE:
    - When you need a synthesized answer, not raw memories
    - For "how did we..." or "what was the solution to..." questions
    - When search results need interpretation and consolidation
    - For getting actionable guidance from past experience
    - When you want cross-session pattern insights

    WHEN NOT TO USE:
    - For simple retrieval (use search_memories instead - faster)
    - When you need to see all raw results (use search_memories)
    - For code search (use search_code)
    - When exploring without a specific question (use search_memories)

    HOW IT WORKS:
    1. Retrieves relevant memories via graph traversal
    2. Synthesizes an answer using an LLM grounded in the evidence
    3. Includes citations [1], [2] referencing specific memories
    4. Highlights cross-session insights (patterns across sessions)

    EXAMPLES:
        # Get a synthesized solution to a past problem
        ask_memories(
            query="What was the solution to the database connection timeout issue?",
            project_id="git:github.com/user/myproject"
        )

        # Understand how something was implemented
        ask_memories(query="How did we implement the authentication feature?")

        # Get patterns from debugging history
        ask_memories(
            query="What patterns have worked for debugging async code?",
            max_memories=12  # Include more context for pattern discovery
        )

        # Understand architectural decisions
        ask_memories(query="Why did we choose PostgreSQL over MongoDB?")

    Args:
        query: Natural language question. Be specific about what you want to know.
               Good: "What caused the memory leak in the image processing pipeline?"
               Bad: "memory issues" (too vague)
        max_memories: Maximum memories to include in LLM context (default: 8).
                      Increase for complex questions needing more evidence.
        max_hops: Graph traversal depth (default: 2). Higher values find more
                  distant but potentially relevant memories.
        project_id: Project isolation. Auto-inferred from cwd if not specified.

    Returns:
        {
            "answer": "Synthesized answer with [1][2] citations...",
            "memories_used": 5,
            "cross_session_insights": ["Pattern found across 3 sessions..."],
            "confidence": "high|medium|low",
            "sources": [{"uuid": "...", "content": "...", "citation": 1}, ...]
        }
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
    """Multi-hop reasoning over memory graph for complex questions.

    PURPOSE: Perform deep reasoning that requires following chains of evidence
    through the memory graph. Use this for questions that need connecting
    multiple memories to form conclusions.

    WHEN TO USE:
    - Questions requiring connecting multiple pieces of evidence
    - Finding evolution/history of features or decisions
    - Discovering patterns across different debugging sessions
    - Tracing cause-effect chains through memory graph
    - When ask_memories gives incomplete answers needing more depth

    WHEN NOT TO USE:
    - Simple fact lookup (use search_memories)
    - Direct questions with obvious answers (use ask_memories)
    - Code search (use search_code)

    HOW IT WORKS:
    1. Vector search finds entry points into the graph
    2. Graph traversal follows relationships to related memories
    3. Semantic path scoring evaluates evidence chains
    4. Returns conclusions with proof chains showing reasoning path

    EXAMPLES:
        # Trace the evolution of a feature
        reason_memories(
            query="How did the authentication feature evolve?",
            max_hops=3  # Allow deeper traversal for history
        )

        # Find debugging patterns
        reason_memories(query="What debugging patterns work for database issues?")

        # Trace cause-effect
        reason_memories(
            query="What led to the decision to migrate to PostgreSQL?",
            project_id="git:github.com/user/myproject"
        )

        # Connect related solutions
        reason_memories(query="Find solutions related to connection timeouts")

    Args:
        query: Complex question requiring multi-hop reasoning. Be specific
               about what connections you're looking for.
        max_hops: Maximum graph traversal depth (1-3). Higher = more connections
                  but slower. Use 3 for historical/evolutionary questions.
        min_score: Minimum relevance score threshold (0.0-1.0). Lower values
                   include more distant but potentially relevant memories.
        project_id: Project isolation. Auto-inferred from cwd if not specified.

    Returns:
        {
            "conclusions": [
                {
                    "uuid": "...",
                    "content": "...",
                    "type": "lesson_learned",
                    "score": 0.85,
                    "proof_chain": ["memory-1 → memory-2 → memory-3"],
                    "hops": 2
                },
                ...
            ]
        }
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

    PURPOSE: Obtain the stable project identifier for use with memory tools.
    The project_id ensures memories are isolated per-project and remain
    consistent even if the project is moved or cloned elsewhere.

    WHEN TO USE:
    - At session start to get the current project's ID
    - When you need to explicitly pass project_id to other tools
    - To verify which project you're working in
    - When setting up project isolation for memory operations

    PROJECT_ID HIERARCHY (most stable first):
    1. git:github.com/user/repo - From git remote (survives moves/clones)
    2. uuid:550e8400-... - From .simplemem.json config (explicit control)
    3. hash:a1b2c3d4... - From project marker files (package.json, etc.)
    4. path:/Users/... - Absolute path fallback (least stable)

    EXAMPLES:
        # Get ID for current working directory
        get_project_id()
        # Returns: {"project_id": "git:github.com/user/myproject", "id_type": "git", ...}

        # Get ID for a specific path
        get_project_id("~/code/another-project")
        # Returns: {"project_id": "hash:a1b2c3d4e5f6", "id_type": "hash", ...}

        # Use the returned project_id with memory tools
        result = get_project_id()
        search_memories(query="...", project_id=result["project_id"])

    Args:
        path: Path to resolve. Defaults to current working directory.
              Can be absolute or relative (will be resolved).

    Returns:
        On success: {
            "project_id": "git:github.com/user/repo",  # Use this with other tools
            "id_type": "git",                          # How the ID was determined
            "id_value": "github.com/user/repo",        # The ID without prefix
            "project_name": "repo",                    # Human-readable name
            "path": "/Users/.../repo",                 # Resolved absolute path
            "message": "Project identified via git..."
        }
        On error: {"error": "..."}
    """
    try:
        if path:
            resolved_path = Path(path).expanduser().resolve()
        else:
            resolved_path = Path.cwd().resolve()

        project_id = generate_project_id(resolved_path)
        id_type, id_value = parse_project_id(project_id)
        project_name = extract_project_name(project_id)

        log.info(f"get_project_id called: {project_id} (type={id_type})")
        return {
            "project_id": project_id,
            "id_type": id_type,
            "id_value": id_value,
            "project_name": project_name,
            "path": str(resolved_path),
            "message": f"Project identified via {id_type}. Use this project_id with memory tools for isolation.",
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
    """Index a Claude Code session trace into searchable memory summaries.

    PURPOSE: Transform raw Claude Code session traces into structured,
    searchable memories. Creates hierarchical summaries that capture
    what happened in the session for future retrieval.

    WHEN TO USE:
    - After completing significant work in a session
    - When user requests indexing of their sessions
    - To build memory from historical sessions (use discover_sessions first)
    - As part of regular maintenance to keep memory up-to-date

    WHEN NOT TO USE:
    - For sessions still in progress (wait until complete)
    - If session was trivial with no useful learnings
    - Repeatedly on same session (it's idempotent but wastes resources)

    HOW IT WORKS:
    1. Reads the session trace file from ~/.claude/projects/
    2. Splits into logical chunks based on activity
    3. Generates summaries using fast LLM (gemini-flash-lite)
    4. Creates memories: 1 session_summary + 5-15 chunk_summaries
    5. Links memories with relationships for graph traversal
    6. Auto-extracts project_id from trace file path

    WORKFLOW:
        # Discover what sessions exist
        discover_sessions(days_back=7)

        # Index a specific session
        result = process_trace(session_id="abc-123-def")

        # Check progress for background jobs
        job_status(job_id=result["job_id"])

    Args:
        session_id: UUID of the Claude Code session to index.
                    Find session IDs using discover_sessions().
        background: Run in background (default: True). Large sessions
                    may take minutes; background prevents timeout.
                    Use job_status() to check progress.

    Returns:
        If background=True: {"job_id": "...", "status": "submitted"}
        If background=False: {
            "session_summary_id": "...",
            "chunk_count": 8,
            "message_count": 156,
            "project_id": "git:github.com/user/repo"
        }
        On error: {"error": "..."}
    """
    try:
        log.info(f"process_trace called (session_id={session_id})")

        reader = await _get_reader()

        # First find the session path to extract project_id
        session_path = await asyncio.to_thread(reader.find_session_path, session_id)
        if session_path is None:
            return {"error": f"Session {session_id} not found"}

        # Infer project_id from the trace file path (Claude's encoded directory name)
        project_id = infer_project_from_session_path(session_path)
        log.info(f"Inferred project_id from session path: {project_id}")

        # Read trace content
        trace_content = await asyncio.to_thread(reader.read_trace_file_by_path, session_path)
        if trace_content is None:
            return {"error": f"Failed to read trace file for session {session_id}"}

        # Send to backend for processing with project_id
        client = await _get_client()
        return await client.process_trace(
            session_id=session_id,
            trace_content=trace_content,
            background=background,
            project_id=project_id,
        )
    except BackendError as e:
        log.error(f"process_trace failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def discover_sessions(
    days_back: int = 30,
    limit: int = 20,
    offset: int = 0,
    group_by: str | None = None,
    include_indexed: bool = True,
) -> dict:
    """Discover available Claude Code sessions for potential indexing.

    Lightweight scan that reads file metadata only (no LLM calls).
    Use this to explore historical sessions before batch indexing.

    Args:
        days_back: Only include sessions modified within this many days (default: 30)
        limit: Maximum number of sessions to return (default: 20)
        offset: Number of sessions to skip for pagination (default: 0)
        group_by: Optional grouping - "project" or "date" (default: None, flat list)
        include_indexed: Include already-indexed sessions in results (default: True)

    Returns:
        Dict containing:
        - sessions: List of session metadata (or grouped dict if group_by specified)
        - total_count: Total sessions found
        - has_more: Whether more sessions are available
        - indexed_count: How many are already indexed
        - unindexed_count: How many are not yet indexed
    """
    try:
        log.info(f"discover_sessions called (days_back={days_back}, limit={limit}, offset={offset})")

        reader = await _get_reader()
        data = await asyncio.to_thread(reader.discover_sessions, days_back, limit, offset)

        sessions = data.get("sessions", [])
        result = {
            "sessions": sessions,
            "total_count": data.get("total", len(sessions)),
            "has_more": data.get("has_more", False),
            "indexed_count": 0,
            "unindexed_count": data.get("total", len(sessions)),
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
    project_id: str | None = None,
    project_root: str | None = None,  # Deprecated: use project_id
) -> dict:
    """Search indexed code for implementations, patterns, and functionality.

    PURPOSE: Find relevant code snippets using semantic search. Unlike grep/ripgrep
    which match exact text, this finds code by MEANING - "authentication handler"
    will find login functions even if they don't contain those exact words.

    WHEN TO USE:
    - Finding implementations: "user authentication", "database connection pool"
    - Finding patterns: "error handling pattern", "retry logic"
    - Understanding structure: "API endpoints", "middleware functions"
    - Before implementing: Check if similar code exists
    - Debugging: Find code related to an error

    WHEN NOT TO USE:
    - For exact text matches (use grep/ripgrep instead)
    - For files that haven't been indexed (use index_directory first)
    - For memory/insight search (use search_memories instead)

    PREREQUISITE: The codebase must be indexed first using index_directory().
    If no results found, the codebase may not be indexed.

    EXAMPLES:
        # Find authentication code
        search_code(query="user login authentication handler")

        # Find database patterns
        search_code(
            query="connection pool database initialization",
            limit=20
        )

        # Search in specific project
        search_code(
            query="API rate limiting middleware",
            project_id="git:github.com/user/myproject"
        )

        # Find error handling
        search_code(query="exception handling retry logic")

    Args:
        query: Natural language description of code you're looking for.
               Be descriptive: "user authentication JWT token validation"
               is better than just "auth".
        limit: Maximum results (default: 10). Increase for broader search.
        project_id: Filter to specific project (preferred). Auto-inferred
                    from cwd if not specified.
        project_root: DEPRECATED - use project_id instead.

    Returns:
        On success: {
            "results": [
                {
                    "file_path": "/path/to/file.py",
                    "line_start": 45,
                    "line_end": 78,
                    "content": "def authenticate_user(...)...",
                    "score": 0.89
                },
                ...
            ]
        }
        On error: {"error": "...", "results": []}
    """
    try:
        # Resolve project_id from path if only project_root provided
        resolved_project_id = project_id
        if not resolved_project_id and project_root:
            log.warning("search_code: project_root is deprecated, use project_id instead")
            resolved_project_id = generate_project_id(project_root)
        elif not resolved_project_id:
            # Auto-infer from current working directory
            resolved_project_id = _resolve_project_id(None)

        log.info(f"search_code called (query='{query[:50]}...', project_id={resolved_project_id})")
        client = await _get_client()
        return await client.search_code(
            query=query,
            limit=limit,
            project_root=project_root,  # Still pass for backwards compat
            project_id=resolved_project_id,
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
    """Index a directory for semantic code search.

    PURPOSE: Build a searchable index of code files for semantic search.
    This enables search_code() to find implementations by meaning,
    not just exact text matches.

    WHEN TO USE:
    - At project start: Index the codebase for semantic search
    - After major changes: Re-index to include new code
    - When search_code returns no results (codebase not indexed)
    - Setting up a new project for SimpleMem

    WHEN NOT TO USE:
    - On already-indexed codebases (unless you need to refresh)
    - For temporary/generated directories
    - For node_modules, .git, or other dependency folders (auto-excluded)

    HOW IT WORKS:
    1. Scans directory for matching source files
    2. Splits files into semantic chunks (functions, classes, etc.)
    3. Generates embeddings for each chunk
    4. Stores in vector database for semantic search
    5. Associates with project_id for isolation

    DEFAULT FILE PATTERNS:
    - Python: **/*.py
    - TypeScript: **/*.ts, **/*.tsx
    - JavaScript: **/*.js, **/*.jsx

    EXAMPLES:
        # Index current project (most common)
        index_directory(path=".")

        # Index with custom patterns
        index_directory(
            path="/path/to/project",
            patterns=["**/*.py", "**/*.rs", "**/*.go"]
        )

        # Re-index without clearing (add new files only)
        index_directory(path=".", clear_existing=False)

        # Index and wait for completion
        result = index_directory(path=".", background=False)
        # Takes longer but returns stats immediately

    WORKFLOW:
        # 1. Index the codebase
        result = index_directory(path=".")

        # 2. Check progress (for background jobs)
        job_status(job_id=result["job_id"])

        # 3. Now search_code will work
        search_code(query="authentication handler")

    Args:
        path: Directory to index. Can be absolute or relative.
        patterns: Glob patterns for files to include.
                  Default: ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"]
        clear_existing: Clear existing index for this project (default: True).
                        Set False to add files incrementally.
        background: Run in background (default: True). Large codebases
                    may take minutes; background prevents timeout.

    Returns:
        If background=True: {"job_id": "...", "status": "submitted", "message": "..."}
        If background=False: {
            "files_indexed": 156,
            "chunks_created": 1247,
            "project_id": "git:github.com/user/repo"
        }
        On error: {"error": "..."}
    """
    try:
        directory = Path(path).expanduser().resolve()
        if not directory.exists():
            return {"error": f"Directory not found: {path}"}

        # Generate stable project_id
        project_id = generate_project_id(directory)
        log.info(f"index_directory called (path={path}, project_id={project_id}, background={background})")

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

        # Send to backend for indexing with both project_id and project_root
        client = await _get_client()
        result = await client.index_code(
            project_root=str(directory),
            project_id=project_id,
            files=files,
            clear_existing=clear_existing,
            background=background,
        )

        # Include project_id in response
        if isinstance(result, dict):
            result["project_id"] = project_id

        return result
    except BackendError as e:
        log.error(f"index_directory failed: {e}")
        return {"error": e.detail}


@mcp.tool()
async def code_stats(
    project_id: str | None = None,
    project_root: str | None = None,  # Deprecated: use project_id
) -> dict:
    """Get statistics about the code index.

    Args:
        project_id: Optional - filter to specific project (preferred)
        project_root: Optional - filter by path (deprecated, use project_id)

    Returns:
        Statistics including chunk count and unique files
    """
    try:
        # Resolve project_id from path if only project_root provided
        resolved_project_id = project_id
        if not resolved_project_id and project_root:
            log.warning("code_stats: project_root is deprecated, use project_id instead")
            resolved_project_id = generate_project_id(project_root)
        elif not resolved_project_id:
            # Auto-infer from current working directory
            resolved_project_id = _resolve_project_id(None)

        log.info(f"code_stats called (project_id={resolved_project_id})")
        client = await _get_client()
        return await client.code_stats(
            project_root=project_root,
            project_id=resolved_project_id,
        )
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
