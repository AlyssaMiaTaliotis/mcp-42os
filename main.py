# main.py
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
import datetime

# Init FastMCP
mcp = FastMCP("memory-layer")

# In-memory store for demo
memory_store = []

@mcp.tool()
async def store_memory(content: str, tags: list[str] = [], source: str = "unknown") -> str:
    """Stores a piece of memory with optional tags and metadata"""
    memory = {
        "content": content,
        "tags": tags,
        "source": source,
        "timestamp": datetime.datetime.now().isoformat()
    }
    memory_store.append(memory)
    return f"Stored memory: {content[:50]}..."

@mcp.tool()
async def query_memory(tag: str = "") -> list[dict]:
    """Returns memory entries that contain the given tag"""
    return [m for m in memory_store if tag in m["tags"]]

@mcp.tool()
async def list_memories() -> list[str]:
    """List all stored memory summaries"""
    return [f"{i+1}. {m['content'][:50]}..." for i, m in enumerate(memory_store)]

@mcp.tool()
async def delete_memory(index: int) -> str:
    """Deletes a memory entry by its index in the list."""
    try:
        deleted = memory_store.pop(index)
        return f"Deleted memory: '{deleted['content'][:50]}...'"
    except IndexError:
        return f"No memory found at index {index}."

@mcp.tool()
async def clear_memory() -> str:
    """Clears all stored memories"""
    memory_store.clear()
    return "Memory store cleared."

# App
def create_app():
    sse = SseServerTransport("/messages/")
    async def handle_sse(request: Request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read, write):
            await mcp._mcp_server.run(read, write, mcp._mcp_server.create_initialization_options())
    return Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ])

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
