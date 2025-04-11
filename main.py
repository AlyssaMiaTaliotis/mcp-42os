# main.py

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

# === Persistent storage config ===
MEMORY_FILE = "memory_store.json"

# In-memory store for demo
memory_store = []

def save_to_disk():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_store, f, indent=2)

def load_from_disk():
    global memory_store
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory_store = json.load(f)

# === Embedding + FAISS Setup ===
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
dimension = 384  # for this model
index = faiss.IndexFlatL2(dimension)
vector_store_ids = []  # maps FAISS index -> memory_store index

def load_embeddings():
    for i, m in enumerate(memory_store):
        embedding = embedding_model.encode([m["content"]])
        index.add(np.array(embedding).astype("float32"))
        vector_store_ids.append(i)

# === MCP setup ===
mcp = FastMCP("memory-layer")

@mcp.tool()
async def store_memory(content: str, tags: list[str] = [], source: str = "unknown") -> str:
    """
    Store a memory with optional tags and source.
    """
    memory = {
        "content": content,
        "tags": tags,
        "source": source,
        "timestamp": datetime.datetime.now().isoformat()
    }
    memory_store.append(memory)

    # Embed + store in FAISS
    embedding = embedding_model.encode([content])
    index.add(np.array(embedding).astype("float32"))
    vector_store_ids.append(len(memory_store) - 1)

    save_to_disk()
    return f"Stored memory: {content[:50]}..."

@mcp.tool()
async def query_memory(prompt: str, top_k: int = 3, min_score: float = 0.0) -> list:
    """
    Semantic memory search using FAISS. Returns top_k matches above min_score.
    """
    if len(memory_store) == 0:
        return ["No memory available."]
    if index.ntotal == 0:
        return ["Memory index is empty."]

    embedding = embedding_model.encode([prompt])
    distances, indices = index.search(np.array(embedding).astype("float32"), top_k)

    results = []
    for score, i in zip(distances[0], indices[0]):
        if i < len(memory_store):
            similarity = 1 / (1 + score)  # L2 to similarity score
            if similarity >= min_score:
                memory = memory_store[vector_store_ids[i]]
                results.append(f"{memory['content']} (score: {similarity:.2f})")

    return results if results else ["No relevant memories found."]

@mcp.tool()
async def list_memories() -> list[str]:
    """List all stored memory summaries"""
    return [f"{i+1}. {m['content'][:50]}..." for i, m in enumerate(memory_store)]

@mcp.tool()
async def delete_memory(index: int) -> str:
    """Deletes a memory entry by its index in the list."""
    # Note: FAISS is not updated in this prototype
    try:
        deleted = memory_store.pop(index)
        save_to_disk()
        return f"Deleted memory: '{deleted['content'][:50]}...'"
    except IndexError:
        return f"No memory found at index {index}."

@mcp.tool()
async def clear_memory() -> str:
    """Clears all stored memories"""
    memory_store.clear()
    # Note: FAISS is not cleared in this prototype
    save_to_disk()
    return "Memory store cleared."

# === Server App ===
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
    load_from_disk()
    load_embeddings()
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
