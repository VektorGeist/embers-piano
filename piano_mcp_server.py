"""piano_mcp_server.py - Ember's Piano MCP server.

A Model Context Protocol server over stdio that exposes the
piano as one tool. The server is a front door only: it takes
the score the model wrote, plays it through the piano, writes
the audio file, and returns the render record unchanged. It
imposes no order, sets no defaults inside arguments, and
decides nothing.

Start it with:

    python3 piano_mcp_server.py

It adds one dependency: the official mcp Python package,
version 2.3.0.

Tool exposed:

- embers_piano_render: Render a score on the piano. Takes
  the score the model wrote, plays it through the piano,
  writes the audio file, and returns the render record.
  Parameters: score, the score artifact object with parts,
  each with a name, an optional gain, and notes with start,
  midi, vel, and dur; and name, a short file stem for this
  render. The render is written to renders/<name>.wav beside
  this server file. The piano applies no faults through this
  door: the faults argument is always the empty list.

A renderer error is returned as an MCP tool error carrying
the exception's message. This file reads no credentials and
makes no network calls.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

import music_renderer

SERVER_NAME = "embers-piano"

_RENDERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")

_TOOL = Tool(
    name="embers_piano_render",
    description=(
        "Render a score on the piano. Takes the score the model "
        "wrote, plays it through the piano, writes the audio "
        "file, and returns the render record."
    ),
    inputSchema={
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "name"],
        "properties": {
            "score": {
                "type": "object",
                "description": (
                    "The score artifact object: parts, each with "
                    "a name, an optional gain, and notes with "
                    "start, midi, vel, and dur."
                ),
            },
            "name": {
                "type": "string",
                "description": "A short file stem for this render.",
            },
        },
    },
)

server = Server(SERVER_NAME)


async def _handle_list_tools(ctx, params):
    return ListToolsResult(tools=[_TOOL])


async def _handle_call_tool(ctx, params):
    arguments = params.arguments or {}
    score = arguments.get("score")
    stem = arguments.get("name")
    try:
        os.makedirs(_RENDERS_DIR, exist_ok=True)
        out_path = os.path.join(_RENDERS_DIR, "%s.wav" % (stem,))
        record = music_renderer.renderer(score, out_path, [])
    except Exception as exc:
        return CallToolResult(
            content=[TextContent(type="text", text=str(exc))],
            isError=True,
        )
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(record, ensure_ascii=False))],
        isError=False,
    )


server.add_request_handler("tools/list", ListToolsRequest, _handle_list_tools)
server.add_request_handler("tools/call", CallToolRequestParams, _handle_call_tool)


async def _main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
