"""Fork-only guard: permissions_respond must stay off the MCP wire surface.

Upstream hermes-agent registers permissions_respond in mcp_serve._TOOL_NAMES.
That tool let any MCP bridge client answer Hermes's approval prompts, routing
around the rule that destructive/irreversible actions reach a human. The fork
removed it from the wire surface (14288db2a0, Mark's hand edit) and guarded
the removal in tests/test_mcp_serve.py — the file the sirron-update rebase
takes --ours for that commit (ADDENDUM-1-R-20260925-B item 2), which is
exactly why this guard lives in its own fork-only file: it must survive the
recorded resolution.

Asserted exactly as the fork's original test_permissions_respond_not_registered
asserted it: absent from list_tools, unreachable via call_tool.
R-20260925-B card U5 item 3 (t_86d44fa8), Mark's ruling 09/25 ~1:35pm CT.
"""

import asyncio

import pytest


def _guard_server():
    """A registered-but-unwired MCP server: registration is all this guard probes."""
    pytest.importorskip("mcp", reason="MCP SDK not installed")
    import mcp_serve

    bridge = mcp_serve.EventBridge()
    return mcp_serve.create_mcp_server(event_bridge=bridge)


def test_permissions_respond_not_registered():
    # Guard the removal: permissions_respond must not be reachable over the
    # wire. Its exposure let a bridge client answer Hermes's approval
    # prompts, routing around the human-approval rule for destructive
    # actions. The handler method stays (behaviour still covered); only
    # registration is removed.
    server = _guard_server()
    tool_names = {t.name for t in server._tool_manager.list_tools()}
    assert "permissions_respond" not in tool_names
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        raised = False
        try:
            loop.run_until_complete(
                server.call_tool("permissions_respond", {"id": "x", "decision": "deny"})
            )
        except Exception:
            raised = True
        assert raised, "permissions_respond must be unreachable over the wire"
    finally:
        asyncio.set_event_loop(None)
        loop.close()
