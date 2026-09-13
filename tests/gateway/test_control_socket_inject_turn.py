"""inject-turn control-socket verb — an MCP bridge injects a user turn into an
existing session through the gateway's own inbound path.

Contract under test: exact session_key lookup (unknown keys rejected, never
created), the [CLAUDE] prefix + internal=True on the injected event, the
over-length cap refusing rather than truncating, key/origin mismatch rejecting,
and the verb never posting to the platform itself (only the adapter's inbound
handle_message is touched).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.control_socket import GatewayControlServer
from gateway.run import INJECT_TURN_PREFIX, _INJECT_TURN_MAX_CHARS, _make_inject_turn_handler
from gateway.session import SessionSource, build_session_key

SESSION_KEY = "agent:main:telegram:dm:123456"


class FakeAdapter:
    """Push-capable adapter double: records the admitted event and mimics the
    ``_gateway_accepted = True`` receipt the real adapter stamps on admission.
    ``send`` raising proves the verb never posts to the platform itself."""

    supports_async_delivery = True

    def __init__(self):
        self.events = []

    async def handle_message(self, event):
        self.events.append(event)
        event._gateway_accepted = True

    async def send(self, *args, **kwargs):
        raise AssertionError("inject-turn must not post to the platform")


class FakeStore:
    def __init__(self, entries):
        self._entries = entries

    def lookup_by_session_key(self, session_key):
        return self._entries.get(session_key)


def make_runner(tmp_path, origin, adapter=None, entries=None):
    runner = MagicMock()
    runner.adapters = {}
    runner.session_store = FakeStore(
        entries if entries is not None else {SESSION_KEY: SimpleNamespace(origin=origin, session_id="sess-1")})
    runner._adapter_for_source = lambda source: adapter if adapter is not None else (
        runner.__dict__.setdefault("_adapter", FakeAdapter()))
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    return runner


def origin(chat_id="123456"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
                         user_id=chat_id, user_name="Mark", chat_name="Mark")


def call_handler(runner, params, loop_holder):
    """Run the handler the way the socket does: off-loop (executor thread), with
    the captured main loop; returns the result dict."""

    async def scenario():
        loop = asyncio.get_running_loop()
        loop_holder["loop"] = loop
        handler = _make_inject_turn_handler(runner, loop)
        return await loop.run_in_executor(None, lambda: handler(params))

    return asyncio.run(scenario())


def test_injected_event_reaches_adapter_with_prefix_and_internal(tmp_path):
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "check the tests"}, {})
    assert result == {"ok": True, "accepted": True, "session_key": SESSION_KEY}
    assert len(adapter.events) == 1
    event = adapter.events[0]
    assert event.text == "[CLAUDE] check the tests"
    assert event.internal is True
    assert event._gateway_accepted is True
    # The event's source must re-derive the exact requested key.
    assert build_session_key(event.source) == SESSION_KEY
    assert event.source.profile is None  # default namespace stays byte-identical


def test_source_carries_transport_adapter_ref(tmp_path):
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    call_handler(runner, {"session_key": SESSION_KEY, "text": "hi"}, {})
    ref = getattr(adapter.events[0].source, "_transport_adapter_ref", None)
    assert ref is not None and ref() is adapter


def test_unknown_session_key_rejected_nothing_injected(tmp_path):
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": "agent:main:telegram:dm:999", "text": "hi"}, {})
    assert result["ok"] is False
    assert "unknown session_key" in result["error"]
    assert adapter.events == []


def test_over_length_refused_not_truncated(tmp_path):
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    long_text = "x" * (_INJECT_TURN_MAX_CHARS + 1)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": long_text}, {})
    assert result["ok"] is False
    assert "text too long" in result["error"] and str(_INJECT_TURN_MAX_CHARS) in result["error"]
    assert adapter.events == []


def test_at_cap_accepted(tmp_path):
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "x" * _INJECT_TURN_MAX_CHARS}, {})
    assert result["ok"] is True
    assert len(adapter.events[0].text) == _INJECT_TURN_MAX_CHARS + len("[CLAUDE] ")


def test_missing_or_blank_fields_rejected(tmp_path):
    runner = make_runner(tmp_path, origin(), adapter=FakeAdapter())
    assert call_handler(runner, {"session_key": "", "text": "hi"}, {})["ok"] is False
    assert call_handler(runner, {"session_key": SESSION_KEY, "text": "   "}, {})["ok"] is False
    assert call_handler(runner, {}, {})["ok"] is False


def test_key_origin_mismatch_rejected(tmp_path):
    # The row's origin derives chat_id 654321 while the requested key names 123456:
    # never nearest-match — reject.
    bad = {SESSION_KEY: SimpleNamespace(origin=origin("654321"), session_id="sess-1")}
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, None, adapter=adapter, entries=bad)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "hi"}, {})
    assert result["ok"] is False
    assert "session key mismatch" in result["error"]
    assert adapter.events == []


def test_no_live_adapter_rejected(tmp_path):
    runner = make_runner(tmp_path, origin())
    runner._adapter_for_source = lambda source: None
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "hi"}, {})
    assert result["ok"] is False
    assert "no live adapter" in result["error"]


def test_adapter_not_admitting_reports_error(tmp_path):
    class RejectingAdapter(FakeAdapter):
        async def handle_message(self, event):
            event._gateway_accepted = False  # simulate a closed gate

    runner = make_runner(tmp_path, origin(), adapter=RejectingAdapter())
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "hi"}, {})
    assert result["ok"] is False
    assert "did not admit" in result["error"]


def test_group_key_with_profile_namespace(tmp_path):
    key = "agent:coder:discord:group:789:456"
    src = SessionSource(platform=Platform.DISCORD, chat_id="789", chat_type="group",
                        user_id="456", profile="coder")
    assert build_session_key(src, profile="coder") == key
    adapter = FakeAdapter()
    runner = make_runner(tmp_path, src, adapter=adapter,
                         entries={key: SimpleNamespace(origin=src, session_id="sess-2")})
    runner._adapter_for_source = lambda source: adapter
    result = call_handler(runner, {"session_key": key, "text": "standup ready"}, {})
    assert result["ok"] is True
    assert adapter.events[0].source.profile == "coder"


def test_verb_dispatch_over_control_socket_wire(tmp_path):
    """Full wire shape: request line with params -> one-arg handler sees them."""
    seen = []

    def handler(params):
        seen.append(params)
        return {"ok": True, "accepted": True}

    server = GatewayControlServer(home=tmp_path, verb_handlers={"inject-turn": handler})
    raw = json.dumps({"verb": "inject-turn", "id": 3,
                      "params": {"session_key": SESSION_KEY, "text": "hi"}}).encode()
    response = json.loads(server.handle_request_line(raw).decode())
    assert response["ok"] is True and response["id"] == 3
    assert seen == [{"session_key": SESSION_KEY, "text": "hi"}]
    assert "inject-turn" in json.loads(
        server.handle_request_line(json.dumps({"verb": "nope"}).encode()).decode())["supported_verbs"]


def test_zero_arg_verbs_keep_working(tmp_path):
    """Back-compat: existing zero-arg handlers receive no params."""
    server = GatewayControlServer(home=tmp_path, verb_handlers={"ping": lambda: {"pong": True}})
    response = json.loads(server.handle_request_line(json.dumps({"verb": "ping"}).encode().rstrip(b"\n")).decode())
    assert response["result"] == {"pong": True}


def test_timeout_never_cancels_the_injection(tmp_path, monkeypatch):
    """Regression (review run 721): the old timeout branch called future.cancel(),
    which could abort the injection at its next await while the result still
    claimed acceptance. The fixed branch must NOT cancel: a timeout answers
    pending=True AND the injection still reaches the adapter."""
    import gateway.run as run_mod
    monkeypatch.setattr(run_mod, "_INJECT_TURN_TIMEOUT_SECONDS", 0.1)

    class SlowAdapter(FakeAdapter):
        async def handle_message(self, event):
            await asyncio.sleep(0.5)  # outlasts the monkeypatched timeout
            event._gateway_accepted = True
            self.events.append(event)

    adapter = SlowAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)

    async def scenario():
        loop = asyncio.get_running_loop()
        handler = _make_inject_turn_handler(runner, loop)
        result = await loop.run_in_executor(
            None, lambda: handler({"session_key": SESSION_KEY, "text": "slow turn"}))
        # Let the still-scheduled coroutine finish despite the timeout answer.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if adapter.events:
                break
        return result

    result = asyncio.run(scenario())
    # Timeout must not have cancelled the injection: pending answer AND real admission.
    assert result == {"ok": True, "accepted": True, "session_key": SESSION_KEY, "pending": True}
    assert len(adapter.events) == 1
    assert adapter.events[0].text == "[CLAUDE] slow turn"
    assert adapter.events[0].internal is True


def test_real_socket_roundtrip_injects_turn(tmp_path):
    """End-to-end over a REAL unix socket: MCP-side request line -> params
    dispatch -> handler -> loop marshal -> adapter admission -> response."""
    import sys
    if sys.platform == "win32":
        pytest.skip("unix socket transport")
    from gateway.control_socket import _query_unix_socket

    adapter = FakeAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)

    async def scenario():
        loop = asyncio.get_running_loop()
        server = GatewayControlServer(
            home=tmp_path,
            verb_handlers={"inject-turn": _make_inject_turn_handler(runner, loop)})
        assert await server.start()
        try:
            request = json.dumps({"verb": "inject-turn", "id": 1, "protocol": 1,
                                  "params": {"session_key": SESSION_KEY,
                                             "text": "roundtrip check"}}).encode() + b"\n"
            raw = await loop.run_in_executor(None, lambda: _query_unix_socket(tmp_path, request, 5.0))
            assert raw is not None, "socket answered nothing"
            return json.loads(raw.decode())
        finally:
            await server.stop()

    response = asyncio.run(scenario())
    assert response["ok"] is True
    assert response["result"] == {"ok": True, "accepted": True, "session_key": SESSION_KEY}
    assert adapter.events[0].text == "[CLAUDE] roundtrip check"
    assert adapter.events[0].internal is True
