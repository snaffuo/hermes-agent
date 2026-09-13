"""inject-turn control-socket verb — an MCP bridge injects a user turn into an
existing session through the gateway's own inbound path.

Contract under test: exact session_key lookup (unknown keys rejected, never
created), the [CLAUDE] prefix + internal=True on the injected event, the
over-length cap refusing rather than truncating, key/origin mismatch rejecting,
and the Telegram mirror: the prefixed text is posted to the origin thread at
admission (before enqueue, so the thread reads question -> answer), only for
Telegram origins, and a mirror failure never fails the injection.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.control_socket import GatewayControlServer
from gateway.run import (INJECT_TURN_PREFIX, _INJECT_TURN_MAX_CHARS,
                         _make_inject_turn_handler)
from gateway.session import SessionSource, build_session_key

SESSION_KEY = "agent:main:telegram:dm:123456"


class FakeAdapter:
    """Push-capable adapter double: records the admitted event and mimics the
    ``_gateway_accepted = True`` receipt the real adapter stamps on admission.
    ``send`` records the mirror post in the same timeline so ordering
    (mirror-before-enqueue) is observable; it never blocks admission."""

    supports_async_delivery = True

    def __init__(self):
        self.events = []
        self.sends = []  # (chat_id, content, metadata) in call order

    async def handle_message(self, event):
        self.events.append(event)
        event._gateway_accepted = True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append((chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="m-1")


class RecordingAdapter(FakeAdapter):
    """Adapter double with ONE shared call sequence across send and
    handle_message, so mirror-before-answer ordering is asserted on the real
    in-process timeline, not inferred from two separate lists."""

    def __init__(self):
        super().__init__()
        self.calls = []  # ("mirror", content, metadata) / ("admit", event)

    async def handle_message(self, event):
        self.calls.append(("admit", event))
        await super().handle_message(event)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("mirror", content, metadata))
        return await super().send(chat_id, content, reply_to, metadata)


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
    # AC-3: a Discord-keyed session gets no mirror post (and no error).
    assert adapter.sends == []


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


# ── Telegram mirror at admission (t_3a8e6158) ──────────────────────────────


def test_mirror_posts_prefixed_text_to_origin_dm(tmp_path):
    """AC-1 (DM): a Telegram-DM injection produces BOTH the enqueued event AND
    a mirror post of the identical prefixed text to the origin chat, with no
    thread metadata (DM-keyed origin mirrors to the DM chat)."""
    adapter = RecordingAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "what is 2+2"}, {})
    assert result == {"ok": True, "accepted": True, "session_key": SESSION_KEY}
    admits = [c for c in adapter.calls if c[0] == "admit"]
    mirrors = [c for c in adapter.calls if c[0] == "mirror"]
    assert len(admits) == 1 and len(mirrors) == 1
    # The mirror text is byte-identical to the enqueued turn's text.
    assert mirrors[0][1] == admits[0][1].text == "[CLAUDE] what is 2+2"
    assert mirrors[0][2] is None  # DM: no thread_id


def test_mirror_posts_to_forum_thread_with_thread_id(tmp_path):
    """AC-1 (group topic): a Telegram forum-topic session mirrors into the SAME
    thread the session is keyed to, via metadata.thread_id."""
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="-100123", chat_type="group",
                        thread_id="9880", user_id="555", chat_name="Ops")
    key = build_session_key(src)
    adapter = RecordingAdapter()
    runner = make_runner(tmp_path, src, adapter=adapter,
                         entries={key: SimpleNamespace(origin=src, session_id="sess-9880")})
    result = call_handler(runner, {"session_key": key, "text": "deploy status?"}, {})
    assert result["ok"] is True
    mirrors = [c for c in adapter.calls if c[0] == "mirror"]
    assert len(mirrors) == 1
    assert mirrors[0][1] == "[CLAUDE] deploy status?"
    assert mirrors[0][2] == {"thread_id": "9880"}


def test_mirror_is_a_single_post_per_injection(tmp_path):
    """One injection => exactly one mirror post (no duplicate, no echo loop)."""
    adapter = RecordingAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    call_handler(runner, {"session_key": SESSION_KEY, "text": "count me"}, {})
    assert sum(1 for c in adapter.calls if c[0] == "mirror") == 1


def test_mirror_posts_at_admission_before_answer(tmp_path):
    """AC-1 ordering: the mirrored question posts BEFORE the turn is admitted to
    handle_message (the answer is a background task downstream of admission), so
    the thread always reads question -> answer. This is the mutation-sensitive
    test: an implementation that mirrors AFTER deliver_wake fails here because
    the 'admit' call precedes the 'mirror' call."""
    adapter = RecordingAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    call_handler(runner, {"session_key": SESSION_KEY, "text": "ordered?"}, {})
    kinds = [c[0] for c in adapter.calls]
    assert kinds == ["mirror", "admit"], (
        f"mirror must precede admission (question before answer); got {kinds}")


def test_mirror_failure_is_isolated_injection_still_accepted(tmp_path):
    """A mirror post that raises must NOT fail the injection: the client still
    gets accepted:true, the turn is still enqueued, and the failure is logged."""
    class ExplodingSendAdapter(FakeAdapter):
        async def send(self, *a, **k):
            raise RuntimeError("telegram 429 flood")

    adapter = ExplodingSendAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "still land"}, {})
    # Injection succeeds despite the mirror blowing up.
    assert result == {"ok": True, "accepted": True, "session_key": SESSION_KEY}
    # And the turn really was enqueued (not skipped because the mirror failed).
    assert len(adapter.events) == 1
    assert adapter.events[0].text == "[CLAUDE] still land"


def test_mirror_non_delivery_result_is_isolated(tmp_path):
    """A send() that RETURNS failure (SendResult.success=False) also must not
    fail the injection — the mirror is an observation, not a precondition."""
    class FailingResultAdapter(FakeAdapter):
        async def send(self, *a, **k):
            return SimpleNamespace(success=False, error="not connected", retryable=True)

    adapter = FailingResultAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)
    result = call_handler(runner, {"session_key": SESSION_KEY, "text": "resilient"}, {})
    assert result["ok"] is True and result["accepted"] is True
    assert len(adapter.events) == 1


def test_mirror_send_hang_does_not_block_admission_and_is_logged(tmp_path, caplog,
                                                                 monkeypatch):
    """ISOLATION mutation-proof (t_c66019eb rework, Mark's ruling: isolation is
    absolute). A send() that hangs FOREVER must not block or fail admission:
    the injection still returns accepted:true (NOT the pending-degraded dict),
    the turn is still enqueued, and the hung send surfaces ONLY in the log via
    the mirror task's bounded timeout after admission.

    Mutation-sensitivity: against the OLD implementation that awaited
    ``_mirror_inject_turn_to_origin_thread`` on the admission path, the hung
    send stalls ``_inject()`` forever, ``future.result(_INJECT_TURN_TIMEOUT)``
    fires and degrades to ``pending:true``, and the event is NEVER enqueued —
    this test fails on the result dict and on adapter.events. It passes only
    with create_task + bounded wait_for off the admission path.
    """
    import logging

    import gateway.run as run_mod

    class HangingSendAdapter(FakeAdapter):
        async def send(self, *a, **k) -> SimpleNamespace:
            await asyncio.Event().wait()  # never returns: the hang under test
            raise AssertionError("unreachable")  # type-honest; never executed

    monkeypatch.setattr(run_mod, "_MIRROR_INJECT_TIMEOUT_SECONDS", 0.05)
    adapter = HangingSendAdapter()
    runner = make_runner(tmp_path, origin(), adapter=adapter)

    async def scenario():
        loop = asyncio.get_running_loop()
        handler = _make_inject_turn_handler(runner, loop)
        result = await loop.run_in_executor(
            None, lambda: handler({"session_key": SESSION_KEY, "text": "hung mirror"}))
        # Keep the loop alive past the mirror's bounded timeout so the
        # done-callback fires WHILE the injection has already admitted.
        await asyncio.sleep(0.15)
        return result

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = asyncio.run(scenario())

    # Admission unaffected: accepted:true with NO pending flag, turn enqueued.
    assert result == {"ok": True, "accepted": True, "session_key": SESSION_KEY}, (
        "a hung mirror send must not degrade admission to pending:true")
    assert len(adapter.events) == 1
    assert adapter.events[0].text == "[CLAUDE] hung mirror"
    # The hang is visible ONLY in the log (no exception reached the caller).
    msgs = [r.getMessage() for r in caplog.records if r.name == "gateway.run"]
    assert any("timed out" in m and SESSION_KEY in m for m in msgs), (
        f"expected the mirror timeout in the log; got {msgs}")


def test_non_telegram_origin_no_mirror_post_no_error(tmp_path):
    """AC-3: a non-Telegram-keyed session produces no mirror post and no error,
    and the mirror helper itself short-circuits before touching send()."""
    import gateway.run as run_mod
    src = SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="channel",
                        user_id="U9")
    key = build_session_key(src)
    adapter = RecordingAdapter()
    runner = make_runner(tmp_path, src, adapter=adapter,
                         entries={key: SimpleNamespace(origin=src, session_id="sess-slack")})
    result = call_handler(runner, {"session_key": key, "text": "slack turn"}, {})
    assert result["ok"] is True
    # send() must never be called for a non-Telegram origin.
    assert [c for c in adapter.calls if c[0] == "mirror"] == []
    assert len(adapter.events) == 1


def test_mirror_helper_gated_directly_on_platform(tmp_path):
    """Direct gate test: _mirror_inject_turn_to_origin_thread returns without
    calling send for any non-Telegram origin, and calls send for Telegram."""
    import asyncio as _aio
    import gateway.run as run_mod

    async def scenario():
        a = RecordingAdapter()
        await run_mod._mirror_inject_turn_to_origin_thread(
            a, SessionSource(platform=Platform.DISCORD, chat_id="x"), "k", "[CLAUDE] hi")
        discord_posts = len([c for c in a.calls if c[0] == "mirror"])
        await run_mod._mirror_inject_turn_to_origin_thread(
            a, SessionSource(platform=Platform.TELEGRAM, chat_id="x"), "k", "[CLAUDE] hi")
        tg_posts = len([c for c in a.calls if c[0] == "mirror"])
        return discord_posts, tg_posts

    discord_posts, tg_posts = _aio.run(scenario())
    assert discord_posts == 0  # non-Telegram: no post
    assert tg_posts == 1       # Telegram: exactly one post
