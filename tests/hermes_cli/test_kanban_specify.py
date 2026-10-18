"""Tests for the specifier module + `hermes kanban specify` CLI surface.

The auxiliary LLM client is mocked — these tests don't hit any network or
real provider. They exercise the prompt plumbing, response parsing, DB
writes, and CLI flag surface.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    """Build a minimal object shaped like an OpenAI chat.completions result.

    The specifier only reads ``resp.choices[0].message.content``, so we
    avoid importing the openai SDK and build the tree with MagicMock.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    """Patch call_llm at its source module — specify_task now routes through
    it (#35566) instead of building a raw client. Returns (patcher, mock) so
    callers can still assert on the call.
    """
    mock_fn = MagicMock(return_value=_fake_aux_response(content))
    return patch("agent.auxiliary_client.call_llm", mock_fn), mock_fn


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    """The LLM path is opt-in since #231: rewrite=True is what lets the reply
    become the card's body (here the card starts with no body at all)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace", rewrite=True)

    assert outcome.ok is True
    assert outcome.task_id == tid
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")


# ---------------------------------------------------------------------------
# keep_body — the body-preserving triage exit (#110339)
# ---------------------------------------------------------------------------

KEEP_BODY = "  **Goal**\nverbatim.\t\n\n**Approach**\n- step  \n" + (
    "Filler past the 4000-char prompt-field truncation limit. " * 100
) + "\n  "


def test_specify_task_keep_body_skips_llm_entirely(kanban_home):
    """AC-1/AC-6: the aux client is patched to RAISE — the keep-body path must
    never reach it (branch is before _call_aux, no import of the aux client),
    and a >4000-char body lands byte-identical (raw column, never the
    truncated prompt field)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="keep", body=KEEP_BODY, triage=True)

    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        outcome = spec.specify_task(tid, author="ace", keep_body=True)

    assert outcome.ok is True
    assert outcome.body_preserved is True
    boom.assert_not_called()
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert len(KEEP_BODY) > 4000
    assert task.body == KEEP_BODY
    assert task.title == "keep"
    assert events[-1].payload == {"changed_fields": [], "verb": "specify", "body_preserved": True}


def test_specify_task_keep_body_survives_aux_unavailable(kanban_home, monkeypatch):
    """AC-6 literally: aux client genuinely unavailable (import fails), NOT a
    mocked success — the flag path still exits triage."""
    import builtins
    import sys

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="keep", body=KEEP_BODY, triage=True)

    real_import = builtins.__import__

    def no_aux(name, *a, **kw):
        if name.startswith("agent.auxiliary_client"):
            raise ImportError("simulated: auxiliary client unavailable")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_aux)
    sys.modules.pop("agent.auxiliary_client", None)
    try:
        outcome = spec.specify_task(tid, author="ace", keep_body=True)
    finally:
        sys.modules.pop("agent.auxiliary_client", None)
    assert outcome.ok is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == KEEP_BODY


def test_specify_task_keep_body_rejects_non_triage(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="done already")  # -> ready/todo path
    outcome = spec.specify_task(tid, author="ace", keep_body=True)
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


def test_specify_task_keep_body_refuses_null_body(kanban_home):
    """#208 guard: a NULL stored body has nothing to preserve — refuse BEFORE
    specify_triage_task runs, with the message verbatim; the card stays in
    triage. (Before the guard, keep-body promoted it verbatim and created the
    empty-body card #208 measured.)"""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no body", triage=True)  # body stays NULL

    with patch("hermes_cli.kanban_db.specify_triage_task") as promote:
        outcome = spec.specify_task(tid, author="ace", keep_body=True)

    promote.assert_not_called()
    assert outcome.ok is False
    assert outcome.body_preserved is False
    assert outcome.reason == (
        "specify: stored body is empty; nothing to preserve "
        "(pass --rewrite to have the specifier write one)"
    )
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert task.status == "triage"
    assert task.body is None
    assert events == []


def test_specify_task_keep_body_refuses_blank_body(kanban_home):
    """#208 guard, blank half: a whitespace-only body is refused the same way."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="blank body", body=" \n\t ", triage=True)

    with patch("hermes_cli.kanban_db.specify_triage_task") as promote:
        outcome = spec.specify_task(tid, author="ace", keep_body=True)

    promote.assert_not_called()
    assert outcome.ok is False
    assert outcome.reason == (
        "specify: stored body is empty; nothing to preserve "
        "(pass --rewrite to have the specifier write one)"
    )
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert task.status == "triage"
    assert task.body == " \n\t "
    assert events == []







# ---------------------------------------------------------------------------
# CLI wiring — argparse + _cmd_specify
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)




def test_cli_specify_tenant_filter(kanban_home, capsys):
    """The --all sweep honours --tenant. Since #231 the sweep takes the
    body-preserving default, so the inside task carries a body and the aux
    client is patched to RAISE — the LLM must never be reached (the raise is
    the control that the default really is preserve-only)."""
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", body="**Goal**\ntenant body.", triage=True, tenant="proj-a",
        )

    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    boom.assert_not_called()
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}
    assert [row["body_preserved"] for row in lines] == [True]

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted, body byte-identical.
        promoted = kb.get_task(conn, inside)
        assert promoted.status in {"todo", "ready"}
        assert promoted.body == "**Goal**\ntenant body."


def test_cli_keep_body_end_to_end(kanban_home, capsys):
    """AC-2 (CLI, supported verbs only, no LLM): create --triage → specify
    --keep-body → claim reaches running with the body byte-identical."""
    rc = _run_cli("create", "keep e2e", "--triage", "--body", KEEP_BODY, "--json")
    assert rc == 0
    tid = jsonlib.loads(capsys.readouterr().out)["id"]

    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        rc = _run_cli("specify", tid, "--keep-body", "--json")
    assert rc == 0
    row = jsonlib.loads(capsys.readouterr().out.strip())
    assert row == {"task_id": tid, "ok": True, "reason": "specified (verbatim)",
                   "new_title": None, "body_preserved": True}
    boom.assert_not_called()

    rc = _run_cli("claim", tid)
    assert rc == 0
    capsys.readouterr()

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        comments = kb.list_comments(conn, tid)
    assert task.status == "running"
    assert task.body == KEEP_BODY
    # AC-4: the audit event names the verb and records preservation.
    assert events[-1].payload == {"changed_fields": [], "verb": "specify", "body_preserved": True}
    assert any("body preserved byte-for-byte" in c.body for c in comments)


def test_cli_keep_body_flag_is_specify_only(kanban_home):
    """AC-3 (parser): --keep-body and --rewrite are specify-only —
    _triage_sweep_args is shared with decompose, which must NOT gain either."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", "specify", "t_x", "--keep-body"])
    assert ns.keep_body is True
    # decompose never received the flag: argparse must reject it there.
    with pytest.raises(SystemExit):
        root.parse_args(["kanban", "decompose", "t_x", "--keep-body"])
    # #231: the rewriting exit is its own explicit opt-in.
    ns3 = root.parse_args(["kanban", "specify", "t_x", "--rewrite"])
    assert ns3.rewrite is True
    assert ns3.keep_body is False
    with pytest.raises(SystemExit):
        root.parse_args(["kanban", "decompose", "t_x", "--rewrite"])
    # specify with neither flag: both off (preservation is the default).
    ns2 = root.parse_args(["kanban", "specify", "t_x"])
    assert ns2.keep_body is False
    assert ns2.rewrite is False


def test_cli_specify_default_preserves_body(kanban_home, capsys):
    """#231 AC: `specify <id>` with NO flags preserves the body byte-for-byte and
    never calls the aux client — the default used to rewrite it instead."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="default", body=KEEP_BODY, triage=True)

    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        rc = _run_cli("specify", tid, "--json")
    assert rc == 0
    boom.assert_not_called()
    row = jsonlib.loads(capsys.readouterr().out.strip())
    assert row == {"task_id": tid, "ok": True, "reason": "specified (verbatim)",
                   "new_title": None, "body_preserved": True}

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert len(KEEP_BODY) > 4000
    assert task.body == KEEP_BODY
    assert task.title == "default"


def test_cli_rewrite_flag_condenses(kanban_home, capsys):
    """AC-3: the LLM pass (and its wholesale body replacement) now requires the
    explicit --rewrite flag."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body=KEEP_BODY, triage=True)
    content = jsonlib.dumps({"title": "Condensed", "body": "**Goal**\nshort."})
    p, mock_fn = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", tid, "--rewrite", "--json")
    assert rc == 0
    row = jsonlib.loads(capsys.readouterr().out.strip())
    assert row["ok"] is True
    assert row["body_preserved"] is False
    assert mock_fn.called
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == "**Goal**\nshort."
    events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert events[-1].payload == {"changed_fields": ["title", "body"]}


def test_specify_task_default_is_preserve_only(kanban_home):
    """Module level: no flags -> preserving exit, LLM untouched."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mod", body="**Goal**\nkept.\n", triage=True)
    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.ok is True
    assert outcome.body_preserved is True
    boom.assert_not_called()
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "**Goal**\nkept.\n"


def test_specify_task_rewrite_flag_replaces_body(kanban_home):
    """Module level: rewrite=True is the opt-in that lets the reply replace."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mod", body="old body", triage=True)
    content = jsonlib.dumps({"title": "New title", "body": "new body"})
    p, mock_fn = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace", rewrite=True)
    assert outcome.ok is True
    assert outcome.body_preserved is False
    assert mock_fn.called
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == "new body"
    assert task.title == "New title"


def test_specify_task_legacy_keep_body_false_still_rewrites(kanban_home):
    """Back-compat: an out-of-tree caller passing keep_body=False keeps the
    pre-#231 meaning (run the LLM pass) — only the DEFAULT flipped."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mod", body="old body", triage=True)
    content = jsonlib.dumps({"title": "T", "body": "legacy rewrite"})
    p, mock_fn = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace", keep_body=False)
    assert outcome.ok is True
    assert outcome.body_preserved is False
    assert mock_fn.called
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "legacy rewrite"


def test_specify_task_rewrite_conflicts_with_keep_body(kanban_home):
    """--rewrite replaces, --keep-body preserves: the contradiction is refused
    loudly (card stays in triage, LLM never called), never resolved by
    precedence."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="both", body="**Goal**\nkept.\n", triage=True)
    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        outcome = spec.specify_task(tid, author="ace", keep_body=True, rewrite=True)
    assert outcome.ok is False
    assert "conflicting request" in outcome.reason
    boom.assert_not_called()
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert task.status == "triage"
    assert task.body == "**Goal**\nkept.\n"
    assert events == []


def test_cli_rewrite_conflicts_with_keep_body_exits_nonzero(kanban_home, capsys):
    """CLI half of the contradiction: exit 1, card untouched, reason on stderr."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="both", body="**Goal**\nkept.\n", triage=True)
    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        rc = _run_cli("specify", tid, "--rewrite", "--keep-body")
    assert rc == 1
    boom.assert_not_called()
    assert "conflicting request" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_cli_specify_empty_body_refusal_names_rewrite(kanban_home, capsys):
    """#208 guard, now on the default path: a NULL body is refused with an
    actionable reason and the card stays in triage — no silent LLM fill-in."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no body", triage=True)
    boom = MagicMock(side_effect=AssertionError("aux client must not be called"))
    with patch("agent.auxiliary_client.call_llm", boom):
        rc = _run_cli("specify", tid)
    assert rc == 1
    boom.assert_not_called()
    assert "--rewrite" in capsys.readouterr().err
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"



