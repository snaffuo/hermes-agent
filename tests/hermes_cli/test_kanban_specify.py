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
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

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
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", triage=True, tenant="proj-a",
        )

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}


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
    """AC-3 (parser): --keep-body is specify-only — _triage_sweep_args is
    shared with decompose, which must NOT gain the flag."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", "specify", "t_x", "--keep-body"])
    assert ns.keep_body is True
    # decompose never received the flag: argparse must reject it there.
    with pytest.raises(SystemExit):
        root.parse_args(["kanban", "decompose", "t_x", "--keep-body"])
    # specify without the flag defaults it off.
    ns2 = root.parse_args(["kanban", "specify", "t_x"])
    assert ns2.keep_body is False


def test_cli_llm_path_still_condenses(kanban_home, capsys):
    """AC-3: specify WITHOUT --keep-body still runs the LLM pass and replaces
    the body (existing condensing behaviour preserved)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body=KEEP_BODY, triage=True)
    content = jsonlib.dumps({"title": "Condensed", "body": "**Goal**\nshort."})
    p, mock_fn = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", tid, "--json")
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



