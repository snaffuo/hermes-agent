"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        triage=True,
    )


def test_specify_promotes_triage_to_todo(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kbc.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


def test_specify_rejects_blank_title(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_records_audit_comment_only_when_author_given(kanban_home):
    # With author → comment added.
    with kbc.connect() as conn:
        tid1 = _create_triage(conn, title="a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    assert len(comments1) == 1
    assert "Specified" in comments1[0].body
    assert comments1[0].author == "ace"

    # Without author → no comment (silent).
    with kbc.connect() as conn:
        tid2 = _create_triage(conn, title="b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert comments2 == []


# ---------------------------------------------------------------------------
# preserve_body — the body-preserving triage exit (#110339)
# ---------------------------------------------------------------------------

# Distinctive multi-section body, deliberately >4000 chars so any path that
# re-sources the body through the 4000-char-truncated prompt field (AC-1
# hazard, Quant advisory) is caught, with awkward whitespace byte-traps at the
# edges (no strip tolerance).
PRESERVE_BODY = (
    "  **Goal**\n"
    "Preserve this byte-for-byte.\t\n\n"
    "**Approach**\n"
    "  1. Step one — trailing spaces   \n"
    "  2. Step two\n"
    + ("Filler paragraph to push this body past the 4000-char prompt-field "
       "truncation limit. " * 90)
    + "\n\n**Acceptance criteria**\n"
    "- [ ] AC-1 body identical\n"
    "- [ ] AC-4 audit records verb\n"
    "\n**Out of scope**\n- Nothing.\n\n  "
)


def _specified_event(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "specified"][-1]


def test_preserve_body_promotes_verbatim(kanban_home):
    """AC-1 (DB layer): triage -> todo with a >4000-char multi-section body
    byte-identical — nothing is stripped, re-wrapped, or truncated."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="keep me", body=PRESERVE_BODY)
        assert kb.specify_triage_task(conn, tid, preserve_body=True, author="spec") is True
        task = kb.get_task(conn, tid)
        events = _specified_event(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task.status in ("todo", "ready")
    assert task.body == PRESERVE_BODY  # exact bytes, incl. leading/trailing runs
    assert len(PRESERVE_BODY) > 4000  # guard: stays a truncation-class test
    # AC-4: verb + body_preserved recorded even though changed_fields is empty.
    assert events.payload == {"changed_fields": [], "verb": "specify", "body_preserved": True}
    # A verbatim promotion changes no fields yet still leaves a human-readable trace.
    assert any("body preserved byte-for-byte" in c.body for c in comments)


def test_preserve_body_records_audit_even_without_author(kanban_home):
    """Audit event lands even with no author (comment is the author-gated part)."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, body=PRESERVE_BODY)
        assert kb.specify_triage_task(conn, tid, preserve_body=True) is True
        events = _specified_event(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert events.payload["body_preserved"] is True
    assert events.payload["verb"] == "specify"
    assert comments == []


def test_llm_path_audit_unchanged(kanban_home):
    """AC-3 (DB layer): a normal specify keeps the old payload shape — no
    verb/body_preserved keys leak into the LLM-path audit event."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough", body="one-liner")
        kb.specify_triage_task(conn, tid, title="Rough", body="**Goal**\nx", author="a")
        events = _specified_event(conn, tid)
    assert events.payload == {"changed_fields": ["title", "body"]}


