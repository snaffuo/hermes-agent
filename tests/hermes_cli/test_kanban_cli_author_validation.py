"""#110081: CLI comment/attach must validate --author against the calling profile.

The tool path (tools/kanban_tools.py) already derives author from HERMES_PROFILE
and ignores caller-supplied identity. The CLI path used to honor a raw
``--author`` value, letting any profile store a comment or attachment under a
foreign author (the same forgery class as the t_c66019eb fabricated-reviewer
comment). These tests pin the refusal, the legit paths, and the audited
operator override.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def task_id(kanban_home):
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="author validation", assignee="coder")


def _comments(tid):
    with kbc.connect_closing() as conn:
        return kb.list_comments(conn, tid)


def _events(tid):
    with kbc.connect_closing() as conn:
        return kb.list_events(conn, tid)


# --- AC-1: spoof refused ------------------------------------------------------

def test_comment_foreign_author_refused(kanban_home, task_id, monkeypatch):
    """--author reviewer from a coder-identified session is REFUSED; nothing stores."""
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = kc.run_slash(f"comment {task_id} fake verdict --author reviewer")
    assert "does not match the calling profile identity 'coder'" in out
    # names the correct invocation
    assert "Re-run without --author" in out
    assert "--as-operator" in out
    # nothing was stored
    assert _comments(task_id) == []
    assert not any(e.kind == "commented" for e in _events(task_id))


def test_attach_foreign_author_refused(kanban_home, task_id, monkeypatch, tmp_path):
    """AC-3 twin: --author reviewer on attach is refused too; no attachment row."""
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    src = tmp_path / "evidence.txt"
    src.write_bytes(b"probe")
    out = kc.run_slash(f"attach {task_id} {src} --author reviewer")
    assert "does not match the calling profile identity 'coder'" in out
    with kbc.connect_closing() as conn:
        assert kb.list_attachments(conn, task_id) == []


# --- AC-2: legit paths unchanged ----------------------------------------------

def test_comment_plain_stores_derived_author(kanban_home, task_id, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = kc.run_slash(f"comment {task_id} plain note")
    assert "Comment added" in out
    assert [c.author for c in _comments(task_id)] == ["coder"]


def test_comment_matching_author_stores_declared(kanban_home, task_id, monkeypatch):
    """--author coder from a coder session keeps working (scripted usage)."""
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = kc.run_slash(f"comment {task_id} scripted note --author coder")
    assert "Comment added" in out
    assert [c.author for c in _comments(task_id)] == ["coder"]


def test_comment_matching_author_casefold(kanban_home, task_id, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = kc.run_slash(f"comment {task_id} note --author Coder")
    assert "Comment added" in out
    assert [c.author for c in _comments(task_id)] == ["Coder"]


def test_attach_matching_author_stores(kanban_home, task_id, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    src = tmp_path / "evidence.txt"
    src.write_bytes(b"probe")
    out = kc.run_slash(f"attach {task_id} {src} --author coder")
    assert "Attached" in out
    with kbc.connect_closing() as conn:
        atts = kb.list_attachments(conn, task_id)
    assert [a.uploaded_by for a in atts] == ["coder"]


# --- AC-5: operator override path, audited -------------------------------------

def test_operator_override_stores_with_audit_event(kanban_home, task_id, monkeypatch):
    """Interactive shell (no HERMES_KANBAN_TASK) may post under a different name —
    the override is recorded as an author_override audit event."""
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = kc.run_slash(
        f"comment {task_id} operator note --author Agent0-orchestrator --as-operator"
    )
    assert "Comment added" in out
    assert [c.author for c in _comments(task_id)] == ["Agent0-orchestrator"]
    ov = [e for e in _events(task_id) if e.kind == "author_override"]
    assert len(ov) == 1
    assert ov[0].payload == {
        "verb": "comment",
        "declared_author": "Agent0-orchestrator",
        "profile_identity": "default",
    }


def test_operator_override_refused_from_worker_session(kanban_home, task_id, monkeypatch):
    """A worker-scoped session (HERMES_KANBAN_TASK set) cannot invoke the override."""
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    out = kc.run_slash(f"comment {task_id} forged --author reviewer --as-operator")
    assert "only valid from an interactive operator shell" in out
    assert _comments(task_id) == []
    assert not any(e.kind == "author_override" for e in _events(task_id))


def test_operator_override_attach_audited(kanban_home, task_id, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    src = tmp_path / "evidence.txt"
    src.write_bytes(b"probe")
    out = kc.run_slash(
        f"attach {task_id} {src} --name marked.txt --author reviewer --as-operator"
    )
    assert "Attached" in out
    with kbc.connect_closing() as conn:
        atts = kb.list_attachments(conn, task_id)
    assert [a.uploaded_by for a in atts] == ["reviewer"]
    ov = [e for e in _events(task_id) if e.kind == "author_override"]
    assert len(ov) == 1
    payload = ov[0].payload or {}
    assert payload["verb"] == "attach"
    assert payload["declared_author"] == "reviewer"
    assert payload["profile_identity"] == "default"
