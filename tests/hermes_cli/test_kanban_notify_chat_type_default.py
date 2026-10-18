"""#272 follow-up: a NEW notify subscription must not default ``chat_type='dm'``
when the chat_id can only be a group/supergroup.

``hermes_cli/kanban_db_notify.add_notify_sub`` defaulted every subscription
created without an explicit ``--chat-type`` to ``'dm'``.  Telegram DM chat ids
are positive user ids; groups, supergroups and channels are negative.  The
default therefore stamped an internally impossible pairing
(``chat_type='dm'`` on a negative chat id) onto 86 live rows on 2026-10-18.
The recorded ``chat_type`` is what the active-wake path replays into
``build_session_key()``, so the woken turn keyed a DM session instead of the
originating group topic.

The default now derives from the chat id shape:

* negative chat_id  -> ``'group'``  (cannot be a DM on any platform whose ids
  follow Telegram's sign convention)
* otherwise         -> ``'dm'``     (unchanged historical default, including a
  positive chat_id carrying a ``thread_id`` -- a Telegram DM topic)

An explicit ``--chat-type`` still wins, and a ``None`` re-subscribe still
leaves the recorded value untouched.
"""

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

# The live group/supergroup id the 86 mismatched rows carried (t_06f17ad4 recon).
GROUP_CHAT_ID = "-1003864867366"
# A Telegram DM chat id is a positive user id (same recon capture).
DM_CHAT_ID = "640466638"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _row(conn, tid):
    subs = kbn.list_notify_subs(conn, tid)
    assert len(subs) == 1
    return subs[0]


def test_capture_discriminator_fix_present_in_tree_under_test():
    """Capture discriminator: the #272 derivation helper exists in the tree.

    Names the loaded module path in the failure message so a run either
    proves which tree executed or fails loudly against a tree without the fix.
    """
    helper = getattr(kbn, "default_chat_type_for_chat_id", None)
    assert callable(helper), (
        "tree under test lacks the #272 chat_type derivation helper "
        f"(kanban_db_notify loaded from {kbn.__file__})"
    )


def test_default_chat_type_is_group_for_negative_chat_id(kanban_home):
    """A group/topic subscription created without chat_type lands 'group', not 'dm'."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="group default", assignee="worker1")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id=GROUP_CHAT_ID, thread_id="19336",
        )
        row = _row(conn, tid)
    finally:
        conn.close()
    assert row["chat_type"] == "group", (
        "a negative (group) chat_id must not be stamped 'dm' — the wake path "
        "replays chat_type into build_session_key()"
    )


def test_default_chat_type_stays_dm_for_positive_chat_id_with_thread(kanban_home):
    """A Telegram DM topic (positive chat id + thread id) must stay 'dm'."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="dm topic default", assignee="worker1")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id=DM_CHAT_ID, thread_id="20197",
        )
        row = _row(conn, tid)
    finally:
        conn.close()
    assert row["chat_type"] == "dm"


def test_default_chat_type_stays_dm_for_positive_chat_id_without_thread(kanban_home):
    """The historical default is unchanged for an ordinary DM."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="plain dm default", assignee="worker1")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id=DM_CHAT_ID,
        )
        row = _row(conn, tid)
    finally:
        conn.close()
    assert row["chat_type"] == "dm"


@pytest.mark.parametrize("explicit", ["dm", "group", "channel", "thread"])
def test_explicit_chat_type_wins_over_derived_default(kanban_home, explicit):
    """An explicit value wins even when it contradicts the chat id shape."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=f"explicit {explicit}", assignee="worker1")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id=GROUP_CHAT_ID, thread_id="19336", chat_type=explicit,
        )
        row = _row(conn, tid)
    finally:
        conn.close()
    assert row["chat_type"] == explicit


def test_none_resubscribe_keeps_recorded_group_chat_type(kanban_home):
    """A None re-subscribe (mode-only change) must not clobber the derived value."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="group re-subscribe", assignee="worker1")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id=GROUP_CHAT_ID, thread_id="19336",
        )
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id=GROUP_CHAT_ID, thread_id="19336", delivery_mode="wake",
        )
        row = _row(conn, tid)
    finally:
        conn.close()
    assert row["chat_type"] == "group"
    assert row["delivery_mode"] == "wake"


def test_cli_notify_subscribe_without_chat_type_lands_group_row(kanban_home):
    """The real CLI verb, invoked without --chat-type, must persist 'group'."""
    import argparse

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cli group sub", assignee="worker")

    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "notify-subscribe", tid, "--platform", "telegram",
        f"--chat-id={GROUP_CHAT_ID}", "--thread-id", "19336",
    ])
    assert kc.kanban_command(args) == 0

    with kbc.connect() as conn:
        row = _row(conn, tid)
    assert row["chat_type"] == "group"
    assert row["thread_id"] == "19336"


def test_cli_notify_subscribe_explicit_chat_type_still_wins(kanban_home):
    """An explicit --chat-type on the CLI overrides the derived default."""
    import argparse

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cli explicit sub", assignee="worker")

    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "notify-subscribe", tid, "--platform", "telegram",
        f"--chat-id={GROUP_CHAT_ID}", "--thread-id", "19336",
        "--chat-type", "thread",
    ])
    assert kc.kanban_command(args) == 0

    with kbc.connect() as conn:
        row = _row(conn, tid)
    assert row["chat_type"] == "thread"
