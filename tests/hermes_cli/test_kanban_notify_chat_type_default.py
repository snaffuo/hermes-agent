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

# Synthetic ids: the same shapes as the live capture (a negative supergroup id and
# a positive user id) without embedding a real chat.  The live values are not
# credentials, but they are not needed here either.
GROUP_CHAT_ID = "-1001234567890"
# A Telegram DM chat id is a positive user id.
DM_CHAT_ID = "123456789"


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


# ---------------------------------------------------------------------------
# #272 follow-up, second site: _inherit_notify_subs (create_task parents /
# link_tasks / decompose) copied a parent's chat_type with
# COALESCE(chat_type, 'dm').  A parent row from a pre-chat_type database has
# NULL, so the child of a legacy negative-chat-id parent was stamped 'dm' —
# the same impossible pairing #272 removed from add_notify_sub.
# ---------------------------------------------------------------------------

def _legacy_parent_with_null_chat_type(conn, chat_id, *, title):
    """A parent task carrying a pre-``chat_type`` subscription row (NULL column)."""
    parent = kb.create_task(conn, title=title, assignee=None)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, thread_id,"
            " chat_type, notifier_profile, delivery_mode, created_at, last_event_id)"
            " VALUES (?, 'telegram', ?, '19336', NULL, 'default', 'notify', 0, 0)",
            (parent, chat_id),
        )
    return parent


def test_inherited_sub_from_null_chat_type_parent_is_not_dm_on_negative_chat_id(kanban_home):
    """A NULL chat_type parent on a negative chat id must not stamp 'dm' on the child."""
    conn = kbc.connect()
    try:
        parent = _legacy_parent_with_null_chat_type(conn, GROUP_CHAT_ID, title="legacy group parent")
        child = kb.create_task(conn, title="legacy group child", assignee="worker1")
        kb.link_tasks(conn, parent, child)
        subs = kbn.list_notify_subs(conn, child)
    finally:
        conn.close()
    assert len(subs) == 1, f"expected exactly one inherited sub, got {len(subs)}"
    assert subs[0]["chat_type"] == "group", (
        "_inherit_notify_subs stamped 'dm' onto a negative (group) chat id — the "
        "#272 impossible pairing, one function over"
    )


def test_inherited_sub_from_null_chat_type_parent_stays_dm_on_positive_chat_id(kanban_home):
    """The legacy DM behaviour is preserved: NULL chat_type + positive id -> 'dm'."""
    conn = kbc.connect()
    try:
        parent = _legacy_parent_with_null_chat_type(conn, DM_CHAT_ID, title="legacy dm parent")
        child = kb.create_task(conn, title="legacy dm child", assignee="worker1")
        kb.link_tasks(conn, parent, child)
        subs = kbn.list_notify_subs(conn, child)
    finally:
        conn.close()
    assert len(subs) == 1, f"expected exactly one inherited sub, got {len(subs)}"
    assert subs[0]["chat_type"] == "dm"


def test_inherited_sub_keeps_a_recorded_chat_type(kanban_home):
    """A parent that DID record a chat_type passes it through unchanged."""
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="recorded parent", assignee=None)
        kbn.add_notify_sub(
            conn, task_id=parent, platform="telegram",
            chat_id=GROUP_CHAT_ID, thread_id="19336", chat_type="channel",
        )
        child = kb.create_task(conn, title="recorded child", assignee="worker1")
        kb.link_tasks(conn, parent, child)
        subs = kbn.list_notify_subs(conn, child)
    finally:
        conn.close()
    assert len(subs) == 1, f"expected exactly one inherited sub, got {len(subs)}"
    assert subs[0]["chat_type"] == "channel"


# ---------------------------------------------------------------------------
# The inheritance rule lives twice on purpose: as Python in
# kanban_db_notify.default_chat_type_for_chat_id (kanban_db cannot import it —
# cycle) and as the SQL expression kanban_db._INHERITED_CHAT_TYPE_SQL.  This
# parity test is what keeps them from diverging.
# ---------------------------------------------------------------------------

CHAT_ID_SHAPES = [
    None, "", "   ", "-1001234567890", "-100", "-0", "-1", "-",
    "-abc", "-123abc", "123abc-", "123456789", "0", "+123",
    "  -1001234567890  ", " -100 ", "chat1", "00-1", "1",
]


def test_sql_chat_type_rule_agrees_with_python_helper(kanban_home):
    """The SQL inheritance expression must match the Python helper on every shape."""
    conn = kbc.connect()
    try:
        mismatches = []
        for shape in CHAT_ID_SHAPES:
            sql_value = conn.execute(
                "SELECT " + kb._INHERITED_CHAT_TYPE_SQL + " AS ct"
                "  FROM (SELECT ? AS chat_id, NULL AS chat_type) AS parent_row",
                (shape,),
            ).fetchone()["ct"]
            py_value = kbn.default_chat_type_for_chat_id(shape)
            if sql_value != py_value:
                mismatches.append((shape, sql_value, py_value))
        # A recorded value is never re-derived, in either expression.
        for recorded in ("dm", "group", "channel", "thread"):
            passthrough = conn.execute(
                "SELECT " + kb._INHERITED_CHAT_TYPE_SQL + " AS ct"
                "  FROM (SELECT ? AS chat_id, ? AS chat_type) AS parent_row",
                (GROUP_CHAT_ID, recorded),
            ).fetchone()["ct"]
            if passthrough != recorded:
                mismatches.append((f"recorded chat_type={recorded!r}", passthrough, recorded))
    finally:
        conn.close()
    assert not mismatches, (
        "the SQL inheritance expression and kanban_db_notify.default_chat_type_for_chat_id "
        f"disagree (shape, sql, python): {mismatches}"
    )

