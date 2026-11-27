# -*- coding: utf-8 -*-
"""本地登记库（SQLite）。

一次确认对应一次登记：registration_id 唯一，提交成功后才允许播报“已登记”。
重复确认复用已保存结果，不会重复写入；写入失败保留草稿并返回失败，由策略
决定重试。工程日志写进 dialog_log 表，便于判断问题发生在哪一轮。
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS registrations (
    registration_id TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    profile_id      TEXT,
    department      TEXT NOT NULL,
    doctor          TEXT,
    visit_date      TEXT NOT NULL,
    period          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    snapshot_json   TEXT NOT NULL,
    source_json     TEXT
);

CREATE TABLE IF NOT EXISTS dialog_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_id     INTEGER NOT NULL,
    phase       TEXT,
    action      TEXT,
    detail      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dialog_log_session ON dialog_log(session_id, turn_id);
"""


class SaveError(RuntimeError):
    """保存失败（模拟磁盘/权限问题或业务校验失败）。"""


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RegistrationStore(object):
    def __init__(self, path=":memory:"):
        self.path = path
        self._fault = False
        if path not in (":memory:", "", None):
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ---- 故障注入（测试与演示用）----
    def fail_next_save(self):
        self._fault = True

    # ---- 写入 ----
    def save_registration(self, draft, department, date, period, doctor=None):
        """保存一条登记，返回 (registration_id, created)。

        同一 registration_id 只允许一条记录：已存在时直接返回 created=False，
        调用方据此播报“已经登记过”，不产生第二条记录。
        """
        if self._fault:
            self._fault = False
            raise SaveError("模拟写入失败")

        existing = self.get_registration(draft.registration_id)
        if existing is not None:
            return draft.registration_id, False

        fields = draft.to_confirm_fields()
        sources = dict((f, v.get("source")) for f, v in fields.items())
        try:
            cur = self.conn.execute(
                "INSERT INTO registrations (registration_id, session_id, profile_id,"
                " department, doctor, visit_date, period, created_at, snapshot_json, source_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (draft.registration_id, draft.session_id, draft.profile_id,
                 department, doctor, date, period, _now(),
                 json.dumps(fields, ensure_ascii=False),
                 json.dumps(sources, ensure_ascii=False)),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # 并发或重复提交：以已存在记录为准
            self.conn.rollback()
            return draft.registration_id, False
        except Exception as exc:  # pragma: no cover - 真实磁盘异常
            self.conn.rollback()
            raise SaveError(str(exc))
        if cur.rowcount != 1:
            return draft.registration_id, False
        return draft.registration_id, True

    # ---- 读取 ----
    def next_registration_id(self):
        """分配一个可读的登记编号，例如 REG-0001；唯一约束由数据库保证。"""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM registrations").fetchone()
        return "REG-%04d" % (int(row["n"]) + 1)

    def count_by_session(self, session_id):
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM registrations WHERE session_id = ?",
            (session_id,)).fetchone()
        return int(row["n"])

    def get_registration(self, registration_id):
        row = self.conn.execute(
            "SELECT * FROM registrations WHERE registration_id = ?", (registration_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_registrations(self):
        return self.conn.execute("SELECT COUNT(*) AS n FROM registrations").fetchone()["n"]

    def list_registrations(self):
        rows = self.conn.execute(
            "SELECT * FROM registrations ORDER BY created_at, registration_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def has_duplicates(self):
        """同一 registration_id 是否出现多条（唯一约束下应始终为 0）。"""
        rows = self.conn.execute(
            "SELECT registration_id, COUNT(*) AS n FROM registrations"
            " GROUP BY registration_id HAVING n > 1"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- 日志 ----
    def log_turn(self, session_id, turn_id, phase, action, detail=None):
        self.conn.execute(
            "INSERT INTO dialog_log (session_id, turn_id, phase, action, detail, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (session_id, int(turn_id), phase, action,
             json.dumps(detail or {}, ensure_ascii=False), _now()),
        )
        self.conn.commit()

    def list_dialog_log(self, session_id=None):
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM dialog_log WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM dialog_log ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
