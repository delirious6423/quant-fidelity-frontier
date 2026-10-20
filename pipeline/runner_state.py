"""Local durable task journal. SQLite transactions own claims; workers own no files.

One local disk, multiple worker processes, one short-lived orchestration lock.
FULL synchronous commits persist payload/result before success is returned. Use
SQLite backup plus the workspace files for transfer, never copy a live DB file.
"""
from contextlib import contextmanager, closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class State:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY, slot TEXT NOT NULL, model TEXT NOT NULL,
              stage TEXT NOT NULL, generation TEXT NOT NULL, payload TEXT NOT NULL,
              state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
              available REAL NOT NULL DEFAULT 0, lease REAL, token TEXT, result TEXT,
              reason TEXT, created INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS groups (name TEXT PRIMARY KEY, served INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT,
              task TEXT, event TEXT, reason TEXT, at REAL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def event(db, task, event, reason=None):
        db.execute('INSERT INTO events(task,event,reason,at) VALUES (?,?,?,?)', (task, event, reason, time.time()))

    def enqueue(self, slot, model, stage, generation, payload):
        task_id = digest([slot, generation, payload])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT id FROM tasks WHERE slot=? AND active=1 AND id!=?', (slot, task_id)).fetchall()
            for row in old:
                db.execute("UPDATE tasks SET active=0,state='blocked',reason='source or task version invalidated',token=NULL WHERE id=?", (row['id'],))
                self.event(db, row['id'], 'invalidated')
            existing = db.execute('SELECT active FROM tasks WHERE id=?', (task_id,)).fetchone()
            if existing and not existing['active']:
                # A -> B -> A does not resurrect a result or an expired lease.
                db.execute("UPDATE tasks SET active=1,state='pending',attempts=0,result=NULL,token=NULL,available=0,reason='reactivated evidence' WHERE id=?", (task_id,))
            if not existing:
                seq = db.execute('SELECT COALESCE(MAX(created),0)+1 FROM tasks').fetchone()[0]
                db.execute('INSERT INTO tasks(id,slot,model,stage,generation,payload,state,created) VALUES (?,?,?,?,?,?,?,?)',
                           (task_id, slot, model, stage, generation, encoded(payload), 'pending', seq))
                self.event(db, task_id, 'enqueued')
            db.execute('INSERT OR IGNORE INTO groups VALUES (?,0)', (model + ':' + stage,))
        return task_id

    def invalidate_missing(self, model, slots):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute('SELECT id,slot FROM tasks WHERE model=? AND active=1', (model,)).fetchall():
                if row['slot'] not in slots:
                    db.execute("UPDATE tasks SET state='blocked',active=0,token=NULL,reason='no longer eligible' WHERE id=?", (row['id'],))
                    self.event(db, row['id'], 'invalidated', 'no longer eligible')

    def claim(self, *, lease_seconds=300, limit=8, max_attempts=3, now=None, eligible_ids=None):
        if not 1 <= limit <= 32 or lease_seconds <= 0:
            raise ValueError('invalid concurrency or lease')
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT * FROM tasks WHERE active=1 AND state='running' AND lease<=?", (now,)).fetchall():
                state = 'blocked' if row['attempts'] >= max_attempts else 'failed'
                db.execute('UPDATE tasks SET state=?,available=?,token=NULL,reason=? WHERE id=?',
                           (state, now + min(60, 2 ** row['attempts']), 'lease expired', row['id']))
                self.event(db, row['id'], state, 'lease expired')
            if db.execute("SELECT COUNT(*) FROM tasks WHERE active=1 AND state='running'").fetchone()[0] >= limit:
                return None
            extra = '' if eligible_ids is None else ' AND t.id IN (' + ','.join('?' for _ in eligible_ids) + ')'
            row = db.execute('''SELECT t.* FROM tasks t JOIN groups g ON g.name=t.model||':'||t.stage
                WHERE t.active=1 AND t.state IN ('pending','failed') AND t.available<=? AND t.attempts<?''' + extra + '''
                ORDER BY g.served,t.created LIMIT 1''', (now, max_attempts, *(eligible_ids or []))).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE tasks SET state='running',token=?,lease=?,attempts=attempts+1 WHERE id=?", (token, now + lease_seconds, row['id']))
            clock = db.execute('SELECT COALESCE(MAX(served),0)+1 FROM groups').fetchone()[0]
            db.execute('UPDATE groups SET served=? WHERE name=?', (clock, row['model'] + ':' + row['stage']))
            self.event(db, row['id'], 'claimed')
            return dict(id=row['id'], token=token, lease=now + lease_seconds, payload=json.loads(row['payload']), stage=row['stage'])

    def complete(self, task_id, token, result, *, now=None):
        now = time.time() if now is None else now
        raw = encoded(result)
        conflict = False
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row and row['active'] and row['state'] == 'succeeded' and row['token'] == token:
                if row['result'] == raw:
                    return 'duplicate'  # transport redelivery, not a second semantic verdict
                db.execute("UPDATE tasks SET state='blocked',reason='conflicting duplicate result' WHERE id=?", (task_id,))
                self.event(db, task_id, 'blocked', 'conflicting duplicate result')
                conflict = True
            elif not row or not row['active'] or row['state'] != 'running' or row['token'] != token or row['lease'] <= now:
                raise ValueError('stale, expired or conflicting completion')
            else:
                db.execute("UPDATE tasks SET state='succeeded',result=?,reason=NULL WHERE id=?", (raw, task_id))
                self.event(db, task_id, 'succeeded')
        if conflict:
            raise ValueError('conflicting duplicate result blocks this task')
        return 'persisted'

    def fail(self, task_id, token, reason, *, blocked=False, now=None):
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM tasks WHERE id=? AND token=? AND active=1 AND state='running' AND lease>?", (task_id, token, now)).fetchone()
            if not row:
                raise ValueError('stale failure')
            state = 'blocked' if blocked or row['attempts'] >= 3 else 'failed'
            db.execute('UPDATE tasks SET state=?,available=?,reason=?,token=NULL WHERE id=?',
                       (state, now + min(60, 2 ** row['attempts']), reason, task_id))
            self.event(db, task_id, state, reason)

    def retry(self, task_id, reason):
        if not reason.strip():
            raise ValueError('retry requires an explicit reason')
        with self.connect() as db:
            n = db.execute("UPDATE tasks SET state='pending',attempts=0,available=0,reason=? WHERE id=? AND active=1 AND state IN ('failed','blocked')", (reason, task_id)).rowcount
            if n != 1:
                raise ValueError('only active failed/blocked tasks can be retried')
            self.event(db, task_id, 'manual retry', reason)

    def tasks(self, active=True):
        with self.connect() as db:
            return [dict(r, payload=json.loads(r['payload']), result=json.loads(r['result']) if r['result'] else None)
                    for r in db.execute('SELECT * FROM tasks' + (' WHERE active=1' if active else '') + ' ORDER BY created')]

    def backup(self, target):
        if Path(target).exists():
            raise ValueError('backup target exists')
        with self.connect() as source, closing(sqlite3.connect(target)) as dest:
            source.backup(dest)
