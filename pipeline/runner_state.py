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
        self._shared = None
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
            CREATE INDEX IF NOT EXISTS tasks_created ON tasks(created);
            CREATE INDEX IF NOT EXISTS tasks_dispatch ON tasks(model||':'||stage, created)
              WHERE active=1 AND state IN ('pending','failed');
            CREATE INDEX IF NOT EXISTS tasks_slot ON tasks(slot, active);
            CREATE INDEX IF NOT EXISTS tasks_model ON tasks(model, active, state);
            CREATE INDEX IF NOT EXISTS tasks_ready ON tasks(active, state, available);
            ''')

    def _open(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db

    @contextmanager
    def connect(self):
        if self._shared is not None:
            yield self._shared  # inside transaction(): one commit for the whole burst
            return
        db = self._open()
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        """Group a burst of journal writes (an enqueue of every eligible card) into one
        durable commit instead of one FULL-synchronous commit per task. Nothing is
        visible to workers until the burst commits; an interrupted burst enqueues
        nothing and the next reconcile recomputes it. Single-threaded use only."""
        if self._shared is not None:
            yield
            return
        db = self._open()
        self._shared = db
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield
        finally:
            self._shared = None
            db.close()

    @staticmethod
    def begin(db):
        if not db.in_transaction:
            db.execute('BEGIN IMMEDIATE')

    @staticmethod
    def event(db, task, event, reason=None):
        db.execute('INSERT INTO events(task,event,reason,at) VALUES (?,?,?,?)', (task, event, reason, time.time()))

    def enqueue(self, slot, model, stage, generation, payload):
        task_id = digest([slot, generation, payload])
        with self.connect() as db:
            self.begin(db)
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
            self.begin(db)
            for row in db.execute('SELECT id,slot FROM tasks WHERE model=? AND active=1', (model,)).fetchall():
                if row['slot'] not in slots:
                    db.execute("UPDATE tasks SET state='blocked',active=0,token=NULL,reason='no longer eligible' WHERE id=?", (row['id'],))
                    self.event(db, row['id'], 'invalidated', 'no longer eligible')

    @contextmanager
    def claim_scope(self, eligible_ids):
        """Load file-adapter eligibility once, without SQLite's IN-parameter limit.

        The temporary table belongs to this connection, not durable task state.
        Each yielded claim commits independently before returning. No transaction
        remains open while a worker reads a response or performs external work.
        Other processes still arbitrate leases and concurrency in the main DB.
        """
        db = self._open()
        try:
            with db:
                db.execute('CREATE TEMP TABLE eligible (id TEXT PRIMARY KEY) WITHOUT ROWID')
                db.executemany('INSERT OR IGNORE INTO eligible VALUES (?)', ((i,) for i in eligible_ids))
            def claim(**kwargs):
                with db:
                    return self._claim(db, restricted=True, **kwargs)
            yield claim
        finally:
            db.close()

    def claim(self, *, eligible_ids=None, **kwargs):
        if eligible_ids is not None:
            with self.claim_scope(eligible_ids) as claim:
                return claim(**kwargs)
        with self.connect() as db:
            return self._claim(db, **kwargs)

    def _claim(self, db, *, lease_seconds=300, limit=8, max_attempts=3, now=None, restricted=False):
        if not 1 <= limit <= 32 or lease_seconds <= 0:
            raise ValueError('invalid concurrency or lease')
        now = time.time() if now is None else now
        self.begin(db)
        for row in db.execute("SELECT id,attempts FROM tasks WHERE active=1 AND state='running' AND lease<=?", (now,)).fetchall():
            state = 'blocked' if row['attempts'] >= max_attempts else 'failed'
            db.execute('UPDATE tasks SET state=?,available=?,token=NULL,reason=? WHERE id=?',
                       (state, now + min(60, 2 ** row['attempts']), 'lease expired', row['id']))
            self.event(db, row['id'], state, 'lease expired')
        if db.execute("SELECT COUNT(*) FROM tasks WHERE active=1 AND state='running'").fetchone()[0] >= limit:
            return None
        extra = ' AND id IN (SELECT id FROM temp.eligible)' if restricted else ''
        # One oldest eligible candidate per model/stage, then least-recently served.
        # The partial index excludes finished work; full evidence is read only for
        # the selected task, never for a catalogue-wide sort on every claim.
        row = db.execute("""SELECT t.* FROM groups g JOIN tasks t ON t.id=(
            SELECT id FROM tasks WHERE model||':'||stage=g.name AND active=1
            AND state IN ('pending','failed') AND available<=? AND attempts<?""" + extra + """
            ORDER BY created LIMIT 1) ORDER BY g.served,t.created LIMIT 1""", (now, max_attempts)).fetchone()
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
            self.begin(db)
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
            self.begin(db)
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

    @staticmethod
    def _row(row, with_payload):
        task = dict(row, result=json.loads(row['result']) if row['result'] else None)
        # Payloads carry whole numbered source cards: decode only when asked for.
        task['payload'] = json.loads(row['payload']) if with_payload else None
        return task

    def tasks(self, active=True, *, model=None, stage=None, states=None, with_payload=True):
        """Filtered listing in creation order; filters run in SQL, never by decoding every payload."""
        where, args = ['active=1'] if active else [], []
        for column, value in (('model', model), ('stage', stage)):
            if value is not None:
                where.append(column + '=?'); args.append(value)
        if states is not None:
            states = tuple(states)
            if not states:
                return []
            where.append('state IN (' + ','.join('?' for _ in states) + ')'); args.extend(states)
        columns = '*' if with_payload else 'id,slot,model,stage,generation,NULL AS payload,state,attempts,available,lease,token,result,reason,created,active'
        query = 'SELECT ' + columns + ' FROM tasks' + (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY created'
        with self.connect() as db:
            return [self._row(r, with_payload) for r in db.execute(query, args)]

    def get(self, task_id):
        """One task by id, with its payload; None when unknown."""
        with self.connect() as db:
            row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        return self._row(row, True) if row else None

    def payload_text(self, task_id):
        """The stored canonical JSON text of one task's payload (no decode/re-encode)."""
        with self.connect() as db:
            row = db.execute('SELECT payload FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row['payload']

    def current(self, slot):
        """The active task of a slot (at most one), with its payload."""
        with self.connect() as db:
            row = db.execute('SELECT * FROM tasks WHERE slot=? AND active=1 ORDER BY created DESC LIMIT 1', (slot,)).fetchone()
        return self._row(row, True) if row else None

    def counts(self):
        with self.connect() as db:
            return {r['state']: r['n'] for r in db.execute('SELECT state, COUNT(*) AS n FROM tasks WHERE active=1 GROUP BY state')}

    def backup(self, target):
        if Path(target).exists():
            raise ValueError('backup target exists')
        with self.connect() as source, closing(sqlite3.connect(target)) as dest:
            source.backup(dest)
