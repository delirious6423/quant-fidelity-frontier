"""Confined local shadow workspace; no publication capability or remote exists here."""
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
ALLOWED = {'qcl_newmodel.py', 'autoextract.py', 'detect_copies.py', 'qcl_mapset.py',
           'qcl_queue.py', 'qcl_watch.py', 'qcl_tiebreak.py', 'gen_auto.py', 'build.py'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from deploy_state import write_json
    write_json(path, value)


def write_text(path, text):
    """Atomic text write (temporary file, fsync, rename) for already-encoded JSON."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install(path, value):
    """Write a validated output unless the file already holds exactly this value.

    Materialization re-installs every succeeded result on each reconcile; an
    unchanged map must not cost a rewrite and fsync per task per round.
    """
    path = Path(path)
    try:
        if path.exists() and read(path) == value:
            return False
    except (OSError, ValueError):
        pass  # unreadable or partial: replace it
    write(path, value)
    return True


def safe_tree(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('symlinks are not permitted in shadow inputs/state')


class Workspace:
    def __init__(self, path):
        raw = Path(path).absolute()
        if raw.resolve() != raw or raw == CODE or CODE in raw.parents:
            raise ValueError('workspace must be outside the repository, without symlink ancestors')
        self.path = raw
        safe_tree(raw)
        self.config = read(raw / 'runner.json')
        if self.config.get('mode') != 'shadow' or self.config.get('version') != 1:
            raise ValueError('only shadow workspaces are supported')
        models = self.config.get('models')
        if not isinstance(models, list) or len(set(models)) != len(models) or any(
                not isinstance(slug, str) or not slug or Path(slug).name != slug or slug in ('.', '..')
                for slug in models):
            raise ValueError('invalid registered model slug')
        if any(p.name == '.git' for p in raw.rglob('.git')):
            raise ValueError('shadow workspace must have no Git repository')

    @classmethod
    def create(cls, path, models):
        raw = Path(path).absolute()
        if raw.resolve() != raw or raw == CODE or CODE in raw.parents or raw.exists():
            raise ValueError('use a new workspace outside the repository')
        for slug in models:
            if not slug or Path(slug).name != slug or slug in ('.', '..'):
                raise ValueError('invalid model slug')
            safe_tree(CODE / 'models' / slug)
        raw.mkdir(parents=True)
        for slug in models:
            shutil.copytree(CODE / 'models' / slug, raw / 'models' / slug,
                            ignore=shutil.ignore_patterns('site', 'site_dev', '__pycache__', '.gatecache*'))
        write(raw / 'runner.json', {'version': 1, 'mode': 'shadow', 'models': models,
                                    'input_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()})
        return cls(raw)

    @classmethod
    def create_ingest(cls, path, slug, model):
        from qcl_fetch import valid_model_id
        if not valid_model_id(model) or not slug or Path(slug).name != slug or slug in ('.', '..'):
            raise ValueError('invalid new model identity')
        w = cls.create(path, [])
        root = w.path / 'models' / slug
        write(root / 'model.json', {'model': model, 'slug': slug, 'name': model.split('/')[-1],
                                   'others': [], 'max_size_gb': 1000, 'size_note': 'Sizes as reported by sources.',
                                   'pin_note': 'Direct HTTP sources pinned by commit and exact byte hash.',
                                   'coverage_note': 'Claims only; incomplete sources remain blocked.'})
        write(root / 'audit/runner_fetch.json', {'status': 'blocked', 'reason': 'new model requires collection'})
        w.config.update(models=[slug], ingestion=True)
        write(w.path / 'runner.json', w.config)
        return cls(w.path)

    def model(self, slug):
        if slug not in self.config['models']:
            raise ValueError('model is not registered in this workspace')
        result = self.path / 'models' / slug
        if result.resolve() != result or not result.is_relative_to(self.path / 'models'):
            raise ValueError('unsafe model path')
        return result

    @contextmanager
    def lock(self):
        # Never held while awaiting a provider. Kernel releases on process death.
        with open(self.path / 'orchestrator.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('another orchestrator owns this shadow workspace') from None
            yield

    def run(self, slug, script, *args):
        if script not in ALLOWED:
            raise ValueError('operation has no shadow capability: ' + script)
        root = self.model(slug)
        safe_tree(self.path)
        source = CODE / script if script == 'build.py' else CODE / 'pipeline' / script
        env = {k: v for k, v in os.environ.items() if not k.startswith('QCL_')}
        env['QCL_ROOT'] = str(root)
        # No shell, caller-supplied executable, production state machine or git.
        result = subprocess.run([sys.executable, str(source), *map(str, args)], cwd=self.path,
                                env=env, capture_output=True, text=True)
        with open(self.path / 'operations.jsonl', 'a') as log:
            log.write(json.dumps({'model': slug, 'script': script, 'rc': result.returncode}) + '\n')
        if result.returncode:
            raise ValueError(f'{script} failed ({result.returncode}): {(result.stdout + result.stderr)[-1200:]}')
        return result.stdout

    def publish(self):
        raise ValueError('shadow runner has no publication capability')

    def backup(self, target, state):
        target = Path(target).absolute()
        if target.exists() or target.resolve() != target or target == CODE or CODE in target.parents or self.path in target.parents:
            raise ValueError('backup needs a new external directory')
        with self.lock(), ExitStack() as stack:
            for path in sorted((self.path / 'ledgers').glob('*/collector.lock')):
                lock = stack.enter_context(open(path, 'a'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError('collector active; retry backup after it checkpoints') from None
            safe_tree(self.path)
            shutil.copytree(self.path, target, ignore=shutil.ignore_patterns('state.sqlite*', 'orchestrator.lock'))
            state.backup(target / 'state.sqlite')
            manifest = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in target.rglob('*') if p.is_file()}
            write(target / 'backup-manifest.json', manifest)
        return target


def restore(source, target):
    source, target = Path(source), Path(target).absolute()
    safe_tree(source)
    if target.exists() or target.resolve() != target or target == CODE or CODE in target.parents:
        raise ValueError('restore needs a new external directory')
    manifest = read(source / 'backup-manifest.json')
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file() and p.name != 'backup-manifest.json'}
    if manifest != actual:
        raise ValueError('backup missing, altered or extra files')
    shutil.copytree(source, target)
    (target / 'backup-manifest.json').unlink()
    return Workspace(target)
