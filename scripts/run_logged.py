#!/usr/bin/env python3
"""Execute an exact command and preserve its output, timing and return code."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    argv = sys.argv[1:]
    if argv and argv[0] == '--':
        argv = argv[1:]
    if not argv:
        raise SystemExit('Usage: python scripts/run_logged.py COMMAND [ARGS ...]')
    results = ROOT / os.environ.get('STITCH_RESULTS', 'results')
    logs = results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    record = {'id': stamp, 'argv': argv, 'command': shlex.join(argv),
              'cwd': os.getcwd(), 'started_utc': stamp}
    journal = results / 'commands.jsonl'
    # Persist intent before execution, including interrupted commands.
    with journal.open('a') as f:
        f.write(json.dumps(dict(record, event='start')) + '\n')
    start = time.perf_counter()
    environment = dict(os.environ, PYTHONPYCACHEPREFIX=str(ROOT/'.cache/pycache'),
                       MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),
                       XDG_CACHE_HOME=str(ROOT/'.cache'),
                       PIP_CACHE_DIR=str(ROOT/'.cache/pip'),
                       TORCH_HOME=str(ROOT/'models/torch'))
    with (logs / (stamp + '.log')).open('wb') as output:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=0, env=environment)
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line.decode('utf-8', errors='replace'), end='', flush=True)
        code = process.wait()
    record.update(event='end', returncode=code, elapsed_s=time.perf_counter()-start,
                  log=str((logs / (stamp+'.log')).relative_to(ROOT)))
    with journal.open('a') as f:
        f.write(json.dumps(record) + '\n')
    raise SystemExit(code)

if __name__ == '__main__':
    main()
