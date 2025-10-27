import subprocess
import sys
import os

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Movie recommender.py'))

def test_smoke_runs_quickly():
    # Run a quick smoke with no evaluation and only 1 recommendation to keep test fast
    proc = subprocess.run([sys.executable, SCRIPT, '--no-eval', '--topn', '1', '--users', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    assert proc.returncode == 0, f"Script exited non-zero. stderr:\n{err}\nstdout:\n{out}"
    assert 'Top 1 hybrid recommendations' in out
