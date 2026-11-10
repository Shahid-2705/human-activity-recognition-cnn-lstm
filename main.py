"""
One command to run the whole project:

    python main.py

Order: (1) npm install if needed, (2) start the Node UI, (3) start the Python sensor API.
Both run in this terminal; press Ctrl+C to stop them together.

Options:  --no-browser   do not open the browser automatically
Env:      PORT (UI, default 3000), API_PORT (Python API, default 8000)
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
PORT = os.environ.get("PORT", "3000")
API_PORT = os.environ.get("API_PORT", "8000")
IS_WIN = os.name == "nt"


def venv_python():
    sub = os.path.join("Scripts", "python.exe") if IS_WIN else os.path.join("bin", "python")
    path = os.path.join(ROOT, ".venv", sub)
    return path if os.path.exists(path) else None


def ensure_project_python():
    """Re-run with the project's .venv interpreter when it exists and we are not using it."""
    py = venv_python()
    if py and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(py)):
        print(f"[main] Using project virtual environment: {py}")
        sys.exit(subprocess.call([py, os.path.abspath(__file__), *sys.argv[1:]]))


def ensure_python_packages():
    try:
        import fastapi, uvicorn, torch  # noqa: F401
    except ImportError:
        print("[main] Installing Python packages (first run can take a few minutes)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])


def npm_path():
    npm = shutil.which("npm.cmd" if IS_WIN else "npm") or shutil.which("npm")
    if not npm:
        sys.exit("[main] Node.js / npm not found. Install Node 18+ from https://nodejs.org and try again.")
    return npm


def ensure_node_packages(npm):
    if not os.path.isdir(os.path.join(WEB, "node_modules")):
        print("[main] 1/3 npm install ...")
        subprocess.check_call([npm, "install"], cwd=WEB)
    else:
        print("[main] 1/3 npm packages already installed.")


def pump(proc, tag):
    """Forward a child's output to this terminal with a prefix."""
    for line in iter(proc.stdout.readline, ""):
        print(f"[{tag}] {line.rstrip()}", flush=True)


def spawn(cmd, cwd, tag, env=None):
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    threading.Thread(target=pump, args=(proc, tag), daemon=True).start()
    return proc


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if IS_WIN:  # npm.cmd spawns node as a child; kill the whole tree
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()


def wait_for(url, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    ensure_project_python()
    ensure_python_packages()
    npm = npm_path()
    ensure_node_packages(npm)

    env = {**os.environ, "START_API": "0", "PORT": PORT, "API_PORT": API_PORT}
    procs = []
    try:
        print(f"[main] 2/3 starting Node UI on http://localhost:{PORT}")
        procs.append(spawn([npm, "start"], WEB, "node", env))

        print(f"[main] 3/3 starting Python API on http://127.0.0.1:{API_PORT} (loading model, ~20s)")
        procs.append(spawn(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", API_PORT],
            ROOT, "python", env,
        ))

        if wait_for(f"http://127.0.0.1:{API_PORT}/api/status", 120):
            print(f"\n[main] Ready -> http://localhost:{PORT}   (Ctrl+C to stop)\n", flush=True)
            if not args.no_browser:
                webbrowser.open(f"http://localhost:{PORT}")
        else:
            print("[main] Python API did not start in time; see the [python] log above.")

        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        print("[main] A process exited; shutting down.")
    except KeyboardInterrupt:
        print("\n[main] Stopping...")
    finally:
        for p in procs:
            kill_tree(p)


if __name__ == "__main__":
    main()
