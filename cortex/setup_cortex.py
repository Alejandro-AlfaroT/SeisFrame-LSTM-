"""One-command CORTEX setup for a Windows machine.

From the repository root, in the PyCharm terminal (OpPy environment):

    python cortex/setup_cortex.py

Creates the backend's Python environment, installs and builds the interface, imports the
SeisFrame Obsidian vault from OneDrive, adds a CORTEX shortcut to the desktop and Start menu,
and starts CORTEX. Safe to run again: finished steps are skipped or refreshed, and re-importing
only updates notes that changed on disk.

Options: --vault PATH (vault folder, if it is not found in OneDrive), --skip-import,
--no-shortcut, --no-start, --yes (answer yes to the Node.js install prompt).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CORTEX = Path(__file__).resolve().parent
BACKEND = CORTEX / "backend"
FRONTEND = CORTEX / "frontend"
VENV_PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
LAUNCHER = CORTEX / "Start CORTEX.cmd"
ICON = CORTEX / "assets" / "cortex.ico"
VAULT_NAME = "SeisFrame-Obsidian-Starter"
PROJECT = "SeisFrame V6"
STEPS = 5


class SetupError(Exception):
    pass


def step(number: int, text: str) -> None:
    print(f"\n[{number}/{STEPS}] {text}", flush=True)


def run(cmd: list[object], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    args = [str(c) for c in cmd]
    print("    > " + " ".join(f'"{a}"' if " " in a else a for a in args), flush=True)
    if subprocess.run(args, cwd=cwd, env=env).returncode != 0:
        raise SetupError(f"this command failed: {Path(args[0]).name} {' '.join(args[1:3])} ...")


def ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def python_environment() -> None:
    if sys.version_info < (3, 11):
        raise SetupError(
            f"CORTEX needs Python 3.11 or newer; this is {sys.version.split()[0]} ({sys.executable}).\n"
            "    Activate the OpPy environment (conda activate OpPy) and run this again."
        )
    if VENV_PYTHON.exists():
        print("    environment already exists: cortex/backend/.venv")
    else:
        run([sys.executable, "-m", "venv", BACKEND / ".venv"])
    probe = (
        "import ssl, sqlite3; "
        "sqlite3.connect(':memory:').execute('create virtual table t using fts5(x)')"
    )
    if subprocess.run([str(VENV_PYTHON), "-c", probe]).returncode != 0:
        raise SetupError(
            "the new environment cannot load SSL or SQLite full-text search.\n"
            "    Delete cortex/backend/.venv and run this script with a python.org Python 3.12\n"
            "    (winget install Python.Python.3.12, then: py -3.12 cortex/setup_cortex.py)."
        )
    run([VENV_PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", BACKEND / "requirements.txt"])


def find_npm() -> Path | None:
    found = shutil.which("npm")
    if found:
        return Path(found)
    for folder in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs",
    ):
        if (folder / "npm.cmd").exists():
            return folder / "npm.cmd"
    return None


def build_interface(assume_yes: bool) -> None:
    npm = find_npm()
    if npm is None:
        print("    Node.js is not installed. It is needed once, to build the interface.")
        if shutil.which("winget") and ask("    Install Node.js LTS now with winget? [y/N] ", assume_yes):
            run(["winget", "install", "--id", "OpenJS.NodeJS.LTS", "--exact", "--source", "winget",
                 "--accept-package-agreements", "--accept-source-agreements"])
            npm = find_npm()
        if npm is None:
            raise SetupError(
                "install Node.js LTS (https://nodejs.org, or: winget install OpenJS.NodeJS.LTS), then run this again."
            )
    # A fresh Node install is not on this terminal's PATH yet; npm's build scripts need `node`.
    env = {**os.environ, "PATH": f"{npm.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    run([npm, "ci", "--no-audit", "--no-fund"], cwd=FRONTEND, env=env)
    run([npm, "run", "build"], cwd=FRONTEND, env=env)


def find_vault(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise SetupError(f"--vault folder not found: {path}")
        return path
    roots = [os.environ.get(k) for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    roots.append(str(Path.home() / "OneDrive"))
    # The vault has lived at the OneDrive root and under ResearchProject; look up to two levels down.
    for root in dict.fromkeys(Path(r) for r in roots if r):
        for pattern in (VAULT_NAME, f"*/{VAULT_NAME}", f"*/*/{VAULT_NAME}"):
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_dir() and not any(part.startswith(".") for part in candidate.relative_to(root).parts):
                    return candidate
    return None


def import_vault(explicit: str | None) -> None:
    vault = find_vault(explicit)
    if vault is None:
        print(f"    {VAULT_NAME} was not found in OneDrive; skipping the import.")
        print('    Import it later with: python cortex/setup_cortex.py --vault "<folder>"')
        return
    print(f"    vault: {vault}")
    run([VENV_PYTHON, BACKEND / "run.py", "import-md", vault, "--route", "by_type", "--project", PROJECT])


def ps_quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def shortcuts() -> None:
    script = f"""
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
  $link = Join-Path $dir 'CORTEX.lnk'
  $s = $shell.CreateShortcut($link)
  $s.TargetPath = {ps_quote(LAUNCHER)}
  $s.WorkingDirectory = {ps_quote(CORTEX)}
  $s.IconLocation = {ps_quote(f"{ICON},0")}
  $s.Description = 'Start CORTEX and open it in the browser'
  $s.Save()
  Write-Output "    shortcut: $link"
}}
"""
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SetupError(f"could not create the shortcuts: {result.stderr.strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up CORTEX on this Windows machine.")
    parser.add_argument("--vault", help=f"path to the {VAULT_NAME} folder (default: found in OneDrive)")
    parser.add_argument("--skip-import", action="store_true", help="do not import the vault")
    parser.add_argument("--no-shortcut", action="store_true", help="do not create desktop / Start menu shortcuts")
    parser.add_argument("--no-start", action="store_true", help="do not start CORTEX at the end")
    parser.add_argument("--yes", action="store_true", help="install Node.js without asking if it is missing")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print("This setup script is for Windows.", file=sys.stderr)
        return 2
    print(f"Setting up CORTEX in {CORTEX}")
    try:
        step(1, "Python environment for the server")
        python_environment()
        step(2, "Building the interface")
        build_interface(args.yes)
        step(3, "Importing the SeisFrame vault")
        if args.skip_import:
            print("    skipped (--skip-import)")
        else:
            import_vault(args.vault)
        step(4, "Desktop and Start menu shortcuts")
        if args.no_shortcut:
            print("    skipped (--no-shortcut)")
        else:
            shortcuts()
        step(5, "Starting CORTEX")
        if args.no_start:
            print("    skipped (--no-start)")
        else:
            os.startfile(LAUNCHER)  # same as double-clicking it: starts the server, opens the browser
            print("    CORTEX is starting; your browser will open at http://127.0.0.1:8420")
    except SetupError as exc:
        print(f"\nSetup stopped: {exc}", file=sys.stderr)
        return 1
    print("\nDone. Next time, double-click CORTEX on the desktop or search for CORTEX in the Start menu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
