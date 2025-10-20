#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import config

# Constants
GIT_CLEAN_CMD = ['git', 'clean', '-fd']
GIT_RESET_HARD_CMD = ['git', 'reset', '--hard']
GIT_CL_UPLOAD_BASE_CMD = ['git', 'cl', 'upload', '--send-mail', '--force']
MAX_API_RETRIES = 4
API_BACKOFF_BASE = 1.5

# --- Logging ---
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Subprocess helpers (cwd optional) ---
def run_command(cmd, check=True, capture_output=True, text=True, cwd=None):
    logging.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    try:
        proc = subprocess.run(cmd, check=check, capture_output=capture_output, text=text, cwd=cwd)
        return (proc.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(e.cmd)} (exit {e.returncode})\n{stderr}") from e

def run_command_no_raise(cmd, cwd=None):
    logging.debug("Running (no-raise): %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)

# --- Resolve chromium/src ---
def resolve_chromium_src():
    cfg = getattr(config, 'CHROMIUM_SRC_PATH', None)
    if cfg:
        p = Path(cfg).expanduser().resolve()
        if p.is_dir():
            return str(p)
        logging.warning("Configured CHROMIUM_SRC_PATH not found: %s", p)

    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent / 'src'
    if candidate.is_dir():
        return str(candidate.resolve())

    candidate2 = script_dir.parent.parent / 'src'
    if candidate2.is_dir():
        return str(candidate2.resolve())

    raise RuntimeError("Could not resolve chromium/src path. Set CHROMIUM_SRC_PATH in config.py.")

# --- Queue helpers ---
def read_file_paths_raw(queue_file_path):
    try:
        with open(queue_file_path, 'r', encoding='utf-8') as fh:
            return [line.rstrip('\n') for line in fh.readlines() if line.strip()]
    except FileNotFoundError:
        logging.error("Queue file not found: %s", queue_file_path)
        return []

def remove_processed_path(queue_file_path, processed_path):
    processed = processed_path.strip()
    try:
        with open(queue_file_path, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
        with open(queue_file_path, 'w', encoding='utf-8') as fh:
            for line in lines:
                if line.strip() != processed:
                    fh.write(line)
    except IOError as e:
        logging.error("Failed to update queue file %s: %s", queue_file_path, e)

# --- Gemini integration (delegated to gemini.py) ---
def initialize_gemini_client():
    # delegate to gemini module to keep exact behavior; import local to avoid circular import
    from gemini import initialize_gemini_client as _init
    return _init()

def call_gemini_with_retries(model, prompt):
    from gemini import call_gemini_with_retries as _call
    return _call(model, prompt)

# --- Prompt builder (delegated to gemini.py) ---
def build_prompt(file_path, original_content):
    from gemini import build_prompt as _build
    return _build(file_path, original_content)

# --- File processing ---
def process_file_with_abs_path(model, abs_path):
    logging.info("Processing file on disk: %s", abs_path)
    # delegate full conversion to gemini.convert_file_with_gemini to preserve original code exactly
    from gemini import convert_file_with_gemini
    return convert_file_with_gemini(model, abs_path)

# --- Git helpers (cwd = chromium/src) ---
def ensure_clean_workspace(cwd):
    run_command(GIT_RESET_HARD_CMD, cwd=cwd)
    run_command(GIT_CLEAN_CMD, cwd=cwd)

def current_branch(cwd):
    return run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=cwd)

def head_sha(cwd):
    return run_command(['git', 'rev-parse', 'HEAD'], cwd=cwd)

def get_next_automate_branch(cwd):
    """
    Find existing local branches named automateN and pick next N.
    If none exist, return 'automate1'.
    """
    proc = run_command_no_raise(['git', 'branch', '--list', 'automate*'], cwd=cwd)
    out = (proc.stdout or "").strip()
    max_n = 0
    if out:
        for line in out.splitlines():
            name = line.strip().lstrip('*').strip()
            if name.startswith('automate'):
                suffix = name[len('automate'):]
                if suffix.isdigit():
                    n = int(suffix)
                    if n > max_n:
                        max_n = n
    return f"automate{max_n + 1}"

def create_temporary_branch(cwd):
    branch = get_next_automate_branch(cwd)
    run_command(['git', 'checkout', '-b', branch], cwd=cwd)
    return branch

def checkout_branch(branch, cwd):
    run_command(['git', 'checkout', branch], cwd=cwd)

def file_has_changes(filepath, cwd):
    proc = run_command_no_raise(['git', 'diff', '--quiet', '--', filepath], cwd=cwd)
    if proc.returncode != 0:
        return True
    proc2 = run_command_no_raise(['git', 'diff', '--cached', '--quiet', '--', filepath], cwd=cwd)
    return proc2.returncode != 0

def commit_and_upload(filepath, cwd):
    run_command(['git', 'add', filepath], cwd=cwd)
    filename = os.path.basename(filepath)
    title = f"[webaudio-testharness] Migrate {filename}"
    body = f"Convert {filename} from the legacy audit.js runner to pure testharness.js\n\nBug: {config.BUG_ID}"
    msg = f"{title}\n\n{body}"
    run_command(['git', 'commit', '-m', msg], cwd=cwd)

    cmd = list(GIT_CL_UPLOAD_BASE_CMD)
    if getattr(config, 'GERRIT_REVIEWERS', None):
        cmd.extend(['-r', ','.join(config.GERRIT_REVIEWERS)])
    run_command(cmd, cwd=cwd)

# --- Run single web-test and capture logs ---
def run_single_web_test(chromium_src, rel_test_path):
    # place logs next to the chromium workspace (parent of chromium/src)
    logs_dir = Path(chromium_src).parent.resolve() / 'automate_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = rel_test_path.replace('/', '_').replace('\\', '_')
    log_path = logs_dir / f"{safe_name}.log"

    cmd = [
        './third_party/blink/tools/run_web_tests.py',
        '--target=Default',
        rel_test_path
    ]
    logging.info("Running web-test: %s", " ".join(cmd))
    proc = run_command_no_raise(cmd, cwd=chromium_src)

    with open(log_path, 'wb') as fh:
        if proc.stdout:
            fh.write(proc.stdout.encode('utf-8', errors='replace'))
        if proc.stderr:
            fh.write(b"\n=== STDERR ===\n")
            fh.write(proc.stderr.encode('utf-8', errors='replace'))

    if proc.returncode == 0:
        logging.info("Web test PASSED for %s (log: %s).", rel_test_path, log_path)
        return True
    else:
        logging.warning("Web test FAILED for %s (returncode=%s). Log: %s", rel_test_path, proc.returncode, log_path)
        return False
