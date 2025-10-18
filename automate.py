#!/usr/bin/env python3
"""
automate_convert_and_upload.py
Placed in: Automate Cl's

Behavior:
- Reads file paths from file_paths.txt in the script directory.
- Interprets each path as relative to chromium/src (config.CHROMIUM_SRC_PATH) unless absolute.
- Runs all git operations inside chromium/src.
- DOES NOT run gclient sync or any pre-sync actions.
- Uses branch names automate1, automate2, ... and never deletes branches.
- Runs the specified single web test per file; if it fails, the file remains in the queue.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import config

# Constants
GIT_CLEAN_CMD = ['git', 'clean', '-fd']
GIT_RESET_HARD_CMD = ['git', 'reset', '--hard']
GIT_CL_UPLOAD_BASE_CMD = ['git', 'cl', 'upload', '--send-mail', '--force']
MAX_API_RETRIES = 4
API_BACKOFF_BASE = 1.5

# Conversion prompt (strict mechanical conversion)
CONVERSION_INSTRUCTIONS = """
You are an expert Chromium WebAudio engineer. Perform a strict, mechanical conversion
of the provided legacy `audit.js`-based test file into an equivalent `testharness.js`
file. DO NOT add explanations, new behavior, tests, or examples. Output only the full
converted file content — nothing else.

MANDATORY RULES (apply every point exactly once; do not add extra code or commentary):

1. Preserve comments exactly as they appear in the original file. Do not modify, move,
   or delete any comment text or formatting.

2. Preserve original function, variable, and constant names exactly as they are.
   - Only rename identifiers when the original name is clearly and unambiguously
     too vague (for example a single-letter variable used widely with ambiguous intent).
   - If you rename to improve clarity, do so minimally and do NOT add inline comments
     or extra explanatory text.

3. Use `const` for values that are immutable; otherwise use `let`/`var` as appropriate
   to keep semantics unchanged.

4. Replace legacy factory calls with modern constructors where equivalent constructors
   exist (e.g., prefer `new AudioBuffer(...)` over deprecated factory helpers).

5. Keep Chromium test support files intact. Do NOT remove or change references to
   helper infrastructure (audit-util.js, audioparam-testing.js, etc.). The only allowed
   removal is the test runner dependency on `audit.js` itself — remove references to
   `audit.js` and convert runner-specific calls accordingly.

6. Replace `should()`/audit-runner-specific assertions with testharness equivalents,
   preserving assertion semantics exactly.

7. Replace `.beConstantValueOf(x)` occurrences with:
   `assert_array_equals(actual, new Float32Array(actual.length).fill(x), desc)`

8. When a promise chain wraps a `context.startRendering()` or similar API, convert to
   `await` form while preserving behavior. Example:
   `return context.startRendering().then(buffer => { ... })`
   becomes:
   `const resultBuffer = await context.startRendering(); ...`

9. Where the test body is entirely synchronous (no `await`, no Promises, no asynchronous
   helpers), use `test(function () { ... });` (the synchronous `test()` form). Where the
   body needs async, use `async` and `await` idioms used in WPT.

10. Prefer arrow function syntax for anonymous functions where it's a safe, behavior-neutral
    mechanical replacement (e.g. `function(x) {` -> `x => {`). Do not change `this`-sensitive
    functions where arrow functions would alter semantics.

11. Use template string literals (`` `...` ``) instead of string concatenation where that is
    a direct mechanical replacement and preserves semantics.

12. Merge tests only when they rely on shared initialized state in different files or
    separate tests such that conversion to testharness.js would create ordering hazards.
    - In short: if audit.js relies on sequential execution (shared state across test bodies),
      convert those dependent tests into a single consolidated test so behavior remains identical.
    - Do not invent new test logic beyond merging to preserve execution order.

13. Do not add or remove helper files, new dependencies, or test assertions except as required
    to migrate away from `audit.js` runner calls. The goal is a mechanical conversion, nothing else.

14. DO NOT add inline commentary, logging, or metadata into the produced file. The converted
    file must be a valid testharness.js test file and nothing else.

If any transformation is ambiguous, default to preserving original behavior and names.
Now: convert the file content (which follows) into the new file content and return only
the converted file (no explanation, no metadata, no leading/trailing whitespace).
"""

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

# --- Gemini integration ---
def initialize_gemini_client():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment (.env missing or key absent).")
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    except Exception:
        return genai

def call_gemini_with_retries(model, prompt):
    last_exc = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            logging.info("Calling Gemini (attempt %d/%d)...", attempt, MAX_API_RETRIES)
            if hasattr(model, 'generate_content'):
                resp = model.generate_content(prompt)
                new_text = getattr(resp, 'text', None) or getattr(resp, 'content', None) or str(resp)
            elif hasattr(model, 'generate'):
                resp = model.generate(prompt)
                new_text = getattr(resp, 'text', None) or getattr(resp, 'content', None) or str(resp)
            elif hasattr(genai, 'generate'):
                resp = genai.generate(model=config.GEMINI_MODEL_NAME, prompt=prompt)
                new_text = resp.get('output', resp.get('text', None)) or str(resp)
            elif callable(model):
                resp = model(prompt)
                new_text = getattr(resp, 'text', None) or getattr(resp, 'content', None) or str(resp)
            else:
                raise RuntimeError("No known generate method available on Gemini client.")
            if not new_text or not str(new_text).strip():
                raise RuntimeError("Gemini returned empty response.")
            return str(new_text)
        except Exception as e:
            last_exc = e
            wait = (API_BACKOFF_BASE ** (attempt - 1)) + (attempt * 0.1)
            logging.warning("Gemini attempt %d failed: %s. Backing off %.1fs", attempt, e, wait)
            time.sleep(wait)
    raise RuntimeError("All Gemini attempts failed.") from last_exc

# --- Prompt builder ---
def build_prompt(file_path, original_content):
    return (
        f"{CONVERSION_INSTRUCTIONS}\n\n"
        f"File path: {file_path}\n"
        f"Original file content below (begin):\n---\n{original_content}\n---\n"
        "Provide the full converted file content now. ONLY the file content — no commentary.\n"
    )

# --- File processing ---
def process_file_with_abs_path(model, abs_path):
    logging.info("Processing file on disk: %s", abs_path)
    p = Path(abs_path)
    if not p.exists():
        raise FileNotFoundError(f"{abs_path} does not exist")

    original = p.read_text(encoding='utf-8')
    prompt = build_prompt(str(abs_path), original)
    new_content = call_gemini_with_retries(model, prompt)

    if new_content.strip() == original.strip():
        logging.info("Gemini returned no changes for %s", abs_path)
        return "no-change"

    tmp = p.with_suffix(p.suffix + '.converted.tmp')
    tmp.write_text(new_content, encoding='utf-8')
    os.replace(str(tmp), str(p))
    logging.info("Wrote converted content to %s", abs_path)
    return "modified"

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

# --- Main ---
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Convert Chromium tests and upload to Gerrit.")
    parser.add_argument('n', type=int, help="Number of file paths to process from queue.")
    args = parser.parse_args()

    # Load .env from script dir
    script_dir = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=script_dir / '.env')
    logging.info("Loaded .env from %s", script_dir)

    # Resolve chromium/src
    try:
        chromium_src = resolve_chromium_src()
    except Exception as e:
        logging.critical("Cannot locate chromium/src: %s", e)
        sys.exit(1)
    logging.info("Using chromium/src at: %s", chromium_src)

    # Initialize Gemini model
    try:
        model = initialize_gemini_client()
    except Exception as e:
        logging.critical("Gemini init failed: %s", e)
        sys.exit(1)

    # Read raw paths from queue file (next to script)
    queue_file = script_dir / config.INPUT_FILE_PATH
    raw_paths = read_file_paths_raw(queue_file)
    if not raw_paths:
        logging.info("No paths in queue.")
        return

    # Translate raw paths -> absolute paths on disk (relative to chromium/src)
    abs_path_map = []
    for raw in raw_paths[:args.n]:
        raw_stripped = raw.strip()
        if os.path.isabs(raw_stripped):
            abs_path = raw_stripped
        else:
            abs_path = str(Path(chromium_src) / raw_stripped)
        abs_path_map.append((raw_stripped, abs_path))

    original_branch = current_branch(cwd=chromium_src)
    original_head = head_sha(cwd=chromium_src)
    logging.info("Repo original branch=%s head=%s", original_branch, original_head)

    for raw_rel, abs_path in abs_path_map:
        tmp_branch = None
        try:
            logging.info("=== START %s (absolute: %s) ===", raw_rel, abs_path)

            # ensure clean workspace inside chromium/src
            ensure_clean_workspace(cwd=chromium_src)

            # create temp branch inside chromium/src with name automateN (never delete)
            tmp_branch = create_temporary_branch(cwd=chromium_src)
            logging.info("Created branch: %s", tmp_branch)

            # Convert the file (abs_path is used to read/write the file on disk)
            result = process_file_with_abs_path(model, abs_path)

            if result == "no-change":
                # nothing to commit; switch back and leave branch as-is; remove from queue
                checkout_branch(original_branch, cwd=chromium_src)
                logging.info("No changes for %s. Removed from queue.", raw_rel)
                remove_processed_path(queue_file, raw_rel)
                continue

            # run the web test on the converted file (relative path)
            relpath_for_git = os.path.relpath(abs_path, start=chromium_src)
            test_ok = run_single_web_test(chromium_src, relpath_for_git)
            if not test_ok:
                logging.warning("Test failed for %s; aborting commit and leaving file in queue.", relpath_for_git)
                # rollback repo to original HEAD (but keep branch)
                try:
                    checkout_branch(original_branch, cwd=chromium_src)
                    run_command(['git', 'reset', '--hard', original_head], cwd=chromium_src)
                except Exception as e:
                    logging.error("Rollback after test failure encountered an error: %s", e)
                # DO NOT remove from queue; DO NOT delete branch
                continue

            # Only commit/upload if git sees changes relative to HEAD
            if not file_has_changes(relpath_for_git, cwd=chromium_src):
                logging.info("Git sees no changes for %s after conversion; skipping commit and removing from queue.", raw_rel)
                checkout_branch(original_branch, cwd=chromium_src)
                remove_processed_path(queue_file, raw_rel)
                continue

            # commit & upload
            commit_and_upload(relpath_for_git, cwd=chromium_src)

            # success -> switch back, keep branch (do not delete), remove from queue
            checkout_branch(original_branch, cwd=chromium_src)
            remove_processed_path(queue_file, raw_rel)
            logging.info("Uploaded and removed %s from queue. Note: branch %s retained.", raw_rel, tmp_branch)

        except KeyboardInterrupt:
            logging.warning("Interrupted by user. Attempting rollback.")
            try:
                checkout_branch(original_branch, cwd=chromium_src)
                run_command(['git', 'reset', '--hard', original_head], cwd=chromium_src)
            except Exception as e:
                logging.error("Rollback failed: %s", e)
            break
        except Exception as e:
            logging.error("Error processing %s: %s", raw_rel, e)
            logging.info("Rolling back to original branch & head.")
            try:
                checkout_branch(original_branch, cwd=chromium_src)
                run_command(['git', 'reset', '--hard', original_head], cwd=chromium_src)
            except Exception as re:
                logging.warning("Partial rollback failure: %s", re)
            logging.warning("Left %s in queue for retry. Branch %s (if created) is retained.", raw_rel, tmp_branch)
            continue

    logging.info("All done.")

if __name__ == "__main__":
    main()

