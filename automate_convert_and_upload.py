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
import config

# import helpers from core and gemini modules
from core import (
    setup_logging,
    run_command,
    run_command_no_raise,
    resolve_chromium_src,
    read_file_paths_raw,
    remove_processed_path,
    ensure_clean_workspace,
    current_branch,
    head_sha,
    create_temporary_branch,
    checkout_branch,
    process_file_with_abs_path,
    run_single_web_test,
    file_has_changes,
    commit_and_upload,
)

from gemini import initialize_gemini_client

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
