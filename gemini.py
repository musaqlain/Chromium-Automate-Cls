#!/usr/bin/env python3
import os
import logging
import time
from pathlib import Path
import google.generativeai as genai
import config

# Constants (relevant for Gemini retries)
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

def build_prompt(file_path, original_content):
    return (
        f"{CONVERSION_INSTRUCTIONS}\n\n"
        f"File path: {file_path}\n"
        f"Original file content below (begin):\n---\n{original_content}\n---\n"
        "Provide the full converted file content now. ONLY the file content — no commentary.\n"
    )

def convert_file_with_gemini(model, file_path):
    logging.info("Processing file on disk: %s", file_path)
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    original = p.read_text(encoding='utf-8')
    prompt = build_prompt(str(file_path), original)
    new_content = call_gemini_with_retries(model, prompt)

    if new_content.strip() == original.strip():
        logging.info("Gemini returned no changes for %s", file_path)
        return "no-change"

    tmp = p.with_suffix(p.suffix + '.converted.tmp')
    tmp.write_text(new_content, encoding='utf-8')
    os.replace(str(tmp), str(p))
    logging.info("Wrote converted content to %s", file_path)
    return "modified"
