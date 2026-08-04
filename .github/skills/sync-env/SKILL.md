---
name: sync-env
description: "Sync .env.example into .env while preserving values for overlapping keys and pruning keys that exist only in .env. Use when .env.example has new params, updated comments, or new sections that should be merged into .env without overwriting real credentials already set for shared keys. Triggers: 'sync env', 'update .env from .env.example', 'merge env example', 'bring in new .env changes', 'env out of sync'."
argument-hint: "Optionally specify the path to .env or .env.example if non-standard"
---

# Sync .env.example → .env

Merges new keys, comments, and sections from `.env.example` into `.env`, preserving values for shared keys and removing keys that exist only in `.env`.

## When to Use

- `.env.example` has been updated with new parameters not yet in `.env`
- New comment blocks or sections were added to `.env.example`
- You want `.env` to match the template shape exactly (no stale local-only keys)

## Procedure

**Goal: `.env` and `.env.example` should be line-for-line identical, EXCEPT where a variable is active in `.env` with a different value than the placeholder in `.env.example`.**

### 1. Read both files in parallel

Read the full contents of `.env` and `.env.example` side by side, noting line structure and order.

### 2. Diff: find what is new in `.env.example`

Identify everything present in `.env.example` but absent from `.env`. The goal is full line-for-line parity:
- **Every comment line** in `.env.example` must appear in `.env` in the same position
- **Every blank line** must match
- **Every section header** must match
- **All documented variables** (including those documented as comments, even if the variable is active elsewhere) must be present
- **Every KEY=value that is not overridden in `.env`** must be copied from `.env.example`

Specifically, identify:
- Comment lines not yet in `.env` (including multi-line comment blocks)
- Section headers or dividers not yet in `.env`
- Commented-out variables (documentation examples) not yet in `.env`
- Blank lines missing from `.env`

### 3. Identify and remove local-only keys from `.env`

Identify any uncommented `KEY=value` lines in `.env` that do **not** exist in `.env.example`:
- These are **local-only variables** unique to your `.env` setup
- **Remove them** — they should not appear in the synced result
- This ensures `.env` matches `.env.example`'s key set exactly (modulo active values)

### 4. Identify overlapping params — preserve `.env` values

For any key that exists in **both** files:
- Keep the **value from `.env`** (never overwrite with the placeholder from `.env.example`)
- If the key is **active in `.env` but commented in `.env.example`**, keep **one active declaration only** in `.env`:
  - Do **not** keep/add a duplicate commented `# KEY=...` example line for that key
  - Place/preserve the active `KEY=value` at the template location for that key
  - Remove any duplicate second declaration elsewhere in `.env`
- Preserve all surrounding documentation and structure

### 5. Apply edits

Use `multi_replace_string_in_file` to apply all changes in one pass:
- Insert new sections at the same relative position they appear in `.env.example`
- Preserve blank lines and section separators to maintain readability
- For new variables, use the **default value from `.env.example`** (since no override exists in `.env`)
- For updated header/section comments, splice in the new text while keeping the rest of the file intact
- Remove any local-only `KEY=value` lines that do not exist in `.env.example`

### 6. Verify

After editing, re-read the affected regions to confirm:
- **Line-for-line parity**: Every line in `.env.example` (comments, blanks, structure) now appears in `.env`, EXCEPT that active variables in `.env` use their actual values instead of the `.env.example` placeholders
- All new keys from `.env.example` are now present in `.env`
- No existing `.env` values were changed
- No placeholder values (e.g. `your_*_here`) were introduced for keys that already had real values
- No local-only keys remain in `.env` (every active key exists in `.env.example`)

## Rules

| Situation | Action |
|-----------|--------|
| Key in `.env.example` only | Add to `.env` with the example's default value |
| Key in both files | Keep `.env` value; do not touch it |
| Key in `.env` only | Remove from `.env` |
| Key commented in `.env.example`, active in `.env` | Keep a single active `KEY=value` line (using `.env` value) at that position; do not also keep/add `# KEY=...` for the same key |
| Comment/section only in `.env.example` | Add to `.env` at the matching position |
| Comment text updated in `.env.example` | Replace old comment text in `.env` with the new wording |
| Header comment updated in `.env.example` | Merge new lines into `.env` header |

## Notes

- **Line-for-line parity is the goal**: `.env` and `.env.example` should be structurally identical, with only the values of active variables differing in `.env`.
- Never overwrite real credentials with placeholder strings like `your_*_here`
- Maintain the same section ordering as `.env.example` for consistency
- If `.env` does not exist yet, create it as a direct copy of `.env.example`
- If a local-only key is still required, add it to `.env.example` first, then sync
- **Single declaration per key**: If a key is active in `.env`, keep exactly one active `KEY=value` declaration. Do not keep/add an extra `# KEY=...` commented assignment line for the same key.
- **Avoid duplicate variable declarations in comments**: If a variable name appears inside a comment block (e.g., as an example or explanation like `# SMART_ASSIGNMENT_USE_ROUTE_SLOT_SCORING=false`), and that variable is already active elsewhere in the file, do NOT uncomment or activate it mid-comment. The comment line(s) should remain as pure comments. The active variable declaration should appear only once at its designated location. Preserve the comment verbatim from `.env.example` to maintain documentation context.
