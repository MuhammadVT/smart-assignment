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

### 1. Read both files in parallel

Read the full contents of `.env` and `.env.example` side by side.

### 2. Diff: find what is new in `.env.example`

Identify everything present in `.env.example` but absent from `.env`:
- **New variables** (uncommented `KEY=value` lines not in `.env`)
- **New commented-out variables** (e.g. `# KEY=value`) that serve as documentation
- **New comment blocks / section headers** that provide context
- **Updated header comments** (e.g. new usage notes at the top of the file)

### 3. Diff: find local-only keys in `.env`

Identify keys present in `.env` but absent from `.env.example`:
- **Local-only variables** (uncommented `KEY=value` lines not declared in `.env.example`)
- Mark these keys for **removal** so `.env` matches the template key set

### 4. Identify overlapping params — preserve `.env` values

For any key that exists in **both** files:
- Keep the **value from `.env`** (never overwrite with the placeholder from `.env.example`)
- Only bring in structural changes: updated inline comments on the same line, if the value itself is not changed

### 5. Apply edits

Use `multi_replace_string_in_file` to apply all changes in one pass:
- Insert new sections at the same relative position they appear in `.env.example`
- Preserve blank lines and section separators to maintain readability
- For new variables, use the **default value from `.env.example`** (since no override exists in `.env`)
- For updated header/section comments, splice in the new text while keeping the rest of the file intact
- Remove any local-only `KEY=value` lines that do not exist in `.env.example`

### 6. Verify

After editing, re-read the affected regions to confirm:
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
| Comment/section only in `.env.example` | Add to `.env` at the matching position |
| Header comment updated in `.env.example` | Merge new lines into `.env` header |

## Notes

- Never overwrite real credentials with placeholder strings like `your_*_here`
- Maintain the same section ordering as `.env.example` for consistency
- If `.env` does not exist yet, create it as a direct copy of `.env.example`
- If a local-only key is still required, add it to `.env.example` first, then sync
