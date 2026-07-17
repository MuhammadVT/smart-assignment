#!/usr/bin/env bash
# Stand up a local, self-hosted Langfuse instance under Podman for reviewing
# eval-run traces and attaching human feedback (scores). See README.md in this
# directory for the full workflow (eval -> traces -> annotation queue).
#
# This script does not maintain its own copy of the Langfuse stack -- it
# fetches Langfuse's own docker-compose.yml (cached locally after the first
# run) and drives it with whichever Podman compose front-end is available, so
# it never drifts from the upstream service topology, images, or version pins.
#
# Usage:
#   ./podman-langfuse.sh up             # fetch (if needed) + start the stack
#   ./podman-langfuse.sh up --refresh   # re-fetch the compose file, then start
#   ./podman-langfuse.sh status         # show container state
#   ./podman-langfuse.sh logs [-f]      # show/follow web+worker logs
#   ./podman-langfuse.sh down           # stop, keep data volumes
#   ./podman-langfuse.sh down --volumes # stop and delete all Langfuse data
#
# Requires: podman, and either `podman compose` (Podman >= 4.1) or the
# standalone `podman-compose` on PATH. Requires network access on first `up`
# to fetch the compose file from raw.githubusercontent.com.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
COMPOSE_URL="https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml"
PROJECT_NAME="smart-assignment-langfuse"

log() { printf '[podman-langfuse] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

require_podman() {
    command -v podman >/dev/null 2>&1 || die "podman not found on PATH. Install Podman first."
}

# Resolve a compose front-end once and cache the chosen invocation in COMPOSE[].
resolve_compose_cmd() {
    if podman compose version >/dev/null 2>&1; then
        COMPOSE=(podman compose)
    elif command -v podman-compose >/dev/null 2>&1; then
        COMPOSE=(podman-compose)
    else
        die "Neither 'podman compose' nor 'podman-compose' is available. Install Podman >= 4.1 (built-in compose) or 'pip install podman-compose'."
    fi
}

fetch_compose_file() {
    local refresh="${1:-false}"
    if [[ -f "${COMPOSE_FILE}" && "${refresh}" != "true" ]]; then
        log "Using cached compose file at ${COMPOSE_FILE} (pass --refresh to re-fetch)."
        return
    fi
    log "Fetching Langfuse's docker-compose.yml from upstream..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${COMPOSE_URL}" -o "${COMPOSE_FILE}.tmp"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "${COMPOSE_URL}" -O "${COMPOSE_FILE}.tmp"
    else
        die "Neither curl nor wget is available to fetch the compose file."
    fi
    mv "${COMPOSE_FILE}.tmp" "${COMPOSE_FILE}"
    log "Saved to ${COMPOSE_FILE}."
}

cmd_up() {
    local refresh=false
    for arg in "$@"; do
        [[ "${arg}" == "--refresh" ]] && refresh=true
    done
    require_podman
    resolve_compose_cmd
    fetch_compose_file "${refresh}"
    log "Starting Langfuse (web, worker, postgres, clickhouse, redis, minio)..."
    "${COMPOSE[@]}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" up -d
    log "Up. Web UI: http://localhost:3000 (first visit: sign up to create the admin account)."
    log "Dev-only secrets ship in the upstream compose -- do not expose this beyond localhost."
}

cmd_status() {
    require_podman
    resolve_compose_cmd
    [[ -f "${COMPOSE_FILE}" ]] || die "No compose file yet -- run '$0 up' first."
    "${COMPOSE[@]}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" ps
}

cmd_logs() {
    require_podman
    resolve_compose_cmd
    [[ -f "${COMPOSE_FILE}" ]] || die "No compose file yet -- run '$0 up' first."
    "${COMPOSE[@]}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" logs "$@" langfuse-web langfuse-worker
}

cmd_down() {
    local volumes=()
    for arg in "$@"; do
        [[ "${arg}" == "--volumes" ]] && volumes=(--volumes)
    done
    require_podman
    resolve_compose_cmd
    [[ -f "${COMPOSE_FILE}" ]] || { log "No compose file found; nothing to stop."; return 0; }
    if [[ "${#volumes[@]}" -gt 0 ]]; then
        log "Stopping Langfuse and deleting all data volumes..."
    else
        log "Stopping Langfuse (data volumes kept)..."
    fi
    "${COMPOSE[@]}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" down "${volumes[@]}"
}

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed '1d;s/^# \{0,1\}//'
}

main() {
    local sub="${1:-}"
    [[ -n "${sub}" ]] && shift || true
    case "${sub}" in
        up)      cmd_up "$@" ;;
        status)  cmd_status "$@" ;;
        logs)    cmd_logs "$@" ;;
        down)    cmd_down "$@" ;;
        ""|-h|--help|help) usage ;;
        *) die "Unknown subcommand '${sub}'. Run '$0 --help'." ;;
    esac
}

main "$@"
