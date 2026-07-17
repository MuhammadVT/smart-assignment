#!/usr/bin/env bash
# Start/stop a local, self-hosted Arize Phoenix instance for reviewing eval-run
# traces and attaching human feedback (annotations). See README.md in this
# directory for the full workflow (eval -> traces -> annotate).
#
# Unlike deployment/langfuse/, this needs NO container runtime -- Phoenix runs
# as a single Python process (`phoenix serve`). Install it into its own
# virtualenv first (it is an external observability backend, not a repo
# dependency):
#     python3 -m venv ~/.venvs/phoenix && ~/.venvs/phoenix/bin/pip install arize-phoenix
#
# Usage:
#   ./phoenix.sh up             # start the server in the background
#   ./phoenix.sh status         # show whether it's running + the data dir
#   ./phoenix.sh logs [-f]      # show/follow server logs (-f to follow)
#   ./phoenix.sh down           # stop, keep local trace data (.data/)
#   ./phoenix.sh down --purge   # stop and delete .data/ (all local trace history)
#
# Env overrides:
#   PHOENIX_BIN           path to the `phoenix` executable (default: resolved via PATH,
#                         falling back to ~/.venvs/phoenix/bin/phoenix if present)
#   PHOENIX_WORKING_DIR   where Phoenix stores its local SQLite trace data
#                         (default: ./.data next to this script)
#   PHOENIX_PORT          UI + OTLP/HTTP port (default: Phoenix's own default, 6006)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
DATA_DIR="${PHOENIX_WORKING_DIR:-${SCRIPT_DIR}/.data}"
PID_FILE="${RUN_DIR}/phoenix.pid"
LOG_FILE="${RUN_DIR}/phoenix.log"

log() { printf '[phoenix] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

resolve_phoenix_bin() {
    if [[ -n "${PHOENIX_BIN:-}" ]]; then
        PHOENIX_BIN_RESOLVED="${PHOENIX_BIN}"
    elif command -v phoenix >/dev/null 2>&1; then
        PHOENIX_BIN_RESOLVED="$(command -v phoenix)"
    elif [[ -x "${HOME}/.venvs/phoenix/bin/phoenix" ]]; then
        PHOENIX_BIN_RESOLVED="${HOME}/.venvs/phoenix/bin/phoenix"
    else
        die "'phoenix' executable not found. Install it first: python3 -m venv ~/.venvs/phoenix && ~/.venvs/phoenix/bin/pip install arize-phoenix (or activate an env that has it and re-run)."
    fi
}

pid_if_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "${pid}"
        return 0
    fi
    return 1
}

cmd_up() {
    resolve_phoenix_bin
    mkdir -p "${RUN_DIR}" "${DATA_DIR}"
    if pid_if_running >/dev/null; then
        log "Already running (pid $(pid_if_running)). Use '$0 status' or '$0 down' first."
        return 0
    fi
    log "Starting Phoenix (data dir: ${DATA_DIR})..."
    (
        export PHOENIX_WORKING_DIR="${DATA_DIR}"
        [[ -n "${PHOENIX_PORT:-}" ]] && export PHOENIX_PORT
        nohup "${PHOENIX_BIN_RESOLVED}" serve >"${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
    )
    sleep 1
    if pid_if_running >/dev/null; then
        log "Started (pid $(pid_if_running)). UI: http://localhost:${PHOENIX_PORT:-6006}"
        log "Logs: $0 logs -f    Data: ${DATA_DIR}"
    else
        die "Phoenix exited immediately -- check ${LOG_FILE} for details."
    fi
}

cmd_status() {
    if pid_if_running >/dev/null; then
        log "Running (pid $(pid_if_running)). Data dir: ${DATA_DIR}"
    else
        log "Not running."
    fi
}

cmd_logs() {
    [[ -f "${LOG_FILE}" ]] || die "No log file yet -- run '$0 up' first."
    if [[ "${1:-}" == "-f" ]]; then
        tail -f "${LOG_FILE}"
    else
        tail -n 200 "${LOG_FILE}"
    fi
}

cmd_down() {
    local purge=false
    for arg in "$@"; do
        [[ "${arg}" == "--purge" ]] && purge=true
    done
    local pid
    if pid="$(pid_if_running)"; then
        log "Stopping Phoenix (pid ${pid})..."
        kill "${pid}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        kill -0 "${pid}" 2>/dev/null && { log "Still running; sending SIGKILL."; kill -9 "${pid}" 2>/dev/null || true; }
    else
        log "Not running."
    fi
    rm -f "${PID_FILE}"
    if [[ "${purge}" == "true" ]]; then
        log "Deleting local trace data at ${DATA_DIR}..."
        rm -rf "${DATA_DIR}"
    fi
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
