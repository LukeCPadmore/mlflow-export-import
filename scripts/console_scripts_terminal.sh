#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
TARGET=""
MODE="all"
HELP_ONLY=false

single_commands=(
  export-experiment
  import-experiment
  export-run
  import-run
  export-model
  import-model
)

bulk_commands=(
  export-experiments
  import-experiments
  export-models
  import-models
  export-all
  import-all
)

all_commands=("${single_commands[@]}" "${bulk_commands[@]}")

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} --target databricks|azureml [--mode single|bulk|all] [--help-only]

Description:
  Terminal runner for core mlflow-export-import console scripts.
  - --help-only: runs "<command> --help" for selected commands.
  - default mode (without --help-only): prints executable command templates.

Examples:
  ${SCRIPT_NAME} --target databricks --mode all --help-only
  ${SCRIPT_NAME} --target azureml --mode single --help-only
  ${SCRIPT_NAME} --target azureml --mode bulk
EOF
}

error() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "INFO: $*"
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    error "Command '${cmd}' not found. Install this repo first (e.g. 'pip install -e .')."
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target)
        [[ $# -ge 2 ]] || error "--target requires a value"
        TARGET="$2"
        shift 2
        ;;
      --mode)
        [[ $# -ge 2 ]] || error "--mode requires a value"
        MODE="$2"
        shift 2
        ;;
      --help-only)
        HELP_ONLY=true
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        error "Unknown option: $1"
        ;;
    esac
  done
}

validate_args() {
  [[ -n "${TARGET}" ]] || error "--target is required (databricks|azureml)"
  case "${TARGET}" in
    databricks|azureml) ;;
    *) error "Invalid --target '${TARGET}'. Use databricks|azureml." ;;
  esac

  case "${MODE}" in
    single|bulk|all) ;;
    *) error "Invalid --mode '${MODE}'. Use single|bulk|all." ;;
  esac
}

preflight_common() {
  local cmd
  for cmd in "${all_commands[@]}"; do
    require_command "${cmd}"
  done
}

preflight_databricks() {
  local uri="${MLFLOW_TRACKING_URI:-}"
  if [[ -n "${uri}" ]]; then
    if [[ "${uri}" != databricks* ]]; then
      error "For target=databricks, MLFLOW_TRACKING_URI must be 'databricks' or 'databricks://<PROFILE>'."
    fi
  else
    if [[ -z "${DATABRICKS_HOST:-}" || -z "${DATABRICKS_TOKEN:-}" ]]; then
      error "Set MLFLOW_TRACKING_URI=databricks (or databricks://<PROFILE>), or set DATABRICKS_HOST and DATABRICKS_TOKEN."
    fi
    info "MLFLOW_TRACKING_URI is not set. Using DATABRICKS_HOST/DATABRICKS_TOKEN."
  fi
}

preflight_azureml() {
  local uri="${MLFLOW_TRACKING_URI:-}"
  [[ -n "${uri}" ]] || error "For target=azureml, MLFLOW_TRACKING_URI is required."

  if [[ "${uri}" != http://* && "${uri}" != https://* && "${uri}" != azureml://* ]]; then
    error "For target=azureml, MLFLOW_TRACKING_URI must start with http://, https://, or azureml://."
  fi

  if [[ -z "${AZURE_CLIENT_ID:-}" && -z "${AZURE_TENANT_ID:-}" && -z "${AZURE_CLIENT_SECRET:-}" && -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
    info "No AZURE_* variables found. Ensure you are authenticated (for example with 'az login') before running real export/import commands."
  fi
}

resolve_commands() {
  case "${MODE}" in
    single) echo "${single_commands[@]}" ;;
    bulk) echo "${bulk_commands[@]}" ;;
    all) echo "${all_commands[@]}" ;;
  esac
}

run_help_commands() {
  local commands=("$@")
  local cmd
  for cmd in "${commands[@]}"; do
    echo
    echo "===== ${cmd} --help ====="
    "${cmd}" --help
  done
}

print_single_templates() {
  cat <<'EOF'
export-experiment --experiment <SRC_EXPERIMENT_NAME_OR_ID> --output-dir <OUTPUT_DIR>
import-experiment --experiment-name <DST_EXPERIMENT_NAME> --input-dir <OUTPUT_DIR>

export-run --run-id <RUN_ID> --output-dir <OUTPUT_DIR>
import-run --input-dir <OUTPUT_DIR>

export-model --model <SRC_MODEL_NAME> --output-dir <OUTPUT_DIR>
import-model --model <DST_MODEL_NAME> --input-dir <OUTPUT_DIR>
EOF
}

print_bulk_templates() {
  cat <<'EOF'
export-experiments --output-dir <OUTPUT_DIR>
import-experiments --input-dir <OUTPUT_DIR>

export-models --output-dir <OUTPUT_DIR>
import-models --input-dir <OUTPUT_DIR>

export-all --output-dir <OUTPUT_DIR>
import-all --input-dir <OUTPUT_DIR>
EOF
}

print_templates() {
  echo
  echo "Target: ${TARGET}"
  if [[ "${TARGET}" == "databricks" ]]; then
    cat <<'EOF'
Environment examples:
  export MLFLOW_TRACKING_URI=databricks
  # or
  export MLFLOW_TRACKING_URI=databricks://<PROFILE>
  # or
  export DATABRICKS_HOST=https://<workspace-host>
  export DATABRICKS_TOKEN=<token>
EOF
  else
    cat <<'EOF'
Environment examples:
  export MLFLOW_TRACKING_URI=<AZUREML_TRACKING_URI>
  # optionally authenticate via az cli:
  # az login
EOF
  fi

  echo
  echo "Command templates:"
  case "${MODE}" in
    single) print_single_templates ;;
    bulk) print_bulk_templates ;;
    all)
      print_single_templates
      echo
      print_bulk_templates
      ;;
  esac
}

main() {
  parse_args "$@"
  validate_args
  preflight_common

  case "${TARGET}" in
    databricks) preflight_databricks ;;
    azureml) preflight_azureml ;;
  esac

  local commands
  read -r -a commands <<< "$(resolve_commands)"

  if [[ "${HELP_ONLY}" == "true" ]]; then
    run_help_commands "${commands[@]}"
  else
    print_templates
  fi
}

main "$@"
