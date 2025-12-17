#!/bin/bash

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  if is_apple_silicon; then
    brew install shellcheck
  fi

  shellcheck -- *.sh */*.sh

  section "Running ruff linter"
  ruff check .

  section "Running ruff formatter check"
  ruff format --check .

  section "Running mypy type checker"
  mypy vllm_metal

  section "Running tests"
  pytest tests/ -v --tb=short

  section "Verifying package import"
  python -c "import vllm_metal; print('vllm_metal imported successfully')"
}

main "$@"
