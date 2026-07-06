#!/usr/bin/env bash
# Shared CI test runner: run every phase's test suite against a target checkout.
#
#   ./12_cicd/run_all_tests.sh [TARGET_DIR]
#
# TARGET_DIR defaults to the repo that contains this script (the local repo). The CI agent
# passes the freshly-cloned workspace instead. Uses one shared venv (12_cicd/.ci-venv) with
# numpy/pyyaml/tokenizers/gguf. Exits non-zero if any suite fails. Reusable by Jenkins too.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"          # .../12_cicd
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${1:-$REPO_DEFAULT}"
CI_VENV="$SCRIPT_DIR/.ci-venv"
PY="$CI_VENV/bin/python"

# Build the shared test venv on first use.
if [ ! -x "$PY" ]; then
  echo "[run_all_tests] creating .ci-venv ..."
  python3 -m venv "$CI_VENV"
  "$CI_VENV/bin/python" -m pip install --quiet --upgrade pip
  "$CI_VENV/bin/python" -m pip install --quiet numpy pyyaml tokenizers gguf
fi

pass=0; fail=0; failed_suites=""
echo "=== running phase test suites against: $TARGET ==="
for testfile in $(ls "$TARGET"/[0-9][0-9]_*/tests/test_*.py 2>/dev/null | sort); do
  suite="$(basename "$(dirname "$(dirname "$testfile")")")"   # e.g. 07_training
  if "$PY" "$testfile" >/tmp/ci_suite_out 2>&1; then
    echo "  PASS  $suite"
    pass=$((pass+1))
  else
    echo "  FAIL  $suite"
    tail -3 /tmp/ci_suite_out | sed 's/^/        /'
    fail=$((fail+1)); failed_suites="$failed_suites $suite"
  fi
done

echo "=== result: $pass passed, $fail failed ==="
if [ "$fail" -ne 0 ]; then
  echo "failed:$failed_suites"
  exit 1
fi
exit 0
