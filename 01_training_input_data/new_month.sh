#!/usr/bin/env bash
# Scaffold a new monthly training batch.
# Usage:  ./01_training_input_data/new_month.sh 2026-08
#         ./01_training_input_data/new_month.sh           # defaults to the current month
set -euo pipefail

batch="${1:-$(date +%Y-%m)}"
root="$(cd "$(dirname "$0")" && pwd)"

if [[ ! "$batch" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Batch id must be YYYY-MM (got: '$batch')" >&2
  exit 1
fi

for sub in raw interim processed; do
  mkdir -p "$root/$sub/$batch"
done
touch "$root/interim/$batch/.gitkeep" "$root/processed/$batch/.gitkeep"

echo "Created batch $batch:"
echo "  drop this month's source docs into  01_training_input_data/raw/$batch/"
echo "  then clean -> 01_training_input_data/interim/$batch/  and tokenize -> 01_training_input_data/processed/$batch/"
echo "  finally append a line to 01_training_input_data/manifests/dataset_manifest.jsonl"
