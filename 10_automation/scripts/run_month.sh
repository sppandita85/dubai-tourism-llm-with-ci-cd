#!/usr/bin/env bash
# Monthly pipeline orchestrator: run every phase for one month in order.
#
#   ./10_automation/scripts/run_month.sh 2026-08            # train from scratch
#   ./10_automation/scripts/run_month.sh 2026-08 continue   # continue latest checkpoint
#   ./10_automation/scripts/run_month.sh                    # defaults to current month, scratch
#
# Assumes this month's raw documents already live in 01_training_input_data/raw/<batch>/.
set -euo pipefail

BATCH="${1:-$(date +%Y-%m)}"
MODE="${2:-scratch}"                 # scratch | continue
if ! [[ "$BATCH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Batch id must be YYYY-MM (got: '$BATCH')" >&2; exit 2
fi

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
TOKPY="02_tokenization/.venv/bin/python"
TRAINPY="07_training/.venv/bin/python"
FTPY="09_finetuning/.venv/bin/python"
EVALPY="08_evaluation/.venv/bin/python"
VERSION="v-$BATCH"
LOGDIR="10_automation/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$BATCH.log"

echo "=== monthly run: batch=$BATCH mode=$MODE version=$VERSION ===" | tee "$LOG"

# 0. Ensure the tokenizer vocab has been extracted (one-time).
if [ ! -f "02_tokenization/vocab/nomic-embed-text/tokenizer.json" ]; then
  echo "[0/4] extracting tokenizer vocab from Ollama GGUF" | tee -a "$LOG"
  $TOKPY 02_tokenization/scripts/extract_vocab.py 2>&1 | tee -a "$LOG"
fi

echo "[1/4] clean raw -> interim" | tee -a "$LOG"
$TOKPY 02_tokenization/scripts/clean_batch.py "$BATCH" 2>&1 | tee -a "$LOG"

echo "[2/4] tokenize interim -> processed" | tee -a "$LOG"
$TOKPY 02_tokenization/scripts/tokenize_batch.py "$BATCH" 2>&1 | tee -a "$LOG"

echo "[3/4] train ($MODE)" | tee -a "$LOG"
if [ "$MODE" = "continue" ]; then
  LATEST="$(ls -td checkpoints/*/ 2>/dev/null | head -1 || true)"
  if [ -z "$LATEST" ]; then
    echo "  no existing checkpoint; falling back to from-scratch" | tee -a "$LOG"
    MODE="scratch"
  fi
fi
if [ "$MODE" = "continue" ]; then
  $FTPY 09_finetuning/scripts/finetune.py --from "${LATEST}model.npz" \
        --batch "$BATCH" --version "$VERSION" 2>&1 | tee -a "$LOG"
else
  $TRAINPY 07_training/scripts/train.py "$BATCH" --version "$VERSION" 2>&1 | tee -a "$LOG"
fi

echo "[4/4] evaluate" | tee -a "$LOG"
$EVALPY 08_evaluation/scripts/evaluate.py \
        --checkpoint "checkpoints/$VERSION/model.npz" --batch "$BATCH" 2>&1 | tee -a "$LOG"

echo "=== done: $BATCH -> checkpoints/$VERSION (log: $LOG) ===" | tee -a "$LOG"
