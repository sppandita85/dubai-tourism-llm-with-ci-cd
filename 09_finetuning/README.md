# Fine-Tuning Phase

Continues training an **existing checkpoint** on new data, instead of starting from random
weights. This is the "continued / incremental" arm of the monthly retraining strategy: last
month's model keeps its knowledge and just adapts to the newest batch — usually with a
**lower learning rate** and **fewer steps** than training from scratch.

## Training from scratch vs. fine-tuning
| | Training phase | Fine-tuning phase |
|---|---|---|
| Starting weights | random | a saved checkpoint |
| Data | all batches (or one) | typically the newest batch |
| Learning rate | higher (0.0006) | lower (0.0002) |
| Manifest `strategy` | `from_scratch` | `continued` |

Both use the **same** trainer, optimizer, and dataloader from `07_training/src/` — fine-tuning
just loads a checkpoint first and points training at the new data.

## Layout
```
finetuning/
├── config/finetune.yaml     # lr, steps, batch_size (no model config — comes from checkpoint)
├── scripts/finetune.py      # CLI: load checkpoint -> continue on a batch -> new checkpoint
└── tests/test_finetuning.py # continuing lowers loss and changes weights
```

## Run
```bash
PY=09_finetuning/.venv/bin/python
$PY 09_finetuning/scripts/finetune.py --from checkpoints/v0.1.0/model.npz \
    --batch 2026-08 --version v0.2.0
$PY 09_finetuning/tests/test_finetuning.py
```
It checks the batch's `vocab_size` matches the checkpoint (same tokenizer), reports val loss
before/after, saves `checkpoints/<version>/model.npz`, and logs a `continued` run to
`01_training_input_data/manifests/training_runs.jsonl`.
