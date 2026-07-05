# Deployment Phase (push to Ollama)

Takes a trained checkpoint (`checkpoints/<version>/model.npz`) and makes it a **runnable
Ollama model**, so you can `ollama run llm-stepbystep:<version>` and chat with the model you
trained. This is the last step of the monthly loop — the fine-tuned model goes live.

## How it works
Ollama runs **GGUF** models in architectures `llama.cpp` recognizes. Our model is a
GPT-2-style decoder, so the phase:
1. **Exports** `model.npz` → a `gpt2`-architecture **GGUF** (`export_gguf.py`) — hyperparameters,
   the weight tensors in llama.cpp's layout (Q/K/V fused into `attn_qkv`, linear weights
   transposed to ggml's `(in,out)` convention), and the **tokenizer copied from the
   nomic-embed-text GGUF** (the exact tokenizer the model was trained with).
2. **Writes a Modelfile** (`FROM ./model.gguf` + generation parameters).
3. Runs **`ollama create <model_name>:<version>`** to register it locally.
4. Optionally **`ollama push`** to the ollama.com registry (needs a namespace + sign-in).

## Layout
```
10_deployment/
├── config/deploy.yaml       # model name, generation params, registry push settings
├── src/export_gguf.py       # model.npz -> gpt2 GGUF (verified against real Ollama)
├── scripts/deploy.py        # export -> Modelfile -> ollama create [-> push] -> log
└── tests/test_deployment.py # validates the exported GGUF (arch/metadata/tensors)
```
The GGUF and Modelfile are written next to the checkpoint (`checkpoints/<version>/`).

## Run
```bash
PY=10_deployment/.venv/bin/python
$PY 10_deployment/scripts/deploy.py --checkpoint checkpoints/v0.2.0/model.npz --version v0.2.0
ollama run llm-stepbystep:v0.2.0 "Dubai is"      # then chat with your model
```
To publish to the ollama.com registry, set `registry_namespace` in `deploy.yaml` (and
`push_to_registry: true`) or pass `--push`, after `ollama` is signed in.

## Reality check
The conversion is real and verified end to end: the exported GGUF loads in Ollama and
generates text using the trained weights + WordPiece tokenizer. Because the model is small
and trained briefly on one document, the text is semi-coherent (Dubai-themed) rather than
fluent — expected at this scale. The **deployment mechanism** is what matters here; quality
scales with model size, data, and training steps.
