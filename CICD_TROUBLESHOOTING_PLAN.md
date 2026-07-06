# Plan: Troubleshooting subagent + Jenkins CI/CD

## Context
The project is an 11-phase, monthly-retrainable LLM pipeline (`01_…`–`11_automation/`) on
GitHub. It has **no CI/CD** and only ad-hoc logging (JSONL manifests +
`11_automation/logs/<batch>.log` written by the orchestrator's `tee`). The goal is
(1) a **Claude Code troubleshooting subagent** that inspects the pipeline's logging and the
CI/CD to diagnose failures, and (2) **Jenkins** as the CI/CD system so there is something to
troubleshoot. Outcome: every push can be validated by a Jenkins pipeline that runs all phase
tests, and a specialized agent can read the pipeline logs + Jenkins build output and pinpoint
root cause.

Environment reality (verified): no Jenkins, no Java, but **brew** is present (Docker is not
used — user preference). Jenkins runs **natively via Homebrew** (`brew install jenkins-lts`
pulls in the required Java) on the Mac, whose `python3` runs the pipeline. Jenkins' first-time
setup is interactive, so the user stands Jenkins up; everything else is scripted. CI test venv
deps = `numpy pyyaml tokenizers gguf`. `10_deployment` test self-skips without Ollama;
`02_tokenization` roundtrip works from the committed vocab.

## Deliverables

### 1. `.claude/agents/pipeline-troubleshooter.md` — the subagent
Frontmatter: `name: pipeline-troubleshooter`, a `description` that triggers on "diagnose /
troubleshoot pipeline run or CI failure", `tools: Read, Grep, Glob, Bash` (read-only
investigation). System prompt contents:
- Map of the 11 phases and the data flow (raw→tokenize→…→train→eval→deploy).
- **Where state/logs live**: `11_automation/logs/<batch>.log`,
  `01_training_input_data/manifests/{dataset_manifest,training_runs}.jsonl`, `checkpoints/`,
  per-phase `<phase>/.venv`.
- **How to read Jenkins**: console text via REST
  (`curl -u USER:TOKEN http://localhost:8080/job/<job>/lastBuild/consoleText`), the build log
  under `$JENKINS_HOME/jobs/<job>/builds/<n>/log`, and reproduce locally with
  `12_cicd/run_all_tests.sh`.
- **Common failure modes → checks**: `ModuleNotFoundError` → phase venv missing/broken;
  Ollama down / nomic blob absent → tokenizer extract + deploy; `tokenizer.json` missing →
  run `extract_vocab.py`; loss NaN/flat → lr/gradient; GGUF `ollama create` fail →
  arch/tensor mismatch; Jenkins → missing `python3`/Java on the node, PATH, venv step.
- **Method**: reproduce → read the specific failing log → isolate the phase → check the last
  successful manifest record → propose the minimal fix. Read-only; recommend, don't apply
  unless asked.

### 2. `Jenkinsfile` (repo root — Jenkins default path)
Declarative pipeline:
- `agent any`; `options { timestamps() }`.
- **Stage Setup**: `bash 12_cicd/run_all_tests.sh --setup-only` (create `.ci-venv`, install deps).
- **Stage Test**: `bash 12_cicd/run_all_tests.sh` (runs all 11 suites; non-zero on any fail).
- `post { always { archiveArtifacts '12_cicd/logs/*.log' } ; failure { echo "see pipeline-troubleshooter" } }`.
Keeping logic in the shell script (not inline Groovy) means the same command runs in Jenkins
and locally, which the troubleshooting agent relies on.

### 3. `12_cicd/` folder
- `run_all_tests.sh` — creates/reuses `12_cicd/.ci-venv` (`numpy pyyaml tokenizers gguf`),
  runs every `<phase>/tests/*.py` with it, tees to `12_cicd/logs/tests.log`, exits non-zero
  on any failure. `--setup-only` flag just builds the venv. Reused by Jenkins **and** humans.
- `start_jenkins.sh` — native launcher: checks for `jenkins-lts` (hints `brew install
  jenkins-lts` if missing), runs `brew services start jenkins-lts` (or foreground
  `jenkins-lts`), and prints the URL + where the initial admin password lives
  (`~/.jenkins/secrets/initialAdminPassword`). No Docker.
- `README.md` — CI/CD overview: install/run Jenkins natively via Homebrew, the one-time
  interactive setup, create a *Pipeline from SCM* job pointing at this repo's `Jenkinsfile`,
  what the stages do, where logs land, and how to invoke the troubleshooter.
- `.gitignore` — `.ci-venv/`, `logs/`.

### 4. Root `README.md` — add a short **CI/CD** section (Jenkins + the troubleshooting agent).

## Notes / reuse
- Tests are plain scripts importing phase code via `sys.path`; one shared CI venv runs them
  all (no per-phase venvs needed in CI). Mirror the existing test-invocation pattern already
  used in this repo's manual sweeps.
- Jenkinsfile stays at repo root for zero-config "Pipeline from SCM"; all real work is in
  `12_cicd/run_all_tests.sh`.

## Verification
1. `bash 12_cicd/run_all_tests.sh` → all 11 suites report pass (deployment self-skips if no
   Ollama). This is exactly what Jenkins executes, so green here == green in CI.
2. **Agent smoke test**: temporarily break one phase (e.g. rename `03_embeddings/.venv` or
   delete `02_tokenization/vocab/.../tokenizer.json`), invoke the `pipeline-troubleshooter`
   agent, confirm it names the failing phase + root cause + fix, then restore.
3. Confirm the agent is discoverable (appears in the available-agents list) and reads
   `manifests/*.jsonl` + `11_automation/logs/`.
4. Jenkins (user-driven, native): `brew install jenkins-lts` → `12_cicd/start_jenkins.sh` →
   interactive setup → create a *Pipeline from SCM* job on this repo's `Jenkinsfile`; the
   Setup+Test stages run green. Optionally lint via the Jenkins declarative linter. (Full
   Jenkins run needs the user's one-time UI setup; no Docker.)
