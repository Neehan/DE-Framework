# Discovery-Execution Framework for Mathematical Reasoning

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12+, Bash, and a running local Docker Engine with Linux containers. Use Linux, macOS, or WSL2.

From the repository root:

```bash
cp .env.example .env
# Configure provider credentials and any CODEX_AUTH_FILE_<n> paths in .env.
bash scripts/setup.sh
source .venv/bin/activate
```

Place `aobench.jsonl` and `imoproofbench.jsonl` in `datasets/`, or set `HF_DATASET_REPO_URL` in `.env` to download missing files during setup.

## Supported models

- `claude-opus-4-8`
- `litellm/gpt-5.4`
- `litellm/gpt-5.5`
- `litellm/gpt-5.6-sol`
- `muse-spark-1.2-contributor`

## Experiments

Each 1× block allows 200,000 output tokens across solve, critique, and revise phases.

| Experiment | Budget | Input |
| --- | --- | --- |
| `unaided` | 1–8× | Problem only; choose `--compute-multiplier-k`. |
| `no-sketch` | 1× | Problem only; separate sketch-control experiment. |
| `oracle-sketch` | 1× | Reference sketch at the start. |
| `alt-sketch` | 1× | Alternative sketch at the start. |
| `oracle-execution` | 8× | Reference sketch at the start. |
| `continue-at-3x` | 3× prefix + 1× | Continue the shared unaided prefix without a sketch. |
| `continue-oracle-at-3x` | 3× prefix + 1× | Fork the same prefix, then supply the reference sketch. |

Independent `unaided` runs at 1× refer to the `parallel-8` arms in paper.

## Run and audit

After setup, run this example from the repository root. Choose any experiment listed above; omit `--compute-multiplier-k` for experiments with fixed budgets. Replace `aobench` with `imoproofbench` for the other dataset. Optionally add `--problems <id> ...` or `--domain <domain>` to select a subset.

```bash
python -m launcher.run --dataset aobench --model claude-opus-4-8 --experiment unaided --compute-multiplier-k 8 --seeds 1 2 3 --max-concurrency 3
python -m launcher.audit --dataset aobench --model claude-opus-4-8 --experiment unaided --compute-multiplier-k 8 --seeds 1 2 3 --max-concurrency 3
```

Outputs are saved in `results/<dataset>/<encoded-model>/<experiment>-<k>x/<problem>/seed_<seed>/`, with compiled `audit.jsonl` in the experiment directory.

## Frameworks

After auditing, fit SG, DE, R-SG, and R-DE using all available seeds:

```bash
python -m launcher.frameworks --model claude-opus-4-8
```

Both datasets and all problems are selected by default; filter with `--dataset`, `--problems`, or `--domain`. Problems need at least 24 audited unaided-1x seeds and 3 complete oracle-execution seeds. Each regularized framework learns its prior across all selected problems for the model. Predictions use `N=1,2,4` with checkpoints `K=1,...,8//N`. Outputs are `results/<dataset>/<model>/frameworks/<sg|de|r-sg|r-de>/fit.json` and `predictions.jsonl`.
