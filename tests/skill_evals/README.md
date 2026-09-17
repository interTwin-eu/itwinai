# Skill evals

Behavioural tests for the skills in `skills/`, run with `claude plugin eval`, not pytest. Each
case gives Claude a task in a scratch workspace, once with the plugin and once without, and
grades what it did. Checks that need no model live in `tests/test_claude_plugin.py`.

`claude plugin eval` refuses an eval directory inside `skills/`, and looks in `evals/` unless
told otherwise, so every command below passes `--eval-dir`.

## Running

From the repository root:

```bash
# Fast cases: no shell, a few minutes each
claude plugin eval . --eval-dir tests/skill_evals --tag fast --scaffold --allow-tools Write Edit

# One case, one run, while iterating on the skill
claude plugin eval . --eval-dir tests/skill_evals --case hpo-re-entry --runs 1 \
  --scaffold --allow-tools Write Edit

# End to end: needs Bash, hence a working sandbox (bubblewrap and socat on Linux)
claude plugin eval . --eval-dir tests/skill_evals --tag e2e --scaffold \
  --allow-tools Write Edit Bash
```

`--scaffold` is required: every case builds its workspace with `scaffold.sh`. Scaffolds clone
`itwinai-plugin-template` from GitHub, and `e2e-fno-darcy` layers a venv over the itwinai
environment in `$ITWINAI_EVAL_PYTHON`, defaulting to this repository's `.venv`.

Each case runs three times per arm by default, so the full suite is not cheap. Results go to
`tests/skill_evals/results/`, which is ignored by Git.
