# Skill evals

Behavioural tests for the skills in this plugin, run with `claude plugin eval`, not pytest.
Each case gives Claude a task in a scratch workspace, once with the plugin and once without,
and grades what it did. Checks that need no model live in `tests/test_claude_plugin.py`.

They need a model, so they are deliberately not part of CI. Run them by hand after changing
the skill. `evals/` is the directory the runner looks in by default, so no flag is needed, but
the target must be the plugin, not the repository root.

## Running

From the repository root:

```bash
# Fast cases: no shell, a few minutes each
claude plugin eval claude-plugin/itwinai --tag fast --scaffold --allow-tools Write Edit

# One case, one run, while iterating on the skill
claude plugin eval claude-plugin/itwinai --case hpo-re-entry --runs 1 \
  --scaffold --allow-tools Write Edit

# End to end: needs Bash, hence a working sandbox (bubblewrap and socat on Linux)
claude plugin eval claude-plugin/itwinai --tag e2e --scaffold \
  --allow-tools Write Edit Bash
```

`--scaffold` is required: every case builds its workspace with `scaffold.sh`. Scaffolds clone
`itwinai-plugin-template` from GitHub, and `e2e-fno-darcy` layers a venv over the itwinai
environment in `$ITWINAI_EVAL_PYTHON`, defaulting to this repository's `.venv`.

Each case runs three times per arm by default, so the full suite is not cheap. Results go to
`claude-plugin/itwinai/evals/results/`, which is ignored by Git.
