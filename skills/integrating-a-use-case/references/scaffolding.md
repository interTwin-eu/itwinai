# Scaffolding the plugin repository

## Why a plugin and not a folder in itwinai

Plugin code lives in the scientist's own repository and installs into the `itwinai.plugins.*`
namespace, so it is importable as if it shipped with itwinai while staying independently
versioned and released. See `docs/getting-started/plugins.rst` in the itwinai repository for the
namespace-package mechanics.

## Create the repository

`https://github.com/interTwin-eu/itwinai-plugin-template` is a GitHub *template* repository. Use
"Use this template", or fork, or copy it. Do not open pull requests against it.

It arrives with the CI, linters, licence, `CITATION.cff`, Dockerfiles and a `tests/` directory
already wired. Keep all of it.

## Choose the package name

This is the `<name>` in `itwinai.plugins.<name>`, and it is painful to change later because it
appears in every `_target_` in `config.yaml`.

Rules, each of which an existing plugin gets wrong:

- **Lowercase.** `itwinai.plugins.Glitchflow` exists in the wild and is inconsistent with every
  other plugin. Use `glitchflow`.
- **A valid Python identifier.** It cannot start with a digit, which is why the 3DGAN plugin's
  package is `tdgan` and not `3dgan`.
- **Globally unique.** Two installed plugins claiming the same name will shadow each other.
  Check the published list in `docs/getting-started/plugins-list.rst` first.
- **No underscore-prefixed or dunder names**, and do not reuse an itwinai module name such as
  `torch`, `loggers` or `pipeline`.

The repository name conventionally ends in `-plugin` (`hython-itwinai-plugin`,
`pulsar-plugin`); the *package* name does not carry the suffix.

## Rename the template

1. `git mv src/itwinai/plugins/my_awesome_plugin src/itwinai/plugins/<name>`
2. Delete the example modules (`awesome_module.py`, `plugin_subfolder/`,
   `another_plugin_subfolder/`) once you have somewhere for the real code to go.
3. In `pyproject.toml` set `[project] name`, `description`, `authors`, and the dependency list.
   Keep `itwinai[torch]`.
4. Update `include` under `[tool.setuptools.packages.find]` - see below.
5. Update `README.md`, `AUTHORS.md` and `CITATION.cff`.

## The `include` list is the most common packaging mistake

`[tool.setuptools.packages.find] include` must name **every** package directory, not just the
top one. A subfolder missing from this list installs as nothing, and the failure appears much
later as an `ImportError` from inside `exec-pipeline`, where it looks like a config problem.

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = [
    "itwinai.plugins.fno",
    "itwinai.plugins.fno.data",
    "itwinai.plugins.fno.models",
]
```

Every directory listed needs an `__init__.py`. Whenever you add a subpackage during Phase 2, add
it here in the same edit.

## Gate

```bash
uv pip install -e .
python -c "import itwinai.plugins.<name>; print('ok')"
```

The template ships `tests/test_plugin_import.py` in some versions; if yours does not, the
hython plugin has one worth copying. An import test is cheap and catches exactly the `include`
mistake above.
