# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Consistency checks for the Claude Code plugin shipped in this repository.

The skill is installed on users' machines as a pinned snapshot, so drift between it and the
codebase is invisible to them until it generates a broken configuration. These tests enforce
the parts of that contract which can be checked mechanically. See the "Claude Code Skill"
section of CLAUDE.md.
"""

import ast
import importlib
import inspect
import json
import os
import re
import textwrap
import typing
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / ".claude-plugin"
SKILL_DIR = REPO_ROOT / "skills" / "integrating-a-use-case"
EVALS_DIR = REPO_ROOT / "evals" / "integrating-a-use-case"
FNO_FIXTURE = EVALS_DIR / "fixtures" / "fno-plugin"
FNO_TUTORIAL = REPO_ROOT / "tutorials" / "claude-skill" / "fno-darcy" / "train.py"
TUTORIAL_DOC = (
    REPO_ROOT / "docs" / "tutorials" / "claude-skill" / "integrate-a-new-use-case.rst"
)

# env-files/torch/skinny.Dockerfile copies the plugin, the skill and the evals into the test
# container, but not docs/. Anything else missing means a partial source tree.
pytestmark = pytest.mark.skipif(
    not (PLUGIN_DIR.is_dir() and SKILL_DIR.is_dir()),
    reason="Claude plugin files are not present (running against a partial source tree)",
)


def _package_version() -> str:
    match = re.search(r'^version = "(.+?)"', (REPO_ROOT / "pyproject.toml").read_text(), re.M)
    assert match, "could not read version from pyproject.toml"
    return match.group(1)


@pytest.fixture
def plugin_manifest() -> dict:
    return json.loads((PLUGIN_DIR / "plugin.json").read_text())


@pytest.fixture
def marketplace_manifest() -> dict:
    return json.loads((PLUGIN_DIR / "marketplace.json").read_text())


def test_manifests_exist():
    assert (PLUGIN_DIR / "plugin.json").is_file()
    assert (PLUGIN_DIR / "marketplace.json").is_file()


def test_marketplace_declares_the_plugin(marketplace_manifest):
    names = [plugin["name"] for plugin in marketplace_manifest["plugins"]]
    assert "itwinai" in names


def test_plugin_version_matches_package(plugin_manifest):
    """Bump plugin.json together with pyproject.toml. The skill compares its own version
    against the installed itwinai to warn users about skew; a stale string disables that."""
    assert plugin_manifest["version"] == _package_version()


def test_marketplace_versions_match_package(marketplace_manifest):
    assert marketplace_manifest["metadata"]["version"] == _package_version()
    for plugin in marketplace_manifest["plugins"]:
        assert plugin["version"] == _package_version()


def test_skill_declares_target_version():
    """SKILL.md states the itwinai version it was written against; keep it in step."""
    match = re.search(r"targets itwinai (\d+\.\d+\.\d+)", (SKILL_DIR / "SKILL.md").read_text())
    assert match, "SKILL.md must state the itwinai version it targets"
    assert match.group(1) == _package_version()


def test_skill_frontmatter_is_well_formed():
    text = (SKILL_DIR / "SKILL.md").read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields = dict(re.findall(r"^(\w+):\s*(.+)$", match.group(1), re.M))
    # The skill name must match its directory, otherwise it cannot be invoked.
    assert fields.get("name") == SKILL_DIR.name
    assert fields.get("description"), "a description is required for the skill to be matched"


def test_reference_links_resolve():
    """Every references/*.md path mentioned in the skill must exist."""
    broken = []
    for markdown_file in SKILL_DIR.rglob("*.md"):
        for match in re.finditer(r"`(references/[\w\-./]+\.md)`", markdown_file.read_text()):
            if not (SKILL_DIR / match.group(1)).exists():
                broken.append(f"{markdown_file.name} -> {match.group(1)}")
    assert not broken, f"broken reference links: {broken}"


def test_every_reference_is_reachable_from_skill_md():
    """References are loaded lazily from SKILL.md; one it never names is never read."""
    linked = set(
        re.findall(r"`(references/[\w\-./]+\.md)`", (SKILL_DIR / "SKILL.md").read_text())
    )
    orphans = {
        f"references/{p.name}" for p in (SKILL_DIR / "references").glob("*.md")
    } - linked
    assert not orphans, f"not linked from SKILL.md: {sorted(orphans)}"


# --------------------------------------------------------------------------------------
# Drift: everything the skill names must still exist in itwinai
# --------------------------------------------------------------------------------------


def _skill_code() -> str:
    """Inline code spans and fenced blocks of the skill, where commands and paths appear."""
    chunks = []
    for path in sorted(SKILL_DIR.rglob("*.md")):
        text = path.read_text()
        chunks += re.findall(r"```\w*\n(.*?)```", text, re.S)
        chunks += re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", text, flags=re.S))
    return "\n".join(chunks)


def test_cited_cli_commands_exist():
    import typer

    from itwinai.cli import app

    commands = set(typer.main.get_command(app).commands)
    cited = set(re.findall(r"(?<![\w.])itwinai ([a-z][a-z-]*)", _skill_code()))
    assert cited, "no itwinai commands found; did the extraction break?"
    assert not cited - commands, f"unknown itwinai commands: {sorted(cited - commands)}"


def _resolve(dotted: str):
    """Import the longest importable module prefix of a dotted path, then walk attributes."""
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(dotted)


def test_cited_python_objects_exist():
    text = "\n".join(p.read_text() for p in SKILL_DIR.rglob("*.md"))
    # A trailing ":" marks an OmegaConf resolver, checked separately. Plugins are examples.
    cited = {
        match
        for match in re.findall(r"\bitwinai(?:\.\w+)+(?![\w:])", text)
        if not match.startswith("itwinai.plugins")
    }
    missing = []
    for dotted in sorted(cited):
        try:
            _resolve(dotted)
        except (ImportError, AttributeError):
            missing.append(dotted)
    assert not missing, f"the skill names objects that do not exist: {missing}"


def test_cited_source_paths_exist():
    # Not preceded by "/", so URLs are skipped. itwinai/plugins/ paths belong to plugin repos.
    cited = {
        path
        for path in re.findall(
            r"(?<![\w/.-])(?:src/)?(itwinai/[\w/.-]+?)/?(?=`|\s|$)", _skill_code()
        )
        if not path.startswith("itwinai/plugins/")
    }
    cited |= {
        f"../{path}"
        for path in re.findall(
            r"\b((?:docs|use-cases|tutorials)/[\w/.-]+\.\w+)", _skill_code()
        )
        if (REPO_ROOT / path.split("/")[0]).is_dir()
    }
    missing = sorted(path for path in cited if not (REPO_ROOT / "src" / path).exists())
    assert not missing, f"the skill points at paths that do not exist: {missing}"


def test_cited_resolvers_are_registered():
    text = "\n".join(p.read_text() for p in SKILL_DIR.rglob("*.md"))
    cited = set(re.findall(r"\bitwinai\.(\w+):", text))
    cli_source = (REPO_ROOT / "src" / "itwinai" / "cli.py").read_text()
    registered = set(re.findall(r'register_new_resolver\(\s*"itwinai\.(\w+)"', cli_source))
    assert cited, "no resolvers found; did the extraction break?"
    assert not cited - registered, f"unregistered resolvers: {sorted(cited - registered)}"


def test_strategies_listed_in_skill_match_trainer():
    from itwinai.torch.trainer import TorchTrainer

    annotation = inspect.signature(TorchTrainer.__init__).parameters["strategy"].annotation
    text = (SKILL_DIR / "references" / "distributed.md").read_text()
    listed = re.search(r"currently (`\w+`(?:, `\w+`)*)", text)
    assert listed, "distributed.md must list the allowed strategies"
    assert set(re.findall(r"`(\w+)`", listed.group(1))) == set(typing.get_args(annotation))


# --------------------------------------------------------------------------------------
# Drift: YAML in the skill, the tutorial and the eval fixture must validate against itwinai
# --------------------------------------------------------------------------------------


def _yaml_blocks() -> list:
    blocks = []
    for path in sorted(SKILL_DIR.rglob("*.md")):
        for i, body in enumerate(re.findall(r"```yaml\n(.*?)```", path.read_text(), re.S)):
            blocks.append(
                pytest.param(yaml.safe_load(textwrap.dedent(body)), id=f"{path.name}-{i}")
            )
    if TUTORIAL_DOC.is_file():
        pattern = r"\.\. code-block:: yaml\n\n((?:(?:   .*)?\n)+)"
        for i, body in enumerate(re.findall(pattern, TUTORIAL_DOC.read_text())):
            blocks.append(
                pytest.param(yaml.safe_load(textwrap.dedent(body)), id=f"tutorial-{i}")
            )
    if FNO_FIXTURE.is_dir():
        config = yaml.safe_load((FNO_FIXTURE / "config.yaml").read_text())
        blocks.append(pytest.param(config, id="fno-fixture-config"))
    return blocks


def _accepted_kwargs(obj) -> set[str] | None:
    """Constructor argument names, or None if it takes **kwargs and so accepts anything."""
    parameters = inspect.signature(obj).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters) - {"self"}


class _ConfigChecker:
    def __init__(self) -> None:
        from itwinai.torch.config import TrainingConfiguration
        from itwinai.torch.trainer import TorchTrainer

        self.trainer_kwargs = _accepted_kwargs(TorchTrainer)
        self.config_fields = set(TrainingConfiguration.model_fields)
        self.problems: list[str] = []

    def check(self, block: dict) -> list[str]:
        block = dict(block)
        if "slurm_config" in block:
            self._check_slurm(block.pop("slurm_config"))
        self._walk(block, path="")

        top_level_targets = [
            v.get("_target_", "") for v in block.values() if isinstance(v, dict)
        ]
        is_pipeline_file = any(t.endswith(".Pipeline") for t in top_level_targets)
        is_search_space = block and all(
            isinstance(v, dict) and "type" in v for v in block.values()
        )
        if block and not is_pipeline_file and not is_search_space:
            # A fragment pasted under the trainer step
            self._check_trainer_keys(block, path="<trainer step>")
        if is_search_space:
            self._check_search_space(block, path="", known_keys=None)
        return self.problems

    def _walk(self, node, path: str) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                self._walk(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        target = node.get("_target_")
        if target:
            self._check_target(node, target, path)
        for key, value in node.items():
            self._walk(value, f"{path}.{key}" if path else key)

    def _check_target(self, node: dict, target: str, path: str) -> None:
        from itwinai.torch.trainer import TorchTrainer

        try:
            obj = _resolve(target)
        except (ImportError, AttributeError):
            if target.startswith("itwinai.plugins."):
                # An illustrative plugin class; a *Trainer is assumed to extend TorchTrainer
                if target.endswith("Trainer"):
                    self._check_trainer_keys(node, path)
                return
            self.problems.append(f"{path}: _target_ {target} does not import")
            return
        if inspect.isclass(obj) and issubclass(obj, TorchTrainer):
            self._check_trainer_keys(node, path)
            return
        accepted = _accepted_kwargs(obj)
        unknown = set(node) - {"_target_"} - (set(node) if accepted is None else accepted)
        if unknown:
            self.problems.append(f"{path}: {target} does not accept {sorted(unknown)}")

    def _check_trainer_keys(self, node: dict, path: str) -> None:
        unknown = set(node) - {"_target_"} - self.trainer_kwargs
        if unknown:
            self.problems.append(f"{path}: TorchTrainer does not accept {sorted(unknown)}")
        if isinstance(node.get("config"), dict):
            unknown = set(node["config"]) - self.config_fields
            if unknown:
                self.problems.append(
                    f"{path}.config: not TrainingConfiguration fields {sorted(unknown)}"
                )
        if isinstance(node.get("ray_search_space"), dict):
            self._check_search_space(
                node["ray_search_space"], f"{path}.ray_search_space", self.config_fields
            )

    def _check_search_space(self, space: dict, path: str, known_keys: set[str] | None) -> None:
        tune = pytest.importorskip("ray.tune")
        for name, spec in space.items():
            if known_keys is not None and name not in known_keys:
                self.problems.append(f"{path}.{name}: not a TrainingConfiguration field")
            if isinstance(spec, dict) and not hasattr(tune, spec.get("type", "")):
                self.problems.append(
                    f"{path}.{name}: ray.tune has no sampler {spec['type']!r}"
                )

    def _check_slurm(self, section: dict) -> None:
        from pydantic import ValidationError

        from itwinai.slurm.configuration import MLSlurmBuilderConfig

        fields = MLSlurmBuilderConfig.model_fields
        for key in sorted(set(section) - set(fields)):
            self.problems.append(f"slurm_config.{key}: not a field of MLSlurmBuilderConfig")
        concrete = {
            key: value
            for key, value in section.items()
            if key in fields and not (isinstance(value, str) and "${" in value)
        }
        try:
            MLSlurmBuilderConfig.model_validate(concrete)
        except ValidationError as error:
            for detail in error.errors():
                if detail["type"] != "missing":
                    location = ".".join(map(str, detail["loc"]))
                    self.problems.append(f"slurm_config.{location}: {detail['msg']}")


@pytest.mark.parametrize("block", _yaml_blocks())
def test_yaml_matches_itwinai(block):
    """Catches renamed or removed arguments that the skill, the tutorial or the fixture
    still use. Values interpolated with ${} are not checked, only keys and literals."""
    problems = _ConfigChecker().check(block)
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------------------
# Eval suite: cheap checks before spending money on `claude plugin eval`
# --------------------------------------------------------------------------------------

_EVAL_CASES = sorted(p.parent for p in EVALS_DIR.glob("*/case.yaml"))


def test_eval_suite_exists():
    assert _EVAL_CASES, f"no eval cases under {EVALS_DIR}"


@pytest.mark.parametrize("case_dir", _EVAL_CASES, ids=lambda p: p.name)
def test_eval_case_is_well_formed(case_dir):
    case = yaml.safe_load((case_dir / "case.yaml").read_text())
    assert case["schema_version"] == "1.1"
    assert case["name"] == case_dir.name

    scaffold = case_dir / case["context"]["scaffold_script"]
    assert scaffold.is_file() and os.access(scaffold, os.X_OK), (
        f"{scaffold} must be executable"
    )

    graders = case["graders"]
    assert len({g["name"] for g in graders}) == len(graders), "grader names must be unique"
    # A tool call is a trajectory, not a result: every case needs at least one outcome grader
    assert any(g["type"] != "tool_used" for g in graders)

    allowed_tools = set(case["execution"]["allowed_tools"])
    for grader in graders:
        if grader["type"] == "regex":
            re.compile(grader["pattern"])
        if grader["type"] != "tool_used":
            continue
        if grader["tool"] == "Skill":
            assert grader["input_match"] == SKILL_DIR.name, "eval targets a renamed skill"
        if grader.get("min", 1) >= 1:
            assert grader["tool"] in allowed_tools, f"{grader['name']} needs {grader['tool']}"


def _top_level_definitions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    return {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


@pytest.mark.skipif(not FNO_TUTORIAL.is_file(), reason="tutorials/ not present")
def test_fno_fixture_matches_tutorial_script():
    """The fixture is the tutorial's plugin, so moved code must stay identical to train.py."""
    original = _top_level_definitions(FNO_TUTORIAL)
    for module in ("darcy.py", "models.py", "losses.py"):
        for name, dump in _top_level_definitions(
            FNO_FIXTURE / "src" / "itwinai" / "plugins" / "fno" / module
        ).items():
            assert dump == original.get(name), (
                f"{module}:{name} differs from {FNO_TUTORIAL.name}"
            )


@pytest.mark.functional
def test_fno_fixture_trains(tmp_path, monkeypatch):
    """Gate 3 on the fixture plugin. It is the tutorial's finished code and the starting point
    of two evals, so if it stops training both are wrong."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import itwinai.plugins

    plugins_path = [
        *itwinai.plugins.__path__,
        str(FNO_FIXTURE / "src" / "itwinai" / "plugins"),
    ]
    monkeypatch.setattr(itwinai.plugins, "__path__", plugins_path)
    monkeypatch.chdir(tmp_path)

    config = OmegaConf.merge(
        OmegaConf.load(FNO_FIXTURE / "config.yaml"),
        # rfft2 keeps grid_size // 2 + 1 modes on one axis, so modes must shrink with the grid
        {"epochs": 1, "n_train": 32, "grid_size": 16, "modes": 4, "width": 8},
    )
    instantiate(config.training_pipeline, _convert_="all").execute()

    assert (tmp_path / "mllogs" / "mlflow").is_dir()
