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

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / ".claude-plugin"
SKILL_DIR = REPO_ROOT / "skills" / "integrating-a-use-case"

# The test container built by env-files/torch/skinny.Dockerfile copies only pyproject.toml,
# src, tests and use-cases, and .dockerignore strips "**/*.md" on top of that, so neither the
# manifests nor the skill exist there. These checks are meaningful only against a full
# checkout: run them locally, or with `make test-local`.
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
