# shellcheck shell=bash
# Sourced by every scaffold.sh. `claude plugin eval` runs scaffolds with the empty eval
# workspace as the working directory, and with HOME and TMPDIR pointing inside the sandbox.

FIXTURES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FIXTURES/../../.." && pwd)"

copy_plugin_template() {
    local tmp
    tmp="$(mktemp -d)"
    git clone --quiet --depth 1 https://github.com/interTwin-eu/itwinai-plugin-template "$tmp"
    rm -rf "$tmp/.git"
    cp -r "$tmp/." .
    rm -rf "$tmp"
}

copy_fno_plugin() {
    cp -r "$FIXTURES/fno-plugin/." .
}

# A venv layered over an existing itwinai installation, so a run never downloads torch.
# uv only inspects the venv's own site-packages, so without uv.toml `uv pip install -e .`
# would resolve itwinai[torch] from PyPI again and shadow the itwinai under test.
make_itwinai_venv() {
    local base_python="${ITWINAI_EVAL_PYTHON:-$REPO_ROOT/.venv/bin/python}"
    local base_site site
    base_site="$("$base_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    uv venv --quiet .venv --python "$base_python"
    site="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    echo "import site; site.addsitedir('$base_site')" >"$site/itwinai_base.pth"
    printf '#!%s\nfrom itwinai.cli import app\n\napp()\n' "$PWD/.venv/bin/python" >.venv/bin/itwinai
    chmod +x .venv/bin/itwinai
    printf '[pip]\nno-deps = true\nno-build-isolation = true\n' >uv.toml
}
