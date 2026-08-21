#!/usr/bin/env bash
# Provision the dev container. Kept as a script rather than a JSON one-liner so
# it can be re-run by hand (`bash .devcontainer/setup.sh`) when a build fails
# partway, and so each step's output is readable in the creation log.
set -euo pipefail

echo "==> System packages (OCR + PDF rendering)"
sudo apt-get update -qq
# Deliberately no `apt upgrade`: it re-downloads most of the base image for no
# benefit and was the main reason container builds took double-digit minutes.
if [ -f packages.txt ]; then
    sudo xargs -a packages.txt apt-get install -y -qq
fi

echo "==> PyTorch (CPU build)"
# sentence-transformers depends on torch, and the default PyPI wheel bundles
# ~2.5 GB of CUDA libraries that a Codespace has no GPU to use. Installing the
# CPU wheel first means the dependency is already satisfied when
# requirements.txt is processed, cutting the download to roughly 200 MB.
#
# This is an optimisation, not a requirement: --index-url replaces PyPI outright
# rather than adding to it, so if that host is unreachable or stops serving one
# of torch's own dependencies, the install fails. Under `set -e` that would
# abort provisioning entirely and leave the container with nothing installed —
# a far worse outcome than a slow build. So a failure here is tolerated and
# requirements.txt is left to pull whatever torch it resolves.
if pip install --user --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu; then
    echo "    CPU wheel installed."
else
    echo "    WARNING: CPU wheel unavailable; falling back to the default"
    echo "    PyPI build. Expect a slower build and a larger image."
fi

echo "==> Project dependencies"
pip install --user --no-cache-dir -r requirements.txt

echo "==> Test dependencies"
pip install --user --no-cache-dir pytest

# data/ is gitignored, so it does not exist on a fresh clone. Create it now so
# there is somewhere obvious to drop a PDF.
mkdir -p data

echo "==> Setup complete."
