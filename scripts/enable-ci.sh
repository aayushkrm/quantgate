#!/usr/bin/env bash
# Turn GitHub Actions on. Needs a token with the `workflow` scope:
#   gh auth refresh -h github.com -s workflow
set -euo pipefail
mkdir -p .github/workflows
cp ci-templates/*.yml .github/workflows/
git add .github/workflows
git commit -m "ci: enable GitHub Actions workflows"
git push
