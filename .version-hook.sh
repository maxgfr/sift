#!/bin/bash
#
# Bump every place the workspace records its version. Run by semantic-release's `exec`
# plugin during `prepare`, before the `git` plugin commits Cargo.toml and Cargo.lock.
#
# A single-package crate needs one edit. This is a workspace, and it needs three — miss any
# and the release is tagged but the tree does not build:
#
#   1. `[workspace.package] version`, which every member inherits.
#   2. `[workspace.dependencies] sift-core`'s `version` requirement. sift-cli and
#      sift-bench depend on sift-core by path *and* version; once the crate is 0.2.0 a
#      `^0.1.0` requirement no longer matches and cargo refuses to resolve.
#   3. The member entries in Cargo.lock, which is tracked because this workspace ships
#      binaries.
#
# The `cargo metadata --locked` at the end is the guard: it fails loudly if the three ever
# disagree, so a bad bump stops the release instead of producing an unbuildable tag.

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Error: version number required" >&2
  exit 1
fi
NEW_VERSION="$1"

# 1. The workspace version. Anchored to the start of a line, which in the root manifest
#    matches only the `[workspace.package]` entry — the sift-core dependency spells its
#    version inline, mid-line.
sed -i.bak -E "s/^version = \".*\"$/version = \"${NEW_VERSION}\"/" Cargo.toml
rm -f Cargo.toml.bak

# 2. The internal path dependency's version requirement.
sed -i.bak -E \
  "s|^(sift-core = \{ path = \"crates/sift-core\", version = )\"[^\"]*\"|\1\"${NEW_VERSION}\"|" \
  Cargo.toml
rm -f Cargo.toml.bak

# 3. Every workspace member in the lockfile. `n` advances to the line after the name, which
#    is where cargo writes the version.
for crate in sift-core sift-cli; do
  sed -i.bak "/^name = \"${crate}\"$/{n;s/^version = \".*\"$/version = \"${NEW_VERSION}\"/;}" Cargo.lock
  rm -f Cargo.lock.bak
done

echo "Bumped workspace to ${NEW_VERSION}"

# Refuse to hand a broken tree to the tagger.
if ! cargo metadata --locked --format-version 1 >/dev/null; then
  echo "Error: Cargo.toml and Cargo.lock disagree after the bump" >&2
  exit 1
fi

echo "Cargo.toml and Cargo.lock agree at ${NEW_VERSION}"
