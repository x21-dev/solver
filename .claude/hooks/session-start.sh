#!/bin/bash
# Sets up D for Claude Code on the web: LDC 1.42.0 (with dub), the way
# satisfaction's and pleaser's hooks do. solver has no dub dependencies, so
# afterwards `dub test --compiler=ldc2` and the worked example in
# docs/example build as they are. To build pleaser against this checkout,
# open the session on pleaser instead (its hook fetches the rest).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

LDC_VERSION=1.42.0
DLANG_DIR="$HOME/dlang"
LDC_DIR="$DLANG_DIR/ldc2-$LDC_VERSION-linux-x86_64"

if [ ! -x "$LDC_DIR/bin/ldc2" ]; then
  echo "Installing LDC $LDC_VERSION"
  mkdir -p "$DLANG_DIR"
  curl -fsSL --retry 3 \
    "https://github.com/ldc-developers/ldc/releases/download/v$LDC_VERSION/ldc2-$LDC_VERSION-linux-x86_64.tar.xz" \
    | tar -xJ -C "$DLANG_DIR"
fi
export PATH="$LDC_DIR/bin:$PATH"
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$LDC_DIR/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

# dub defaults: LDC, and the official registry only. The mirrors aren't
# reachable here and trying them costs half a minute per failed download.
mkdir -p "$HOME/.dub"
if [ ! -f "$HOME/.dub/settings.json" ]; then
  cat > "$HOME/.dub/settings.json" <<'JSON'
{
  "defaultCompiler": "ldc2",
  "skipRegistry": "standard",
  "registryUrls": ["https://code.dlang.org"]
}
JSON
fi

echo "D toolchain ready: $(ldc2 --version | head -1)"
