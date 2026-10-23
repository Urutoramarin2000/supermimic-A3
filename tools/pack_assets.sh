#!/usr/bin/env bash
# Bundles the robot model and the reference motions (neither is in git) into one tarball for a new server.
# Run it where both exist, copy the tarball over and unpack it in the repository root:
#   tools/pack_assets.sh /path/to/a3_pingpong.xml /path/to/a3_31dof_paddle_50hz [supermimic_a3_assets.tar.gz]
#   tar xzf supermimic_a3_assets.tar.gz -C /path/to/supermimic-A3
# which gives the default config paths assets/a3_racket/a3_pingpong.xml (+ meshes) and data/a3_31dof_paddle_50hz/.
set -euo pipefail
USAGE="usage: tools/pack_assets.sh <a3_pingpong.xml> <motion_dir> [out.tar.gz]"
XML=${1:?$USAGE}
MOTIONS=${2:?$USAGE}
OUT=${3:-supermimic_a3_assets.tar.gz}
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
ls "$MOTIONS"/*.npz > /dev/null

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/assets/a3_racket" "$STAGE/data/a3_31dof_paddle_50hz"
# Meshes are found through <compiler meshdir="..."> relative to the XML, so keep that layout.
MESHDIR=$(sed -n 's/.*<compiler[^>]*meshdir="\([^"]*\)".*/\1/p' "$XML" | head -1)
if [ -n "$MESHDIR" ]; then
  mkdir -p "$(dirname "$STAGE/assets/a3_racket/$MESHDIR")"
  cp -r "$(dirname "$XML")/$MESHDIR" "$STAGE/assets/a3_racket/$MESHDIR"
else
  cp -r "$(dirname "$XML")/." "$STAGE/assets/a3_racket/"
fi
cp "$XML" "$STAGE/assets/a3_racket/a3_pingpong.xml"
cp "$MOTIONS"/*.npz "$STAGE/data/a3_31dof_paddle_50hz/"

tar czf "$OUT" -C "$STAGE" assets data
echo "robot model $(du -sh "$STAGE/assets" | cut -f1), $(ls "$STAGE"/data/*/*.npz | wc -l | tr -d ' ') motions" \
  "$(du -sh "$STAGE/data" | cut -f1) -> $OUT"
