#!/usr/bin/env bash
## example to use this sh:
## chmod +x batch_convert.sh
## ./batch_convert.sh /path/to/your/folder

set -euo pipefail

IN_DIR="${1:-.}"
OUT_DIR="${IN_DIR%/}/out_2560x1440_25fps"
mkdir -p "$OUT_DIR"

shopt -s nullglob
for f in "$IN_DIR"/*.mp4; do
  base="$(basename "$f")"
  name="${base%.*}"
  out="$OUT_DIR/${name}_2560x1440_25fps.mp4"

  ffmpeg -hide_banner -y -i "$f" \
    -map 0:v:0 -an \
    -vf "fps=25,scale=2560:1440" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    "$out"
done

