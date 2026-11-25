#!/bin/bash
# Downloads MVFA data, zero-shot checkpoints and CLIP weights to the HDD.
# The HDD (NTFS/fuseblk) rejects chmod, which gdown needs, so each file is
# downloaded to a temp dir on the SSD, copied without permissions, then removed.
set -u
. "$(cd "$(dirname "$0")" && pwd -P)/../shared/config.sh"   # TASK4_PYTHON, TASK4_HDD, TASK4_DL_TMP
G=${GDOWN:-$(dirname "$TASK4_PYTHON")/gdown}
R=$TASK4_HDD
TMP=$TASK4_DL_TMP
mkdir -p "$TMP"

fetch() {  # fetch <gdrive-id> <dest-dir> <name>
  local id=$1 dest=$2 name=$3
  if [ -f "$dest/$name.done" ]; then echo "$name already done"; return 0; fi
  rm -f "$TMP/$name"
  if $G -q "$id" -O "$TMP/$name" && cat "$TMP/$name" > "$dest/$name" && [ "$(stat -c%s "$TMP/$name")" = "$(stat -c%s "$dest/$name")" ]; then
    touch "$dest/$name.done"; echo "$name ok $(du -h "$dest/$name" | cut -f1)"
  else
    echo "$name FAILED"
  fi
  rm -f "$TMP/$name"
}

cd $R/clip && [ -s ViT-L-14-336px.pt ] && echo "CLIP already done"
rm -f $R/data/*.part
fetch 1nGhcK32CrkgTR5Rav6rNfptUHaASfRnU $R/ckpt zero-shot.zip
fetch 1xriF0uiwrgoPh01N6GlzE5zPi_OIJG1I $R/data Liver.tar.gz
fetch 1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK $R/data Brain.tar.gz
fetch 1BqDbK-7OP5fUha5zvS2XIQl-_t8jhTpX $R/data Retina_RESC.tar.gz
fetch 1GqT0V3_3ivXPAuTn4WbMM6B9i0JQcSnM $R/data Retina_OCT2017.tar.gz
fetch 15DhnBAk-h6TGLTUbNLxP8jCCDtwOHAjb $R/data Chest.tar.gz
fetch 1hueVJZCFIZFHBLHFlv1OhqF8SFjUVHk6 $R/data Histopathology_AD.tar.gz
rmdir "$TMP" 2>/dev/null
echo ALL_DONE
