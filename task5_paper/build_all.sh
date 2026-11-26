#!/bin/bash
# Rebuild paper/inputs/results.json from every Task 4 round (run from paper/). Rounds are never pooled.
cd "$(dirname "$0")"
python -E -P tools/build_results.py --round r1=inputs/task4_r1 --round r2=inputs/task4_r2 --round r3=inputs/task4_r3 --round fu=inputs/task4_fu \
 --pair r1=mh:mh_ctrl --retention r1=m:b0 \
 --main-seeds r2=0,1,2 --pair r2=g:mil --pair r2=pu:mil --pair r2=mil:mvfa --pair r2=g:mvfa --pair r2=pu:mvfa \
 --pair r2=g_m100:mvfa --pair r2=pu_m100:mvfa --retention r2=mil:pu:g --best-epoch-exclude r2=g_m10,g_m25 \
 --main-seeds r3=3,4,5 --pair r3=mil:mvfa --pair r3=pu_m100:mvfa --pair r3=mil_m100:mvfa --pair r3=mil_m10:mvfa \
 --pair r3=mil_m25:mvfa --retention r3=mil:mil_m10:mil_m25:mil_m100:pu_m100 \
 --holm "r3=1:H2:mil-mvfa@Brain,1:H3:pu_m100-mvfa,2:H4:mil_m100-mvfa,2:H5:mil_m10-mvfa" \
 --main-seeds fu=8,9,10,11,12 --pair fu=mil_m10:mvfa --retention fu=mil_m10 \
 --out inputs/results.json
