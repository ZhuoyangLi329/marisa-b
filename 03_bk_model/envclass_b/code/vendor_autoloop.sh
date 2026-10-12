#!/bin/bash
# 自动 vendor 补拷循环：ModuleNotFoundError 的模块从旧 codes 拷入 vendor/py。
cd /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main >/dev/null 2>&1
for round in $(seq 1 15); do
  echo "=== round $round ==="
  if timeout 5400 python3 code/run_envclass_b_kmax_scan.py > scan_run.log 2>&1; then
    echo SCAN_COMPLETE
    grep -E 'kmax=|saved:' scan_run.log | tail -46
    exit 0
  fi
  missing=$(grep -oP "No module named '\K[^']+" scan_run.log | head -1)
  if [ -z "$missing" ]; then
    echo NO_MISSING_MODULE_OTHER_ERROR
    tail -8 scan_run.log
    exit 1
  fi
  echo "vendoring: $missing"
  ok=0
  for t in $(seq 1 30); do
    if cp "/pscratch/sd/l/lzy/reconstruction-png/codes/$missing.py" vendor/py/ 2>/dev/null; then
      ok=1
      break
    fi
    sleep 8
  done
  if [ $ok -eq 0 ]; then
    echo "COPY_FAIL_$missing"
    exit 1
  fi
done
echo LOOP_EXHAUSTED
exit 1
