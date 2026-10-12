#!/bin/bash
# 一次性把旧仓库 B 端依赖拷入 v2 vendor/（躲开旧仓库的元数据抖动）。
# 每个文件重试直到成功（抖动窗口分钟级）。
set -u
SRC=/pscratch/sd/l/lzy/reconstruction-png
DST=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/vendor
mkdir -p "$DST/py" "$DST/data"

copy_retry() {
  local rel=$1 out=$2
  for i in $(seq 1 40); do
    if cp "$SRC/$rel" "$DST/$out" 2>/dev/null && [ -s "$DST/$out" ]; then
      echo "OK  $out ($(stat -c%s "$DST/$out") bytes)"
      return 0
    fi
    sleep 8
  done
  echo "FAIL $rel"
  return 1
}

fail=0
copy_retry codes/quijote_halo_z0p5_direct_finite_fnl_models.py py/low.py || fail=1
copy_retry codes/quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models.py py/scan.py || fail=1
copy_retry codes/run_quijote_halo_z0p5_local_png_p112_recovery_profiles.py py/recovery.py || fail=1
copy_retry "codes/run_quijote_halo_z0p5_local_png_p112_full_parameter_profiles_v42p875_priorpm200.py" py/full.py || fail=1
copy_retry codes/quijote_halo_z0p5_shared_fully_tree_context_kmax0p20.py py/high_backend.py || fail=1
copy_retry analysis/quijote_halo_z0p5_pre_p0_fnl250_v1/PRE_P0_FNL250_MATRIX.npz data/PRE_P0_FNL250_MATRIX.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_fnl250_standard_marisa/matrices/STANDARD_FNL250_MATRIX.npz data/STANDARD_FNL250_MATRIX.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_fnl250_standard_marisa/matrices/MARISA_FNL250_MATRIX.npz data/MARISA_FNL250_MATRIX.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1/theory/marisa_adaptive_finite_tree_bases.npz data/marisa_finite_kmax0p08.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p08_v1/theory/marisa_adaptive_finite_tree_bases.json data/marisa_finite_kmax0p08.json || fail=1
copy_retry analysis/quijote_halo_z0p5_pre_direct_finite_tree_kmax0p20_v1/theory/pre_halo_tree_finite_png_z0p5_kmax0p20_nrad4_nmu80.npz data/pre_tree_kmax0p20.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_pre_direct_finite_tree_kmax0p20_v1/theory/pre_halo_tree_finite_png_z0p5_kmax0p20_nrad4_nmu80.json data/pre_tree_kmax0p20.json || fail=1
copy_retry analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p20_v1/theory/marisa_adaptive_finite_tree_bases.npz data/marisa_finite_kmax0p20.npz || fail=1
copy_retry analysis/quijote_halo_z0p5_marisa_direct_finite_tree_kmax0p20_v1/theory/marisa_adaptive_finite_tree_bases.json data/marisa_finite_kmax0p20.json || fail=1
echo "VENDOR_SETUP done fail=$fail"
exit $fail
