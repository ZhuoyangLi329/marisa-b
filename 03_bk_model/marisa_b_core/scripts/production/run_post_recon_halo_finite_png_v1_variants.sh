#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        "usage: $0 refined_v5_qmc|finite_box_v5_qmc|tail_qmax_v5_qmc [parallel_jobs]" \
        "  refined_v5_qmc: full selected 27-bin cache, QMC power 12 / 8 replicates" \
        "  finite_box_v5_qmc: full selected 27-bin cache, qmin=2*pi/L" \
        "  tail_qmax_v5_qmc: bins 6 and 75, qmax 10/40, strictly serial"
}

if (( $# < 1 || $# > 2 )); then
    usage >&2
    exit 2
fi

variant=$1
parallel_jobs=${2:-28}
if [[ ! $parallel_jobs =~ ^[1-9][0-9]*$ ]] \
    || (( parallel_jobs > 28 )); then
    printf '%s\n' \
        "parallel_jobs must be an integer in 1..28" >&2
    exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=${MARISA_B_REPOSITORY_ROOT:-$(cd "$script_directory/../.." && pwd)}
data_root=${MARISA_B_DATA_ROOT:?set MARISA_B_DATA_ROOT to the external data bundle}
run_root=${MARISA_B_RUN_ROOT:-$data_root}
driver=${MARISA_B_POST_PNG_DRIVER:-$repo/build/halo_v1/eft_v2_post_r1_png_template_driver}
power=$data_root/analysis/theory_vectors/marisa_b_v0/marisa_b_pre_gaussian_shellbin_z1_r15_diag15_kmax0p3_nrad3_nmu12_partial446_20260616_plin_z1.dat
png=$data_root/analysis/theory_vectors/marisa_b_v0/marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat
edges=$repo/configs/eft_v2_b000_edges.txt
wrapper=$repo/scripts/run_eft_v2_r0_shard.sh
archive=$run_root/log/post_recon_halo_finite_png_v1_20260727
python=${PYTHON:-python3}
gaussian_gate=$run_root/analysis/post_recon_halo_finite_png_v1_20260727/gaussian_gate_diagnostic.json
production_templates=$run_root/analysis/post_recon_halo_finite_png_v1_20260727/post_r1_gaussian_templates.jsonl
analysis_runner=$repo/scripts/production/run_post_recon_halo_finite_png_v1.py
output=$archive/raw_png/$variant
logs=$archive/logs_png_$variant

case "$variant" in
    refined_v5_qmc)
        expected_jobs=70
        epsrel=0.01
        p13_epsrel=0.01
        qmin=0.0001
        qmc_power=12
        qmc_replicates=8
        ;;
    finite_box_v5_qmc)
        expected_jobs=70
        epsrel=0.02
        p13_epsrel=0.02
        qmin=0.006283185307179586
        qmc_power=11
        qmc_replicates=4
        ;;
    tail_qmax_v5_qmc)
        expected_jobs=8
        epsrel=0.02
        p13_epsrel=0.02
        qmin=0.0001
        qmc_power=11
        qmc_replicates=4
        parallel_jobs=1
        ;;
    *)
        printf 'unregistered variant: %s\n' "$variant" >&2
        usage >&2
        exit 2
        ;;
esac

lock_file=$archive/run_post_recon_halo_finite_png_v1_variants.lock
exec 9>"$lock_file"
if ! flock -n 9; then
    printf 'variant already running; lock=%s\n' "$lock_file" >&2
    exit 2
fi

for required in "$driver" "$power" "$png" "$edges" "$wrapper"; do
    if [[ ! -f $required ]]; then
        printf 'missing required input: %s\n' "$required" >&2
        exit 2
    fi
done

if [[ ! -f $gaussian_gate ]]; then
    printf 'Gaussian production gate is missing: %s\n' \
        "$gaussian_gate" >&2
    exit 2
fi
"$python" - \
    "$gaussian_gate" \
    "$production_templates" \
    "$analysis_runner" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

path = Path(sys.argv[1])
expected_templates = Path(sys.argv[2]).resolve()
expected_runner = Path(sys.argv[3]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("status") != "pass"
    or payload.get("finite_png_gate", {}).get("opened") is not True
):
    raise SystemExit(
        "Gaussian production gate has not passed; refusing PNG numerics"
    )

def sha256(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

inputs = payload.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("Gaussian gate lacks hashed input provenance")
registered = (
    ("templates", "templates_sha256", expected_templates),
    ("runner", "runner_sha256", expected_runner),
)
for path_key, hash_key, expected_path in registered:
    recorded_path = Path(str(inputs.get(path_key, ""))).resolve()
    if (
        recorded_path != expected_path
        or not recorded_path.is_file()
        or inputs.get(hash_key) != sha256(recorded_path)
    ):
        raise SystemExit(
            f"Gaussian gate has stale {path_key} provenance"
        )
for path_key, hash_key in (
    ("matrix", "matrix_sha256"),
    ("contract", "contract_sha256"),
):
    recorded_path = Path(str(inputs.get(path_key, ""))).resolve()
    if (
        not recorded_path.is_file()
        or inputs.get(hash_key) != sha256(recorded_path)
    ):
        raise SystemExit(
            f"Gaussian gate has stale {path_key} provenance"
        )
references = (
    payload.get("production_numerics", {}).get("references")
)
if not isinstance(references, list):
    raise SystemExit("Gaussian gate lacks convergence references")
required_categories = {
    "qmax_uv",
    "loop_quadrature",
    "shell_projector",
}
categories = {
    row.get("category")
    for row in references
    if isinstance(row, dict)
    and row.get("required_for_production") is True
}
if categories != required_categories:
    raise SystemExit(
        "Gaussian gate lacks the exact registered convergence categories"
    )
for row in references:
    if not isinstance(row, dict):
        raise SystemExit("Gaussian gate has an invalid reference row")
    candidate = Path(str(row.get("path", ""))).resolve()
    if (
        not candidate.is_file()
        or row.get("sha256") != sha256(candidate)
    ):
        raise SystemExit(
            f"Gaussian gate has stale reference provenance: {candidate}"
        )

repo = expected_runner.parents[2]
spec = importlib.util.spec_from_file_location(
    "_post_r1_png_variant_verification_validator",
    expected_runner,
)
if spec is None or spec.loader is None:
    raise SystemExit(
        f"cannot import production runner: {expected_runner}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
paths = module.build_paths(
    SimpleNamespace(
        repo_root=repo,
        templates=expected_templates,
        png_cache=(
            repo
            / "analysis/post_recon_halo_finite_png_v1_20260727/"
            / "finite_png_templates.npz"
        ),
    )
)
module.load_verification(paths)
print("Gaussian production gate: pass")
PY

active_drivers=()
while read -r process_id process_state process_name; do
    if [[ $process_state != T* && $process_state != Z* ]] \
        && [[ $process_name == eft_v2_post_r1_ ]]; then
        active_drivers+=("$process_id:$process_state")
    fi
done < <(ps -eo pid=,stat=,comm=)
if (( ${#active_drivers[@]} )); then
    printf 'refusing to compete with active post-R1 drivers: %s\n' \
        "${active_drivers[*]}" >&2
    exit 2
fi

mkdir -p "$output" "$logs"

if [[ -n ${GSL_PREFIX:-} ]]; then
    gsl_lib=$GSL_PREFIX/lib
    if compgen -G "$GSL_PREFIX/lib/*-linux-gnu" >/dev/null; then
        gsl_lib=$(compgen -G "$GSL_PREFIX/lib/*-linux-gnu" | head -n 1)
    fi
    export LD_LIBRARY_PATH="$gsl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MARISA_B_MAX_THREADS=1

png_job_complete() {
    local sector=$1
    local lambda=$2
    local lambda_label=$3
    local shard=$4
    local start=$5
    local stop=$6
    local qmax=$7
    local qmax_label=$8
    local label="${variant}_${sector//-/_}_l${lambda_label}_q${qmax_label}_s${shard}"
    local destination="$output/${sector}_lambda_${lambda_label}_qmax_${qmax_label}_shard_${shard}.jsonl"
    local stderr_path="$logs/$label.err"
    local time_path="$logs/$label.time"
    local exitcode_path="$logs/$label.exitcode"
    [[ -f $destination && -f $stderr_path \
        && -f $time_path && -f $exitcode_path ]] || return 1
    "$python" - \
        "$destination" \
        "$stderr_path" \
        "$time_path" \
        "$exitcode_path" \
        "$driver" \
        "$power" \
        "$png" \
        "$edges" \
        "$analysis_runner" \
        "$sector" \
        "$lambda" \
        "$start" \
        "$stop" \
        "$epsrel" \
        "$p13_epsrel" \
        "$qmin" \
        "$qmax" \
        "$qmc_power" \
        "$qmc_replicates" >/dev/null 2>&1 <<'PY'
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

(
    destination,
    stderr_path,
    time_path,
    exitcode_path,
    driver,
    power,
    png,
    edges,
    runner_path,
) = (Path(value).resolve() for value in sys.argv[1:10])
sector = sys.argv[10]
lambda_value = float(sys.argv[11])
start = int(sys.argv[12])
stop = int(sys.argv[13])
epsrel = float(sys.argv[14])
p13_epsrel = float(sys.argv[15])
qmin = float(sys.argv[16])
qmax = float(sys.argv[17])
qmc_power = int(sys.argv[18])
qmc_replicates = int(sys.argv[19])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")

if exitcode_path.read_text(encoding="utf-8") != "0\n":
    raise ValueError("job exit code is not exactly zero")
if stderr_path.read_bytes() != b"":
    raise ValueError("job stderr is nonempty")
timing = time_path.read_text(encoding="utf-8")
number = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
if re.fullmatch(
    rf"elapsed_seconds={number} max_rss_kb=[0-9]+ "
    rf"user_seconds={number} system_seconds={number}\n?",
    timing,
) is None:
    raise ValueError("job timing record is incomplete or malformed")
if (
    time_path.stat().st_mtime_ns < destination.stat().st_mtime_ns
    or exitcode_path.stat().st_mtime_ns < destination.stat().st_mtime_ns
):
    raise ValueError("job completion markers predate the raw output")

raw_lines = destination.read_text(encoding="utf-8").splitlines()
if len(raw_lines) != 1 + stop - start or any(
    not line.strip() for line in raw_lines
):
    raise ValueError("raw output has the wrong exact record count")
records = [
    json.loads(line, parse_constant=reject_constant)
    for line in raw_lines
]
header = records[0]
spec = importlib.util.spec_from_file_location(
    "_post_r1_png_reuse_validator",
    runner_path,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import runner: {runner_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
validated_header, validated_bins = module.read_jsonl(destination)
module._validate_png_header(
    destination,
    validated_header,
    allowed_qmax=(qmax,),
)
module._validate_png_shard_rows(
    destination,
    validated_header,
    validated_bins,
)
if validated_header != header or validated_bins != records[1:]:
    raise ValueError("raw output changed during validation")
expected_sources = {
    "linear_power": sha256(power),
    "png_table": sha256(png),
    "edge_file": sha256(edges),
    "driver_executable": sha256(driver),
}
source_hashes = header.get("source_hashes")
if (
    not isinstance(source_hashes, dict)
    or set(source_hashes)
    != {*expected_sources, "parameter_registry"}
    or any(
        source_hashes.get(key) != value
        for key, value in expected_sources.items()
    )
    or re.fullmatch(
        r"[0-9a-f]{64}",
        str(source_hashes.get("parameter_registry", "")),
    )
    is None
):
    raise ValueError("raw output has stale source provenance")
expected_integration = {
    "epsrel": epsrel,
    "p13_epsrel": p13_epsrel,
    "qmin": qmin,
    "qmax": qmax,
    "png_ir_cutoff": 0.006283185307179586,
    "png_ir_cutoff_scope": "all_local_primordial_B0_T0_legs",
    "matter_linear_multicenter_qmc": True,
    "b112ii_qmc_power": qmc_power,
    "b112ii_qmc_replicates": qmc_replicates,
}
expected_projection = {
    "measure": (
        "exact float32 FFT-lattice k1-k2-mu plus normalized "
        "Haar orientation cubature"
    ),
    "fft_box_size": 1000,
    "fft_mesh_size": 256,
    "radial_order": 2,
    "angular_order": 8,
    "orientation_orders": [2, 2, 2],
}
expected_reconstruction = {
    "R": 15,
    "b_rec_h": 2.7340475186190334,
    "cell_size": 8,
    "cic_power": 4,
}
if (
    header.get("record") != "header"
    or header.get("schema")
    != "marisa-b-post-r1-finite-png-jsonl-v5"
    or header.get("sector") != sector
    or not math.isclose(
        float(header.get("lambda")),
        lambda_value,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    or header.get("bin_range") != [start, stop]
    or header.get("radial_stop") != 7
    or header.get("shell_projection") != expected_projection
    or header.get("reconstruction") != expected_reconstruction
    or header.get("integration") != expected_integration
    or header.get("lambda_definition")
    != "b1_over_b_rec_h_for_matter_uplift"
):
    raise ValueError("raw output has the wrong exact job signature")
expected_indices = list(range(start, stop))
actual_indices = []
for row in records[1:]:
    if row.get("record") != "bin":
        raise ValueError("raw output contains a non-bin data record")
    index = row.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("raw output has a non-integral bin index")
    actual_indices.append(index)
if actual_indices != expected_indices:
    raise ValueError("raw output has the wrong ordered bin set")
PY
}

run_stamp=$(date '+%Y%m%dT%H%M%S')
retry_archive="$archive/diagnostic_png_incomplete_${variant}_${run_stamp}_$$"
declare -A expected_output_files=()
declare -A expected_log_files=()
reused_jobs=0
archived_artifacts=0
pending_jobs=0
all_jobs_file=$(mktemp)
job_file=$(mktemp)
trap 'rm -f "$all_jobs_file" "$job_file"' EXIT

archive_artifact() {
    local artifact=$1
    local target
    [[ -e $artifact ]] || return 0
    mkdir -p "$retry_archive"
    target="$retry_archive/$(basename "$artifact")"
    if [[ -e $target ]]; then
        target="$retry_archive/${archived_artifacts}_$(basename "$artifact")"
    fi
    [[ ! -e $target ]] || {
        printf 'refusing archive collision: %s\n' "$target" >&2
        return 1
    }
    mv -- "$artifact" "$target"
    archived_artifacts=$((archived_artifacts + 1))
}

register_png_variant_job() {
    local sector=$1
    local lambda=$2
    local lambda_label=$3
    local shard=$4
    local start=$5
    local stop=$6
    local qmax=$7
    local qmax_label=$8
    local label="${variant}_${sector//-/_}_l${lambda_label}_q${qmax_label}_s${shard}"
    local destination="$output/${sector}_lambda_${lambda_label}_qmax_${qmax_label}_shard_${shard}.jsonl"
    local stderr_path="$logs/$label.err"
    local time_path="$logs/$label.time"
    local exitcode_path="$logs/$label.exitcode"
    expected_output_files["$destination"]=1
    expected_log_files["$stderr_path"]=1
    expected_log_files["$time_path"]=1
    expected_log_files["$exitcode_path"]=1
    printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0' \
        "$sector" "$lambda" "$lambda_label" "$shard" \
        "$start" "$stop" "$qmax" "$qmax_label" \
        >>"$all_jobs_file"
    if png_job_complete "$@"; then
        reused_jobs=$((reused_jobs + 1))
        return 0
    fi
    archive_artifact "$destination"
    archive_artifact "$stderr_path"
    archive_artifact "$time_path"
    archive_artifact "$exitcode_path"
    printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0' \
        "$sector" "$lambda" "$lambda_label" "$shard" \
        "$start" "$stop" "$qmax" "$qmax_label" \
        >>"$job_file"
    pending_jobs=$((pending_jobs + 1))
}

archive_unexpected_artifacts() {
    local artifact
    while IFS= read -r -d '' artifact; do
        [[ ${expected_output_files["$artifact"]+registered} ]] \
            || archive_artifact "$artifact"
    done < <(
        find "$output" -mindepth 1 -maxdepth 1 -type f -print0
    )
    while IFS= read -r -d '' artifact; do
        [[ ${expected_log_files["$artifact"]+registered} ]] \
            || archive_artifact "$artifact"
    done < <(
        find "$logs" -mindepth 1 -maxdepth 1 -type f -print0
    )
}

validate_all_jobs() {
    local sector lambda lambda_label shard start stop qmax qmax_label
    local validated=0
    while IFS= read -r -d '' sector \
        && IFS= read -r -d '' lambda \
        && IFS= read -r -d '' lambda_label \
        && IFS= read -r -d '' shard \
        && IFS= read -r -d '' start \
        && IFS= read -r -d '' stop \
        && IFS= read -r -d '' qmax \
        && IFS= read -r -d '' qmax_label; do
        if ! png_job_complete \
            "$sector" "$lambda" "$lambda_label" "$shard" \
            "$start" "$stop" "$qmax" "$qmax_label"; then
            printf 'PNG job failed exact completion validation: %s %s %s %s\n' \
                "$variant" "$sector" "$lambda_label" "$shard" >&2
            return 1
        fi
        validated=$((validated + 1))
    done <"$all_jobs_file"
    if (( validated != expected_jobs )); then
        printf 'variant=%s expected_jobs=%s validated_jobs=%s\n' \
            "$variant" "$expected_jobs" "$validated" >&2
        return 1
    fi
    printf 'variant=%s validated_jobs=%s reused_jobs=%s new_jobs=%s archived_artifacts=%s\n' \
        "$variant" "$validated" "$reused_jobs" "$pending_jobs" \
        "$archived_artifacts"
}

run_png_variant_job() {
    local sector=$1
    local lambda=$2
    local lambda_label=$3
    local shard=$4
    local start=$5
    local stop=$6
    local qmax=$7
    local qmax_label=$8
    local label="${variant}_${sector//-/_}_l${lambda_label}_q${qmax_label}_s${shard}"
    local destination="$output/${sector}_lambda_${lambda_label}_qmax_${qmax_label}_shard_${shard}.jsonl"
    taskset -c 0-27 "$wrapper" \
        "$label" \
        "$destination" \
        "$logs" \
        "$driver" \
        "$power" \
        "$png" \
        "$edges" \
        7 \
        "$start" \
        "$stop" \
        2 \
        8 \
        2 \
        2 \
        2 \
        "$sector" \
        "$lambda" \
        "$epsrel" \
        "$p13_epsrel" \
        "$qmin" \
        "$qmax" \
        0.006283185307179586 \
        "$qmc_power" \
        "$qmc_replicates" \
        15 \
        2.7340475186190334 \
        8 \
        4
}
export -f run_png_variant_job
export variant output logs wrapper driver power png edges
export epsrel p13_epsrel qmin qmc_power qmc_replicates

if [[ $variant == tail_qmax_v5_qmc ]]; then
    # Result-blind endpoint canary: the most squeezed selected bin and the
    # highest-k diagonal bin.  Keep these eight calls strictly serial.
    for flat in 6 75; do
        for sector in matter-linear matter-b112; do
            for qmax_specification in "10 10" "40 40"; do
                read -r qmax qmax_label <<<"$qmax_specification"
                register_png_variant_job \
                    "$sector" 1 1 "$flat" \
                    "$flat" "$((flat + 1))" \
                    "$qmax" "$qmax_label"
            done
        done
    done
    archive_unexpected_artifacts
    while IFS= read -r -d '' sector \
        && IFS= read -r -d '' lambda \
        && IFS= read -r -d '' lambda_label \
        && IFS= read -r -d '' shard \
        && IFS= read -r -d '' start \
        && IFS= read -r -d '' stop \
        && IFS= read -r -d '' qmax \
        && IFS= read -r -d '' qmax_label; do
        run_png_variant_job \
            "$sector" "$lambda" "$lambda_label" "$shard" \
            "$start" "$stop" "$qmax" "$qmax_label"
    done <"$job_file"
    validate_all_jobs
    exit 0
fi

# Bin zero is preregistered as excluded from every inference and numerical
# gate.  All 27 retained kmax<=0.14 bins are included in each cache variant.
starts=(1 15 29 42 54 65 75)
stops=(7 21 34 46 57 67 76)
for shard in "${!starts[@]}"; do
    register_png_variant_job \
        tree-fixed 0 0 "$shard" \
        "${starts[$shard]}" "${stops[$shard]}" 20 20
    for item in "0 0" "1 1" "2 2" "1.37 1p37" "2.2 2p2"; do
        read -r lambda lambda_label <<<"$item"
        register_png_variant_job \
            matter-linear "$lambda" "$lambda_label" "$shard" \
            "${starts[$shard]}" "${stops[$shard]}" 20 20
    done
    for item in "0 0" "1 1" "1.37 1p37" "2.2 2p2"; do
        read -r lambda lambda_label <<<"$item"
        register_png_variant_job \
            matter-b112 "$lambda" "$lambda_label" "$shard" \
            "${starts[$shard]}" "${stops[$shard]}" 20 20
    done
done

archive_unexpected_artifacts
xargs -0 -r -n 8 -P "$parallel_jobs" \
    bash -c 'run_png_variant_job "$@"' _ <"$job_file"
validate_all_jobs
