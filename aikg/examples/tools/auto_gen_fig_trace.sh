#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEVEL2_DIR="${1:-/mnt/lustre-client/zhangzizheng/AIKG/save_data/ds_v4_flash_h100/evolve_database/level1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${LEVEL2_DIR}" ]]; then
    echo "[ERROR] level2 dir not found: ${LEVEL2_DIR}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}" || exit 1
mkdir -p evolve_plots

total=0
success=0
failed=0
skipped=0

while IFS= read -r -d '' target_path; do
    target_dir="$(basename "${target_path}")"
    island_dir="${target_path}/island_0"
    total=$((total + 1))

    if [[ ! -d "${island_dir}" ]]; then
        echo "[WARN] skip ${target_dir}: island_0 not found"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[INFO] processing ${target_dir}"

    if ! "${PYTHON_BIN}" export_evolve_json_list.py --folder_path "${island_dir}"; then
        echo "[ERROR] export failed: ${target_dir}" >&2
        failed=$((failed + 1))
        continue
    fi

    if ! "${PYTHON_BIN}" plot_exp.py --save_name "${target_dir}"; then
        echo "[ERROR] plot failed: ${target_dir}" >&2
        failed=$((failed + 1))
        continue
    fi

    success=$((success + 1))
done < <(
    find "${LEVEL2_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\0' \
        | sort -z -n -t $'\t' -k1,1 \
        | cut -z -f2-
)

rm -f evolve_*.json

echo "[INFO] done. total=${total}, success=${success}, skipped=${skipped}, failed=${failed}"

if [[ "${failed}" -gt 0 ]]; then
    exit 1
fi
