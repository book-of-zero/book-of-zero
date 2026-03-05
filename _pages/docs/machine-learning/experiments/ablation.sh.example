#!/usr/bin/env bash
# Run all ablation variants. Expects configs/ablation_*.yaml and train.py.

set -euo pipefail

CONFIGS=(configs/ablation_*.yaml)

if [ "${#CONFIGS[@]}" -eq 0 ] || [ ! -f "${CONFIGS[0]}" ]; then
    echo "No ablation configs found matching configs/ablation_*.yaml" >&2
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    echo "=== Running: ${config} ==="
    python train.py --config "$config"
done

echo "All ablation variants complete."
