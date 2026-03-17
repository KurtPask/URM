#!/usr/bin/env bash
set -euo pipefail

group_name="sudoku-urm-vs-tarm-probe-$(date +%Y%m%d-%H%M%S)"

WANDB_GROUP="$group_name" bash scripts/URM_sudoku_probe.sh
WANDB_GROUP="$group_name" bash scripts/TARM_sudoku_probe.sh
