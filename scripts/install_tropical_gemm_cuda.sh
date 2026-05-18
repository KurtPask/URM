#!/usr/bin/env bash
set -euo pipefail

pip install maturin
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python
maturin develop --release --features cuda
