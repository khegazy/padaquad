#!/usr/bin/env bash
# Run the library comparison (examples/speed_test/library_comparison.py) and
# write the results the Performance page is built from:
# docs/performance/results_cuda.json and docs/performance/results_cpu.json.
#
# Run it on a compute node that has a GPU and nothing else running (an
# interactive allocation, or a batch or debug-queue job), never on a login
# node: other users' work there changes the timings. The GPU run and the CPU
# run go one after the other, never at the same time.
#
#   examples/speed_test/run_benchmark.sh setup     # once: build the environment
#   examples/speed_test/run_benchmark.sh [run]     # GPU run, then CPU run
#   examples/speed_test/run_benchmark.sh cuda      # GPU run only
#   examples/speed_test/run_benchmark.sh cpu       # CPU run only
#   examples/speed_test/run_benchmark.sh dry       # one timed repeat of each, to /tmp
#
# On Perlmutter, for example:
#   salloc -A <account> -q interactive -C gpu -N 1 --gpus 4 -t 01:00:00
#   examples/speed_test/run_benchmark.sh
#
# Environment variables:
#   PADAQUAD_BENCH_VENV  the environment (default: $SCRATCH/padaquad_bench, or
#                        ~/.cache/padaquad_bench without $SCRATCH)
#   PADAQUAD_BENCH_PYTHON  Python >= 3.11 to build it with (default: python3)
#   UV_CACHE_DIR         uv's download cache (default: next to the environment)
#   REPEATS              timed runs per case (default 7)
#
# After a run: read the new tables (sphinx-build, or render_markdown in the
# script), rewrite "What the results show" in docs/performance.md from them,
# and commit the results files from a clean working tree (each file records
# the commit it was measured at).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${PADAQUAD_BENCH_VENV:-${SCRATCH:-$HOME/.cache}/padaquad_bench}"
PY="$VENV/bin/python"
SCRIPT="$ROOT/examples/speed_test/library_comparison.py"
REPEATS="${REPEATS:-7}"
# JAX must not reserve most of the GPU's memory, which padaquad then cannot
# use for its batches.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

setup() {
  local python="${PADAQUAD_BENCH_PYTHON:-python3}"
  command -v uv >/dev/null || { echo "setup needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  # The CUDA wheels are several GB: keep uv's download cache next to the
  # environment, not in a home directory with a small quota.
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VENV")/padaquad_bench_uv_cache}"
  export UV_LINK_MODE=copy
  uv venv --clear --python "$python" "$VENV"
  # The CUDA builds of torch and JAX, then padaquad itself with the
  # benchmark's libraries (the bench extra of pyproject.toml).
  uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$PY" "jax[cuda12]"
  uv pip install --python "$PY" -e "$ROOT[bench]"
  "$PY" -c "import torch, jax; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('jax', jax.__version__, jax.devices())"
}

check_node() {
  # A login node has no Slurm job; refuse to time anything there.
  if [[ -z "${SLURM_JOB_ID:-}" && "${PADAQUAD_BENCH_ALLOW_ANY_NODE:-0}" != 1 ]]; then
    echo "Run this inside a compute-node allocation (no SLURM_JOB_ID is set)." >&2
    echo "Set PADAQUAD_BENCH_ALLOW_ANY_NODE=1 to run elsewhere, on an otherwise idle machine." >&2
    exit 1
  fi
  [[ -x "$PY" ]] || { echo "No environment at $VENV: run '$0 setup' first." >&2; exit 1; }
}

run_device() {  # run_device cuda|cpu OUTPUT REPEATS
  local device="$1" out="$2" repeats="$3"
  if [[ "$device" == cpu ]]; then
    # JAX defaults to the GPU when its CUDA build is installed.
    JAX_PLATFORMS=cpu "$PY" "$SCRIPT" --device cpu --repeats "$repeats" --out "$out"
  else
    # The CPU platform must stay visible: diffrax's evaluations are counted
    # with jax.debug.callback, which runs there.
    JAX_PLATFORMS=cuda,cpu "$PY" "$SCRIPT" --device cuda --repeats "$repeats" --out "$out"
  fi
}

mode="${1:-run}"
case "$mode" in
  setup) setup ;;
  run)
    check_node
    run_device cuda "$ROOT/docs/performance/results_cuda.json" "$REPEATS"
    run_device cpu "$ROOT/docs/performance/results_cpu.json" "$REPEATS"
    ;;
  cuda | cpu)
    check_node
    run_device "$mode" "$ROOT/docs/performance/results_$mode.json" "$REPEATS"
    ;;
  dry)
    check_node
    # Not /tmp: on a compute node it is local to the node.
    out="$(mktemp -d "$(dirname "$VENV")/padaquad_bench_dry.XXXXXX")"
    run_device cuda "$out/results_cuda.json" 1
    run_device cpu "$out/results_cpu.json" 1
    echo "Dry-run results in $out"
    ;;
  *)
    echo "usage: $0 [setup|run|cuda|cpu|dry]" >&2
    exit 2
    ;;
esac
