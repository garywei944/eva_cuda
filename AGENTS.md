# Repository Guidelines

## Scope

These instructions apply to the entire repository.

## Project Overview

`eva_cuda` is a small CUDA learning and experimentation repository. The examples under
`freeCodeCamp.org/05_Writing_your_First_Kernels/` are standalone programs that track course
progress through CUDA indexing, kernels, profiling, and streams. There is currently no shared
library, build system, or automated test suite.

## Repository Layout

- `01 CUDA Basics/`: thread and block indexing exercises.
- `02 Kernels/`: standalone vector-add and matrix-multiplication exercises.
- `03 Profiling/`: NVTX and Nsight profiling experiments.
- `05 Streams/`: CUDA stream, event, and callback exercises.
- `.vscode/`: CUDA/clangd editor configuration; CUDA is expected under `/opt/cuda`.
- `.pre-commit-config.yaml`: repository-wide formatting and lint hooks.

## Working Practices

- Inspect `git status` before editing and preserve unrelated or concurrent worktree changes.
- Keep each exercise self-contained unless the task explicitly introduces shared infrastructure.
- Prefer small changes that preserve the educational progression of each example.
- Do not commit generated executables, CUDA intermediates, profiler reports, SQLite files, or
  debugger history; these are covered by `.gitignore`.
- Do not force-push or overwrite existing work.

## Build and Verification

Paths contain spaces, so quote source paths in shell commands. Build artifacts should go outside
the source tree when practical. For example:

```bash
nvcc -std=c++17 \
  "freeCodeCamp.org/05_Writing_your_First_Kernels/01 CUDA Basics/01_idxing.cu" \
  -o /tmp/eva_cuda_idxing
```

For every changed CUDA example:

1. Compile it with `nvcc -std=c++17`.
2. Check CUDA API return values and kernel-launch errors.
3. Synchronize before measuring host-side elapsed time.
4. Compare GPU results with a CPU reference or another independently computed expectation.
5. Run formatting checks without modifying unrelated files:

```bash
git ls-files -z '*.cu' | xargs -0 clang-format --dry-run --Werror
```

Use Nsight Systems or Compute Sanitizer when a change affects profiling, memory access, streams,
or synchronization. Report actual command output; do not infer runtime correctness from successful
compilation alone.

## Style

- Follow `.editorconfig`: UTF-8, LF endings, final newline, and two-space indentation for C/CUDA.
- Use C++17-compatible CUDA unless a task explicitly changes the language level.
- Keep kernel launch geometry and memory ownership explicit.
- Prefer descriptive names for host/device pointers and document non-obvious synchronization or
  ordering requirements.
