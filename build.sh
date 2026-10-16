#!/usr/bin/env bash
# Usage: build.sh [--debug|--release] [file.cu ...]   (flag may go anywhere; paths relative to cwd)
#   flag only       -> configure that build directory (cmake -S)
#   files only      -> build them in build64_release
#   flag and files  -> configure, then build them in that directory
# Prints the binary path of every built file.
set -euo pipefail

usage() {
  echo "usage: $0 [--debug|--release] [file.cu ...]" >&2
  exit 1
}

configure=false
type=release
files=()
for arg in "$@"; do
  case "$arg" in
    --debug) type=debug configure=true ;;
    --release) type=release configure=true ;;
    -*) usage ;;
    *) files+=("$arg") ;;
  esac
done
[[ $configure == true || ${#files[@]} -gt 0 ]] || usage

root="$(cd "$(dirname "$0")" && pwd)"
dir="$root/build64_$type"

# Also configure when the directory has never been configured.
if [[ $configure == true || ! -f $dir/CMakeCache.txt ]]; then
  cmake -S "$root" -B "$dir" -DCMAKE_BUILD_TYPE="${type^}"
fi
[[ ${#files[@]} -gt 0 ]] || exit 0

# Same naming rule as CMakeLists.txt: drop .cu, map non [A-Za-z0-9_] chars to _
targets=()
for f in "${files[@]}"; do
  src=$(realpath --relative-to="$root" -- "$f")
  targets+=("$(printf '%s' "${src%.cu}" | tr -c 'A-Za-z0-9_' '_')")
done

cmake --build "$dir" --target "${targets[@]}"
for t in "${targets[@]}"; do
  echo "$dir/$t"
done
