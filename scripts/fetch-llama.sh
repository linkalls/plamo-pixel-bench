#!/usr/bin/env bash
set -euo pipefail

# Pin to a verified PLaMo-3-capable upstream commit by passing LLAMA_REF.
# The default tag is updated only after CI validation.
LLAMA_REF="${LLAMA_REF:-ca89968a23a13e571a6b802a9891d22763153b61}"
DIR="app/src/main/cpp/llama.cpp"
if [ -f "${DIR}/CMakeLists.txt" ]; then
    echo "llama.cpp already present: ${DIR}"
    exit 0
fi
mkdir -p "$(dirname "${DIR}")"
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${DIR}"
if [ "${LLAMA_REF}" != "master" ]; then
    git -C "${DIR}" fetch --depth 1 origin "${LLAMA_REF}"
    git -C "${DIR}" checkout --detach FETCH_HEAD
fi
git -C "${DIR}" rev-parse HEAD

