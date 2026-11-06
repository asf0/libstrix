# syntax=docker/dockerfile:1
# strix-vllm: the asf0/strix-vllm fork + the libstrix plugin, built for gfx1151
# (Strix Halo). ROCm, torch and Triton come as gfx1151 wheels from AMD's index;
# only vLLM's C++ and libstrix compile here. Build from the libstrix checkout:
#
#   docker build -f docker/Dockerfile -t strix-vllm .
#
# STRIX_VLLM_REF pins the fork; libstrix is the build context. A local fork
# checkout (commits not on GitHub yet) comes in as the named context
# strix-vllm-local, a git repository (docker/build-and-push.sh passes a bare
# clone with STRIX_VLLM_LOCAL=/path/to/strix-vllm):
#
#   docker build --build-context strix-vllm-local=/tmp/strix-vllm.git ...

ARG UBUNTU=ubuntu:26.04
# The builder stage starts from this: the local `toolchain` stage by default,
# or a prebuilt toolchain image (CI passes ghcr.io/asf0/strix-vllm-toolchain:<hash>).
ARG TOOLCHAIN=toolchain

# Empty unless --build-context strix-vllm-local=... replaces it.
FROM scratch AS strix-vllm-local

# The XDNA2 NPU shim (npu/, STRIX_ENABLE=npu): a Rust cdylib over hipfire's
# trunk transport and G80 program builder, built against hipfire at HIPFIRE_REF.
# It links only libc and dlopens the HIP runtime torch already loaded.
FROM rust:1.99-trixie AS npu
ARG HIPFIRE_REPO=https://github.com/warpfront/hipfire.git
ARG HIPFIRE_REF=6fded099bf9022ffd431abcfc7ac491e33711c8b
RUN git clone --filter=blob:none "$HIPFIRE_REPO" /src/hipfire \
    && git -C /src/hipfire checkout "$HIPFIRE_REF"
COPY npu /src/libstrix/npu
RUN cd /src/libstrix/npu && cargo build --release \
    && install -D target/release/libstrix_npu.so /stage/libstrix_npu.so

FROM ${UBUNTU} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Triton builds its launcher with the host C compiler and Python headers at
# first use, so both stages need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.14 python3.14-venv python3.14-dev gcc g++ libnuma1 libdrm2 \
        libelf1t64 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3.14 -m venv /opt/venv
# The ROCm SDK lives inside the wheels; these mirror the host env script.
ENV SP=/opt/venv/lib/python3.14/site-packages
ENV ROCM_PATH=${SP}/_rocm_sdk_devel \
    ROCM_HOME=${SP}/_rocm_sdk_devel \
    HIP_PATH=${SP}/_rocm_sdk_devel \
    LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_libraries/lib \
    PYTHONPATH=${SP}/_rocm_sdk_core/share/amd_smi \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151

# Everything that depends only on the pins below (docker/constraints.txt), not
# on the fork or libstrix: built once and
# reused by .github/workflows/image.yml.
FROM base AS toolchain
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
        rocm==10.1.0 rocm-bootstrap==0.1.0 rocm-sdk-core==10.1.0 rocm-sdk-devel==10.1.0 \
        rocm-sdk-libraries==10.1.0 rocm-sdk-device-gfx1151==10.1.0 \
        torch==2.14.0+rocm10.1.0 amd-torch-device-gfx1151==2.14.0+rocm10.1.0 \
        amd-torch-device-gfx115x==2.14.0+rocm10.1.0 \
        torchvision==0.29.0a0+rocm10.1.0 amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0 \
        torchaudio==2.11.0.3+rocm10.1.0 triton==3.8.0+git669b31ac.rocm10.1.0
# Expands the devel tree (hipcc, headers, CMake configs) and links the gfx1151
# device files into it; ROCM_PATH points there.
RUN rocm-sdk init && test -x "$ROCM_PATH/lib/llvm/bin/clang++"
COPY docker/constraints.txt /tmp/constraints.txt
# The rest of the working venv at its exact versions. --no-deps: that venv was
# assembled with several no-deps installs, so the set runs but does not resolve.
# --ignore-requires-python: amd-quark's py3 wheel is capped below 3.14 but runs.
RUN pip install --no-deps --ignore-requires-python -r /tmp/constraints.txt \
    && python -c "import torch; assert torch.version.hip, torch.__version__"

FROM ${TOOLCHAIN} AS builder
ARG STRIX_VLLM_REPO=https://github.com/asf0/strix-vllm.git
ARG STRIX_VLLM_REF=gfx1151
ARG MAX_JOBS=8
RUN --mount=type=bind,from=strix-vllm-local,target=/mnt/strix-vllm \
    if git -C /mnt/strix-vllm rev-parse --git-dir >/dev/null 2>&1; then \
        git -c safe.directory='*' clone /mnt/strix-vllm /src/strix-vllm; \
    else \
        git clone --filter=blob:none "$STRIX_VLLM_REPO" /src/strix-vllm; \
    fi \
    && git -C /src/strix-vllm checkout "$STRIX_VLLM_REF" \
    && git -C /src/strix-vllm rev-parse HEAD > /src/strix-vllm.ref
# The fork clone has no upstream release tags; stamp the release it is based on
# plus the fork commit instead of setuptools-scm's 0.1.devN fallback.
ARG VLLM_BASE_VERSION=0.33.0
RUN cd /src/strix-vllm \
    && VLLM_VERSION_OVERRIDE="$VLLM_BASE_VERSION+strix.g$(cut -c1-10 /src/strix-vllm.ref)" \
       MAX_JOBS=$MAX_JOBS pip install --no-build-isolation --no-deps .
COPY . /src/libstrix
RUN cd /src/libstrix && MAX_JOBS=$MAX_JOBS ./build.sh \
    && pip install --no-build-isolation --no-deps . \
    && (git -c safe.directory='*' -C /src/libstrix rev-parse HEAD || echo unknown) > /src/libstrix.ref
# The runtime needs only _rocm_sdk_core and _rocm_sdk_libraries: no recipe
# loads a file from _rocm_sdk_devel, and nothing compiles with hipcc at
# runtime (Triton has its own LLVM and builds its launcher with gcc; the image
# leaves out tilelang, see docs/docker.md#building-the-image). Also unused at
# runtime: static libraries, rpp (image processing; nothing links it), LLVM's
# tools, headers and MLIR (comgr links only libLLVM and libclang-cpp; the clang
# resource headers and device bitcode stay for hiprtc), hipify, rocgdb and its
# per-Python bindings for versions other than 3.14.
# The big, rarely-changing trees move to /stage so the runtime stage copies
# each as its own layer: an update to vLLM or libstrix then re-downloads only
# the last layer, and no layer approaches the registry's 10 GB limit.
RUN mkdir -p /stage/rocm \
    && mv "$SP"/_rocm_sdk_core "$SP"/_rocm_sdk_libraries /stage/rocm/ \
    && rm -rf "$SP"/_rocm_sdk_devel \
    && cd /stage/rocm \
    && find . -name '*.a' -delete \
    && rm -rf _rocm_sdk_libraries/lib/librpp.so* \
        _rocm_sdk_core/lib/llvm/bin _rocm_sdk_core/lib/llvm/include \
        _rocm_sdk_core/lib/llvm/lib/libMLIR* _rocm_sdk_core/lib/llvm/lib/libmlir_* \
        _rocm_sdk_core/lib/llvm/lib/libclang.so* _rocm_sdk_core/lib/llvm/lib/libomptarget* \
        _rocm_sdk_core/lib/llvm/lib/libLLVMOffload* \
        _rocm_sdk_core/bin/hipify-clang _rocm_sdk_core/bin/rocgdb* _rocm_sdk_core/bin/rocgcore* \
        _rocm_sdk_core/lib/python3.1[0-3] \
    && mv "$SP/torch" "$SP/triton" /stage/

FROM base AS runtime
ARG BUILD_DATE
LABEL org.opencontainers.image.source=https://github.com/asf0/libstrix \
      org.opencontainers.image.description="vLLM (asf0/strix-vllm) with the libstrix plugin for AMD Strix Halo (gfx1151)" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.created=${BUILD_DATE}
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
# Without the devel tree the ROCm root is _rocm_sdk_core.
ENV ROCM_PATH=${SP}/_rocm_sdk_core \
    ROCM_HOME=${SP}/_rocm_sdk_core \
    HIP_PATH=${SP}/_rocm_sdk_core \
    LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_libraries/lib
# Layers from least to most often changed (see the /stage step above).
# Both _rocm_sdk_* trees in one layer.
COPY --from=builder /stage/rocm/ ${SP}/
COPY --from=builder /stage/torch ${SP}/torch
COPY --from=builder /stage/triton ${SP}/triton
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/strix-vllm.ref /src/libstrix.ref /opt/strix/
COPY --from=npu /stage/libstrix_npu.so /opt/strix/lib/libstrix_npu.so
COPY docker/entrypoint.sh /opt/strix/entrypoint.sh
# Compile, Triton and HF caches go to /cache; the recipes mount a persistent
# host directory there by default, or a tmpfs for local test runs.
ENV XDG_CACHE_HOME=/cache \
    VLLM_CACHE_ROOT=/cache/vllm \
    TRITON_CACHE_DIR=/cache/triton \
    HF_HOME=/cache/huggingface
RUN cd /tmp && python -c "import vllm, strix, strix_vllm; print('vllm', vllm.__version__, '| strix kernels', len(strix.kernels()))"
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strix/entrypoint.sh"]
