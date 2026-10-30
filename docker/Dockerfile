# syntax=docker/dockerfile:1
# strix-vllm: the asf0/strix-vllm fork + the libstrix plugin, built for gfx1151
# (Strix Halo). ROCm, torch and Triton come as gfx1151 wheels from AMD's index;
# only vLLM's C++ and libstrix compile here. Build from the libstrix checkout:
#
#   docker build -f docker/Dockerfile -t strix-vllm .
#
# STRIX_VLLM_REF pins the fork; libstrix is the build context.

ARG UBUNTU=ubuntu:26.04

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

FROM base AS builder
ARG STRIX_VLLM_REPO=https://github.com/asf0/strix-vllm.git
ARG STRIX_VLLM_REF=gfx1151
ARG MAX_JOBS=8
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
# tensorizer 2.10.1 reads the class-body __annotations__, which Python 3.14
# (PEP 649) no longer materializes; without this `import vllm` fails.
COPY docker/patch_tensorizer.py /tmp/patch_tensorizer.py
RUN python /tmp/patch_tensorizer.py
RUN git clone --filter=blob:none "$STRIX_VLLM_REPO" /src/strix-vllm \
    && git -C /src/strix-vllm checkout "$STRIX_VLLM_REF" \
    && git -C /src/strix-vllm rev-parse HEAD > /src/strix-vllm.ref
# The fork clone has no upstream release tags; stamp the release it is based on
# plus the fork commit instead of setuptools-scm's 0.1.devN fallback.
ARG VLLM_BASE_VERSION=0.32.1rc0
RUN cd /src/strix-vllm \
    && VLLM_VERSION_OVERRIDE="$VLLM_BASE_VERSION+strix.g$(cut -c1-10 /src/strix-vllm.ref)" \
       MAX_JOBS=$MAX_JOBS pip install --no-build-isolation --no-deps .
COPY . /src/libstrix
RUN cd /src/libstrix && MAX_JOBS=$MAX_JOBS ./build.sh \
    && pip install --no-build-isolation --no-deps . \
    && (git -c safe.directory='*' -C /src/libstrix rev-parse HEAD || echo unknown) > /src/libstrix.ref
# Unused at runtime: rpp (image processing; nothing links it) and the static
# libraries. LLVM stays: hipcc-based JIT (tilelang, aiter) needs it.
# The big, rarely-changing trees move to /stage so the runtime stage copies
# each as its own layer: an update to vLLM or libstrix then re-downloads only
# the last layer, and no layer approaches the registry's 10 GB limit.
# The hard links rocm-sdk init creates become symlinks: COPY would store each
# hard link as a full copy.
COPY docker/dedup_rocm.py /tmp/dedup_rocm.py
RUN rm -f "$ROCM_PATH"/lib/librpp.so* "$ROCM_PATH"/lib/*.a \
    && mkdir -p /stage/rocm \
    && mv "$SP"/_rocm_sdk_* /stage/rocm/ \
    && python /tmp/dedup_rocm.py /stage/rocm \
    && mv "$SP/torch" "$SP/triton" /stage/

FROM base AS runtime
ARG BUILD_DATE
LABEL org.opencontainers.image.source=https://github.com/asf0/libstrix \
      org.opencontainers.image.description="vLLM (asf0/strix-vllm) with the libstrix plugin for AMD Strix Halo (gfx1151)" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.created=${BUILD_DATE}
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
# Layers from least to most often changed (see the /stage step above).
# Every _rocm_sdk_* tree in one layer; their shared libraries are symlinks
# (dedup_rocm.py), so each is stored once.
COPY --from=builder /stage/rocm/ ${SP}/
COPY --from=builder /stage/torch ${SP}/torch
COPY --from=builder /stage/triton ${SP}/triton
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/strix-vllm.ref /src/libstrix.ref /opt/strix/
COPY docker/entrypoint.sh /opt/strix/entrypoint.sh
# Compile, Triton and HF caches go to one volume so restarts skip recompiling.
ENV XDG_CACHE_HOME=/cache \
    VLLM_CACHE_ROOT=/cache/vllm \
    TRITON_CACHE_DIR=/cache/triton \
    HF_HOME=/cache/huggingface
RUN cd /tmp && python -c "import vllm, strix, strix_vllm; print('vllm', vllm.__version__, '| strix kernels', len(strix.kernels()))"
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strix/entrypoint.sh"]
