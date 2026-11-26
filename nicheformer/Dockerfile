# GPU image for the three Nicheformer front-ends in this folder:
#   Python/  - notebook (PyTorch + Lightning, TorchScript export)
#   R/       - R torch inference on the TorchScript model
#   C/       - libtorch C++ embedder (nicheformer_embed)
#
# Build:  docker build -t nicheformer-gpu .
# Run:    docker run --rm -it --gpus all -v "$PWD":/workspace -p 8888:8888 nicheformer-gpu
#
# CUDA 12.6 is the one version supported by both PyTorch (cu126 wheels) and
# R torch (12.6 / 12.8 / 12.9), so everything shares a single toolkit.

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.13.0
ARG TORCH_CUDA=cu126
ARG R_TORCH_CUDA=12.6
# GitHub "user/repo" for the nicheR package used by R/nicheformer.r (left out if empty)
ARG NICHER_REMOTE=""

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/nicheformer/bin:${PATH}

# ---------------------------------------------------------------- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl wget ca-certificates unzip \
        gnupg software-properties-common dirmngr \
        python3 python3-venv python3-dev python3-pip \
        libhdf5-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
        libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev \
        libpng-dev libtiff5-dev libjpeg-dev libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- Python
RUN python3 -m venv ${VIRTUAL_ENV} \
    && pip install --upgrade pip setuptools wheel \
    && pip install torch==${TORCH_VERSION} \
           --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    && pip install \
           pytorch-lightning anndata scanpy numpy scipy scikit-learn numba tqdm \
           "transformers==4.57.6" "tiktoken==0.9.0" "sentencepiece==0.2.1" \
           jupyterlab ipykernel

# ---------------------------------------------------------------- R
# R from CRAN's Ubuntu repo; CRAN packages as Linux binaries from Posit Package Manager.
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | gpg --dearmor -o /usr/share/keyrings/cran.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    && apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' \
        'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))' \
        'options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))' \
        'options(Ncpus = parallel::detectCores())' \
        >> /usr/lib/R/etc/Rprofile.site

# R torch downloads its own libtorch + lantern built for CUDA ${R_TORCH_CUDA}.
ENV CUDA=${R_TORCH_CUDA} \
    TORCH_INSTALL=1 \
    CUDA_HOME=/usr/local/cuda
RUN Rscript -e 'install.packages(c("torch", "BiocManager", "remotes", "IRkernel"))' \
    && Rscript -e 'torch::install_torch()' \
    && Rscript -e 'BiocManager::install(c("SingleCellExperiment", "anndataR", "rhdf5"), ask = FALSE, update = FALSE)' \
    && Rscript -e 'IRkernel::installspec(user = FALSE)' \
    && if [ -n "${NICHER_REMOTE}" ]; then \
           Rscript -e "remotes::install_github('${NICHER_REMOTE}', upgrade = 'never')"; \
       fi

# ---------------------------------------------------------------- C++ (libtorch)
# Build against the libtorch shipped inside the pip torch wheel, so the C++
# binary uses exactly the same torch version that scripted the model.
COPY C/ /opt/nicheformer/src/
RUN TORCH_CMAKE=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") \
    && cmake -S /opt/nicheformer/src -B /opt/nicheformer/build -G Ninja \
           -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}" \
    && cmake --build /opt/nicheformer/build --target nicheformer_embed \
    && mkdir -p /opt/nicheformer/bin \
    && cp /opt/nicheformer/build/nicheformer_embed /opt/nicheformer/bin/

WORKDIR /workspace
EXPOSE 8888
CMD ["bash"]
