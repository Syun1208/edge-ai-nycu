FROM nvidia/cuda:12.8.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 cuda-cuxxfilt-12-8 \
        cuda-nsight-compute-12-8 cuda-nsight-systems-12-8 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
WORKDIR /work
