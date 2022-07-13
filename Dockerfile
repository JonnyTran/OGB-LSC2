FROM mambaorg/micromamba:0.24.0
COPY env.yml /tmp/env.yml
# Install Python package dependencies with mamba
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes
RUN CUDA="cu116" && \
    micromamba install -y pip -n base -c defaults && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html &&\
    pip install torch-geometric

FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
RUN apt-get update && apt-get upgrade -y && apt-get install -y wget \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
SHELL ["/bin/bash", "-c"]
COPY --from=0 /opt/conda/ /opt/conda/
COPY --from=0 /root/ /root/
WORKDIR /root/
ENV PATH /opt/conda/bin:$PATH

# Copy source code & data
COPY src/ src/
COPY notebooks/ notebooks/
COPY data/ data/

# Script which launches RUN commands in Dockerfile
WORKDIR /root
CMD ["/bin/bash"]
