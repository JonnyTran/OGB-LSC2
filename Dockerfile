FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
RUN apt-get update && apt-get upgrade -y && apt-get install -y wget \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
SHELL ["/bin/bash", "-c"]

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Configs
ENV PATH /opt/conda/bin:$PATH
ENV CUDA="cu116"

# Install Python package dependencies with mamba
COPY env.yml /tmp/env.yml
RUN conda install mamba -c conda-forge --name base -y --quiet && \
    mamba env update --name base -f /tmp/env.yml --prune --quiet && \
    mamba clean --all --yes
RUN mamba update jupyter ipywidgets -y -q
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html --quiet && \
    pip install torch-geometric --quiet
### Install any new conda pkgs after this line

# Permissions
ARG NB_USER=jovyan
RUN useradd -ms /bin/bash ${NB_USER}
EXPOSE 8888/tcp
WORKDIR /home/${NB_USER}

# Copy source code & data from host's, with permissions
COPY --chown=${NB_USER} src/ src/
COPY --chown=${NB_USER} notebooks/ notebooks/

# Switch to NB_USER during runtime
USER 1000

# Script which launches RUN commands in Dockerfile
CMD ["/bin/bash"]
