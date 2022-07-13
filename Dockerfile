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
COPY env.yml /tmp/env.yml

# Install Python package dependencies with mamba
RUN /opt/conda/bin/conda install mamba -c conda-forge -y && \
    /opt/conda/bin/mamba env create --name base -f /tmp/env.yml && \
    /opt/conda/bin/mamba clean --all --yes
RUN CUDA="cu116" && \
    /opt/conda/bin/pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html &&\
    /opt/conda/bin/pip install torch-geometric

ENV PATH /opt/conda/bin:$PATH
WORKDIR /root/

# Copy source code & data
COPY src/ src/
COPY notebooks/ notebooks/
COPY data/ data/

# Script which launches RUN commands in Dockerfile
WORKDIR /root
CMD ["/bin/bash"]
