# OGB-LSC2
Docker set up for the [OGB Large Scale Challenge](https://ogb.stanford.edu/neurips2022/) with graph neural networks.

The [jonnytran/ogb-lsc2](https://hub.docker.com/r/jonnytran/ogb-lsc2) docker image stores the OGB datasets at the container's `~/data/` and codes at `~/src/`. When running, the container will write persistent data to the host's `<repo>/data/`, which *should* automatically be saved into the docker image when pushing it to the Docker Hub.

# Requirements
Must have `docker` or `nvidia-docker` if using GPUS.

# Usage
```sh
### Pull
docker pull jonnytran/ogb-lsc2:latest

### Interactive shell
nvidia-docker run --gpus all \
                  -p 8888:8888 \
                  -v $(pwd)/data/:/home/jovyan/data/ \
                  -v $(pwd)/notebooks/:/home/jovyan/notebooks/ \
                  --rm -it jonnytran/ogb-lsc2 /bin/bash

### If running an interactive Jupyter Notebook inside the container
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --autoreload --log-level='ERROR' --allow-root
```
