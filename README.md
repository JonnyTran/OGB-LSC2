# OGB-LSC2
Docker set up for the [OGB Large Scale Challenge](https://ogb.stanford.edu/neurips2022/) with graph neural networks.

The [jonnytran/ogb-lsc2](https://hub.docker.com/r/jonnytran/ogb-lsc2) docker image stores the OGB datasets at the container's `~/data/` and codes at `~/src/`. When running, the container will write persistent data to the host's `<repo_dir>/data/`, which *should* automatically be saved into the docker image when pushing it to the Docker Hub.

This project contains:

- A git repo containing code that you’re working on.
- A Dockerfile that builds that code and downloads the OGB-LSC dataset into a working container.
- A Docker-compose.yml file you use to run that container

# Requirements
Must have `docker` or `nvidia-docker` if using GPUS.

# Usage
## Pull image
```sh
docker pull jonnytran/ogb-lsc2:latest
```

## Run the container with an interactive shell
```sh
cd <repo_root>
chmod -R 777 dataset/
chmod -R 777 notebooks/
```

## Run the container with an interactive shell
```sh
nvidia-docker run --gpus all \
                  -p 8888:8888 \
                  -v $(pwd)/dataset/:/home/jovyan/dataset/ \
                  -v $(pwd)/notebooks/:/home/jovyan/notebooks/ \
                  --rm -it jonnytran/ogb-lsc2 /bin/bash

### Running an interactive Jupyter Notebook inside the container
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --autoreload --log-level='ERROR' --allow-root
```

## Build & Push Changes to Dockerfile
```sh
docker build -t jonnytran/ogb-lsc2:0.1 -t jonnytran/ogb-lsc2:latest .
docker push jonnytran/ogb-lsc2:latest
```
