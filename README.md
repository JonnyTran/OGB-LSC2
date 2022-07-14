# OGB-LSC2
Docker set up for the [OGB Large Scale Challenge](https://ogb.stanford.edu/neurips2022/) with graph neural networks.

The [jonnytran/ogb-lsc2](https://hub.docker.com/r/jonnytran/ogb-lsc2) uses the OGB datasets at the mounted `~/dataset/` directory and codes at `~/src/` copied from this github repo. When running, the container will write persistent data to the host's `<repo_dir>/dataset/` as well as `<repo_dir>/notebooks/` if you're using Jupyter within the .

This project contains:

- A git repo containing code that you’re working on.
- A Dockerfile that builds that code and downloads the OGB-LSC dataset into a working container.
- A Docker-compose.yml file you use to run that container

# Requirements
Must have `docker` or `nvidia-docker` if using GPUS.

# Usage
### Pull image
```sh
docker pull jonnytran/ogb-lsc2:latest
```

### Set correct permissions to mount datasets to the container
```sh
cd <repo_root>
chmod -R 777 dataset/
chmod -R 777 notebooks/
chmod -R 777 src/
```

## Run the container with an interactive shell
```sh
nvidia-docker run --gpus all \
                  -p 8888:8888 \
                  -v $(pwd)/dataset/:/home/jovyan/dataset/ \
                  -v $(pwd)/notebooks/:/home/jovyan/notebooks/ \
                  -v $(pwd)/src/:/home/jovyan/src/ \
                  --rm -it jonnytran/ogb-lsc2 /bin/bash
```

## Running an interactive Jupyter Notebook inside the container
```
env PORT=8888
docker compose up
```

### Build & Push Changes to Dockerfile
```sh
docker build -t jonnytran/ogb-lsc2:<tag> -t jonnytran/ogb-lsc2:latest .
docker push jonnytran/ogb-lsc2:latest
```
