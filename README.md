# OGB-LSC2
Docker set up for the [OGB Large Scale Challenge](https://ogb.stanford.edu/neurips2022/) with graph neural networks.

# Requirements
Must have `docker` or `nvidia-docker` if using GPUS.

# Usage
```sh
### Pull
docker pull jonnytran/ogb-lsc2:latest

### Interactive shell
nvidia-docker run --gpus all \
                  -v $(pwd)/data/:/root/data/ \
                  --rm -it jonnytran/ogb-lsc2:latest /bin/bash

### or run an interactive Jupyter Notebook
nvidia-docker run -p 8888:8888 \
                  -v $(pwd)/data/:/root/data/ \
                  -v $(pwd)/notebooks/:/root/notebooks/ \
                  -e JUPYTER_ENABLE_LAB=yes \
                  -e JUPYTER_TOKEN=docker \
                  --name jupyter \
                  -d jonnytran/ogb-lsc2:latest jupyter lab --no-browser --autoreload --log-level='ERROR'
```
