# OGB-LSC2
Docker set up for the [OGB Large Scale Challenge](https://ogb.stanford.edu/neurips2022/) with graph neural networks.

# Requirements
Must have `docker` or `nvidia-docker` if using GPUS.

# Usage
```sh
### Interactive shell
nvidia-docker run --gpus all --rm -it jonnytran/ogb-lsc2:0.1 /bin/bash

### or run an interactive Jupyter Notebook
nvidia-docker run -p 8888:8888 \
                  -e JUPYTER_ENABLE_LAB=yes \
                  -e JUPYTER_TOKEN=docker \
                  --name jupyter \
                  -d jonnytran/ogb-lsc2:latest
```
