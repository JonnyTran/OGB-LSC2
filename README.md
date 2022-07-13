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
                  -p 8888:8888 \
                  -v $(pwd)/data/:/root/data/ \
                  -v $(pwd)/notebooks/:/root/notebooks/ \
                  --rm -it jonnytran/ogb-lsc2 /bin/bash

### If running an interactive Jupyter Notebook inside the container
jupyter lab --ip 0.0.0.0 --no-browser --autoreload --NotebookApp.iopub_data_rate_limit=2147483647 --port 8888 --log-level='ERROR' --allow-root
```
