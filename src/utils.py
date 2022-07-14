from collections.abc import MutableMapping
from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch
from torch import Tensor


def activation(y_pred: Tensor, loss_type: str):
    # Apply softmax/sigmoid activation if needed
    if "LOGITS" in loss_type or "FOCAL" in loss_type:
        if "SOFTMAX" in loss_type:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)

    elif "NEGATIVE_LOG_LIKELIHOOD" == loss_type or "SOFTMAX_CROSS_ENTROPY" in loss_type:
        y_pred = torch.softmax(y_pred, dim=1)

    return y_pred


def filter_samples(Y_hat: Tensor, Y: Tensor, weights: Tensor):
    if weights is None or not hasattr(weights, 'shape') or weights.shape == None or weights.numel() == 0:
        return Y_hat, Y

    if not isinstance(weights, Tensor):
        weights = torch.tensor(weights)

    idx = torch.nonzero(weights).view(-1)

    if Y.dim() > 1:
        Y = Y[idx, :]
    else:
        Y = Y[idx]

    if Y_hat.dim() > 1:
        Y_hat = Y_hat[idx, :]
    else:
        Y_hat = Y_hat[idx]

    return Y_hat, Y


def filter_samples_weights(Y_hat: Tensor, Y: Tensor, weights, return_index=False):
    if weights is None or \
            (isinstance(weights, (Tensor, np.ndarray)) and weights.shape == None):
        return Y_hat, Y, None

    if isinstance(weights, Tensor):
        idx = torch.nonzero(weights).view(-1)
    else:
        idx = torch.tensor(np.nonzero(weights)[0])

    if return_index:
        return idx

    if Y.dim() > 1:
        Y = Y[idx, :]
    else:
        Y = Y[idx]

    if Y_hat.dim() > 1:
        Y_hat = Y_hat[idx, :]
    else:
        Y_hat = Y_hat[idx]

    return Y_hat, Y, weights[idx]


def tensor_sizes(input: Any) -> Any:
    if isinstance(input, (dict, MutableMapping)):
        return {metapath: tensor_sizes(v) \
                for metapath, v in input.items()}
    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        return [tensor_sizes(v) for v in input]
    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, Tensor) and input.dim() == 0:
                return input.item()

            return list(input.shape)
        else:
            return input


def edge_index_sizes(edge_index_dict):
    output = {}
    for m, edge_index in edge_index_dict.items():
        if isinstance(edge_index, tuple):
            edge_index, values_a = edge_index
        else:
            values_a = None

        if edge_index.size(1) == 0:
            output[m] = None
        else:
            sizes = edge_index.max(1).values.data.tolist()
            if values_a is not None:
                output[m] = (sizes, values_a.shape)
            else:
                output[m] = sizes

    return output
