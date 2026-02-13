import math

import torch


def temporal_encoding(timestamps, d_model=8):
    """
    Sinusoidal temporal encoding similar to Transformer positional encoding.
    timestamps: normalized scalar time values
    """
    pe = torch.zeros((len(timestamps), d_model), dtype=torch.float)

    for pos, t in enumerate(timestamps):
        for i in range(0, d_model, 2):
            div_term = math.exp(-math.log(10000.0) * i / d_model)
            pe[pos, i] = math.sin(float(t) * div_term)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(float(t) * div_term)

    return pe
