import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stratified_edge_masks(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create deterministic stratified masks for edge-level labels.
    """
    n = y.numel()
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    gen = torch.Generator()
    gen.manual_seed(seed)

    for cls in [0, 1]:
        idx = torch.where(y == cls)[0]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=gen)]

        n_train = int(round(train_ratio * perm.numel()))
        n_val = int(round(val_ratio * perm.numel()))
        n_train = min(n_train, perm.numel())
        n_val = min(n_val, perm.numel() - n_train)
        n_test_start = n_train + n_val

        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_test_start]] = True
        test_mask[perm[n_test_start:]] = True

    return train_mask, val_mask, test_mask
