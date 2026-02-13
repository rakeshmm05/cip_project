import torch
import torch.nn as nn


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else torch.tensor(1.0),
        )

    def forward(self, logits, targets):
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_term = (1.0 - pt).pow(self.gamma)
        return (alpha_t * focal_term * bce).mean()


def train_model(model, data, epochs=50, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_mask = getattr(data, "train_mask", torch.ones_like(data.y, dtype=torch.bool))
    y_train = data.y[train_mask]
    class_counts = torch.bincount(y_train, minlength=2).float()
    if class_counts[1] == 0 or class_counts[0] == 0:
        pos_weight = torch.tensor(1.0, dtype=torch.float)
    else:
        pos_weight = class_counts[0] / class_counts[1]

    criterion = FocalBCEWithLogitsLoss(
        gamma=2.0,
        alpha=0.75,
        pos_weight=pos_weight,
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        temporal = getattr(data, "temporal", None)
        logits = model(data.x, data.edge_index, data.edge_attr, temporal)
        loss = criterion(logits[train_mask], data.y[train_mask].float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
