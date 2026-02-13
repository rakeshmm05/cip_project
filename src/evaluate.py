import torch


def _binary_clf_curve(y_true, y_score):
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order].float()
    y_score = y_score[order]
    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1.0 - y_true, dim=0)
    return fps, tps, y_score


def _average_precision(y_true, y_score):
    pos_total = y_true.sum().item()
    if pos_total == 0:
        return 0.0
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    precision = tps / (tps + fps).clamp(min=1.0)
    recall = tps / pos_total
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    return torch.sum((recall[1:] - recall[:-1]) * precision[1:]).item()


def _roc_auc(y_true, y_score):
    pos_total = y_true.sum().item()
    neg_total = (1 - y_true).sum().item()
    if pos_total == 0 or neg_total == 0:
        return 0.0
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fpr = fps / neg_total
    tpr = tps / pos_total
    fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
    tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    return torch.trapz(tpr, fpr).item()


@torch.no_grad()
def evaluate_model(model, data, threshold=0.5, tune_threshold=False, mask=None):
    model.eval()
    temporal = getattr(data, "temporal", None)
    logits = model(data.x, data.edge_index, data.edge_attr, temporal)
    y = data.y if mask is None else data.y[mask]
    logits = logits if mask is None else logits[mask]
    probs = torch.sigmoid(logits)

    def metrics_for_threshold(t):
        preds = (probs >= t).long()
        acc = (preds == y).float().mean().item()
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return acc, precision, recall, f1

    if tune_threshold:
        best_t = threshold
        best_f1 = -1.0
        for t in torch.linspace(0.05, 0.95, steps=19):
            _, _, _, f1_t = metrics_for_threshold(float(t))
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = float(t)
        threshold = best_t

    acc, precision, recall, f1 = metrics_for_threshold(threshold)
    pr_auc = _average_precision(y.float(), probs)
    roc_auc = _roc_auc(y.float(), probs)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Threshold: {threshold:.2f}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold,
    }
