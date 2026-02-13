from src.graph_builder import build_window_graph_from_csv
from src.evaluate import evaluate_model
from src.model_gat_edge import EdgeAwareGAT
from src.train import train_model
from src.utils import make_stratified_edge_masks, set_seed


def main():
    set_seed(42)
    data = build_window_graph_from_csv(
        "dataset/ieee_oran_context_violation_dataset.csv",
        window_seconds=2,
        malicious_min_degradations=3,
    )
    train_mask, val_mask, test_mask = make_stratified_edge_masks(data.y, seed=42)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = EdgeAwareGAT(
        node_dim=data.num_node_features,
        edge_dim=data.edge_attr.shape[1],
        hidden_dim=32,
        temporal_dim=data.temporal.shape[1],
    )

    model = train_model(model, data, epochs=120, lr=0.003)
    val_metrics = evaluate_model(model, data, tune_threshold=True, mask=data.val_mask)
    evaluate_model(
        model,
        data,
        threshold=val_metrics["threshold"],
        tune_threshold=False,
        mask=data.test_mask,
    )


if __name__ == "__main__":
    main()
