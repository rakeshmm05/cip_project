from src.graph_builder import build_graph_from_csv


def load_oran_dataset(csv_path):
    return build_graph_from_csv(csv_path)
