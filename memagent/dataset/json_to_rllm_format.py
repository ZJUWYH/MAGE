"""
Convert the preprocessed JSON dataset to rllm DatasetRegistry format (parquet).

This script only depends on rllm (not agentdojo). Run it in the rllm venv
after agentdojo.py has produced the JSON file.

Pipeline:
    1. (agentdojo venv)  python -m dataset.agentdojo          -> dataset/agentdojo/all_samples.json
    2. (rllm venv)       python -m dataset.json_to_rllm_format -> registered parquet train/test splits

Each data item has the format:
    {"segments": List, "labels": List, "file_path": str}
"""

import argparse
import json
import random

from rllm.data.dataset import DatasetRegistry


def convert_json_to_rllm(
    json_path: str = "dataset/agentdojo/all_samples.json",
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_name: str = "agentdojo",
):
    """
    Load preprocessed samples from JSON, split into train/test, and register
    with rllm DatasetRegistry (saves parquet + verl parquet).

    Args:
        json_path: Path to the JSON file produced by agentdojo.py.
        train_ratio: Fraction of data used for training (rest is test). Default 0.9.
        seed: Random seed for reproducibility.
        dataset_name: Name for the registered dataset (default: agentdojo).
    """
    # ------------------------------------------------------------------
    # 1. Load JSON
    # ------------------------------------------------------------------
    with open(json_path, "r") as f:
        dataset = json.load(f)

    print(f"[json_to_rllm] Loaded {len(dataset)} samples from {json_path}")

    total_segments = sum(len(d["segments"]) for d in dataset)
    total_harmful = sum(sum(d["labels"]) for d in dataset)
    total_harmless = total_segments - total_harmful
    print(f"[json_to_rllm] Total segments: {total_segments} "
          f"(harmful: {total_harmful}, harmless: {total_harmless})")

    if len(dataset) == 0:
        print("[json_to_rllm] ERROR: No samples found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Shuffle and split into train / test (90% / 10%)
    # ------------------------------------------------------------------
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split_idx = int(len(dataset) * train_ratio)
    train_data = [dataset[i] for i in indices[:split_idx]]
    test_data = [dataset[i] for i in indices[split_idx:]]

    print(f"[json_to_rllm] Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Label distribution per split
    def _label_stats(data_list):
        segs = sum(len(d["segments"]) for d in data_list)
        harmful = sum(sum(d["labels"]) for d in data_list)
        return segs, harmful, segs - harmful

    tr_segs, tr_h, tr_s = _label_stats(train_data)
    te_segs, te_h, te_s = _label_stats(test_data)
    print(f"[json_to_rllm] Train segments: {tr_segs} (harmful: {tr_h}, harmless: {tr_s})")
    print(f"[json_to_rllm] Test  segments: {te_segs} (harmful: {te_h}, harmless: {te_s})")

    # ------------------------------------------------------------------
    # 3. Serialize complex fields to JSON strings for parquet compatibility
    #    (pyarrow cannot handle nested lists of dicts with varying schemas)
    #    The env deserializes them back when loading.
    # ------------------------------------------------------------------
    def _serialize_for_parquet(data_list):
        serialized = []
        for d in data_list:
            serialized.append({
                "segments": json.dumps(d["segments"]),
                "labels": json.dumps(d["labels"]),
                "file_path": d["file_path"],
            })
        return serialized

    train_serialized = _serialize_for_parquet(train_data)
    test_serialized = _serialize_for_parquet(test_data)

    # ------------------------------------------------------------------
    # 4. Register with rllm DatasetRegistry (saves parquet + verl parquet)
    # ------------------------------------------------------------------
    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_serialized, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_serialized, "test")

    print(f"[json_to_rllm] Registered '{dataset_name}' train split: {train_dataset.get_data_path()}")
    print(f"[json_to_rllm] Registered '{dataset_name}' test  split: {test_dataset.get_data_path()}")
    print(f"[json_to_rllm] Verl train path: {train_dataset.get_verl_data_path()}")
    print(f"[json_to_rllm] Verl test  path: {test_dataset.get_verl_data_path()}")

    # ------------------------------------------------------------------
    # 5. Print a sample
    # ------------------------------------------------------------------
    sample = train_data[0]
    print(f"\n[json_to_rllm] Sample train example:")
    print(f"  file_path: {sample['file_path']}")
    print(f"  num_segments: {len(sample['segments'])}")
    print(f"  labels: {sample['labels']}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    # python -m dataset.json_to_rllm_format --json_path dataset/stac/all_samples.json --dataset_name stac
    # python -m dataset.json_to_rllm_format --json_path dataset/agentdojo/all_samples.json --dataset_name agentdojo
    # python -m dataset.json_to_rllm_format --json_path dataset/harmbench/all_samples.json --dataset_name harmbench
    # python -m dataset.json_to_rllm_format --json_path dataset/agentdojov2/all_samples.json --dataset_name agentdojov2
    # python -m dataset.json_to_rllm_format --json_path dataset/agentdojov3/all_samples.json --dataset_name agentdojov3
    # python -m dataset.json_to_rllm_format --json_path dataset/stac/all_samples_v2v3.json --dataset_name stac_v2v3
    parser = argparse.ArgumentParser(
        description="Convert agentdojo JSON to rllm DatasetRegistry parquet format"
    )
    parser.add_argument(
        "--json_path", type=str, default="dataset/agentdojo/all_samples.json",
        help="Path to the JSON file produced by agentdojo.py (default: dataset/agentdojo/all_samples.json)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9,
        help="Fraction of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="agentdojo",
        help="Name for the registered dataset (default: agentdojo)",
    )
    args = parser.parse_args()

    convert_json_to_rllm(
        json_path=args.json_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )
