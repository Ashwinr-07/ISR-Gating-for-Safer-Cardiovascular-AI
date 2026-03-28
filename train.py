"""
Entry point: train Clinical-Longformer on MIMIC-IV Cardiac Disease dataset.

Usage:
    python train.py
    python train.py --data_path /path/to/data --output_dir ./outputs
"""

import argparse
import json
import os

from config import DATA_PATH, OUTPUT_DIR
from src.data.processor import MIMICCardiacDataProcessor
from src.models.train import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = MIMICCardiacDataProcessor(args.data_path)
    processor.load_data()
    processor.create_labels()
    train_df, val_df, test_df = processor.prepare_dataset()

    label_map = dict(enumerate(processor.label_encoder.classes_))
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    train(train_df, val_df, test_df, label_map, args.output_dir)
    processor.save(os.path.join(args.output_dir, "processor.pkl"))
    print(f"\nTraining complete. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
