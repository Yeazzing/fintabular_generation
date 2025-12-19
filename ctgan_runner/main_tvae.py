
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from pytz import timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_REPO = REPO_ROOT / "ctgan_legacy"    
sys.path.insert(0, str(LEGACY_REPO))
from ctgan.synthesizers.tvae import TVAE  

tm = datetime.now(timezone('Asia/Seoul'))
tm_log = tm.strftime('%Y%m%d_%I%M%S')
train_output_dir = str(Path(__file__).parents[0]/'train_log'/ str(tm_log))

output_dir = str(Path(__file__).parents[0]/'pred_result'/ 'tvae')  


def build_train_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True, help="Training CSV path")
    p.add_argument("--metadata", type=str, required=True, help="metadata.json path")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--save_dir", type=str, default=train_output_dir, help="Directory to save trained model")
    return p


def build_infer_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--weight_path", type=str, required=True, help="Path to trained model weight")
    p.add_argument("--output_dir", type=str, default=output_dir)
    p.add_argument("--num_samples", type=int, default=60000, help="Number of rows to sample")

    return p


def load_discrete_columns(metadata_path: str):
    with open(metadata_path) as f:
        metadata = json.load(f)

    cols = []
    for col, info in metadata["columns"].items():
        if info.get("sdtype") != "numerical":
            cols.append(col)
    return cols


def prepare_data(df, discrete_columns):
    df.columns = [col.replace('_', '') for col in df.columns]
    
    #결측치 처리
    df = df.replace('_', np.nan)
    df['1순위신용체크구분'] = df['1순위신용체크구분'].fillna('누락') 
    df = df.dropna(axis=1)

    for cat_attr in discrete_columns:
        df[cat_attr] = cat_attr + '_' + df[cat_attr].astype(str)
        
    return df


def main():
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=["train", "infer"], required=True)
    mode_args, remaining_argv = mode_parser.parse_known_args()

    if mode_args.mode == "train":
        args = build_train_parser().parse_args(remaining_argv)

        df = pd.read_csv(args.data_path)
        discrete_columns = load_discrete_columns(args.metadata)
        df = prepare_data(df, discrete_columns)

        model = TVAE(epochs=args.epochs)

        os.makedirs(args.save_dir, exist_ok=True)
        model.fit(df, discrete_columns, save_dir=args.save_dir)
        model.save(os.path.join(args.save_dir, "final.pt"))

        print(f"Saved model to: {args.save_dir}")

    else:  # infer
        args = build_infer_parser().parse_args(remaining_argv)
        
        if not args.weight_path:
            raise ValueError("--weight_path is required in infer mode.")

        model = TVAE.load(args.weight_path)
        sampled = model.sample(args.num_samples)

        os.makedirs(args.output_dir, exist_ok=True)
        sampled.to_csv(args.output_dir+"/pred.csv", index=False)

        print(f"Saved samples to: {args.output_dir}")


if __name__ == "__main__":
    main()
    print("finish")
