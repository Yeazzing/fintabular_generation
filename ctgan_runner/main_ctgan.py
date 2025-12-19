import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from pytz import timezone
from tqdm import tqdm
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_REPO = REPO_ROOT / "ctgan_legacy"   
sys.path.insert(0, str(LEGACY_REPO))      
from ctgan.synthesizers.ctgan import CTGAN 

tm = datetime.now(timezone('Asia/Seoul'))
tm_log = tm.strftime('%Y%m%d_%I%M%S')
train_output_dir = str(Path(__file__).parents[0]/'train_log'/ str(tm_log))

output_dir = str(Path(__file__).parents[0]/'pred_result'/ 'ctgan')  

def build_train_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=str, required=True, help="metadata.json path")
    p.add_argument('--data_path', type=str, required=True, help="Training CSV path")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--generator_lr", type=float, default=2e-4)
    p.add_argument("--discriminator_lr", type=float, default=2e-4)
    p.add_argument("--generator_decay", type=float, default=1e-6)
    p.add_argument("--discriminator_decay", type=float, default=0.0)
    p.add_argument("--generator_dim", type=str, default="256,256")
    p.add_argument("--discriminator_dim", type=str, default="256,256")
    p.add_argument("--save_dir", type=str, default=train_output_dir, help="Directory to save trained model")
    return p

def build_infer_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--weight_path", type=str, required=True, help="Path to trained model weight")
    p.add_argument('--output_dir', type=str, default=output_dir)
    p.add_argument(
        '-n',
        '--num-samples', type=int, default=100,
        help="Number of rows to sample (used when --condition_csv is not given)",
    )
    p.add_argument("--condition_csv", type=str, help="CSV that has label column for counts (optional)")
    p.add_argument("--condition_col", type=str, default="LifeStage")
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

        gen_dim = [int(x) for x in args.generator_dim.split(",")]
        disc_dim = [int(x) for x in args.discriminator_dim.split(",")]

        model = CTGAN(
            epochs=args.epochs,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            generator_dim=gen_dim,
            discriminator_dim=disc_dim,
            generator_lr=args.generator_lr,
            discriminator_lr=args.discriminator_lr,
            generator_decay=args.generator_decay,
            discriminator_decay=args.discriminator_decay
        )

        os.makedirs(args.save_dir, exist_ok=True)

        model.fit(df, discrete_columns=discrete_columns)  #모델 내부 자체에서 전처리 하는데 시간 1시간 이상 소요됨
        model.save(args.save_dir+"/final.pt")
        print(f"Saved model to: {args.save_dir}")

    else:  # infer
        args = build_infer_parser().parse_args(remaining_argv)
        
        if not args.weight_path:
            raise ValueError("--weight_path is required in infer mode.")

        model = CTGAN.load(args.weight_path)

        # 1) eval 데이터의 condition 카운트 기반 샘플링 
        if args.condition_csv:
            cond_df = pd.read_csv(args.condition_csv)
            label_counts = cond_df['label'].value_counts()  #읽은 df에서 'label' 컬럼 (condition 컬럼) 기준 카운트 집계
            all_samples = []
            for label_value, n in tqdm(label_counts.items(), desc="Generating samples per label"):
                try:
                    sampled = model.sample(
                        n,
                        args.condition_col,
                        label_value,
                    )
                    sampled["condition_label"] = label_value
                    all_samples.append(sampled)
                except Exception as e:
                    print(f"Error generating for {label_value}: {e}")

            final_df = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()


        # 2) N개 샘플링
        else:
            n = args.num_samples if args.num_samples > 0 else 0
            if n <= 0:
                raise ValueError("Provide --condition_csv or set --num_samples > 0 for infer.")
            final_df = model.sample(n)
            
        os.makedirs(args.output_dir, exist_ok=True)
        final_df.to_csv(args.output_dir+"/pred.csv", index=False)
        print(f"Saved samples to: {args.output_dir}")


if __name__ == "__main__":
    main()
    print("finish")
