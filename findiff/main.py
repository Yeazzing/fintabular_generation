import argparse
from train import train
from infer import infer
from pathlib import Path

output_dir = str(Path(__file__).parents[0]/'pred_result'/ 'result1')  

def build_common_parser():
    """train/infer 공통으로 쓰는 옵션들"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--seed',  type=int,   default=1234)  #random seed
    p.add_argument('--condition_cols', type=list, default=['LifeStage']) 
    p.add_argument('--data_path', type=str, required=True)
    return p

def build_train_parser(common_parser):
    parser = argparse.ArgumentParser(parents=[common_parser])
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--epochs',  type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpointing_steps', type=int, default=50)  
    return parser

def build_infer_parser(common_parser):
    parser = argparse.ArgumentParser(parents=[common_parser])
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--weight_path', type=str, required=True)
    return parser

def main():
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=["train", "infer"], required=True)
    mode_args, remaining_argv = mode_parser.parse_known_args()

    common_parser = build_common_parser()

    if mode_args.mode == "train":
        parser = build_train_parser(common_parser)
        args = parser.parse_args(remaining_argv)
        train(args)
    else:  
        parser = build_infer_parser(common_parser)
        args = parser.parse_args(remaining_argv)
        infer(args)

if __name__ == "__main__":
    main()