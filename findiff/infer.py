import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
from pathlib import Path
import random
import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset import FinancialDataset
from MLPSynthesizer import MLPSynthesizer
from BaseDiffuser import BaseDiffuser

def infer(args):
    cat_emb_dim = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    diffusion_steps = 500
    mlp_layers = [1024, 1024, 1024, 1024] # set number of neurons per layer
    activation = 'lrelu' # set non-linear activation function
    diffusion_beta_start = 1e-4
    diffusion_beta_end = 0.02

    scheduler = 'linear'  #diffusion scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datapreparing = FinancialDataset(path=args.data_path, cat_emb_dim=cat_emb_dim, label_cols=args.condition_cols)
    traindata, valdata = datapreparing.prepare_data()
    
    #1/10크기로 샘플링 해서 사용 (원래 60만개)
    g = torch.Generator().manual_seed(args.seed)

    n_total = len(valdata)
    n_small = n_total // 10
    indices = torch.randperm(n_total, generator=g)[:n_small].tolist()

    pd.Series(indices).to_csv(os.path.join(args.output_dir, "indices.csv"), index=False)
    val_subset_small = torch.utils.data.Subset(valdata.dataset, indices)
    
    test_dataloader = DataLoader(
    dataset=val_subset_small, 
    batch_size=1, 
    num_workers=0, 
    shuffle=False 
    )
    
    synthesizer_model = MLPSynthesizer(
    d_in=datapreparing.encoded_dim,
    hidden_layers=mlp_layers,
    activation=activation,
    n_cat_tokens=datapreparing.n_cat_tokens,
    n_cat_emb=cat_emb_dim,
    embedding_learned=False,
    n_classes=datapreparing.n_classes
    ).to(device)
    
    diffuser_model = BaseDiffuser(
    total_steps=diffusion_steps,
    beta_start=diffusion_beta_start,
    beta_end=diffusion_beta_end,
    scheduler=scheduler,
    device=device
    )
    
    synthesizer_model.load_state_dict(torch.load(args.weight_path)) 
    
    eval_progress_bar = tqdm(range(len(test_dataloader)), position=0, leave=True)
    
    pred = []
    labels = []
    
    for i, (batch_cat, batch_num, batch_y) in enumerate(test_dataloader):
        batch_cat = batch_cat.to(device)
        batch_num = batch_num.to(device)
        batch_y = batch_y.to(device)
    
        samples = torch.randn((len(batch_y), datapreparing.encoded_dim), device=device)

        with torch.no_grad():
            for t in range(diffusion_steps):
                timesteps = torch.full((len(batch_y),), t, dtype=torch.long, device=device)

                # run synthesizer model forward pass
                model_out = synthesizer_model(x=samples.float(), timesteps=timesteps, label=batch_y.to(device))

                # run diffuser model forward pass
                samples = diffuser_model.p_sample_gauss(model_out, samples, timesteps)
        
        # split sample into numeric and categorical parts
        samples = samples.detach().cpu().numpy()
        samples_num = samples[:, datapreparing.cat_dim:]
        samples_cat = samples[:, :datapreparing.cat_dim]

        # denormalize numeric attributes
        z_norm_df = pd.DataFrame(samples_num, columns=datapreparing.num_attrs)
        
        for attr_idx, attr_name in enumerate(datapreparing.num_attrs):
            z_norm_df[attr_name] = np.maximum(datapreparing.num_scalers[attr_name].inverse_transform(z_norm_df[[attr_name]]),0)
        
        # get embedding lookup matrix
        embedding_lookup = synthesizer_model.get_embeddings().cpu()

        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        samples_cat = samples_cat.reshape(-1, len(datapreparing.cat_attrs), cat_emb_dim)

        # compute pairwise distances
        distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(samples_cat))

        # get the closest distance based on the embeddings that belong to a column category
        z_cat_df = pd.DataFrame(index=range(len(samples_cat)), columns=datapreparing.cat_attrs)

        nearest_dist_df = pd.DataFrame(index=range(len(samples_cat)), columns=datapreparing.cat_attrs)

        # iterate over categorical attributes
        for attr_idx, attr_name in enumerate(datapreparing.cat_attrs):

            attr_emb_idx = list(datapreparing.vocab_per_attr[attr_name])
            attr_distances = distances[:, attr_emb_idx, attr_idx]

            nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
            nearest_idx = nearest_idx.cpu().numpy()

            z_cat_df[attr_name] = np.array(attr_emb_idx)[nearest_idx]  # need to map emb indices back to column indices
            nearest_dist_df[attr_name] = nearest_values.cpu().numpy()
            
        z_cat_df = z_cat_df.apply(datapreparing.label_encoder.inverse_transform)
        #label = datapreparing.condition_encoders.inverse_transform(batch_y.cpu().numpy())

        samples_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)
        pred.append(samples_decoded)
        #labels.append(label)
        
        eval_progress_bar.update(1)
        
    pred_df = pd.concat(pred)
    pred_df.to_csv(os.path.join(args.output_dir, "pred.csv"), index=False)
    # label_df = pd.DataFrame(labels, columns=['label'])
    # label_df.to_csv(os.path.join(output_dir, "label.csv"), index=False)

if __name__ == "__main__":
    output_dir = str(Path(__file__).parents[0]/'pred_result'/ 'result1')  
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int,   default=1234)  #random seed 
    parser.add_argument('--data_path', type=str, default='/workspace/data/201807_회원정보.csv')
    parser.add_argument('--condition_cols', type=list, default=['LifeStage'])  
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--weight_path', type=str, default='/workspace/Yeazzing/FinDiff/train_log/20251105_022210/best/synthesizer_model.pt')
    args = parser.parse_args()
    infer(args)
    print(1)