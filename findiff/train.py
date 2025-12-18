import pandas as pd
import numpy as np
import math

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import argparse

from tqdm import tqdm
from pathlib import Path

import wandb
from datetime import datetime
from pytz import timezone

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset import FinancialDataset
from MLPSynthesizer import MLPSynthesizer
from BaseDiffuser import BaseDiffuser

def train(args):
    args.cat_emb_dim = 2  # set dimension of categorical embeddings
    mlp_layers = [1024, 1024, 1024, 1024] # set number of neurons per layer
    activation = 'lrelu' # set non-linear activation function
    diffusion_steps = 500

    # set diffusion start and end betas
    diffusion_beta_start = 1e-4
    diffusion_beta_end = 0.02

    scheduler = 'linear'  #diffusion scheduler

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    #데이터 준비
    datapreparing = FinancialDataset(path=args.data_path, cat_emb_dim=args.cat_emb_dim, label_cols=args.condition_cols) 
    traindata, valdata = datapreparing.prepare_data()
    
    train_dataloader = DataLoader(
    dataset=traindata, 
    batch_size=args.batch_size, 
    num_workers=0, 
    shuffle=True 
    )
    
    val_dataloader = DataLoader(
    dataset=valdata, 
    batch_size=args.batch_size, 
    num_workers=0, 
    shuffle=False 
    )
    
    #모델 정의
    synthesizer_model = MLPSynthesizer(
    d_in=datapreparing.encoded_dim,
    hidden_layers=mlp_layers,
    activation=activation,
    n_cat_tokens=datapreparing.n_cat_tokens,
    n_cat_emb=args.cat_emb_dim,
    n_classes=datapreparing.n_classes,
    embedding_learned=True
    ).to(device)
    
    diffuser_model = BaseDiffuser(
    total_steps=diffusion_steps,
    beta_start=diffusion_beta_start,
    beta_end=diffusion_beta_end,
    scheduler=scheduler,
    device=device
    )
    
    # determine synthesizer model parameters
    parameters = filter(lambda p: p.requires_grad, synthesizer_model.parameters())

    # init Adam optimizer
    optimizer = optim.Adam(parameters, lr=args.lr)

    # init learning rate scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=False)

    # int mean-squared-error loss
    loss_fnc = nn.MSELoss()
    
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None:
        checkpointing_steps = int(checkpointing_steps)
    
    tm = datetime.now(timezone('Asia/Seoul'))
    tm_log = tm.strftime('%Y%m%d_%I%M%S')
    
    args.output_dir = str(Path(__file__).parents[0]/'train_log'/ str(tm_log))
    print(f"logs and train related things will be saved under {args.output_dir}")
    
    wandb.init(project='findiff',config=args, config_exclude_keys=['wandb_project'],sync_tensorboard=True,reinit=True)
    workstation_name=os.environ['WORKS_ID'] if 'WORKS_ID' in os.environ else 'aimlk'
    wandb.run.name = tm_log+'-'+workstation_name
    

    train_epoch_losses = []
    completed_steps = 0
    synthesizer_model.train()
    
    pbar = tqdm(iterable=range(args.epochs), position=0, leave=True)
    
    for epoch in pbar:

        batch_losses = []

        train_progress_bar = tqdm(range(len(train_dataloader)), position=1, leave=False)
        for batch_cat, batch_num, batch_y in train_dataloader:

            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)
            
            # sample diffusion timestep
            timesteps = diffuser_model.sample_timesteps(n=batch_cat.shape[0])

            # determine categorical embeddings
            batch_cat_emb = synthesizer_model.embed_categorical(x_cat=batch_cat)

            # concatenate categorical and numerical embeddings
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

            # add diffuser gaussian noise
            batch_noise_t, noise_t = diffuser_model.add_gauss_noise(x_num=batch_cat_num, t=timesteps)
            #  a data tensor with injected noise, noise itself

            # conduct synthesizer model forward pass
            predicted_noise = synthesizer_model(x=batch_noise_t, timesteps=timesteps, label=batch_y)

            # compute training batch loss
            batch_loss = loss_fnc(input=noise_t, target=predicted_noise)

            # reset model gradients
            optimizer.zero_grad()

            # run model backward pass
            batch_loss.backward()

            # optimize model parameters
            optimizer.step()

            # collect training batch losses
            batch_losses.append(batch_loss.detach().cpu().numpy())
            train_progress_bar.update(1)

        # determine mean training epoch loss
        batch_losses_mean = np.mean(np.array(batch_losses))

        # update learning rate scheduler
        lr_scheduler.step()

        # collect mean training epoch loss
        train_epoch_losses.append(batch_losses_mean)

        # prepare and set training epoch progress bar update
        pbar.set_description('epoch: {}, train-loss: {}'.format( str(epoch).zfill(4), str(batch_losses_mean)))
        
        completed_steps += 1
        
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    os.makedirs(output_dir,  exist_ok=True)
                    torch.save(synthesizer_model.state_dict(), output_dir +"/synthesizer_model.pt")
    
        #eval
        eval_epoch_losses = []
        synthesizer_model.eval()

        eval_batch_losses = []
        best_loss = np.inf
        
        eval_progress_bar = tqdm(range(len(val_dataloader)), position=1, leave=False)
        for batch_cat, batch_num, batch_y in val_dataloader:
            with torch.no_grad():

                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)
                
                # sample diffusion timestep
                timesteps = diffuser_model.sample_timesteps(n=batch_cat.shape[0])

                # determine categorical embeddings
                batch_cat_emb = synthesizer_model.embed_categorical(x_cat=batch_cat)

                # concatenate categorical and numerical embeddings
                batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

                # add diffuser gaussian noise
                batch_noise_t, noise_t = diffuser_model.add_gauss_noise(x_num=batch_cat_num, t=timesteps)

                # conduct synthesizer model forward pass
                predicted_noise = synthesizer_model(x=batch_noise_t, timesteps=timesteps, label=batch_y)

                # compute training batch loss
                eval_batch_loss = loss_fnc(input=noise_t, target=predicted_noise)

                # collect training batch losses
                eval_batch_losses.append(eval_batch_loss.detach().cpu().numpy())

            # determine mean training epoch loss
            eval_batch_losses_mean = np.mean(np.array(eval_batch_losses))

            # collect mean training epoch loss
            eval_epoch_losses.append(eval_batch_losses_mean)
            
            eval_progress_bar.update(1)
            
        #save
        result = {}
        result["epoch"] = epoch+1,
        result["step"] = completed_steps
        result["train_loss"] = batch_losses_mean
        result["val_loss"] = eval_batch_losses_mean

        wandb.log(result)

        if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                best_output_dir = "best"
                best_output_dir = os.path.join(args.output_dir, best_output_dir)
                os.makedirs(best_output_dir,  exist_ok=True)
                torch.save(synthesizer_model.state_dict(), best_output_dir +"/synthesizer_model.pt")

    
    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int,   default=1234)  #random seed
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--epochs',  type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpointing_steps', type=int, default=50)  
    parser.add_argument('--data_path', type=str, default='/workspace/data/201807_회원정보.csv')
    parser.add_argument('--condition_cols', type=list, default=['LifeStage'])  
    args = parser.parse_args()
    train(args)
    wandb.finish() 