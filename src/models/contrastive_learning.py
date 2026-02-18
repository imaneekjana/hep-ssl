import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

import torch
from torch.utils.data import IterableDataset

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

import h5py
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from livelossplot import PlotLosses


"""

Contrastive Learning training protocol

"""


class Contrastive_Learning(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def forward(self, view):

        latent_vec = self.model(view)

        return latent_vec
        
    def test(self, loader,desc=''):
        
        modelh = self.model
        
        modelh.eval()
        
        loss_ = 0
        
        with torch.no_grad():
        
            for view1, view2 in loader:
                
                view1 = view1.to(self.args.device)
                view2 = view2.to(self.args.device)

                feat1 = self.forward(view1)
                feat2 = self.forward(view2)

                features = torch.cat((feat1, feat2), dim=0)

                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)
                loss_ += loss.item()

                
    
        
        return loss_/len(loader)
    
    
    

    def train(self, train_loader, val_loader, save_model = False, folder = '', wandb_ = False, **key_name):

        #Scaler = GradScaler(enabled=self.args.fp16_precision)

        Scaler = GradScaler()

        liveloss = PlotLosses()

        if wandb_ == True:

            import wandb

            wandb_key = key_name.get("key")
            
            if wandb_key is None:
               raise ValueError("wandb_=True but no wandb_key provided")
        
            wandb.login(key=wandb_key)
        
            wandb.init(project=key_name.get("name"), config=vars(self.args))

        
        n_iter = 0
    

        for epoch_counter in tqdm(range(self.args.epochs),desc='epoch'):
            loss_train = 0
            
            for view1, view2 in train_loader:
                
                view1 = view1.to(self.args.device)
                view2 = view2.to(self.args.device)

                feat1 = self.forward(view1)
                feat2 = self.forward(view2)

                features = torch.cat((feat1, feat2), dim=0)

                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)
                loss_train += loss.item()

                self.optimizer.zero_grad()

                Scaler.scale(loss).backward()

                Scaler.step(self.optimizer)
                Scaler.update()
                
                n_iter += 1
            
            #loss_train = self.test(train_loader).item()
            loss_train = loss_train/len(train_loader)
            loss_val = self.test(val_loader,desc='validation')
            
            # LiveLossPlot logging
            liveloss.update({
                'loss_train': loss_train,
                'loss_val': loss_val
             })
            liveloss.send()

            if wandb_ == True:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val})
            

            if save_model ==True:
                
                if epoch_counter%10==0:
                    save_path = folder + f"model_epoch_{epoch_counter+1}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                    #wandb.save(save_path)

                    
        if save_model ==True:
            save_path = folder+f"model_final.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
               

        

        if wandb_ == True:
            wandb.save(save_path)
            wandb.finish()
