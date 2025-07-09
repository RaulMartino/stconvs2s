import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os
from sklearn.preprocessing import MinMaxScaler

from model.stconvs2s import STConvS2S_R, STConvS2S_C
from model.baselines import *
from model.ablation import *
 
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import NetCDFDataset
from tool.loss import RMSELoss, MAELoss
from tool.utils import Util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device
        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'
        self.step = str(config.step)
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = self.dataset_name + '_step' + self.step
                
    def run_model(self, number):
        self.__define_seed(number)
        validation_split = 0.2
        test_split = 0.2
        # Loading the dataset
        ds = xr.open_mfdataset(self.dataset_file).load()
        if (self.config.small_dataset):
            ds = self._create_stratified_small_dataset(ds, target_samples=1000)

        train_dataset = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split)
        val_dataset   = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_validation=True)
        test_dataset  = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_test=True)

        # === Log1p transform for precipitation (Y) ===
        for dataset_name, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
            y_np = dataset.y.detach().cpu().numpy()
            y_log = np.log1p(y_np)
            dataset.y = torch.tensor(y_log, dtype=dataset.y.dtype, device=dataset.y.device)
            if self.config.verbose:
                print(f"Applied log1p to Y in {dataset_name} set. Before: min={y_np.min()}, max={y_np.max()} | After: min={y_log.min()}, max={y_log.max()}")

        # === MinMaxScaler feature scaling for all channels in X ===
        use_min_max = True
        if use_min_max:
            for channel_idx in range(train_dataset.X.shape[1]):
                # Fit scaler on training data for this channel
                train_data = train_dataset.X[:, channel_idx].detach().cpu().numpy()
                original_shape = train_data.shape
                reshaped = train_data.reshape(-1, 1)
                scaler = MinMaxScaler().fit(reshaped)
                scaled_train = scaler.transform(reshaped).reshape(original_shape)
                train_dataset.X[:, channel_idx] = torch.tensor(
                    scaled_train, dtype=train_dataset.X.dtype, device=train_dataset.X.device
                )
                # Apply scaler to val and test
                for split_name, dataset in zip(["val", "test"], [val_dataset, test_dataset]):
                    data = dataset.X[:, channel_idx].detach().cpu().numpy()
                    reshaped = data.reshape(-1, 1)
                    scaled = scaler.transform(reshaped).reshape(data.shape)
                    dataset.X[:, channel_idx] = torch.tensor(
                        scaled, dtype=dataset.X.dtype, device=dataset.X.device
                    )
                    if self.config.verbose:
                        print(f"Scaled {split_name} data for channel index {channel_idx} using training scaler")

        if (self.config.verbose):
            print('[X_train] Shape:', train_dataset.X.shape)
            print('[y_train] Shape:', train_dataset.y.shape)
            print('[X_val] Shape:', val_dataset.X.shape)
            print('[y_val] Shape:', val_dataset.y.shape)
            print('[X_test] Shape:', test_dataset.X.shape)
            print('[y_test] Shape:', test_dataset.y.shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')
                        
        params = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)
        
        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
            'convlstm': STConvLSTM,
            'predrnn': PredRNN,
            'mim': MIM,
            'conv2plus1d': Conv2Plus1D,
            'conv3d': Conv3D,
            'enc-dec3d': Endocer_Decoder3D,
            'ablation-stconvs2s-nocausalconstraint': AblationSTConvS2S_NoCausalConstraint,
            'ablation-stconvs2s-notemporal': AblationSTConvS2S_NoTemporal,
            'ablation-stconvs2s-r-nochannelincrease': AblationSTConvS2S_R_NoChannelIncrease,
            'ablation-stconvs2s-c-nochannelincrease': AblationSTConvS2S_C_NoChannelIncrease,
            'ablation-stconvs2s-r-inverted': AblationSTConvS2S_R_Inverted,
            'ablation-stconvs2s-c-inverted': AblationSTConvS2S_C_Inverted,
            'ablation-stconvs2s-r-notfactorized': AblationSTConvS2S_R_NotFactorized,
            'ablation-stconvs2s-c-notfactorized': AblationSTConvS2S_C_NotFactorized
        }
        if not(self.config.model in models):
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')
            quit()
            
        # Creating the model    
        model_bulder = models[self.config.model]
        model = model_bulder(train_dataset.X.shape, self.config.num_layers, self.config.hidden_dim, 
                             self.config.kernel_size, self.device, self.dropout_rate, int(self.step))
        model.to(self.device)
        criterion = MAELoss()
        opt_params = {'lr': 0.001, 
                      'alpha': 0.9, 
                      'eps': 1e-6}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        util = Util(self.config.model, self.dataset_type, self.config.version, self.filename_prefix)
        
        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(model, criterion, optimizer, train_loader,  val_loader, util) 
                                                 
        eval_info = self.__load_and_evaluate(model, criterion, optimizer, test_loader, 
                                             train_info['train_time'], util)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}


    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.config.epoch, 
                          self.device, util, self.config.verbose, self.config.patience, self.config.no_stop)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, criterion, optimizer, test_loader, train_time, util):  
        evaluator = Evaluator(model, criterion, optimizer, test_loader, self.device, util, self.step)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        if (self.config.chirps):
            dataset_file = 'data/output_07_07.nc'
            dataset_name = 'chirps'
        else:
            dataset_file = 'data/output_07_07.nc'
            dataset_name = 'cfsr'
        
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return dropout_rate
    
    def _create_stratified_small_dataset(self, ds, target_samples=500):
        """
        Create a stratified small dataset that maintains precipitation level proportions.
        
        Args:
            ds: Original xarray dataset
            target_samples: Number of samples to include in small dataset
            
        Returns:
            Stratified xarray dataset subset
        """
        import numpy as np
        
        # Get precipitation data (Y) - assume last channel is precipitation
        y_data = ds.y.values[:, :, :, :, -1]  # Last channel
        
        # Calculate mean precipitation per sample
        sample_means = np.mean(y_data, axis=(1, 2, 3))
        
        # Define precipitation bins (in log1p scale since data will be transformed)
        bins = [0, np.log1p(5), np.log1p(25), np.log1p(50), np.inf]
        bin_labels = ['0-5mm', '5-25mm', '25-50mm', '50+mm']
        
        # Assign each sample to a precipitation level
        sample_levels = np.digitize(sample_means, bins) - 1
        sample_levels = np.clip(sample_levels, 0, len(bin_labels) - 1)
        
        # Calculate original proportions
        original_counts = np.bincount(sample_levels, minlength=len(bin_labels))
        original_props = original_counts / len(sample_levels)
        
        # Calculate target counts for small dataset
        target_counts = (original_props * target_samples).astype(int)
        
        # Ensure at least 1 sample from each extreme class (if available)
        min_extreme_samples = 1
        for level in [2, 3]:  # 25-50mm and 50+mm classes
            if original_counts[level] > 0 and target_counts[level] == 0:
                target_counts[level] = min_extreme_samples
                print(f"  Forced at least {min_extreme_samples} sample(s) for class {bin_labels[level]}")
        
        # Adjust to ensure we get exactly target_samples
        diff = target_samples - target_counts.sum()
        if diff > 0:
            # Add to most frequent class
            target_counts[np.argmax(original_counts)] += diff
        elif diff < 0:
            # Remove from most frequent class (but not below forced minimums)
            most_frequent = np.argmax(original_counts)
            reduction = min(-diff, target_counts[most_frequent] - 1)  # Keep at least 1
            target_counts[most_frequent] -= reduction
        
        # Sample from each level
        selected_indices = []
        for level in range(len(bin_labels)):
            level_indices = np.where(sample_levels == level)[0]
            n_to_sample = min(target_counts[level], len(level_indices))
            
            if n_to_sample > 0:
                sampled_indices = np.random.choice(level_indices, n_to_sample, replace=False)
                selected_indices.extend(sampled_indices)
        
        selected_indices = np.array(selected_indices)
        
        # Print stratification results
        print(f"\n=== Stratified Small Dataset ({target_samples} samples) ===")
        print("Original dataset proportions:")
        for i, label in enumerate(bin_labels):
            print(f"  {label}: {original_counts[i]:,} ({original_props[i]:.1%})")
        
        print("\nSmall dataset proportions:")
        small_counts = np.bincount([sample_levels[i] for i in selected_indices], minlength=len(bin_labels))
        small_props = small_counts / len(selected_indices)
        for i, label in enumerate(bin_labels):
            print(f"  {label}: {small_counts[i]:,} ({small_props[i]:.1%})")
        
        # Create subset dataset
        stratified_ds = ds.isel(sample=selected_indices)
        
        return stratified_ds