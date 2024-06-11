#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:23:25 2024

@author: peter
"""

#%%

import numpy as np
import torch
import pathlib


#%%

class SystemManager():
    

#%%

    def __init__( self,
                  model,
                  train_dataloader,
                  valid_dataloader,
                  test_dataloader,
                  criterion,
                  optimizer,
                  scheduler,
                  config ):
        
        
        self.model            = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader  = test_dataloader
        self.criterion        = criterion
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.config           = config
        self.epoch_loss       = []
        self.epoch_accuracy   = []
        
        
#%%

    def utilities_training_functions( self ):
        
        
        for path in self.config.ALL_TRAINING_PATHS:
            
            if not path.is_dir():
                
                path.mkdir( parents  = True, 
                            exist_ok = False )
                

#%%


    def save_model( self,
                    BEST_MODEL_FILENAME = None ):
        
        if BEST_MODEL_FILENAME is not None:
            
            torch.save( self.model.state_dict(), BEST_MODEL_FILENAME )
            
        else:
            
            torch.save( self.model.state_dict(), self.config.BEST_MODEL_PATH / self.config.BEST_MODEL_NAME )
            
    
#%%

    def load_trained_model( self,
                            TRAINED_MODEL_PATH = None ):
        
        if TRAINED_MODEL_PATH is None:
            
            self.model.load_state_dict( torch.load( self.config.TRAINED_MODEL_PATH ) )
            
        else:
            
            self.model.load_state_dict( torch.load( TRAINED_MODEL_PATH ) )
    
    
#%%

    def train( self ):
        
        
        #*********************
        # TRAINING MISCELLANEA
        #*********************
        self.utilities_training_functions()
        
        
        WIDTH_FORMAT = len( str( self.config.NUM_EPOCHS ) )
        for epoch in range( self.config.NUM_EPOCHS ):
            
            train_epoch_loss     = 0.0
            train_epoch_accuracy = 0.0
            train_total_correct  = 0
            train_total_samples  = 0
            for batch_idx, ( batch_data, batch_label ) in enumerate( self.train_dataloader, 0):
                
                self.optimizer.zero_grad()
                
                batch_data  = batch_data.to( self.config.DEVICE )
                batch_label = batch_label.to( self.config.DEVICE )
                y_pred      = self.model( batch_data )
                loss        = self.criterion(y_pred, batch_label)
    
                _, predicted = torch.max(y_pred, 1)
                
                train_epoch_loss += loss.item()
                
                # Update the running total of correct predictions and samples
                train_total_correct += (predicted == batch_label).sum().item()
                train_total_samples += batch_label.size(0)


                loss.backward()
                self.optimizer.step()
                
    
            self.scheduler.step()
            
            
            #**************************************
            # CALCULATE THE ACCURACY FOR THIS EPOCH
            #**************************************
            train_accuracy = 100 * train_total_correct / train_total_samples

            print(f" Epoch {epoch + 1:{WIDTH_FORMAT}d}/{self.config.NUM_EPOCHS} - Accuracy: {train_accuracy:.4f}% - Loss: {train_epoch_loss:.2f}")   
            
            
            self.epoch_accuracy.append( train_accuracy )
            self.epoch_loss.append( train_epoch_loss )
            
            
        #*********************************
        # SAVING MODEL AFTER TRAININT STEP
        #*********************************
        self.save_model()
    
    
#%%

    def test( self ):
        
        
        self.model.eval()

        test_loss          = 0.0
        test_accuracy      = 0.0
        test_total_correct = 0
        test_total_samples = 0
        with torch.no_grad():
            
            for batch_idx, ( batch_data, batch_label ) in enumerate( self.test_dataloader, 0):
        
                batch_data  = batch_data.to( self.config.DEVICE )
                batch_label = batch_label.to( self.config.DEVICE )
                
                y_pred      = self.model( batch_data )
                loss        = self.criterion(y_pred, batch_label)
    
                _, predicted = torch.max(y_pred, 1)
                
                test_loss += loss.item()
                
                #************************************************************
                # UPDATE THE RUNNING TOTAL OF CORRECT PREDICTIONS AND SAMPLES
                #************************************************************
                test_total_correct += (predicted == batch_label).sum().item()
                test_total_samples += batch_label.size(0)
                
            
            #***************************************
            # CALCULATE THE ACCURACY FOR THE TESTING
            #***************************************
            test_accuracy = 100 * test_total_correct / test_total_samples
    
            print(f"Test Accuracy: {test_accuracy:.2f}% - Loss: {test_loss:.2f}")   


#%%

    def predict( self,
                 sample_data,
                 sample_label ):
    
        self.model.eval()

        with torch.no_grad():

            sample_data = sample_data.to( self.config.DEVICE )
            y_pred      = self.model( sample_data )
            
            _, predicted = torch.max(y_pred, 1)
            
        return predicted
    
    
#%%
