#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:44:02 2024

@author: peter
"""

#%%

import torch
import numpy as np


#%%

class DatasetTON( torch.utils.data.Dataset ):
    
    
    def __init__( self,
                  x_data,
                  y_data ):
        
        self.x_data = x_data
        self.y_data = y_data
        
        
    def __len__( self ):
        
        return len( self.x_data )
    

    def __getitem__( self, 
                     idx ):
        
        data  = torch.from_numpy( self.x_data[idx, :] ).float()
        label = torch.from_numpy( self.y_data[idx] ).squeeze(dim = 0)
        
        return data, label
    
    
#%%


class DatasetTON_WINDOW( torch.utils.data.Dataset ):
    
    
    def __init__( self,
                  x_data,
                  y_data ):
        
        self.x_data = x_data
        self.y_data = y_data
        
        
    def __len__( self ):
        
        return len( self.x_data )
    

    def __getitem__( self, 
                     idx ):
        
        data  = self.x_data[idx].reshape( np.prod( self.x_data[idx].shape[1:] ) )
        data  = torch.from_numpy( data ).float()
        
        label = self.y_data[idx]
        label = torch.from_numpy( np.array(label) )#.unsqueeze(dim = 0)
        
        
        return data, label
    
    
#%%
