#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:42:20 2024

@author: peter
"""

#%%

import torch


#%%

class Network( torch.nn.Module ):
    
    
    def __init__( self,
                  input_size,
                  hidden_layers_size,
                  output_size ):
        
      super().__init__()
      
      #**********************
      # BUILDING THE MODEL
      #**********************
      layers = [ torch.nn.Linear(input_size, hidden_layers_size[0]), 
                 torch.nn.ReLU() ]
      for input_dim, output_dim, in zip( hidden_layers_size[0:-1], hidden_layers_size[1:] ):
          
          layers.append( torch.nn.Linear(input_dim, output_dim) )
          layers.append( torch.nn.ReLU() )
          
    
      layers.append( torch.nn.Linear( hidden_layers_size[-1], output_size) )
      
      self.net = torch.nn.Sequential( *layers )
      
      
    def forward( self,
                 data ):
        
        out = self.net( data )
        
        return out


#%%

class Encoder( torch.nn.Module ):
    """
    Standard encoder module for standard autoencoders with tabular input.
    """
    def __init__( self, 
                  data_size, 
                  hidden_sizes, 
                  latent_size ):
        """
        Args:
            data_size (int): Dimensionality of the input data.
            hidden_sizes (list[int]): Sizes of hidden layers (not including the
                input layer or the latent layer).
            latent_size (int): Size of the latent space.
        """
        super().__init__()

        self.data_size = data_size
        

        #**********************
        # CONSTRUCT THE ENCODER
        #**********************
        encoder_szs    = [data_size] + hidden_sizes
        encoder_layers = []
        for in_sz, out_sz, in zip( encoder_szs[:-1], encoder_szs[1:] ):
            
            encoder_layers.append( torch.nn.Linear(in_sz, out_sz) )
            encoder_layers.append( torch.nn.ReLU() )
            
         
        encoder_layers.append( torch.nn.Linear( encoder_szs[-1], latent_size ) )
        
        self.encoder = torch.nn.Sequential(*encoder_layers)
        

    def encode( self, 
                x ):
        
        return self.encoder(x)
    

    def forward( self, 
                 x ):
        
        x = self.encode(x)
        
        return x

    
#%%


class Decoder( torch.nn.Module ):
    """
    VAE decoder module that models a diagonal multivariate Bernoulli
    distribution with a feed-forward neural net.
    """
    def __init__( self, 
                  data_size, 
                  hidden_sizes, 
                  latent_size ):
        """
        Args:
            data_size (int): Dimensionality of the input data.
            hidden_sizes (list[int]): Sizes of hidden layers (not including the
                input layer or the latent layer).
            latent_size (int): Size of the latent space.
        """
        super().__init__()

        # construct the decoder
        hidden_sizes   = [latent_size] + hidden_sizes
        decoder_layers = []
        for in_sz,out_sz, in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            
            decoder_layers.append( torch.nn.Linear(in_sz, out_sz) )
            decoder_layers.append( torch.nn.ReLU() )
            
            
        decoder_layers.append( torch.nn.Linear(hidden_sizes[-1], data_size) )
        # decoder_layers.append( torch.nn.Sigmoid() )
        
        self.decoder = torch.nn.Sequential(*decoder_layers)
        

    def forward( self, 
                 z ):
        
        return self.decoder(z)
    

#%%

class AutoEncoderModel( torch.nn.Module ):

    
    def __init__( self, 
                  data_size, 
                  encoder_szs, 
                  latent_size, 
                  decoder_szs = None ):
        
        super().__init__()

        # if decoder_szs not specified, assume symmetry
        if decoder_szs is None:
            
            decoder_szs = encoder_szs[::-1]
            

        #************************
        # INSTANTIATE THE ENCODER
        #************************
        self.encoder = Encoder( data_size    = data_size, 
                                hidden_sizes = encoder_szs,
                                latent_size  = latent_size )

        #************************
        # INSTANTIATE THE DECODER
        #************************
        self.decoder = Decoder( data_size    = data_size, 
                                latent_size  = latent_size,
                                hidden_sizes = decoder_szs )

        self.data_size = data_size
        

    def decode( self, 
                z ):
        
        return self.decoder(z)
    

    def forward( self, 
                 x ):
        
        #*****************************************************
        # FLATTENING ALL DIMENSION STARTING FROM THE FIRST ONE
        #*****************************************************
        x   = x.flatten(start_dim = 1)
        
        z   = self.encoder(x)
        out = self.decoder(z)
        
        return out


#%%
