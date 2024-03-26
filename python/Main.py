#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:37:03 2024

@author: peter
"""

#%%

import pathlib
import pandas            as pd
import numpy             as np
import torch
import Configuration     as config
import matplotlib.pyplot as plt


from Model                   import AutoEncoderModel
from Model                   import Network
from sklearn.model_selection import train_test_split
from Dataset_TON_IoT         import DatasetTON_WINDOW
from System_Manager          import SystemManager
from sklearn.preprocessing   import Normalizer
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import MinMaxScaler
from torchsummary            import summary


np.set_printoptions(precision = 3)
np.set_printoptions(linewidth = 1000 )


torch.set_printoptions(precision = 3)
torch.set_printoptions(linewidth = 1000 )


#%%

if __name__ == '__main__':
    
    
    CWD          = pathlib.Path.cwd()
    DATASET_PATH = CWD.parent / 'TON_IoT_Dataset/#Dataset/Train_Test_IoT_dataset'
    INPUT_FILES  = DATASET_PATH.rglob( '*.csv' )
    
    
#%%

    #*******************
    # IMPORTING ALL DATA
    #*******************
    all_dataframes = [ pd.read_csv( INPUT_FILE ) for INPUT_FILE in INPUT_FILES ]
    
    
#%%
    
    
    #************************************************
    # CREATING VARIABLE "DATETIME" FOR ALL DATAFRAMES
    #************************************************
    for dataframe in all_dataframes:
        
        dataframe["datemerge"] = pd.to_datetime( dataframe["date"] + " " + dataframe["time"],
                                                 infer_datetime_format = True )
        
        dataframe.drop( ['date', 'time'], inplace = True, axis = 1 )
        
        
#%%

    #*****************************************************
    # GROUPING DATAFRAME BY DATEMERGE TO MANAGE DUPLICATES
    #*****************************************************
    # all_dataframes = [ dataframe.groupby( ['datemerge', 'type'] ).last() for dataframe in all_dataframes ]
    all_dataframes = [ dataframe.groupby( ['datemerge'] ).last() for dataframe in all_dataframes ]
    
    
#%%

    #******************************
    # CONCATENATION OF ALL DATASETS
    #******************************
    merged_df = pd.concat( all_dataframes, axis = 1 )
    
    
#%%

    #******************************
    # DROPPING A FEW COLUMNS
    #******************************
    merged_df = merged_df.drop( ['label'], axis = 1 )
    

#%%

    #******************************
    # MERGING DUPLICATES COLUMNS
    #******************************
    out = merged_df.loc[:, 'type'].values.astype(str)
    out = np.where( out == 'nan', '', out)
    
    merged_df         = merged_df.drop( ['type'], axis = 1 )
    merged_df['type'] = [ ''.join( set( item ) ) for item in out]
    
    
#%%

    #**************************************************************************
    # FILLING MISSED VALUE WITH THE MEAN OF THE COLUMN IF THE COLUMN IS NUMERIC
    # OR AN APPROPRIATE NUMEBR OF STRING IF THE COLUMN IS AN OBJECT
    #**************************************************************************
    for col in merged_df.columns[ merged_df.isnull().any(axis = 0) ]:
    
        if merged_df[col].dtype.kind in 'biufc':
            
            merged_df[col].fillna( merged_df[col].mean(), inplace = True )
            
        else:
            
            out      = merged_df[col].value_counts()
            
            # nan_cols = merged_df[col].loc[merged_df[col].isnull(), :]
            nan_cols = merged_df[col].loc[ merged_df[col].isnull() ]
            
            merged_df.loc[ nan_cols[0:nan_cols.shape[0]//2].index, col ] = out.index[0]
            
            merged_df.loc[ nan_cols[nan_cols.shape[0]//2:].index, col ]  = out.index[1]
    
    
#%%

    #*************************************
    # RESETTING THE INDEX OF THE DATAFRAME
    #*************************************
    merged_df = merged_df.reset_index( drop = False )
    
    
#%%

    #******************************
    # DROPPING A FEW COLUMNS
    #******************************
    merged_df = merged_df.drop( ['datemerge'], axis = 1 )
    
    
#%%

    #**********************************************
    # STRIPPING WHITE SPACE FROM DATAFRAME COLUMNS
    #**********************************************
    for col in merged_df.columns:
    
        if merged_df[col].dtype.kind not in 'biufc':
    
            merged_df[col] = merged_df[col].str.strip()
            
            
#%%

    #*******************************
    # SEPARATING EATURES FROM LABELS
    #*******************************
    cols = merged_df.columns.values.tolist()
    cols.remove('type')
    
    y_label = merged_df.loc[:, ['type'] ]
    X_input = merged_df.loc[:, cols ]
    
    attacks_classes_dict = { value:key for key, value in enumerate( np.unique( y_label.values ).tolist(), 0 ) }
    
    classes_attacks_dict = { key:value for key, value in enumerate( np.unique( y_label.values ).tolist(), 0 ) }
    
    y_label['type']      = y_label['type'].apply( lambda inp_data: attacks_classes_dict[inp_data ] )
    
    
#%%

    #***********************************************
    # CONVERTING NO NUMERIC DATO TO ONE-HOT ENCODING
    #***********************************************
    for col in X_input.columns:
        
        if X_input[col].dtype.kind not in 'biufc':
            
            X_input = pd.get_dummies( X_input, columns = [col], drop_first = True)
            

#%%

    #*****************************************************
    # SEGMENTING INPUT DATA CREATING BLOCKS OF A GIVE SIZE
    #*****************************************************
    x_shape = (config.NUM_SAMPLES, X_input.shape[1])
    y_shape = (config.NUM_SAMPLES, y_label.shape[1])
    X_input_window = np.lib.stride_tricks.sliding_window_view( X_input, x_shape)[::config.WINDOW_SIZE, :]
    y_input_window = np.lib.stride_tricks.sliding_window_view( y_label, y_shape)[::config.WINDOW_SIZE, :]


    #**************************************************
    # DEFINING TEST SAMPLE WITH A SPECIFIC WINDOW SIZE
    #**************************************************
    # x_shape = (config.NUM_SAMPLES, X_train.shape[1])
    # y_shape = (config.NUM_SAMPLES, y_train.shape[1])


#%%

    #**********************************
    # SQUEEZING TRAIN NUMPY LABEL ARRAY
    #**********************************
    y_input_window = np.squeeze( y_input_window, axis = -1)
    y_input_window = np.squeeze( y_input_window, axis = 1)
    y_input_window = y_input_window[:, -1]
    
    
#%%

    #**********************
    # SPLITTING THE DATASET
    #**********************
    X_train, X_test, y_train, y_test = train_test_split( X_input_window, 
                                                         y_input_window, 
                                                         test_size    = 0.2, 
                                                         shuffle      = True, 
                                                         stratify     = None ,
                                                         random_state = 0 )
    
    
#%%

    #*********************************
    # DATA SCALING USING NORMALIZATION
    #*********************************
    # normalizer = Normalizer()
    
    # normalizer.fit( X_train )
    
    # X_train = normalizer.transform(X_train)
    # X_test  = normalizer.transform(X_test)
    
    
    #***********************************
    # DATA SCALING USING STANDARDIZATION
    #***********************************
    scaler = StandardScaler(with_mean = True, with_std = True)
    
    scaler.fit( X_train.reshape(-1, X_train.shape[-1] ) )
    
    X_train = scaler.transform( X_train.reshape(-1, X_train.shape[-1] ) ).reshape( X_train.shape )
    X_test  = scaler.transform( X_test.reshape(-1, X_test.shape[-1] ) ).reshape( X_test.shape )


    #***********************************
    # DATA SCALING USING MINMAXSCALER
    #***********************************
    # min_max_scaler = MinMaxScaler()
    
    # min_max_scaler.fit(X_train)
    
    # X_train = min_max_scaler.transform(X_train)
    # X_test  = min_max_scaler.transform(X_test)


#%%


    #***********************
    # DEFINING TRAIN DATASET
    #***********************
    train_dataset = DatasetTON_WINDOW( X_train, 
                                       y_train )
        
    
    #**************************
    # DEFINING TRAIN DATALOADER
    #**************************
    train_dataloader = torch.utils.data.DataLoader( dataset     = train_dataset, 
                                                    shuffle     = True, 
                                                    batch_size  = config.TRAIN_BATCH,
                                                    num_workers = 0 )


    # #***********************
    # # DEFINING TEST DATASET
    # #***********************
    # test_dataset = DatasetTON_WINDOW( np.array( X_test ), np.array( y_test ) )
    
    
    # #**************************
    # # DEFINING TEST DATALOADER
    # #**************************
    # test_dataloader = torch.utils.data.DataLoader( dataset     = test_dataset, 
    #                                                 shuffle     = True, 
    #                                                 batch_size  = config.TEST_BATCH,
    #                                                 num_workers = 0 )
    
    
#%%

    NUMBER_FEATURES = X_input.shape[1]
    NUMBER_FEATURES = np.prod( x_shape )
    NUMBER_ATTACKS  = len( attacks_classes_dict.keys() )
    
    
    #************************
    # DEFINING THE MAIN MODEL
    #************************
    model = Network(NUMBER_FEATURES, [32, 32], NUMBER_ATTACKS)
    # model = Network(NUMBER_FEATURES, [64, 64], NUMBER_ATTACKS)
    
    
    # #************************************
    # # DEFINING THE AUTOENCODER MODEL
    # #************************************
    # model = AutoEncoderModel( data_size   = 400, 
    #                           encoder_szs = [300, 200, 100], 
    #                           latent_size = 100,
    #                           decoder_szs = [300, 200, 100] )

    
    
#%%

    #***************************
    # DEFINING THE LOSS FUNCTION
    #***************************
    criterion = torch.nn.CrossEntropyLoss()


#%%

    #************************
    # DEFINING THE OPTIMIZER
    #************************
    optimizer = torch.optim.SGD( model.parameters(),
                                  lr       = config.LEARNING_RATE,
                                  momentum = config.MOMENTUM )
    
    
#%%

    #************************
    # DEFINING THE SCHEDULER
    #************************
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, 
                                                      milestones = [10, 20],
                                                      gamma      = 0.1,
                                                      verbose    = False )
    
    
#%%

    #*************************
    # DEFINE A SYSTEM MANAGER
    #*************************
    manager = SystemManager( train_dataloader = train_dataloader,
                              valid_dataloader = None,
                              test_dataloader  = None, 
                              model            = model,
                              criterion        = criterion,
                              optimizer        = optimizer,
                              scheduler        = scheduler,
                              config           = config )
    
    
#%%

    #**************************
    # RUNNING THE TRAINING STEP
    #**************************
    manager.train()
    
    
#%%

    #*************************
    # RUNNING THE TESTING STEP
    #*************************
    # manager.test()
    

#%%
   
    #*************************
    # LOADING TRAINED MODEL
    #*************************
    # manager.load_trained_model()


    # #**********************
    # # RUN THE PREDICT STEP
    # #**********************
    # STRING_FORMAT = max( list( map( len, classes_attacks_dict.values() ) ) )
    # for data, label in test_dataloader:
        
    #     predicted_attack_id = manager.predict( data, label )
        
    #     predicted_attack    = classes_attacks_dict[predicted_attack_id.item()]
    #     ground_truth_attach = classes_attacks_dict[label.item()]
    #     matching            = True if predicted_attack_id == label else False
        
    #     print( f'Ground Truth Attack : {ground_truth_attach:{STRING_FORMAT}s} ===> Predicted Attack : {predicted_attack:{STRING_FORMAT}s} ===> {matching}')
        
        
#%%
