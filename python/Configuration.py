#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:09:58 2024

@author: peter
"""

#%%

import pathlib
import os
import torch


from datetime import datetime


#%%

#*****************************
# DEFINE CURRENT DATE AND TIME
#*****************************
DATE_TIME_STRING = datetime.now().strftime("DATE_%d_%m_%Y_TIME_%H_%M_%S")


#*********************************************
# DEFINING THE ROOT PATH
#*********************************************
ROOT_PATH = pathlib.Path.cwd()


#**************************************************
# DEFINING THE PATH TO THE SESSION OUTPUT DIRECTORY
#**************************************************
RUN_PATH = ROOT_PATH / '_'.join( ["RUN_PATH", DATE_TIME_STRING])


#***************************************************************
# DEFINING THE PATH WHERE TO SAVE THE BEST MODEL
#***************************************************************
BEST_MODEL_PATH = RUN_PATH / "BEST_MODEL"


#***************************************************************
# DEFINING THE NAME OF THE MDOEL TO SAVE
#***************************************************************
BEST_MODEL_NAME = 'Best_Model.pth'


#***************************************************************
# DEFINING THE PATH WHERE SAVING METRICS
#***************************************************************
OUTPUT_PATH = RUN_PATH / "OUTPUT"


#*******************************************
# LIST OF ALL PATHS USED IN THE TRAINNG STEP
#*******************************************
ALL_TRAINING_PATHS = [RUN_PATH, BEST_MODEL_PATH, OUTPUT_PATH]


#*********************************************
# DEFINING TRAINED RUN PATH
#*********************************************
TRAINED_RUN_FOLDER_PATH = pathlib.Path( './RUN_PATH_DATE_25_03_2024_TIME_11_28_10' )


#*********************************************
# DEFINING TRAINED MODEL TO LOAD PATH
#*********************************************
TRAINED_MODEL_PATH = TRAINED_RUN_FOLDER_PATH / 'BEST_MODEL/Best_Model.pth'


#********************
# TRAINING ATTRIBUTES
#********************
NUM_EPOCHS    = 30
TRAIN_BATCH   = 32
TEST_BATCH    = 1
OUTPUT_DIM    = 10
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
WINDOW_SIZE   = 20
NUM_SAMPLES   = 30
DEVICE        = torch.device( "cpu" )