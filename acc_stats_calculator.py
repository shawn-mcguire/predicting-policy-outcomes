#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:54:03 2020

@author: shawnmcguire
"""

"""
for a selected model, calculates RF mean and std of
    1. accuracy
    2. balanced accuracy
    3. AUC
from user selected number of runs

model A:  p90, netIGA
model B:  p90, 43 IGs, policy domains
model C:  p90, 14 IGs, policy domains
model D:  p90, 43 IGs, policy areas
"""

import pandas as pd
from policy_predictor import policy_predictor
from policy_predictor import mean_feature_importance

# %% 
''' USER ENTRIES '''
    
# model A is default if model flags below are all false
model_B_flag = True
model_C_flag = False
model_D_flag = False

# retro_flag.  True for retrodition, False for random draw
retro_flag = False

runs = 25

# %%

# instantiate df
df= pd.DataFrame(columns = ('acc', 'bal_acc', 'AUC'))

df_mean_imp_IGs = mean_feature_importance();
df_mean_imp_IGs.to_pickle('./df_mean_imp_IGs.pkl')

# run multiple times
for n in range(runs):
    accuracy, balanced_accuracy, auc_score = policy_predictor(model_B_flag, model_C_flag, model_D_flag, retro_flag, True);
    df.loc[n] = [accuracy, balanced_accuracy, auc_score]

acc_mean = df['acc'].mean()
acc_std = df['acc'].std()
bal_acc_mean = df['bal_acc'].mean()
bal_acc_std = df['bal_acc'].std()
auc_mean = df['AUC'].mean()
auc_std = df['AUC'].std()

    
    


