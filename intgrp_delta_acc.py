#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:57:03 2020

@author: shawnmcguire
"""

'''
calculates the accuracy difference for each interest group between model B (p90, IGs, PDs and model A (p90, IGNA)   
'''

import pandas as pd
from policy_predictor import policy_predictor, acc_calculator, delta_acc_plotter

#%%
'''
user entries
'''
# select # of runs
runs = 25
# select retrodiction (true) or full data set random draw (false)
retro_flag = False

#%%
'''
run model B multiple times to create acc_table_IndIG with mean accuracies

'''
# run model B once to instantiate dataframe, df
test_features_int_grp, predictions, test_labels, data, feature_imp = policy_predictor(True, False, False, retro_flag, False);
dfB = acc_calculator(True, test_features_int_grp, predictions, test_labels)

# run multiple times
for n in range(runs):
    test_features_int_grp, predictions, test_labels, data, feature_imp = policy_predictor(True, False, False, retro_flag, False);
    acc_table_IndIG = acc_calculator(True, test_features_int_grp, predictions, test_labels)
    dfB['acc_run_' + str(n)] = acc_table_IndIG['acc']
    dfB['n'] = dfB['n'] + acc_table_IndIG['n'] # running sum of test cases

# calculate mean test cases, mean accuracy. create acc_table_IndIG
dfB['n'] = (dfB['n']/(runs + 1)).astype(float).round(0).astype(int)
dfB['mean_acc'] = dfB.drop(['n'], axis = 1).sum(axis = 1)/(runs + 1)
acc_table_IndIG = dfB[['int_grp_name', 'n', 'mean_acc']]

#%%
'''
run model A multiple times to create acc_table_IGNA with mean accuracy

'''
## run model A once to instantiate the dataframe, dfA
predictions, test_labels, data = policy_predictor(False, False, False, retro_flag, False);
dfA = acc_calculator(False, test_features_int_grp, predictions, test_labels)

# run multiple times
for n in range(runs):
    predictions, test_labels, data = policy_predictor(False, False, False, retro_flag, False);
    acc_table_IGNA = acc_calculator(False,test_features_int_grp, predictions, test_labels)
    dfA['acc_run_' + str(n)] = acc_table_IGNA['acc']

# calculate mean accuracy and create acc_table_IGNA
dfA['mean_acc'] = dfA.drop(['n'], axis = 1).sum(axis = 1)/(runs + 1)
acc_table_IGNA = dfA[['int_grp_name', 'n', 'mean_acc']]

#%%
'''
calculate mean standard deviation, delta_acc_std, from delta accuracies from user selected 'n' runs

'''
# calculate std by from individual acc runs
dfA_temp = dfA.drop(['int_grp_name','n', 'mean_acc'], axis = 1)
dfB_temp = dfB.drop(['int_grp_name','n', 'mean_acc'], axis = 1)

# df3 = df of delta accuracy for EACH run
df3 = dfB_temp.subtract(dfA_temp)

# calculate std
df3['delta_acc_std'] = df3.std(axis=1)

# create acc_table with delta accuracy column, 'delta_acc' 
acc_table_IGNA.columns = ['int_grp_name', 'n', 'acc_IGNA']
acc_table_IndIG.columns = ['int_grp_name', 'n', 'acc_IndIG']
acc_table = pd.concat([acc_table_IndIG, acc_table_IGNA['acc_IGNA']], axis = 1)
acc_table['delta_acc'] = acc_table['acc_IndIG'] - acc_table['acc_IGNA']
acc_table = pd.concat([acc_table, df3['delta_acc_std']], axis = 1)

# create bar plot
delta_acc_plotter(acc_table_IndIG, acc_table_IGNA)
    
