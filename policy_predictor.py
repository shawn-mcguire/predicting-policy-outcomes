#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:51:34 2020

@author: shawnmcguire
"""


"""
--------------------------------------------------------------
--------------------function defs ----------------------------
--------------------------------------------------------------
"""

def policy_predictor(model_B_flag, model_C_flag, model_D_flag, retrodiction_flag, acc_stats_flag):
    '''
    produces single run modeling results
        
    1.  reads in csv data with independent and dependent (outcome) variables
    2.  selects model via model flags (model A is default if model flags false)
        Model A: p90, NetIGA
        Model B: p90, policy domains, individual interest groups
        Model C: P90, policy domains, top 14 Interest groups
        Model D: p90, policy areas, individual interest groups
    3.  selects random train/test vs retrodiction via retrodiction_flag       
    4.  returns accuracy values if acc_stats_flag = True
    '''    
 
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import auc
    
    # read data
    data = pd.read_csv('./gilens_data_sm_copy.csv')
    
    #interest group columns 
    interestGroupIndices = np.arange(8,51) #interest group columns 
    
    # filter data, select proper rows, drop switcher, etc.
    data = data.drop(['pred10_sw'], axis = 1)
    data = data.drop(['pred50_sw'], axis = 1)
    data = data.iloc[0:1836] 
    data = data.drop([ 'switcher'], axis = 1) 
    data = data.drop(['pred90 - pred10'], axis = 1) 
    
    ''' USER ENTRIES '''  
    MAX_DEPTH_A = 3
    MAX_DEPTH_B = 5  # model C = 5 as well
    MAX_DEPTH_D = 8
    CLASS_WEIGHT = "balanced"
    
    # year to split train and test sets for retrodiction
    SPLIT_YEAR = 1997
    
    ''' BUILD MODEL '''    
    # select model data for chosen model
    if model_B_flag: 
        MAX_DEPTH = MAX_DEPTH_B
        data = data.drop(['IntGrpNetAlign'], axis = 1);
        data = data.drop(['XL_AREA'], axis = 1)
        
    elif model_C_flag:    
        MAX_DEPTH = MAX_DEPTH_B
        data = data.drop(['XL_AREA'], axis = 1)
        # drop lower importance features code
        df_mean_imp_IGs = pd.read_pickle('./df_mean_imp_IGs.pkl')
        df = df_mean_imp_IGs.tail(-14)
        for name in df.index.values:
            if name in data.columns:
                data = data.drop([name], axis=1)
                
    elif model_D_flag:
        MAX_DEPTH = MAX_DEPTH_D
        data = data.drop(['IntGrpNetAlign'], axis = 1);
        data = data.drop(['ECYN', 'SWYN', 'FPYN', 'RLYN', 'GNYN'], axis = 1)
        
    else: # select model data for model A
        MAX_DEPTH = MAX_DEPTH_A
        data = data.drop(['XL_AREA'], axis = 1)
        data = data.drop(data.columns[interestGroupIndices], axis = 1); 
        data = data.drop(['ECYN', 'SWYN', 'FPYN', 'RLYN', 'GNYN'], axis = 1)
        
    # hot code data (XL_Area is non-numeric)
    data = pd.get_dummies(data);
      
    # if retrodiction flag true: split into train and test sets via SPLIT_YEAR 
    if retrodiction_flag:
        dataTrainID = data.YEAR < SPLIT_YEAR;
        dataTrain = data.loc[dataTrainID].drop(['YEAR', 'OutcomeYear'], axis = 1);
        dataTest = data.loc[~dataTrainID].drop(['YEAR', 'OutcomeYear'], axis = 1);
    # else take a random draw of train/test cases 
    else:
        dataTrain = data.sample(frac = .65)
        dataTrain = dataTrain.drop(['YEAR', 'OutcomeYear'], axis = 1)
        ind = dataTrain.index
        dataTest = data.drop(ind)
        dataTest = dataTest.drop(['YEAR', 'OutcomeYear'], axis = 1)
        
    # separate into train/test features and labels
    train_features = dataTrain.drop('Binary Outcome', axis = 1);
    train_labels = dataTrain['Binary Outcome'];
    test_features = dataTest.drop('Binary Outcome', axis = 1);
    test_labels = dataTest['Binary Outcome'];
    
    # instantiate model and train 
    clf = RandomForestClassifier(n_estimators = 200, max_depth = MAX_DEPTH, class_weight = CLASS_WEIGHT);
    print(clf.get_params());
    clf.fit(train_features,train_labels);
    
    ''' CALCULATE PERFORMANCE METRICS '''   
    # predict and get results
    predictions = clf.predict(test_features);
    accuracy = metrics.accuracy_score(test_labels, predictions);
    precision = precision_score(test_labels, predictions);
    recall = recall_score(test_labels, predictions);
    f1 = f1_score(test_labels,predictions);
    conf_matrix = metrics.confusion_matrix(test_labels, predictions);
    
    # calculate sens and spec
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    balanced_accuracy = (sens + spec) / 2
    
    # calculate predictor importance
    feature_names = list(train_features.head(0))
    feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
    
    # calculate auc_score from fpr, tpr
    prob_scores = clf.predict_proba(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, prob_scores[:,1], pos_label=1)      
    auc_score = auc(fpr, tpr) 
   
    ''' RETURN RESULTS '''   
    # if acc_stats_flag: return accuracy stats
    if acc_stats_flag:
        return accuracy, balanced_accuracy, auc_score   
    
    # model B: create train features dataframe with individual int groups (for delta comparison plot with model A) 
    elif model_B_flag:
#        test_features = test_features.drop(['ECYN', 'FPYN', 'SWYN', 'RLYN', 'GNYN'], axis = 1)
        test_features_int_grp = test_features
        return test_features_int_grp, predictions, test_labels, data, feature_imp
    # else return model A data
    else:
        return predictions, test_labels, data
        
#%%

def acc_calculator(model_B_flag, test_features_int_grp, predictions, test_labels):

    '''
     acc_calculator calculates accuracy of predictions based one of two selected models
     1. model B: p90 + individual interest groups + policy domains
     2. model A: p90 + IntGrpNetAlign + policy domains      
     and returns:
         acc_table_IndIG for model B
         acc_table_IGNA for model A
    '''
   
    import pandas as pd
    
    #int_grp_name = 'AARP'
    names = list(test_features_int_grp)
    names.remove('pred90_sw')
    
    # create output table, acc_table
    acc_table = pd.DataFrame(columns = ('int_grp_name', 'n','acc'))
    p = 0
    for n in names:
        int_grp_name = n
    
        # create 'data' dataframe for each int group (cols = int grp subset, pred, and outcome
        int_grp = pd.DataFrame(test_features_int_grp[int_grp_name])
        int_grp.reset_index(drop=True, inplace=True)
        pred = pd.DataFrame(predictions, columns = ['pred'])
        pred.reset_index(drop=True, inplace=True)
        outcome = pd.DataFrame(test_labels)
        outcome.reset_index(drop=True, inplace=True)
        data = pd.concat([int_grp, pred, outcome], axis = 1)
        
        # drop -1, 0, and 1 from int grp alignment
        data = data.drop(data[(data[int_grp_name] < 2) & (data[int_grp_name] > -2)].index)
        data['acc'] = 0
           
        # calculate accuracy
        for i in data.index:
            data.loc[i,'acc'] = 0
            if (data.loc[i,'pred'] == 1) & (data.loc[i,'Binary Outcome'] == 1):
                data.loc[i,'acc'] = 1    
            if (data.loc[i,'pred'] == 0) & (data.loc[i,'Binary Outcome'] == 0):
                data.loc[i,'acc'] = 1       
        acc = (data['acc'].sum())/(len(data))
        numCases = len(data)
        acc_table.loc[p] = [int_grp_name, numCases, acc]
            
        p = p + 1
    # if model_B_flag is True    
    if model_B_flag:
        acc_table_IndIG =  acc_table
        return acc_table_IndIG
    # else, model A
    else:
        acc_table_IGNA = acc_table
        return acc_table_IGNA
    
    #%%
        
def delta_acc_plotter(acc_table_IndIG, acc_table_IGNA):
  
    '''    
    delta_acc_plotter creates horizontal bar plot of accuracy difference 
    (model B - model A) for each interest group    
    '''
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
      
    # create acc_table with relative accuracy column, 'rel_acc' 
    acc_table_IGNA.columns = ['int_grp_name', 'n', 'acc_IGNA']
    acc_table_IndIG.columns = ['int_grp_name', 'n', 'acc_IndIG']
    acc_table = pd.concat([acc_table_IndIG, acc_table_IGNA['acc_IGNA']], axis = 1)
    acc_table['rel_acc'] = acc_table['acc_IndIG'] - acc_table['acc_IGNA']

    # eliminate intgrps with low # of cases, n
    acc_table = acc_table[(acc_table['n'] > 20)]
    
    # sort acc_table by ascending 'n' column and reset index

    acc_table.reset_index(inplace=True, drop=True)
    
    rel_acc = acc_table['rel_acc']
    
    # bar plot of rel_ac vs ig 
    fig, ax = plt.subplots()
    
    y_labels = acc_table.int_grp_name

    y_pos = np.arange(len(y_labels))
    x_value = rel_acc
    colormat = np.where(x_value>0, 'g','y')
    ax.barh(y_pos, x_value, align='center', color = colormat)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize = 14)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Model Accuracy Change', fontsize = 14)
    ax.set_title('Delta accuracy on interest group subsets: \n Model B  - Model A \n random draw',wrap = True, fontsize = 14)
    plt.show()
       
    #%%
    
def mean_feature_importance():

    """
    calculates mean feature importance values of n runs
    
    """
    
    import pandas as pd
    from policy_predictor import policy_predictor

    ''' USER ENTRIES '''        
    model_B_flag = True
    model_C_flag = False
    model_D_flag = False
    retro_flag = False
    acc_stats_flag = False
    
    runs = 30
    
    # run model once to instantiate df
    test_features_int_grp, predictions, test_labels, data, feature_imp = policy_predictor(model_B_flag, model_C_flag, model_D_flag, retro_flag, False);
    df = feature_imp.to_frame()
    
    # run multiple times
    for n in range(runs):
        test_features_int_grp, predictions, test_labels, data, feature_imp = policy_predictor(model_B_flag, model_C_flag, model_D_flag, retro_flag, False);
        df = pd.concat([df, feature_imp.to_frame()], axis=1, sort=False)
    
    # get mean feature importances
    df['mean_imp'] = df.mean(axis=1)
    df_mean_imp = df.copy()
    df_mean_imp = df_mean_imp.sort_values(by='mean_imp', ascending = False)
    df_mean_imp = df_mean_imp['mean_imp'].to_frame()
    
    # drop PDs from list
    df_mean_imp = df_mean_imp.drop(['ECYN', 'SWYN', 'FPYN', 'RLYN', 'GNYN'])
    
    return df_mean_imp
    


    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
