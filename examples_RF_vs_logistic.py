"""
This script reproduces results in the the paper 'Predicting United States policy outcomes with
Random Forests', by Shawn McGuire and Charles Delahunt. It examines the RF vs Logistic example for
Defense Contractors and P90 in Foreign policy (section 4.1.1 and Figure 1 in paper). It can also
run a similar example, that of NRA vs P90 in Gun Policy.

Copyright (c) 2020 Charles B. Delahunt.  delahunt@uw.edu
MIT License

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

"""
Note re plot window location:
1. To print plots in separate windows, run the following on the command line:
%matplotlib qt
2. To revert back to plots appearing in console:
%matplotlib inline
"""
#%%
''' USER ENTRIES '''
# Choose which example to look at: FPYN (defense and P90) or GNYN (NRA and P90):
whichDomainExample = 'Foreign'  # 'Foreign' -> Foreign Policy example. 'Guns' -> Gun Policy example.

''' END USER ENTRIES '''

#%%
''' FUNCTION DEFS '''

def findMaxAccFunction(scores, labels):
    ''' Calculate the maximun accuracies given the model scores and true labels
    Inputs:
        scores = n x 1 np vector with values in [0,1]
        labels = n x 1 np vector with values in {0, 1}
    Outputs:
        maxBalancedAcc = scalar in [0, 1]
        thresh = scaler in [0,1]
        '''

    x = np.linspace(0, 1, 101)
    rawAcc = np.zeros(x.shape)
    sens = np.zeros(x.shape)
    spec = np.zeros(x.shape)
    balancedAcc = np.zeros(x.shape)
    for i in range(len(x)):
        TP = np.sum(np.logical_and(scores >= x[i], labels == 1))
        FN = np.sum(np.logical_and(scores < x[i], labels == 1))
        FP = np.sum(np.logical_and(scores >= x[i], labels == 0))
        TN = np.sum(np.logical_and(scores < x[i], labels == 0))
        rawAcc[i] = (TP + TN) / (TP + TN + FP + FN)
        sens[i] = TP / (TP + FN)
        if TP + FN == 0:
            sens[i] = -10000  # Flag for bad train-test split
        spec[i] = TN / (FP + TN)
        #precision[i] = TP / (TP + FP)
        balancedAcc[i] = (sens[i] + spec[i]) / 2
    maxBalancedAcc = max(balancedAcc)
    threshInd = np.where(balancedAcc == maxBalancedAcc)[0]
    thresh = x[threshInd[int(np.floor(len(threshInd)/2))]]  # Take middle value

    return maxBalancedAcc, thresh

# End findMaxAccFunction
#---------------------------------------------------

#%%
def comparePlotsFunction(rfYHat, logYHat, y, xLabelStr):
    ''' Make three subplots, showing true classes; logistic predictions; and RF predictions.
    Two parts: 1. Collect the accuracy stats to color-code the plots; 2. Make plots.
    Inputs:
        rfYHat: vector of ints.
        logYHat: vector of ints.
        y: vector of ints.
        xLabelStr: str.
    Outputs:
        matplotlib figure.
        Strings to console.
    '''
    
    # 1. Assemble accuracy statistics:
    rfTP = np.logical_and(rfYHat == 1, y == 1)
    rfTN = np.logical_and(rfYHat == 0, y == 0)
    rfFP = np.logical_and(rfYHat == 1, y == 0)
    rfFN = np.logical_and(rfYHat == 0, y == 1)
    logTP = np.logical_and(logYHat == 1, y == 1)
    logTN = np.logical_and(logYHat == 0, y == 0)
    logFP = np.logical_and(logYHat == 1, y == 0)
    logFN = np.logical_and(logYHat == 0, y == 1)

    tp = rfTP
    tn = rfTN
    fp = rfFP
    fn = rfFN
    sens = np.sum(tp)/ (np.sum(tp) + np.sum(fn))
    spec = np.sum(tn) / (np.sum(tn) + np.sum(fp))
    rfBalAcc = np.round(100* 0.5*(sens + spec))

    tp = logTP
    tn = logTN
    fp = logFP
    fn = logFN
    sens = np.sum(tp)/len(tp) / (np.sum(tp)/len(tp) + np.sum(fn)/len(fn))
    spec = np.sum(tn)/len(tn) / (np.sum(tn)/len(tn) + np.sum(fp)/len(fp))
    logBalAcc = np.round(100* 0.5*(sens + spec))

    print('RF bal acc = ' + str(rfBalAcc) + '%, logistic bal acc = ' + str(logBalAcc) + '%')

    # 2. Make the figure:
    plt.figure()
    
    yLabelStr = 'P 90'

    plt.subplot(1, 3, 1)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', markersize=6, label='Positive')
    plt.plot(x[y == 0, 0], x[y == 0, 1], 'bd', markersize=8, markerfacecolor='w', label='Negative')
    plt.ylabel(yLabelStr, fontsize=14, fontweight='bold')
    plt.xlabel(xLabelStr, fontsize=14, fontweight='bold')
    plt.title('true labels', fontsize=14, fontweight='bold')
    plt.legend(loc='lower center', prop={'size':12, 'weight': 'bold'})
    if fpynFlag:
        plt.hlines([0.485], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.3, 0.9, 'm')
    else:
        plt.hlines([0.7], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.2, 0.9, 'm')

    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.xlim(-2.5, 2.5)
    if fpynFlag:
        plt.ylim(0.15, 0.92)
    else:
        plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.plot(x[logTP, 0], x[logTP, 1], 'bo', markersize=6, label='TP')
    plt.plot(x[logTN, 0], x[logTN, 1], 'bd', markersize=8, markerfacecolor='w', label='TN')
    plt.plot(x[logFN, 0], x[logFN, 1], 'ro', markersize=6, label='FN')
    plt.plot(x[logFP, 0], x[logFP, 1], 'rd', markersize=8, markerfacecolor='w', label='FP')
    plt.title('Logistic', fontsize=14, fontweight='bold')
    plt.legend(loc='lower center', prop={'size':12, 'weight': 'bold'})
    if fpynFlag:
        plt.hlines([0.485], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.3, 0.9, 'm')
    else:
        plt.hlines([0.7], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.2, 0.9, 'm')
    plt.yticks(size=12, weight='bold')
    plt.xticks(size=12, weight='bold')
    plt.xlim(-2.5, 2.5)
    if fpynFlag:
        plt.ylim(0.15, 0.92)
    else:
        plt.ylim(0, 1)

    plt.subplot(1, 3, 3)
    plt.plot(x[rfTP, 0], x[rfTP, 1], 'bo', markersize=6, label='TP')
    plt.plot(x[rfTN, 0], x[rfTN, 1], 'bd', markersize=8, markerfacecolor='w', label='TN')
    plt.plot(x[rfFN, 0], x[rfFN, 1], 'ro', markersize=6, label='FN')
    plt.plot(x[rfFP, 0], x[rfFP, 1], 'rd', markersize=8, markerfacecolor='w', label='FP')
    plt.title('Random Forest ', fontsize=14, fontweight='bold')
    plt.legend(loc='lower center', prop={'size':12, 'weight': 'bold'})
    if fpynFlag:
        plt.hlines([0.485], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.3, 0.9, 'm')
    else:
        plt.hlines([0.7], -2.2, -0.5, 'm')
        plt.vlines([-0.5], 0.2, 0.9, 'm')
    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.xlim(-2.5, 2.5)
    if fpynFlag:
        plt.ylim(0.15, 0.92)
    else:
        plt.ylim(0, 1)

# End comparePlotsFunction
#-----------------------------------------------------------
''' END FUNCTION DEFS'''

''' BEGIN MAIN: '''

#%% Load the main dataset:
dataFolder = 'data/'
dataFull = pd.read_csv(dataFolder + 'gilens_data_sm_copy.csv')
dataFull = dataFull.iloc[0:1836]    # All rows after 1835 are nan

policyDomainList = ['ECYN', 'FPYN', 'SWYN', 'RLYN', 'GNYN', 'MISC']
allInterestGroupIndices = np.arange(12, 55)

fpynFlag = whichDomainExample == 'Foreign'

#%% Extract features and labels

if fpynFlag:  # Case 1: Look at FPYN, P90 and defense contractors.
    #  Restrict to those cases where defense had an opinion:
    data = dataFull[np.logical_and(dataFull['FPYN'] == 1, dataFull['Defense Contractors'] != 0)]

    # Get preferences of P90 and defense contractors
    P90 = data.pred90_sw.values
    defense = data['Defense Contractors'].values
    outcome = data['Binary Outcome'].values
    x = np.zeros([len(P90), 2])
    x[:, 0] = defense
    x[:, 1] = P90
    y = outcome.flatten()
    xLabelStr = 'Defense'

else:  # Case 2: Look at Guns, P 90 and NRA:
    # Restrict to cases where defense had an opinion:
    data = dataFull[np.logical_and(dataFull['GNYN'] == 1,
                                   dataFull['National Rifle Association'] != 0)]

    # Get preferences of P90 and NRA:
    P90 = data.pred90_sw.values
    nra = data['National Rifle Association'].values
    outcome = data['Binary Outcome'].values
    x = np.zeros([len(P90), 2])
    x[:, 0] = nra
    x[:, 1] = P90
    y = outcome.flatten()
    xLabelStr = 'NRA'

#%%
# Remove cases that have identical positives and negatives, ie any pair of cases with identical
#   feature values but different outcomes.
counts = np.ones(y.shape)
keep = np.ones(y.shape)
for i in range(len(y)):
    if keep[i] > 0:
        this = x[i, :]
        dups = np.where(np.logical_and.reduce((x[:, 0] == this[0], x[:, 1] == this[1],
                                               keep == 1)))[0]
        counts[i] = 10*len(dups) + np.sum(y[dups])
        if len(dups) > 1:
            currentLabel = y[dups[-1]]
            if  y[dups[0]] == currentLabel:
                keep[dups[0]] == 0
            if y[dups[0]] != currentLabel:  # If a pair of opposite labels, remove both.
                keep[dups[0]] = 0
                keep[dups[-1]] = 0
x = x[keep == 1, :]
y = y[keep == 1]

#%%
#Logistic:
model = LogisticRegression(class_weight='balanced')
model = model.fit(x, y)
logProbs = model.predict_proba(x)
logProbs = logProbs[:, 1]
logYHat = model.predict(x)

logRawAcc = int(np.round(100* len(np.where(y == logYHat)[0]) / len(y)))
logThresh = 0.5

print('logistic model: coefs for [' + xLabelStr + ', P90] = ' + str(np.round(model.coef_, 3))
      + ', intercept = ' + str(np.round(model.intercept_, 3)))

#%%
#  Random forest:
rfModel = RandomForestClassifier(max_depth=4, class_weight='balanced')
rfModel = rfModel.fit(x, y)
rfProbs = rfModel.predict_proba(x)
rfProbs = rfProbs[:, 1]
rfYHat = rfModel.predict(x)

rfRawAcc = int(np.round(100*len(np.where(rfYHat == y)[0]) / len(y)))
rfThresh = 0.5

#%%
# Plot true classes, logistic estimates, and RF estimates:
comparePlotsFunction(rfYHat, logYHat, y, xLabelStr)

'''
MIT license:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
