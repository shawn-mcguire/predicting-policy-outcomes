
"""
This is a main script to reproduce results in the paper 'Predicting United States policy outcomes
with Random Forests', by Shawn McGuire and Charles Delahunt.
 
It run a single setup (ie a hyperparameter and feature set combo). It will loop through each Policy
Domain that is specified (or just once, if all Policy Domains combined is specified). For each
Policy Domain, it will train a Random Forest (or xgBoost) model, and optionally a Logistic model
and a Neural Net model, on the cases within that Policy Domain. For each model, it repeats the
training for the specified number of runs to collect accuracy statistics. Many plots and console
printouts are optional. Results are saved to .csv file. 
 
Copyright (c) 2020 Charles B. Delahunt.  delahunt@uw.edu
MIT License

""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

# import support functions:
from support_functions import plotFeatureStatsFunction, generateSetupStringFunction, \
generateTrainTestSplitFunction, parseDataframeFunction, calculateAccuracyStatsOnTestSetFunction, \
calculatePrintAndSaveModelStatsOverAllRunsFunction

"""
Note re plot window location:
1. To print plots in separate windows, run the following on the command line:
%matplotlib qt
2. To revert back to plots appearing in console:
%matplotlib inline
"""
#%%

"""
USER ENTRIES:
These are divided into entries you might modify, and entries (created for various experiments) that
you are unlikely to modify. So in most cases only the first set of entries will be relevant.
"""

''' 1. Parameters that might be modified: '''
numRuns = 3    # train this many models, to get accuracy statistics
saveResultsStub = 'resultsOneSetup_27july2020' # becomes part of filename

startYearForTest = 2005  # 2005 -> random train/test split from all data.
#   1997 -> test set is all samples >= 1997 (retrodiction)
runLogisticRegressionFlag = True  # True -> run logistic regression (as in Gilen's model) in
#   addition to Random Forest. Random Forest is always run.
oneModelPerPolicyDomainFlag = False  # To select most salient IGs for each policy domain, use True.
#--------------------------------------------
# Interest group features:
useIGNAFlag = False  # True -> use netIGA as feature. False -> do not use.
# Some of the next few conditions have logical dependencies, eg they are mutually exclusive. Such
#   cases are called out before training begins.
useIndividualInterestGroupsFlag = True  # Must be True for the next three flags to be relevant.
useAllInterestGroupsFlag = False
useRfChosenInterestGroupsFlag = True  # True -> use RF-chosen, False -> use
#   logistic-chosen.
useShortInterestGroupListFlag = False  # True -> use short list, False AND useAllIGsList == False ->
#    use medium list.
# A list of Interest Groups and their column numbers are given at the end of this script.
#-------------------------------------
# policy domain feature:
usePolicyDomainsFlag = False  # True -> use policy domains as features.
# NOTE: this script does not allow for using Policy Areas as features (this is
#   an artifact, and could be changed).
#---------------------------------------------
# Economic class features:
use90PercentFlag = True
use50PercentFlag = False
use10PercentFlag = False
use90Minus10PercentFlag = False
#----------------------------------------------------
# Output and plotting flags:
printDifferenceInRfChosenVsLogChosenFeaturesFlag = True #  Table 4 in paper. NOTE: To reproduce
#   these results, use ONLY IGs as features, ie set both usePolicyDomainsFlag and
#   use90PercentFlag = False.
showFeaturePlotsFlag = True   # Happens once per setup. This prints to console (and text file) RF
#   feature importance, IG-outcome correlations, and IG at-bats, and shows a bar plot of RF feature
#   importance. To see the most salient IGs for each policy domain, set this = True and
#   oneModelPerPolicyDomainFlag = True.
numFeaturesToPlot = 44  # 44 -> all IGs and also P90.
showAccuracyPlotsFlag = True  # Plots accuracy results of each run separately. So if numRuns > 1,
#  set = False to prevent figure overload.

# Outputs less likely to be used:
printToConsoleFlag = True # Only used in plotResultsFunction( )
printLogisticBetasFlag = True   # To examine which features logistic regression weights.
showFeatureImportanceVsAtBatsScatterplotFlag = False   # Print scatterplots of importance score vs
#  at-bats. This is voided if 'printLogisticBetasFlag' == False.

printTestCasesPerRunFlag = True
printExtraSetupDetailsFlag = True
plotAccuraciesByThresholdsFlag = False  # ! LOTS OF FIGURES (numRuns x numSetups x numModelTypes)
# ---------- END OF ENTRIES MOST LIKELY TO BE MODIFIED -----------------------
#%%
''' 2. Parameters that can likely be left alone: '''

logisticPenalty = 'l2'  # 'none', 'l1', 'l2'. 'l2' is the default. l1 and l2 return similar top
#   features and scores. 'none' returns wacky feature rankings. So sticking with 'l2' is good.

useBalancedModelFlag = True  # whether to use balanced mode when training.

# Choose whether to use Permutation or Gini impurity for RF feature selection:
usePermutationFeatureRankingFlag = False    # False -> use Gini impurity for choosing RF features.
#   True -> use permutation method for choosing RF features, and substitute these IGs in for the
#   usual RF-chosen SHORT list (ie only a short list is defined for permutation features). Gini
#   features work best.
useJointInsteadOfRFIGListsFlag = False  # A one-off, can be ignored. Using combined (union of
#   rf-chosen and log-chosen) features gives worse results than RF-chosen features.

trainFractionDefault = 0.67
ignoreGunCasesFlag = False
ignoreMiscCasesFlag = False  # these are cases that do not fit into the five main policy domains
#   (ie 'Miscellaneous').
xgBoostFlag = False  # True -> use xgBoost instead of RF. False -> standard RF.
runNeuralNetFlag = False  # True -> train a NN in addition to a RF.

# Flags to focus on cases where rich and poor disagree:
disagreementThreshold = 0.05  # To distinguish when 90 and 10 really disagree.
restrictTrainToHighDisagreementFlag = False  # True -> only train on cases where |90-10| > 0.05.
restrictTestToHighDisagreementFlag = False  # True -> only test on cases where |90-10| > 0.05.

# Flags about how thresholds (for test set use) are picked on training sets:
useSimpleMaxForThreshFlag = True  # False -> use a Bayesian-flavored version. See def
#   calculateAccuracyStatsOnTestSetFunction(  )
uncertaintyValue = 0.02  # used when 'useSimpleMaxForThreshFlag' = False.

weightFeatureOutcomeCorrelationsByAtBatsOnly = True  # True -> feature correlation with outcomes is
#   normalized by number of at-bats, not number of total cases.
numInLinspace = 101  # For roc and accuracy curves.
#-------------------------------------------------------------------
# Specialized model parameters (for NN, RF, and xgBoost):
# RF AND xgboost parameters:
maxDepth = 4  # In general try 4. Try 13 when using all interest
#    groups on all data (ie one model for combined domains).
numTrees = 200
# XG Boost parameters:
numRoundsBoosting = 50  # Number of rounds of boosting.
etaForBoosting = 1
# NeuralNet params:
hiddenLayers = (50, 50)  # Try (50,50) for all cases.

#---------------------------------------------------------------
''' 3. Interest group subsets, that were determined to be most salient by
training models and doing feature selection on individual Policy Domains.
Note: Larger gaps between entries in a dict value denote different tiers of
importance (eg most important group vs 2nd most important group). '''

# 1. RF-chosen via Gini impurity:
# Medium subset:
rfChosenInterestGroupsToKeepList = \
{'ECYN':[36, 38, 51, 33], 'FPYN':[29,   20, 52, 24,   46, 14, 33, 13],
 'SWYN':[12, 49,   42, 47, 53, 40, 31],'RLYN':[45, 26, 22,   40, 39, 51],
 'GNYN':[44], 'MISC':[25,   13, 24,   18, 46,   23, 26]}
# Smaller subset:
rfChosenInterestGroupsToKeepListShort = \
{'ECYN':[36, 38], 'FPYN':[29, 20, 52, 24], 'SWYN':[12, 49],'RLYN':[26, 45, 22],
 'GNYN':[44], 'MISC':[25]}
# 2. RF-chosen via Permutation method. The permutation method was unstable, giving different feature
#   sets depending on whether P90 was included as a feature. There are two choices, one of which is
#   commented out.
# Either using only IGs (no P90):
rfPermutationChosenIGsToKeepListShort = \
{'ECYN':[37, 36, 33],'FPYN':[29,   20], 'SWYN':[12,   26, 47],'RLYN':[45],
 'GNYN':[44], 'MISC':[23, 25, 46]}
# Or using IGs and P90:
# rfPermutationChosenIGsToKeepListShort = \
#{'ECYN':[37, 25, 19, 47], 'FPYN':[29, 20, 13], 'SWYN':[12, 49 ,26], 'RLYN':[45], 'GNYN':[],
# 'MISC':[23, 46]}
# 3. Logistic-chosen:
# Medium subset:
logisticChosenInterestGroupsToKeepList = \
{'ECYN':[47, 37, 31, 32], 'FPYN':[37, 46, 29, 52,   14, 24], 'SWYN':[49, 23,   13, 54, 15, 36, 40],
 'RLYN':[45, 31, 16,   34, 42, 26], 'GNYN':[44], 'MISC':[23, 35,   36, 17, 51,   45,44, 22]}
# Smaller subset:
logisticChosenInterestGroupsToKeepListShort = \
{'ECYN':[47, 37],'FPYN':[37,  46, 29, 52], 'SWYN':[49, 23,   13], 'RLYN':[45, 31, 16], 'GNYN':[44],
 'MISC':[23, 35]}
# 4. Combined RF- and logistic-chosen IGs (gives significantly worse results than RF-chosen):
# Medium subset:
jointInterestGroupsToKeepList = \
{'ECYN':[36, 38, 51, 33,   47, 37, 31, 32], 'FPYN':[29, 20, 52, 24, 46, 14, 33, 13,   37, 24],
 'SWYN':[12, 49, 42, 47, 53, 40, 31,   23, 13, 54, 15, 36],
 'RLYN':[45, 26, 22, 40, 39, 51,   31, 16, 34, 42], 'GNYN':[44],
 'MISC':[25, 13, 24, 18, 46, 23, 26,   35, 36, 17, 51, 45, 44, 22]}
# Short subset:
jointInterestGroupsToKeepListShort = \
{'ECYN':[36, 38,   47, 37],'FPYN':[29, 20, 52, 24,   37, 46], 'SWYN':[12, 49,   23, 13],
 'RLYN':[26, 45, 22,   45, 31, 16], 'GNYN':[44], 'MISC':[25,   23]}
 
dataFolder = 'data/'

""" END USER ENTRIES """

#%%
""" MAIN: """

''' First some preparation:'''

# Load the main dataset:
dataFull = pd.read_csv(dataFolder + 'gilens_data_sm_copy.csv')
dataFull = dataFull.iloc[0:1836]    # All rows after 1835 are nan.

policyDomainList = ['ECYN', 'FPYN', 'SWYN', 'RLYN', 'GNYN', 'MISC']  # MISC is defined a couple
#   lines down. It contains all cases not in the 5 defined policy groups.
allInterestGroupIndices = np.arange(12, 55)  # Used if selecting a subset of IGs.

# Add a column to the dataframe to mark cases in MISC:
dataFull['MISC'] = np.round(-1*(dataFull.GNYN + dataFull.FPYN + dataFull.SWYN
                                + dataFull.RLYN + dataFull.ECYN - 1))
# Make a dataframe for saving results:
columnNames = ['modelType', 'balanced', 'policyDomain', 'startYearForTest', 'use90', 'use50',
               'use10', 'use90Minus10', 'useIGNA', 'useIndividualIGs', 'useAllIGs',
               'useRfChosenMediumIGs', 'useRfChosenShortIGs', 'useLogChosenMediumIGs',
               'useLogChosenShortIGs', 'usePolicyDomains', 'useOnly90Minus10Disagreements',
               'trainFraction', 'disagreementThreshold', 'maxDepth', 'hiddenLayers',
               'IGSubsetUsed', 'numRuns', 'rawAccTestMeans', 'rawAccTestMedians', 'rawAccTestStds',
               'balAccTestMeans', 'balAccTestMedians', 'balAccStds']
resultsDataFrame = pd.DataFrame(columns=columnNames)

# Complete the names of the files to save:
balStr = 'UNbalanced'
if  useBalancedModelFlag:
    balStr = 'balanced'
combinedStr = 'combinedDomains'
if oneModelPerPolicyDomainFlag:
    combinedStr = 'oneModelPerDomain'
yearStr = str(startYearForTest)

if useJointInsteadOfRFIGListsFlag:
    rfChosenInterestGroupsToKeepList = jointInterestGroupsToKeepList
    rfChosenInterestGroupsToKeepListShort = jointInterestGroupsToKeepListShort
if usePermutationFeatureRankingFlag:
    rfChosenInterestGroupsToKeepListShort = \
    rfPermutationChosenIGsToKeepListShort       # only SHORT list is modified

saveDataframeResultsFilename = \
'resultsDataframe_' + combinedStr + '_' + balStr + '_' + yearStr + '_' \
+ saveResultsStub + '.csv'  # save results to dataframe
saveStringResultsFilename = \
'resultsAsStrings_' + combinedStr + '_' + balStr + '_' + yearStr + '_' \
+ saveResultsStub + '.txt'  # save results as textfile (same as console output)

fid = open(saveStringResultsFilename, 'a')

# correct a dependent flag if necessary:
if oneModelPerPolicyDomainFlag:
    usePolicyDomainsList = [False]

# Initialize arrays to hold model results from each setup. The second index should match or exceed
#   the number of setups being run:
rfRawAccArray = np.zeros([numRuns, 4, len(policyDomainList)])
rfBalAccArray = np.zeros([numRuns, 4, len(policyDomainList)])
rfAucArray = np.zeros([numRuns, 4, len(policyDomainList)])
logRawAccArray = np.zeros([numRuns, 4, len(policyDomainList)])
logBalAccArray = np.zeros([numRuns, 4, len(policyDomainList)])
logAucArray = np.zeros([numRuns, 4, len(policyDomainList)])

# Some setup conditions cause duplication due to logical dependencies. List some conditions to
#   trigger skipping such setups:
condition1 = useAllInterestGroupsFlag and (useRfChosenInterestGroupsFlag
                                           or useShortInterestGroupListFlag)
# Using all interest groups is mutually exclusive with using only subsets.
condition2 = oneModelPerPolicyDomainFlag and usePolicyDomainsFlag  # These are mutually exclusive
#   conditions.
condition3 = not useIndividualInterestGroupsFlag \
and (useAllInterestGroupsFlag or useRfChosenInterestGroupsFlag
     or useShortInterestGroupListFlag) # The three conditions in parentheses can be active only if
#   'useIndividualInterestGroupsFlag' == True.
if condition1:
    print('Note: Use either all interest groups or just subsets as features, but not both.')
if condition2:
    print('Note: Running one model per policy domain makes Policy Domains a meaningless' \
          + ' feature, though resulting models are still valid.')
if condition3:
    print(" Note: 'useIndividualInterestGroupsFlag' must be True for other interest group flags"
          + " to be meaningful. ")

#%%
# Define some variables:

# 1. use the specified interest group list:
if useRfChosenInterestGroupsFlag:
    if useShortInterestGroupListFlag:
        interestGroupsToKeepList = rfChosenInterestGroupsToKeepListShort
    else:
        interestGroupsToKeepList = rfChosenInterestGroupsToKeepList
else:
    if useShortInterestGroupListFlag:
        interestGroupsToKeepList = logisticChosenInterestGroupsToKeepListShort
    else:
        interestGroupsToKeepList = logisticChosenInterestGroupsToKeepList

if useBalancedModelFlag:
    balanceStr = 'balanced'
else:
    balanceStr = None

if startYearForTest < 2003: # Case: retrodiction
    trainFraction = 1  # Use all pre-cutoff date cases for training
else:
    trainFraction = trainFractionDefault   # < 1,  to divide the full dataset into Train/Test.

# 2. Book-keeping: combine all these interest groups for use in one big model:
a = list(interestGroupsToKeepList.values())
b = []
for i in a:
    for j in i:
        b.append(j)
interestGroupsToKeepGroupedByDomain = b
combinedInterestGroupsToKeep = np.unique(b)
if useAllInterestGroupsFlag:
    combinedInterestGroupsToKeep = allInterestGroupIndices
if not useIndividualInterestGroupsFlag:
    combinedInterestGroupsToKeep = []
#%%
# store setup params for future use:
setupParams = \
 {'numRuns':numRuns, 'startYearForTest':startYearForTest, 'trainFraction':trainFraction,
  'oneModelPerPolicyDomainFlag':oneModelPerPolicyDomainFlag, 'ignoreGunCasesFlag':ignoreGunCasesFlag,
  'ignoreMiscCasesFlag':ignoreMiscCasesFlag, 'xgBoostFlag':xgBoostFlag, 'usePolicyDomainsFlag':
  usePolicyDomainsFlag, 'use90PercentFlag':use90PercentFlag, 'use50PercentFlag':use50PercentFlag,
  'use10PercentFlag':use10PercentFlag, 'use90Minus10PercentFlag':use90Minus10PercentFlag,
  'useIndividualInterestGroupsFlag':useIndividualInterestGroupsFlag, 'useAllInterestGroupsFlag':
  useAllInterestGroupsFlag, 'useIGNAFlag':useIGNAFlag, 'restrictTrainToHighDisagreementFlag':
  restrictTrainToHighDisagreementFlag, 'restrictTestToHighDisagreementFlag':
  restrictTestToHighDisagreementFlag, 'maxDepth':maxDepth, 'numTrees':numTrees, 'numRoundsBoosting':
  numRoundsBoosting, 'etaForBoosting':etaForBoosting, 'hiddenLayers':hiddenLayers,
  'weightFeatureOutcomeCorrelationsByAtBatsOnly':weightFeatureOutcomeCorrelationsByAtBatsOnly,
  'useBalancedModelFlag':useBalancedModelFlag, 'useSimpleMaxForThreshFlag':useSimpleMaxForThreshFlag,
  'uncertaintyValue':uncertaintyValue, 'useRfChosenInterestGroupsFlag':useRfChosenInterestGroupsFlag,
  'useShortInterestGroupListFlag':useShortInterestGroupListFlag, 'disagreementThreshold':
  disagreementThreshold, 'policyDomainList':policyDomainList, 'showAccuracyPlotsFlag':
  showAccuracyPlotsFlag, 'printToConsoleFlag':printToConsoleFlag, 'plotAccuraciesByThresholdsFlag':
  plotAccuraciesByThresholdsFlag, 'useOnly90Minus10Disagreements':restrictTestToHighDisagreementFlag,
  'allInterestGroupIndices': allInterestGroupIndices, 'interestGroupsToKeepGroupedByDomain':
  interestGroupsToKeepGroupedByDomain, 'combinedInterestGroupsToKeep':combinedInterestGroupsToKeep}

#%%
# print out a summary of setup details for the model about to run:
setupStr = generateSetupStringFunction(setupParams)
print()
print(setupStr)
fid.write('' + '\n')
fid.write(setupStr + '\n')

#%%
# Prep for looping through models (either one model for combined policy domains
#   or one model per policy domain):
if oneModelPerPolicyDomainFlag:
    numToLoop = len(policyDomainList)  # To prevent repeated plots
else:
    numToLoop = 1  # If one model only for all domains

# Initialize arrays to store results from all runs and all models:
a = np.zeros((numRuns, numToLoop))
randomForestBalAcc = np.copy(a)
randomForestRawAcc = np.copy(a)
randomForestAuc = np.copy(a)
randomForestSens = np.copy(a)
randomForestSpec = np.copy(a)
randomForestPrecision = np.copy(a)
randomForestRecall = np.copy(a)
randomForestF1 = np.copy(a)
xgBoostBalAcc = np.copy(a)
xgBoostRawAcc = np.copy(a)
xgBoostAuc = np.copy(a)
xgBoostSens = np.copy(a)
xgBoostSpec = np.copy(a)
xgBoostPrecision = np.copy(a)
xgBoostRecall = np.copy(a)
xgBoostF1 = np.copy(a)
xgBoostF1Raw = np.copy(a)
logisticBalAcc = np.copy(a)
logisticRawAcc = np.copy(a)
logisticAuc = np.copy(a)
logisticSens = np.copy(a)
logisticSpec = np.copy(a)
logisticPrecision = np.copy(a)
logisticRecall = np.copy(a)
logisticF1 = np.copy(a)
nNetBalAcc = np.copy(a)
nNetRawAcc = np.copy(a)
nNetAuc = np.copy(a)
nNetSens = np.copy(a)
nNetSpec = np.copy(a)
nNetPrecision = np.copy(a)
nNetRecall = np.copy(a)
nNetF1 = np.copy(a)
acceptTestResultsMatrix = np.ones(a.shape, dtype=bool)  # To mark invalid train/test splits

#%%
# If this is the first Setup, generate train/test splits for each run-ind pair. For each run, the
#   same split will get used for all model types.
dataTrainBooleanArray =  \
[[None for j in range(numToLoop)]  for i in range(numRuns)]  # Makes an array numRuns x numToLoop.
dataTestBooleanArray =  \
[[None for j in range(numToLoop)]  for i in range(numRuns)]

for ind in range(numToLoop):
    for run in range(numRuns):
        if oneModelPerPolicyDomainFlag:
            policyDomainsToKeep = policyDomainList[ind]
        else:
            policyDomainsToKeep = policyDomainList
        setupParams['policyDomainsToKeep'] = policyDomainsToKeep  # Temporary, to make the
        #  domain-specific train/test splits here. We reassign this later.
        dataTrainBoolean, dataTestBoolean = \
             generateTrainTestSplitFunction(dataFull, setupParams)
        dataTrainBooleanArray[run][ind] = dataTrainBoolean
        dataTestBooleanArray[run][ind] = dataTestBoolean

#%%
''' Preparation is done. Now run the model(s):'''

for ind in range(numToLoop):  
    # 'ind' indexes the policy domains we are running over. This is *not* the Setup, which is the 
    # selection of all parameters including features). Either ind = 0 (if one model for policy 
    # combined domains) or ind = 0 (economic), 1 (foreign), 2 (social welfare), 3 (religion),
    # 4 (guns), or 5 (misc). Note that for each 'ind', we may train multiple models (certainly
    # RF or xgBoost, and maybe Logistic and NN).

    # Initialize some matrices for this model:
    featureScoresAllRuns = np.zeros([100, numRuns])  # Overshoot number of features (since currently
    #  unknown) when initializing.
    featureOutcomeCorrelationsAllRuns = np.zeros([100, numRuns])
    lregCoeffs = np.zeros([100, numRuns])  # To hold logistic beta coeffs.
    numAtBatsAllRuns = np.zeros([100, numRuns])
    numTrainCasesAllRuns = np.zeros([numRuns, 1])
    numTestCasesAllRuns = np.zeros([numRuns, 1])
    acceptTestResultsAllRuns = np.ones([numRuns, 1], dtype=bool)  # This will mark whether a run had
    #  Both pass and fail cases in the test set. Make a flag for extracting the cases in this
    #  domain. Do this here since it applies to all runs:
    if oneModelPerPolicyDomainFlag:
        policyDomainsToKeep = policyDomainList[ind]  # Case: one policy domain.
    else:
        policyDomainsToKeep = str(policyDomainList)  # Only for print to console.
#        policyDomainsToKeep = policyDomainsToKeep.replace('[','')
#        policyDomainsToKeep = policyDomainsToKeep.replace(']','')
#        policyDomainsToKeep = policyDomainsToKeep.replace("'",'')

    # Define the set of interest groups to use as features:
    if useAllInterestGroupsFlag:
        interestGroupsToKeep = allInterestGroupIndices
    else: # Use a strict subset of interest groups:
        if oneModelPerPolicyDomainFlag:
            interestGroupsToKeep = interestGroupsToKeepList[policyDomainsToKeep]
        else:  # Case: only one model, so combine all relevant interest groups:
            interestGroupsToKeep = combinedInterestGroupsToKeep
    if not useIndividualInterestGroupsFlag:
        interestGroupsToKeep = []
    setupParams['policyDomainsToKeep'] = policyDomainsToKeep
    setupParams['interestGroupsToKeep'] = interestGroupsToKeep

    #%% Do many runs and save results to the above arrays:
    for run in range(numRuns):
        #%%
        """ Load and parse data"""

        # load the relevant train/test split:
        dataTrainBoolean = dataTrainBooleanArray[run][ind]
        dataTestBoolean = dataTestBooleanArray[run][ind]
#
        # Create a dataframe with the correct columns:
        data = parseDataframeFunction(dataFull, setupParams)

        # Create feature and label arrays:
        dataTrain = data.loc[dataTrainBoolean]
        dataTest = data.loc[dataTestBoolean]

        train_features = dataTrain.drop('Binary Outcome', axis=1).values
        train_labels = dataTrain['Binary Outcome'].values
        test_features = dataTest.drop('Binary Outcome', axis=1).values
        test_labels = dataTest['Binary Outcome'].values

        # misc book-keeping:
        temp = dataTrain.drop('Binary Outcome', axis=1)  # Used as an intermediate var here; has to
        #  be a dataframe
        featureNames = list(temp.head(0))
        numFeaturesToPlot = min(numFeaturesToPlot, len(featureNames))
        # Reduce this param value if the original value exceeds the total number of features.

        # Check num of cases of each type:
        numPosTestCases = np.sum(test_labels == 1)
        numNegTestCases = np.sum(test_labels == 0)
        numNegTrainCases = np.sum(train_labels == 0)
        numPosTrainCases = np.sum(train_labels == 1)
        numTestCasesAllRuns[run] = numPosTestCases + numNegTestCases
        acceptTestResultsAllRuns[run] = numPosTestCases > 0 and numNegTestCases > 0   # False if
        # there are 0 pos or 0 neg test cases.

        # Print warnings re bad train/test splits:
        if numPosTestCases == 0:
            print(policyDomainsToKeep + ', run ' + str(run)
                  + ' Test has 0 pos cases. Excluding this run.')
            fid.write(policyDomainsToKeep + ', run ' + str(run)
                      + ' Test has 0 pos cases. Excluding this run.' + '\n')
        if numNegTestCases == 0:
            print(policyDomainsToKeep + ', run ' + str(run)
                  + ' Test has 0 neg cases. Excluding this run.')
            fid.write(policyDomainsToKeep + ', run ' + str(run)
                      + ' Test has 0 neg cases. Excluding this run.' + '\n')

        # Record number of at-bats in test set, per interest group:
        for i in range(len(featureNames)):
            if 'pred' in featureNames[i]:
                numAtBatsAllRuns[i, run] = \
                np.sum(np.abs(dataTest[featureNames[i]].values  - 0.5) > 0.1)  # For voter
                #  preferences, consider 0.4 to 0.6 as 'neutral'.
            else:
                numAtBatsAllRuns[i, run] = np.sum(dataTest[featureNames[i]].values != 0)

        #%%
        ''' Always train a random forest model '''

        ''' 1. standard random forest: '''
        if not xgBoostFlag:
            forestType = 'Random forest'
            modelStr = 'randomForest'  # Different capitalization is artifact.

            # Instantiate model and train
            clf = RandomForestClassifier(n_estimators=numTrees, max_depth=maxDepth,
                                         class_weight=balanceStr)

            # Train the model for use on test set
            clf.fit(train_features, train_labels)
            # Run test set through model:
            testProbScores = clf.predict_proba(test_features)  # This returns two columns of
            #  probabiliies
            trainProbScores = clf.predict_proba(train_features)  # For setting thresholds to apply
            #  to test set

            # Record feature importances:
            if usePermutationFeatureRankingFlag:
                # CAUTION: This method is very noisy:
                # 1. We'll need the original model balanced acc:
                # a. Either use variable threshold:
                fullModelAccStats = calculateAccuracyStatsOnTestSetFunction \
                 (trainProbScores, testProbScores, train_labels, test_labels, setupParams, modelStr,
                  fid)
                # b. Or fixed 0.5 threshold:
                testYhat = clf.predict(test_features)
                tp = np.sum(np.logical_and(testYhat == 1, test_labels == 1))
                fn = np.sum(np.logical_and(testYhat == 0, test_labels == 1))
                tn = np.sum(np.logical_and(testYhat == 0, test_labels == 0))
                fp = np.sum(np.logical_and(testYhat == 1, test_labels == 0))
                fullTestBalAcc = 0.5*(tp/(tp+fn) + tn/(tn + fp))

                # 2. Now mix up the values of each feature in turn, then calculate the loss in
                #  accuracy:
                for i in range(len(featureNames)):
                    X = test_features.copy()
                    np.random.shuffle(X[:, i])
                    # goes with (a):
                    thisPermutedSetProbScores = clf.predict_proba(X)
                    thisAccStats = \
                     calculateAccuracyStatsOnTestSetFunction(trainProbScores,
                                                             thisPermutedSetProbScores, train_labels,
                                                             test_labels, setupParams, modelStr,
                                                             fid)
                    featureScoresAllRuns[i, run] = \
                    (fullModelAccStats['balancedAccTest'] - thisAccStats['balancedAccTest']) \
                    / fullModelAccStats['balancedAccTest']

            else:  # Just use the built-in gini impurity
                featureScoresAllRuns[range(len(featureNames)), run] = clf.feature_importances_

            # Eliminate surplus rows:
            featureScoresAllRuns = featureScoresAllRuns[range(len(featureNames)), :]

        ''' 2. xgboost: '''
        if xgBoostFlag:
            forestType = 'xgBoost'
            modelStr = 'xgBoost'

            datasets.dump_svmlight_file(train_features, train_labels, 'xgbTrain.txt')
            xgbTrain = xgb.DMatrix('xgbTrain.txt', silent=True)
            datasets.dump_svmlight_file(test_features, test_labels, 'xgbTest.txt')
            xgbTest = xgb.DMatrix('xgbTest.txt', silent=True)

            # define cost:
            if useBalancedModelFlag:
                scale_pos_weight = numNegTrainCases/numPosTrainCases
            else:
                scale_pos_weight = 1

            xgbParam = {'max_depth':maxDepth, 'eta':etaForBoosting, 'objective':'binary:logistic',
                        'booster':'gblinear', 'scale_pos_weight': scale_pos_weight}
            bst = xgb.train(xgbParam, xgbTrain, numRoundsBoosting)
            # make predictions:
            xgbTestOutput = bst.predict(xgbTest)
            testProbScores = np.zeros((len(xgbTestOutput), 2))
            testProbScores[:, 1] = xgbTestOutput
            testProbScores[:, 0] = 1 - xgbTestOutput
            # predictions on training set, to find a threshold:
            xgbTrainOutput = bst.predict(xgbTrain)
            trainProbScores = np.zeros((len(xgbTrainOutput), 2))
            trainProbScores[:, 1] = xgbTrainOutput
            trainProbScores[:, 0] = 1 - xgbTrainOutput
        # End of xgBoost-specific section.

        # Calculate test set accuracy: ('aS' = accuracy stats). This function includes possible roc,
        #  balAcc, sens, spec plots.
        aS = calculateAccuracyStatsOnTestSetFunction(trainProbScores, testProbScores, train_labels,
                                                     test_labels, setupParams, modelStr, fid)

        # Save to numRuns array:
        if xgBoostFlag:
            xgBoostBalAcc[run, ind] = aS['balancedAccTest']
            xgBoostAuc[run, ind] = aS['aucScore']
            xgBoostSens[run, ind] = aS['sensTest']
            xgBoostSpec[run, ind] = aS['specTest']
            xgBoostPrecision[run, ind] = aS['precisionTest']
            xgBoostRecall[run, ind] = aS['sensTest']
            xgBoostF1[run, ind] = 2*aS['sensTest']*aS['specTest'] \
            / (aS['sensTest'] + aS['specTest'])
            xgBoostRawAcc[run, ind] = aS['rawAccTest']
        else:
            randomForestBalAcc[run, ind] = aS['balancedAccTest']
            randomForestAuc[run, ind] = aS['aucScore']
            randomForestSens[run, ind] = aS['sensTest']
            randomForestSpec[run, ind] = aS['specTest']
            randomForestPrecision[run, ind] = aS['precisionTest']
            randomForestRecall[run, ind] = aS['sensTest']
            randomForestF1[run, ind] = 2*aS['sensTest']*aS['specTest'] \
            / (aS['sensTest'] + aS['specTest'])
            randomForestRawAcc[run, ind] = aS['rawAccTest']

        #%%
        ''' Train a Neural Net '''

        if runNeuralNetFlag:
            neuralNet = MLPClassifier(hiddenLayers, activation='relu', max_iter=5000,
                                      early_stopping=True)
            neuralNet = neuralNet.fit(train_features, train_labels)
            testProbScoresNN = neuralNet.predict_proba(test_features)
            trainProbScoresNN = neuralNet.predict_proba(train_features)  # For setting thresholds to
            #  apply to test set

            # Calculate test set accuracy: ('aS' = accuracy stats). Includes possible roc, balAcc,
            #  sens, spec plots.
            aS = calculateAccuracyStatsOnTestSetFunction(trainProbScoresNN, testProbScoresNN,
                                                         train_labels, test_labels, setupParams,
                                                         'neuralNet', fid)

            # save to numRuns array:
            nNetBalAcc[run, ind] = aS['balancedAccTest']
            nNetRawAcc[run, ind] = aS['rawAccTest']
            nNetAuc[run, ind] = aS['aucScore']
            nNetSens[run, ind] = aS['sensTest']
            nNetSpec[run, ind] = aS['specTest']
            nNetPrecision[run, ind] = aS['precisionTest']
            nNetRecall[run, ind] = aS['sensTest']
            nNetF1[run, ind] = 2*aS['sensTest']*aS['specTest'] \
            / (aS['sensTest'] + aS['specTest'])

        #%%
        ''' Train a logistic regression model (ie Gilens method) '''
        if runLogisticRegressionFlag:
            # Instantiate and train model:
            if logisticPenalty == 'none':
                lreg = LogisticRegression(class_weight=balanceStr, penalty=logisticPenalty,
                                          solver='lbfgs')  # C = 1, fit_intercept = True
            else:
                lreg = LogisticRegression(class_weight=balanceStr, penalty=logisticPenalty)

            # Create a feature scaler and fit it to the training data:
            scaler = StandardScaler()
            trainFeaturesScaled = scaler.fit_transform(train_features)
            testFeaturesScaled = scaler.transform(test_features)

            lreg = lreg.fit(trainFeaturesScaled, train_labels)

            # Run test set througgh model:
            testProbScoresLreg = lreg.predict_proba(testFeaturesScaled)
            trainProbScoresLreg = lreg.predict_proba(trainFeaturesScaled)

            # Calculate test set accuracy: ('aS' = accuracy stats). Includes possible roc, balAcc,
            #  sens, spec plots.
            aS = calculateAccuracyStatsOnTestSetFunction(trainProbScoresLreg, testProbScoresLreg,
                                                         train_labels, test_labels, setupParams,
                                                         'logistic', fid)

            # Save to numRuns array:
            logisticBalAcc[run, ind] = aS['balancedAccTest']
            logisticRawAcc[run, ind] = aS['rawAccTest']
            logisticAuc[run, ind] = aS['aucScore']
            logisticSens[run, ind] = aS['sensTest']
            logisticSpec[run, ind] = aS['specTest']
            logisticPrecision[run, ind] = aS['precisionTest']
            logisticRecall[run, ind] = aS['sensTest']
            logisticF1[run, ind] = 2*aS['sensTest']*aS['specTest'] \
            / (aS['sensTest'] + aS['specTest'])

            # Save betas:
            betas = lreg.coef_[0]
            lregCoeffs[range(len(betas)), run] = betas.transpose()

        #%%
        # See whether each interest group's correlation with the test set outcomes is positive or
        #  negative. Note: this calculation will not be meaningful for policy domain features.
        for i in range(len(featureNames)):
            temp = np.copy(test_labels)
            temp[np.where(temp == 0)[0]] = -1  # Change labels to -1 and 1 to allow inner product
            #  with preferences.
            pref = test_features[:, i]  # This feature's preferences.
            # For pred90 etc, set the middle region (0.4 to 0.6) to 0 and scale to [-2, 2] to match
            #  interest groups.
            if 'pred' in featureNames[i]:
                pref = (pref - 0.5)*4
                pref = np.maximum(pref, -2)  # Only needed if multiplier > 4.
                pref = np.minimum(pref, 2)
                pref[np.where(np.logical_and(pref > -0.4, pref < 0.4))[0]] = 0
            numberAtBats = np.sum(pref != 0)  # For weighting correlation score by at-bats only
            #  rather than all cases
            if weightFeatureOutcomeCorrelationsByAtBatsOnly:
                denom = 2*numberAtBats
            else:
                denom = 2*len(test_labels)
            featureOutcomeCorrelationsAllRuns[i, run] = sum(np.multiply(pref, temp)) / denom

    ''' End of loop through numRuns  '''

    # Save the RF accuracies to the 3-D array: (the second dimension is a carry-over (deadwood here)
    #  from the script that loops over many models + feature sets).
    rfRawAccArray[:, 0, ind] = (randomForestRawAcc[:, ind]).flatten()
    rfBalAccArray[:, 0, ind] = (randomForestBalAcc[:, ind]).flatten()
    rfAucArray[:, 0, ind] = (randomForestAuc[:, ind]).flatten()
    # save the logistic accuracies similarly:
    logRawAccArray[:, 0, ind] = (logisticRawAcc[:, ind]).flatten()
    logBalAccArray[:, 0, ind] = (logisticBalAcc[:, ind]).flatten()
    logAucArray[:, 0, ind] = (logisticAuc[:, ind]).flatten()

    # Print number of test cases to console:
    testCasesMean = np.round(np.mean(numTestCasesAllRuns), 1)
    testCasesStd = np.round(np.std(numTestCasesAllRuns), 1)
    if printTestCasesPerRunFlag:
        print(str(testCasesMean) + ' +/-' + str(testCasesStd) + ' test cases per run')
        fid.write(str(testCasesMean) + ' +/-' + str(testCasesStd) + ' test cases per run' + '\n')
    # 3. Feature importance and feature-outcome correlation plots:
    if showFeaturePlotsFlag and not xgBoostFlag:  # Because we do not have xgBoost feature scores.
        # Cut off the surplus placeholders in featureScores:
        featureScoresAllRuns = featureScoresAllRuns[range(len(featureNames)), :]
        plotFeatureStatsFunction(featureScoresAllRuns, featureOutcomeCorrelationsAllRuns,
                                 numAtBatsAllRuns, featureNames, str(policyDomainsToKeep),
                                 numFeaturesToPlot, testCasesMean, fid)

    # 4. Print out logistic betas, mean and std:
    lregCoeffs = lregCoeffs[range(len(featureNames))]
    if runLogisticRegressionFlag and printLogisticBetasFlag:
        betaMeans = np.round(np.mean(lregCoeffs, axis=1), 3)
        betaStds = np.round(np.std(lregCoeffs, axis=1), 3)
        betaFoms = np.round(np.divide(np.abs(betaMeans), betaStds), 3)  # Use abs(mean)/std as FoM.
        # remove nans:
        betaFoms[np.where(np.isnan(betaFoms))[0]] = 0
        sortedBetaFomIndices = np.flip(np.argsort(betaFoms))
        sortedBetaFoms = betaFoms[sortedBetaFomIndices]
        print('')
        print('logistic regression betas for ' + str(policyDomainsToKeep) + ':')
        for i in range(numFeaturesToPlot):
            print(str(sortedBetaFoms[i]) + '= mu/sigma, ' + str(betaMeans[sortedBetaFomIndices[i]]) \
                  + ' +/- ' + str(betaStds[sortedBetaFomIndices[i]]) + '    '
                  + featureNames[sortedBetaFomIndices[i]])
        fid.write('' + '\n')
        fid.write('logistic regression betas for ' + str(policyDomainsToKeep) + ':'  + '\n')
        for i in range(numFeaturesToPlot):
            fid.write(str(sortedBetaFoms[i]) + '= mu/sigma, ' \
                      + str(betaMeans[sortedBetaFomIndices[i]]) + ' +/- ' \
                      + str(betaStds[sortedBetaFomIndices[i]]) + '    ' \
                      + featureNames[sortedBetaFomIndices[i]]  + '\n')

        # Scatterplot logistic mean feature score and also RF feature score vs number of at-bats for
        #  each IG:
        # We need numAtBats:
        atBatMean = np.mean(numAtBatsAllRuns, axis=1)
        atBatMean = atBatMean[:len(betaMeans)]
        # scale the importance scores so they are comparable:
        scaledRfFeatureMeans = np.mean(featureScoresAllRuns, axis=1)
        scaledRfFeatureMeans = scaledRfFeatureMeans / np.max(scaledRfFeatureMeans)
        scaledBetaMeans = np.abs(betaMeans) / np.max(np.abs(betaMeans))

        if startYearForTest >= 2003 and not xgBoostFlag:  # Since we have no xgBoost feature scores.
            # Calculate linear fits, to plot these:
            r = scaledRfFeatureMeans.reshape(-1, 1)  # To make shape [n, 1] instead of [n,]
            lo = scaledBetaMeans.reshape(-1, 1)
            ab = atBatMean.reshape(-1, 1)
            # use only IGs with >0 at-bats:
            nonZeroAtBatInds = np.where(ab > 1)[0]
            r = r[nonZeroAtBatInds]
            lo = lo[nonZeroAtBatInds]
            ab = ab[nonZeroAtBatInds]

            lin = LinearRegression()
            lin = lin.fit(ab, r)
            mRF = lin.coef_[0]
            bRF = lin.intercept_[0]

            # calculate R^2 scores
            r2RF = metrics.r2_score(r, bRF + mRF*ab)
            r2RF = np.round(r2RF, 2)

            lin = lin.fit(ab, lo)
            mLog = lin.coef_[0]
            bLog = lin.intercept_[0]
            # calculate R^2 scores
            r2Log = metrics.r2_score(lo, bLog + mLog*ab)
            r2Log = np.round(r2Log, 2)

            if showFeatureImportanceVsAtBatsScatterplotFlag:
                plt.figure()
                plt.plot(atBatMean, scaledBetaMeans, 'r.', markersize=12, label='Logistic')
                plt.plot(atBatMean, scaledRfFeatureMeans, 'b.', markersize=12, label='RF')
                plt.legend(loc='lower right', prop={'size':12, 'weight':'bold'})
                plt.xlabel('# at-bats', fontsize=14, fontweight='bold')
                plt.ylabel('importance score', fontsize=14, fontweight='bold')
                plt.title(str(policyDomainsToKeep) +' feature importances, ' +  '. RF R^2 = ' \
                          + str(r2RF) + ', logistic R^2 = ' + str(r2Log),
                          fontsize=14, fontweight='bold')
                plt.xticks(size=12, weight='bold')
                plt.yticks(size=12, weight='bold')
                plt.xlim([0, testCasesMean])
                plt.ylim([0, 1.05])
                plt.grid(b=False)
                # Plot linear fits:
                xAxisVals = [0, int(np.max(ab))]
                plt.plot(xAxisVals, bLog + mLog*xAxisVals, 'r:')
                plt.plot(xAxisVals, bRF + mRF*xAxisVals, 'b:')
                plt.show()

    # Populate acceptableTestResultsMatrix for this policy domain:
    acceptTestResultsMatrix[:, ind] = acceptTestResultsAllRuns.transpose()

''' End of loop through policy domains '''

#%%
''' print out statistics of results to console: '''

# For each model type, calc stats and put them into a string:
# Note that *BalAcc, *Auc etc have one column per model. If oneModelPerPolicyDomainFlag == True,
#   then col 0 to 4 = ECYN, FPYN, SWYN, RLYN, GNYN.

# 1. Random Forest XOR xgBoost:
if not xgBoostFlag:
    accResultDict = {'rawAcc':randomForestRawAcc, 'balAcc':randomForestBalAcc,
                     'AUC': randomForestAuc, 'prec': randomForestPrecision,
                     'recall': randomForestRecall, 'f1': randomForestF1}
else:
    accResultDict = {'rawAcc':xgBoostRawAcc, 'balAcc':xgBoostBalAcc, 'AUC': xgBoostAuc,
                     'prec': xgBoostPrecision, 'recall': xgBoostRecall, 'f1': xgBoostF1}

if xgBoostFlag:
    typeStr = 'xgBoost'
else:
    typeStr = 'random forest'
if useBalancedModelFlag:
    typeStr = typeStr + '_balanced'
else:
    typeStr = typeStr + '_UNbalanced'

resultsDataFrame = \
calculatePrintAndSaveModelStatsOverAllRunsFunction(accResultDict, acceptTestResultsMatrix, typeStr,
                                                   setupParams, resultsDataFrame, fid)

# 2. logistic:
if runLogisticRegressionFlag:
    accResultDict = {'rawAcc':logisticRawAcc, 'balAcc':logisticBalAcc, 'AUC': logisticAuc, 'prec':
                     logisticPrecision, 'recall': logisticRecall, 'f1': logisticF1}

    typeStr = 'logistic'
    if useBalancedModelFlag:
        typeStr = typeStr + '_balanced'
    else:
        typeStr = typeStr + '_UNbalanced'

    resultsDataFrame = \
    calculatePrintAndSaveModelStatsOverAllRunsFunction(accResultDict, acceptTestResultsMatrix,
                                                       typeStr, setupParams, resultsDataFrame, fid)
    # Also print out difference of RF and logistic:
    rfRa = randomForestRawAcc
    rfBa = randomForestBalAcc
    rfAuc = randomForestAuc
    if xgBoostFlag:
        rfRa = xgBoostRawAcc
        rfBa = xgBoostBalAcc
        rfAuc = xgBoostAuc
    accResultDict = {'rawAcc': rfRa - logisticRawAcc, 'balAcc': rfBa - logisticBalAcc,
                     'AUC': rfAuc - logisticAuc}
    typeStr = 'RF minus logistic'
    resultsDataFrame = \
    calculatePrintAndSaveModelStatsOverAllRunsFunction(accResultDict, acceptTestResultsMatrix,
                                                       typeStr, setupParams, resultsDataFrame, fid)
# 3. Neural net:
if runNeuralNetFlag:
    accResultDict = {'rawAcc':nNetRawAcc, 'balAcc':nNetBalAcc, 'AUC': nNetAuc,
                     'prec': nNetPrecision, 'recall': nNetRecall, 'f1': nNetF1}

    typeStr = 'neuralNet'
    if useBalancedModelFlag:
        typeStr = typeStr + '_balanced'
    else:
        typeStr = typeStr + '_UNbalanced'

    resultsDataFrame = \
    calculatePrintAndSaveModelStatsOverAllRunsFunction(accResultDict, acceptTestResultsMatrix,
                                                       typeStr, setupParams, resultsDataFrame, fid)

resultsDataFrame.to_csv(saveDataframeResultsFilename)

fid.close()

#%%
'''
Interest groups column number key:

AARP = 12
AFL CIO = 13
Airlines = 14
American Bankers Association = 15
American Council of Life Insurance = 16
American Farm Bureau Federation = 17
American Federation of State County and Municipal Employees = 18
American Hospital Association = 19
American Israel Public Affairs Committee = 20
American Legion = 21
American Medical Association = 22
Association of Trial Lawyers = 23
Automobile companies = 24
Chamber of Commerce = 25
Christian Coalition = 26
Computer software and hardware = 27
Credit Union National Association = 28
Defense contractors = 29
Electric companies = 30
Health Insurance Association = 31
Independent Insurance Agents of America = 32
International Brotherhood of Teamsters = 33
Motion Picture Association of America = 34
National Association of Broadcasters = 35
National Association of Home Builders = 36
National Association of Manufacturers = 37
National Association of Realtors = 38
National Beer Wholesalers Association = 39
National Education Association = 40
National Federation of Independent Business = 41
National Governors Association = 42
ational Restaurant Association = 43
National Rifle Association = 44
National Right to Life Committee = 45
Oil Companies = 46
Pharmaceutical Research Manufacturers = 47
Recording Industry Association = 48
Securities and investment companies = 49
Telephone companies = 50
Tobbaco companies = 51
United Auto Workers union = 52
Universities = 53
Veterans of Foreign Wars of the US = 54
    '''
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