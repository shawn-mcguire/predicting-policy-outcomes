""" 
This file contains support function defs for the script 'run_models_main.py'.
 
Copyright (c) 2020 Charles B. Delahunt.  delahunt@uw.edu
MIT License

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
"""
--------------------------------------------------------------
--------------------function defs ----------------------------
--------------------------------------------------------------
"""
#%%

def plotResultsFunction(probScores, testLabels, balancedAcc, sens, spec, params, fid):

    '''
    Plots ROC curves and balanced accuracy plots. let n = number of samples. It is only called from
    within calculateAccuracyStatsOnTestSetFunction().
    Inputs:
        probScores = n x 1 np.array (vector) of floats in [0, 1]
        testLabels = n x 1 np.array (vector) of 0s and 1s
        balancedAcc = n x 1 np.array (vector) of floats in [0, 1]
        sens = n x 1 np.array (vector) of floats in [0, 1]
        spec = n x 1 np.array (vector) of floats in [0, 1]
        params = dict of important parameters
        fid: file id, from open()

    Outputs: (depending on flags in 'params')
        Info printed to console
        ROC plot and balanced accuracy (over many thresholds) plot
        '''

    def makeRocAndAccuracyPlotsSubFunction(fpr, tpr, auc_score, threshIndex, rocTitleStr, sens,
                                           spec, balancedAcc, balAccTitleStr):
        ''' Makes two subplots with ROC curve and Balanced accuracies vs thresholds. Each subplot
        uses half of the many argins.
        Inputs:
            fpr: vector of floats
            tpr: vector of floats
            auc_score: float
            threshIndex: int
            rocTitleStr: str
            sens: vector of floats
            spec: vector of floats
            balancedAcc: vector of floats
            balAccTitleStr: str
        Outputs:
            matplotlib figure
            '''
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # ROC curve subplot:
        ax1.plot(fpr, tpr, color='black', lw=2, linestyle='--', label=' AUC = %0.2f' % auc_score)
        ax1.plot(fpr[threshIndex], tpr[threshIndex], 'ko', markersize=8)
        ax1.legend(loc='lower right')
        ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax1.set_xlabel('1 - specificity')
        ax1.set_ylabel('sensitivity')

        ax1.set_title(rocTitleStr)

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.grid(True)

        # Balanced Accuracy subplot
        x = np.linspace(0, 1, 101)
        ax2.plot(x, sens, color='blue', lw=2, linestyle='-', label='sens')
        ax2.plot(x, spec, color='green', lw=2, linestyle='-', label='spec')
        ax2.plot(x, balancedAcc, color='black', lw=2, linestyle='-', label='balanced acc')
        ax2.plot()
        ax2.set_xlabel('accuracy')
        ax2.set_ylabel('threshold')
        ax2.legend(loc='lower right')

        ax2.set_title(balAccTitleStr)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        ax2.grid(True)
        plt.show()
        # End of makeAccuracyPlotsFunction sub-function.
        #-------------------------------------------------------------

    def printSetupParamsToConsoleAndFileFunction(p, auc_score, fid):
        ''' Print key results and parameters to console and to file.
        Inputs:
            p: dict. 
            auc_score: float.
            fid: file ID.
        Outputs:
            print to console and add lines to a text file.
            '''
        numPosTestCases = p['numPosTestCases']
        numNegTestCases = p['numNegTestCases']

        print('\n' + p['modelType'] + ' model for policy domain = ' + p['policyDomainsToKeep'] \
              + ', startYear ' + str(p['startYearForTest']) + '. hiddenLayers = ' \
              + str(p['hiddenLayers']))
        print('numPos = ' + str(numPosTestCases) + ', numNeg = ' + str(numNegTestCases) \
              + ', total = ' + str(numPosTestCases + numNegTestCases))
        print('AUC = ' + str(np.round(auc_score, 2)) + ', maxBalancedAcc = ' \
              + str(np.round(100*p['maxBalancedAccTrain'], 0)) + ' at thresh = ' \
              + str(np.round(p['threshForMaxBalancedAcc'], 2)))
        fid.write('\n' + p['modelType'] + ' model for policy domain = ' + p['policyDomainsToKeep'] \
                  + ', startYear ' + str(p['startYearForTest']) + '. hiddenLayers = ' \
                  + str(p['hiddenLayers']) + '\n')
        fid.write('numPos = ' + str(numPosTestCases) + ', numNeg = ' + str(numNegTestCases) \
                  + ', total = ' + str(numPosTestCases + numNegTestCases) + '\n')
        fid.write('AUC = ' + str(np.round(auc_score, 2)) + ', maxBalancedAcc = ' \
                  + str(np.round(100*p['maxBalancedAccTrain'], 0)) + ' at thresh = ' \
                  + str(np.round(p['threshForMaxBalancedAcc'], 2)) + '\n')
        # End of printSetupParamsToConsoleAndFileFunction sub-function.
        #-------------------------------------------------------------

    p = params
    numPosTestCases = p['numPosTestCases']
    numNegTestCases = p['numNegTestCases']

    # raw materials for ROC curve:
    fpr, tpr, thresholds = metrics.roc_curve(testLabels, probScores[:, 1], pos_label=1)
    threshIndex = np.where(np.abs(thresholds - p['threshForMaxBalancedAcc']) \
                            == np.min(np.abs(thresholds - p['threshForMaxBalancedAcc'])))[0]
    auc_score = auc(fpr, tpr)
      
    # print stats to console: 
    printSetupParamsToConsoleAndFileFunction(p, auc_score, fid)

    if p['showAccuracyPlotsFlag']:
        # ROC curve:
        rocTitleStr = 'ROC, ' +  p['policyDomainsToKeep'] + ': ' + str(numPosTestCases) + ' Pos, ' \
        + str(numNegTestCases) + ' neg'
        balAccTitleStr = 'balanced accuracy.' + p['modelType'] + ', startYear ' \
        + str(p['startYearForTest'])
        makeRocAndAccuracyPlotsSubFunction(fpr, tpr, auc_score, threshIndex, rocTitleStr, sens,
                                           spec, balancedAcc, balAccTitleStr)

# End of plotResultsFunction.
#----------------------------------------------------------------------

def plotFeatureStatsFunction(featureScoresAllRuns, featureOutcomeCorrelationsAllRuns,
                             numAtBatsAllRuns, featureNames, titleStr, numFeaturesToPlot,
                             numTestCases, fid):

    '''
    Plot top N features' importances (based on score mean, not mu/sigma) and also whether features
    have positive or negative correlations with test set outcomes.
    NOTE: feature correlation only makes sense for interest groups.
    Let r = number of runs, m = number of features.
    Inputs:
        featureScoresAllRuns:m x r np.array of floats in [0, 1]. each row are the importances, over
           each run, of a single feature.
        featureOutcomeCorrelationsAllRuns:m x r np.array of floats.
        numAtBatsAllRuns: m x r np.array of ints.
        featureNames: m x 1 np.array of strings.
        titleStr: str. The policy domain name.
        numFeaturesToPlot: int.
        numTestCases: float.
        fid: file id, from open().
    Outputs:
        matplotlib figure: plot of feature importances.
        print to console and file: feature correlations.
    '''

    def printResultsToConsoleAndFileSubFunction(resultTypeStr, titleStr, numFeaturesToPlot, means,
                                                stds, names, fid):
        ''' Print to console and to file.
            Inputs:
                resultTypeStr: str.
                titleStr: str.
                numFeaturesToPlot: int.
                means: vector of floats.
                stds: vector of floats.
                names: vector of strings.
                fid: file identifier.
            Outputs:
                Print str to console.
                Print str to file.
                '''

        print('\n ' + resultTypeStr + ' for ' + titleStr + ' (mean+std): ')
        fid.write('\n ' + resultTypeStr + ' for ' + titleStr + ' (mean+std): ')
        for i in range(numFeaturesToPlot):
            print(str(np.round(means[i], 3)) + ' +/-' \
                  + str(np.round(stds[i], 3)) + ' : ' + names[i])
            fid.write('\n' + str(np.round(means[i], 3)) + ' +/-' \
                  + str(np.round(stds[i], 3)) + ' : ' + names[i])
    # End of printResultsToConsoleAndFileSubFunction
    #-------------------------------------------------

    featureScoreMeans = np.mean(featureScoresAllRuns, axis=1)
    featureScoreStds = np.std(featureScoresAllRuns, axis=1)
    featureScoreFoms = np.divide(featureScoreMeans, featureScoreStds)  # Not currently used. Can be
    #   used to rank features.
    # Knock out NaNs:
    featureScoreFoms[np.where(np.isnan(featureScoreFoms))[0]] = 0

    atBatMeans = np.mean(numAtBatsAllRuns, axis=1)
    atBatStds = np.std(numAtBatsAllRuns, axis=1)

    # Feature correlations: we want to ignore featureCorrelations that are Nan, because these show
    # there were no at-bats in that test set:
    featureCorrelationsMeans = np.zeros([len(featureNames), 1])
    featureCorrelationsStds = np.zeros([len(featureNames), 1])
    for i in range(len(featureNames)):
        temp = featureOutcomeCorrelationsAllRuns[i, :]
        indsToKeep = temp[np.where(np.logical_not(np.isnan(temp)))[0]]
        featureCorrelationsMeans[i] = np.mean(indsToKeep)
        featureCorrelationsStds[i] = np.std(indsToKeep)

    # Back to feature scores:
    sortedIndices = np.flip(np.argsort(featureScoreMeans))[0:numFeaturesToPlot] # Flip to make
    #  descending
    sortedNames = np.array([featureNames[i] for i in sortedIndices])
    sortedMeans = np.array([featureScoreMeans[i] for i in sortedIndices])
    sortedStds = np.array([featureScoreStds[i] for i in sortedIndices])
    sortedCorrelationsMean = np.array([featureCorrelationsMeans[i] for i in sortedIndices])
    sortedCorrelationsStd = np.array([featureCorrelationsStds[i] for i in sortedIndices])
    sortedAtBatMeans = np.array([atBatMeans[i] for i in sortedIndices])
    sortedAtBatStds = np.array([atBatStds[i] for i in sortedIndices])
    sortedFoms = np.array([featureScoreFoms[i] for i in sortedIndices])

    _, ax = plt.subplots(figsize=[8, 4])
    ax.barh(range(numFeaturesToPlot), sortedMeans, xerr=sortedStds, align='center')
    ax.set_yticks(range(len(sortedMeans)))
    ax.set_yticklabels(sortedNames)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance score')
    ax.set_title(titleStr)
    plt.show()

    # Scatterplot feature score vs number of at-bats for each IG:
    plt.figure()
    plt.plot(atBatMeans[:len(featureScoreMeans)], featureScoreMeans, 'b.', markersize=12)
    plt.xlabel('# at-bats')
    plt.ylabel('feature score')
    plt.title(titleStr)
    plt.xlim([0, numTestCases])
    plt.ylim([0, np.max(featureScoreMeans*1.1)])
    plt.grid(b=True)
    plt.show()

    # Print four things to console and file:
    resultTypeStr = 'Feature importance'
    printResultsToConsoleAndFileSubFunction(resultTypeStr, titleStr, numFeaturesToPlot,
                                            sortedMeans, sortedStds, sortedNames, fid)
    resultTypeStr = 'Figure of merit'
    printResultsToConsoleAndFileSubFunction(resultTypeStr, titleStr, numFeaturesToPlot, sortedFoms,
                                            np.zeros(sortedFoms.shape), sortedNames, fid)
    resultTypeStr = 'Feature outcome correlations'
    printResultsToConsoleAndFileSubFunction(resultTypeStr, titleStr, numFeaturesToPlot,
                                            sortedCorrelationsMean, sortedCorrelationsStd,
                                            sortedNames, fid)
    resultTypeStr = '# at-bats'
    printResultsToConsoleAndFileSubFunction(resultTypeStr, titleStr, numFeaturesToPlot,
                                            sortedAtBatMeans, sortedAtBatStds, sortedNames, fid)

# End of plotFeatureStatsFunction.
#----------------------------------------------------------------------

def generateSetupStringFunction(setupParams):
    '''
    Generate a string that contains setup information (model, features). Strings are constructed
    using booleans from setupParams, so may be empty.

    Inputs:
        setupParams = dict
    Outputs:
        setupStr = string
    '''
    sp = setupParams

    gunStr = ', ignore gun cases' * sp['ignoreGunCasesFlag']
    forestStr = 'Random forest' + '.xgBoost' * sp['xgBoostFlag']
    policyDomainStr = ', policyDomains' * sp['usePolicyDomainsFlag']

    voterPrefsStr = ', 90%ile' * sp['use90PercentFlag'] + ', 50%ile' * sp['use50PercentFlag'] \
    + ', 10%ile' * sp['use10PercentFlag'] + ', 90-10%ile' * sp['use90Minus10PercentFlag']

    interestGroupStr = 'IGNA' * sp['useIGNAFlag']
    if sp['useIndividualInterestGroupsFlag']:
        interestGroupStr = interestGroupStr + ', ' + 'all IGs' * sp['useAllInterestGroupsFlag'] \
        + str(sp['interestGroupsToKeepGroupedByDomain']) * (not sp['useAllInterestGroupsFlag'])

    balancedModelStr = 'UN' * (not sp['useBalancedModelFlag']) + 'balanced model'

    if sp['useIndividualInterestGroupsFlag']:
        igSubset = 'all IGs' * sp['useAllInterestGroupsFlag'] \
        + 'RF-chosen IGs' * sp['useRfChosenInterestGroupsFlag'] \
        + 'logistic-chosen IGs' * (not sp['useRfChosenInterestGroupsFlag']) \
        + 'short ' * sp['useShortInterestGroupListFlag'] \
        + 'medium ' * (not sp['useShortInterestGroupListFlag'])
    else:
        igSubset = 'No individual IGs'

    setupStr = 'numRuns ' + str(sp['numRuns']) + '. startYearForTest = ' \
    + str(sp['startYearForTest']) + ', trainFraction ' + str(sp['trainFraction']) + gunStr + '. ' \
    + forestStr + ', ' + balancedModelStr + ', ' + igSubset + ', maxDepth ' + str(sp['maxDepth']) \
    + '. Features = ' + policyDomainStr + voterPrefsStr + interestGroupStr

    return setupStr

# End of generateSetupStrFunction.
#-------------------------------------------------------------------------------

def generateTrainTestSplitFunction(data, setupParams):
    '''
    Given a dataframe and params, make a train test split.

    Inputs:
        data: pandas dataframe. This should be the full (unparsed) dataframe = dataFull.
        setupParams: dict of params

    Outputs:
        dataTrainBoolean: boolean vector
        dataTestBoolean: boolean vector
    '''

    startYear = setupParams['startYearForTest']

    # drop gun dolicy cases and/or MISC policy cases if wished:
    if setupParams['oneModelPerPolicyDomainFlag']:
        inPolicyTrainBoolean = np.logical_and(data.YEAR < startYear,
                                              data[setupParams['policyDomainsToKeep']] >= 1)
        inPolicyTestBoolean = np.logical_and(data.YEAR >= startYear,
                                             data[setupParams['policyDomainsToKeep']] >= 1)

    else: # Case: Single model for combined policy domains:
        # When combining all (or most) policy domains together, we have several cases depending on
        #  which domains are being rejected:
        if setupParams['ignoreMiscCasesFlag']:
            if setupParams['ignoreGunCasesFlag']:
                inPolicyTrainBoolean = np.logical_and.reduce((data.YEAR < startYear,
                                                              data['MISC'] == 0,
                                                              data['GNYN'] == 0))
                inPolicyTestBoolean = np.logical_and.reduce((data.YEAR >= startYear,
                                                             data['MISC'] == 0, data['GNYN'] == 0))
                # Above (and some places below) can use np.all((a,b,c), axis=0) instead of .reduce.
            else:    # Keep guns
                inPolicyTrainBoolean = np.logical_and.reduce((data.YEAR < startYear,
                                                              data['MISC'] == 0))
                inPolicyTestBoolean = np.logical_and.reduce((data.YEAR >= startYear,
                                                             data['MISC'] == 0))

        else:  # We are keeping MISC cases:
            if setupParams['ignoreGunCasesFlag']:
                inPolicyTrainBoolean = np.logical_and.reduce((data.YEAR < startYear,
                                                              data['GNYN'] == 0))
                inPolicyTestBoolean = np.logical_and.reduce((data.YEAR >= startYear,
                                                             data['GNYN'] == 0))
            else:  # Keep guns
                inPolicyTrainBoolean = (data.YEAR < startYear).values
                inPolicyTestBoolean = (data.YEAR >= startYear).values

    # 'inPolicyTrainID' currently contains all samples before 'startYear'. If startYear > 2003,
    #    then inPolicyTestID is empty. In this case we want to  define dataTrainID and dataTestID
    #    so that they are both randomly drawn from inPolicyTrainID.
    if startYear > 2003:
        dataTrainSubset, dataTestSubset, dummy1, dummy2 = \
        train_test_split(range(len(inPolicyTrainBoolean)), np.ones(inPolicyTrainBoolean.shape),
                         train_size=setupParams['trainFraction'])
        # Replace the existing booleans dataTrainBoolean and dataTestBoolean:
        dataTestBoolean = np.zeros(inPolicyTrainBoolean.shape)
        dataTestBoolean[dataTestSubset] = 1
        dataTestBoolean = dataTestBoolean.astype(bool)
        dataTestBoolean = np.logical_and(dataTestBoolean, inPolicyTrainBoolean) # the AND combines
        #  the 1's which show training set and the 1's which show correct policy domain.
        dataTrainBoolean = np.zeros(inPolicyTrainBoolean.shape)
        dataTrainBoolean[dataTrainSubset] = 1
        dataTrainBoolean = dataTrainBoolean.astype(bool)
        dataTrainBoolean = np.logical_and(dataTrainBoolean, inPolicyTrainBoolean)
    else:   # Case: no fussing is needed
        dataTrainBoolean = inPolicyTrainBoolean
        dataTestBoolean = inPolicyTestBoolean

    # If restricting to high 90-10 disagreement, further restrict dataTrainBoolean and/or
    #  dataTestBoolean:
    disagreement = data['pred90 - pred10'].values
    highDisagreementBoolean = np.abs(disagreement) > setupParams['disagreementThreshold']
    if setupParams['restrictTrainToHighDisagreementFlag']:
        dataTrainBoolean = np.logical_and(dataTrainBoolean, highDisagreementBoolean)
    if setupParams['restrictTestToHighDisagreementFlag']:
        dataTestBoolean = np.logical_and(dataTestBoolean, highDisagreementBoolean)

    return dataTrainBoolean, dataTestBoolean

# End of generateTrainTestSplitFunction.
#-----------------------------------------------------------------------------------------------

def parseDataframeFunction(dataFull, setupParams):
    '''
    Starts with the full dataframe. Removes columns and selects the rows (cases) to keep.

    Inputs:
        dataFull: pandas dataframe to be modified
        setupParams: dict containing the necessary flags

    Outputs:
        data: pandas dataframe
    '''
    sp = setupParams

    data = dataFull  # We'll edit and return 'data'.

    # Select proper rows and drop switcher
    data = data.drop(['switcher'], axis=1)

    # Filter features by dropping columns:
    # 1. Individual interest groups and IntGrpNetAlign:
    interestGroupsToDropColumnIndices = sp['allInterestGroupIndices'] \
    [np.logical_not(np.isin(sp['allInterestGroupIndices'], sp['combinedInterestGroupsToKeep']))]
    if not sp['useIndividualInterestGroupsFlag']:
        interestGroupsToDropColumnIndices = np.arange(12, 55)
    data = data.drop(data.columns[interestGroupsToDropColumnIndices], axis=1) # Individual interest
    #  groups.
    if not sp['useIGNAFlag']:
        data = data.drop(['IntGrpNetAlign'], axis=1)

    # 2. Policy domains:
    data = data.drop(['XL_AREA'], axis=1) # Ignore this column for these experiments. But note that
    #   the best accuracies in the paper were attained using XL_AREA rather than Policy Domain as a
    #   feature.
    if not sp['usePolicyDomainsFlag']:
        data = data.drop(sp['policyDomainList'], axis=1)

    if sp['ignoreMiscCasesFlag']:
        data = data.drop(['MISC'], axis=1)

    # 3. Public opinions:
    if not sp['use90PercentFlag']:
        data = data.drop(['pred90_sw'], axis=1)
    if not sp['use50PercentFlag']:
        data = data.drop(['pred50_sw'], axis=1)
    if not sp['use10PercentFlag']:
        data = data.drop(['pred10_sw'], axis=1)
    if not sp['use90Minus10PercentFlag']:
        data = data.drop(['pred90 - pred10'], axis=1)

    # 4. Get rid of some non-feature columns:
    data = data.drop(['YEAR', 'OutcomeYear'], axis=1)

    return data
#-------------------------------------------------------------------

def calculateAccuracyStatsOnTestSetFunction(trainProbScores, testProbScores, trainLabels,
                                            testLabels, setupParams, modelType, fid):
    '''
    Given train and test prob scores, plus params, calculate various accuracy values.

    Inputs:
        trainProbScores: np vector
        testProbScoresTest: np vector
        trainLabels = np vector
        testLabels = np vector
        setupParams: dict
        modelType: string (eg 'logistic')
        fid: file id, from open()

    Outputs:
        accuracyStats: dict
        Also generated, if flags indicate: balanced acc and auc plots.
    '''
    def chooseBestThresholdsSubFunction(balancedAcc, rawAccTrain, trainLabels, setupParams):
        ''' Find best thresholds for use on the test set, according to training set results.
        There are two methods for choosing test set thresholds.
        Inputs:
            balancedAcc: vector of floats.
            rawAccTrain: vector of floats.
            trainLabels: vector of booleans.
            setupParams: dict.
        Outputs:
            threshForMaxBalancedAcc: float scalar.
            threshForMaxRawAcc: float scalar.
            '''
        x = np.linspace(0, 1, len(balancedAcc))
        if setupParams['useSimpleMaxForThreshFlag']:
            # Method 1 (optimistic, assumes test samples will score as high as train samples.
            #   We assume a positive outcome is being optimized by the RF). Take the threshold
            #   that gives highest training set accuracy:
            # 1. For balanced accuracy:
            maxBalancedAcc = np.max(balancedAcc)
            indexForMax = np.where(balancedAcc == maxBalancedAcc)[0]
            if len(indexForMax) > 1:  # Since 'indexForMax' could be a vector.
                middleInd = int(np.floor(len(indexForMax) / 2))   # Choose the middle index
                indexForMax = indexForMax[middleInd]
                # print('caution: RF/xgBoost returned vector indexForMax: ' + str(indexForMax)
                # + '  (domain = ' + setupParams['policyDomainsToKeep'] + ')')
            threshForMaxBalancedAcc = x[indexForMax]

            # 2. For raw accuracy:
            maxRawAcc = np.max(rawAccTrain)
            indexForMax = np.where(rawAccTrain == maxRawAcc)[0]
            if len(indexForMax) > 1:
                middleInd = int(np.floor(len(indexForMax) / 2))   # Choose the middle index
                indexForMax = indexForMax[middleInd]
                # print('caution: RF/xgBoost returned vector indexForMax: ' + str(indexForMax)
                # + '  (domain = ' + setupParams['policyDomainsToKeep'] + ')')
            threshForMaxRawAcc = x[indexForMax]

        else:
            # Method 2 (considers the top candidates, and adjusts threshold up or down according to
            #  pos:neg balance). We consider all thresholds within 'wobbleFactor' of the top score.
            #  These make a plateau. We choose along the plateau according to pos:neg balance.
            uncertaintyValue = setupParams['uncertaintyValue']
            pos = np.sum(trainLabels == 1)  # 'pos' is the numPosTrainCases
            neg = np.sum(trainLabels == 0)  # 'neg' is the numNegTrainCases

            # Threshold for best balanced accuracy:
            maxBalancedAcc = np.max(balancedAcc)
            indsForMax = np.where(balancedAcc >= maxBalancedAcc - uncertaintyValue)[0]  # Take the
            #  first of these indices.
            left = min(indsForMax)
            right = max(indsForMax)
            indexForMax = int(np.round((pos*left + neg*right) / (pos + neg)))
            threshForMaxBalancedAcc = x[indexForMax]

            # Threshold for max raw accuracy:
            maxRawAcc = np.max(rawAccTrain)
            indsForMax = np.where(rawAccTrain >= maxRawAcc - uncertaintyValue)[0]  # Take the first
            # of these indices.
            left = min(indsForMax)
            right = max(indsForMax)
            indexForMax = int(np.round((pos*left + neg*right) / (pos + neg)))
            threshForMaxRawAcc = x[indexForMax]

        return np.array([threshForMaxBalancedAcc]), np.array([threshForMaxRawAcc])
    # End of chooseBestThresholdsSubFunction
    # ------------------------------------------------------

    def calculateSensSpecAtThresholdSubFunction(scores, labels, thresholds):
        ''' Calculate sens, spec, for either a vector of thresholds or a single value.
        Inputs:
            scores: np array vector of floats.
            labels: np array vector of 0s and 1s
            threshold: np array vector of floats.
        Outputs:
            sens: np array vector of float in [0, 1], or maybe  -100 as a flag.
            spec: np array vector of float in [0, 1].
            precision: np array vector of float in [0, 1].
            balancedAcc: np array vector of float in [0, 1].
            rawAcc: np array vector of float in [0, 1].
            '''
        precision = np.zeros(thresholds.shape)
        sens = np.zeros(thresholds.shape)
        spec = np.zeros(thresholds.shape)
        balancedAcc = np.zeros(thresholds.shape)
        rawAcc = np.zeros(thresholds.shape)
        for i in range(len(thresholds)):
            TP = np.sum(np.logical_and(scores[:, 1] >= thresholds[i], labels == 1))
            FN = np.sum(np.logical_and(scores[:, 1] < thresholds[i], labels == 1))
            FP = np.sum(np.logical_and(scores[:, 1] >= thresholds[i], labels == 0))
            TN = np.sum(np.logical_and(scores[:, 1] < thresholds[i], labels == 0))
            sens[i] = TP / (TP + FN)
            if TP + FN == 0:
                sens[i] = -100
            spec[i] = TN / (FP + TN)
            precision[i] = TP / (TP + FP)
            balancedAcc[i] = (sens[i] + spec[i]) / 2
            rawAcc[i] = (TP + TN) / len(labels)
        return sens, spec, precision, balancedAcc, rawAcc
    # End of calculateSensSpecAtThresholdSubFunction.
    # --------------------------------------------------------

    def plotRawAndBalancedAccuraciesPerRunSubFunction(modelTypeStr, trainRaw, testRaw, trainBal,
                                                      testBal, threshForMaxRawAcc,
                                                      threshForMaxBalancedAcc):
        ''' Plots train accuracy and test accuracy (raw and balanced) vs thresholds.
            Inputs:
                modelTypeStr: str.
                trainRaw: vector floats in [0,1].
                testRaw: vector floats in [0,1].
                trainBal: vector floats in [0,1].
                testBal: vector floats in [0,1].
                threshForMaxRawAcc: float.
                threshFormaxBalancedAcc: float.
            Outputs:
                matplotlib plot with 2 subplots.
                '''
        x = np.linspace(0, 1, 101)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, trainRaw*100, 'b')  # Training
        plt.plot(x, testRaw*100, 'r')
        plt.vlines(threshForMaxRawAcc, 0, 100, colors='k')
        plt.grid(b=True)
        plt.ylim(40, 100)
        plt.title(modelTypeStr + '. Raw accuracy.')

        plt.subplot(1, 2, 2)
        plt.plot(x, trainBal*100, 'b')
        plt.plot(x, testBal*100, 'r')
        plt.ylim(40, 100)
        plt.vlines(threshForMaxBalancedAcc, 0, 100, colors='k')
        plt.grid(b=True)
        plt.title('Balanced accuracy. b = train, r = test.')
        plt.show()
    # End of plotRawAndBalancedAccuraciesPerRunSubFunction
    #-------------------------------------------------------

    # Using train results, find best thresholds for use on test set:
    _, _, _, balancedAccTrain, rawAccTrain = \
    calculateSensSpecAtThresholdSubFunction(trainProbScores, trainLabels, np.linspace(0, 1, 101))
    threshForMaxBalancedAcc, threshForMaxRawAcc = \
    chooseBestThresholdsSubFunction(balancedAccTrain, rawAccTrain, trainLabels, setupParams)

    # Now apply these thresholds to the test set: 
    sensTest, specTest, precisionTest, balancedAccTest, rawAccTest = \
    calculateSensSpecAtThresholdSubFunction(testProbScores, testLabels, threshForMaxBalancedAcc)     
    # Get vars for AUC here for saving (it gets calc'ed again inside the plotResultsFunction()):
    fpr, tpr, _ = metrics.roc_curve(testLabels, testProbScores[:, 1], pos_label=1)

    # Collect the Test Set accuracies. These are single values (found using 'threshForMax*').
    accuracyStats = {'rawAccTest':rawAccTest, 'balancedAccTest':balancedAccTest,
                     'aucScore':auc(fpr, tpr), 'sensTest':sensTest, 'specTest':specTest,
                     'precisionTest':precisionTest}

    #---------------------

    # If wished, make plots of raw and balanced accuracies, for train and test sets. CAUTION: THIS
    #  MAKES LOTS OF PLOTS (numSetups x numRuns x 2).
    if setupParams['plotAccuraciesByThresholdsFlag']:
        # W need vectors of test accuracies:
        _, _, _, balancedAccTest, rawAccTest = \
        calculateSensSpecAtThresholdSubFunction(testProbScores, testLabels, np.linspace(0, 1, 101))
        modelTypeStr = modelType  + ', ' + setupParams['policyDomainsToKeep']
        plotRawAndBalancedAccuraciesPerRunSubFunction(modelTypeStr, rawAccTrain, rawAccTest,
                                                      balancedAccTrain, balancedAccTest,
                                                      threshForMaxRawAcc, threshForMaxBalancedAcc)

    # Plot ROC and bal acc-sens-spec, if flags indicate.
    # First make a dict to get params into plotting function:
    # We need a couple strings:
    futurePredictionStr = 'all years' * (setupParams['startYearForTest'] >= 2003) \
        + ('post-' + str(setupParams['startYearForTest'])) * (setupParams['startYearForTest'] < 2003)
    hiddenLayerStr = setupParams['hiddenLayers'] * (modelType == 'neuralNet')
    # These are parameters associated with training. Results on test set are saved elsewhere.
    params = \
    {'policyDomainsToKeep':setupParams['policyDomainsToKeep'], 'numPosTestCases':
     np.sum(testLabels == 1), 'numNegTestCases':np.sum(testLabels == 0), 'showAccuracyPlotsFlag':
     setupParams['showAccuracyPlotsFlag'], 'printToConsoleFlag':setupParams['printToConsoleFlag'],
     'futurePredictionStr':futurePredictionStr, 'threshForMaxBalancedAcc':threshForMaxBalancedAcc[0],
     'threshForMaxRawAcc':threshForMaxRawAcc[0], 'maxBalancedAccTrain':np.max(balancedAccTrain),
     'maxRawAccTrain':np.max(rawAccTrain), 'modelType':modelType, 'hiddenLayers':hiddenLayerStr,
     'startYearForTest':setupParams['startYearForTest']}

    # If we are going to make plots, we'll need as argin (to the plotResultsFunction) vectors of
    #  sens and spec on the test set:
    if setupParams['showAccuracyPlotsFlag']:
        sensTest, specTest, _, balAccTest, rawAccTest = \
        calculateSensSpecAtThresholdSubFunction(testProbScores, testLabels, np.linspace(0, 1, 101))
        plotResultsFunction(testProbScores, testLabels, balAccTest, sensTest,
                            specTest, params, fid)

    return accuracyStats

# calculateAccuracyStatsOnTestSetFunction.
#---------------------------------------------------------------------------

def calculatePrintAndSaveModelStatsOverAllRunsFunction(accResultDict, acceptTestResultsMatrix,
                                                       modelStr, setupParams, resultsDataFrame,
                                                       fid):
    '''
    Given the results for many runs using a given setup + domain, calculate accuracy statistics
    over all runs. Then (a) print them to console, (b) write them to a textfile, (c) save them into
    the results dataframe.

    Inputs:
        accResultDict: dict with entries as laid out below
        acceptTestResultsMatrix: boolean np.array, numRuns x numToLoop
        rawAcc: numRuns x numModels np.array. NOTE: if the setup combines all domains into a single
           model, numModels = 1. Else numModels = 6. This implicit fact is used to decide how to
           save data.
        balAcc: ditto
        AUC: ditto
        balPrec: ditto (not currently used).
        balRecall: ditto (not currently used).
        balF1: ditto (not currently used).
        modelStr: string (short descriptor of model).
        setupParams: dict (contains model + feature details).
        resultsDataFrame: pandas dataframe, the results dataframe we with to add results to.
        fid: file id, from open().

    Outputs:
        resultsDataframe: pandas dataframe, with new rows added.
    '''

    def printAccuracySubFunction(mu, med, sigma, tag):
        '''
        Make a string containing results.
        Inputs:
            mu: np vector (maybe with len 1).
            med: ditto.
            std: ditto.
            tag: str.
        Output:
            resultStr: str.
        '''
        tempStr = ''
        medStr = ''
        for i in range(len(mu)):
            tempStr = tempStr + str(mu[i]) + '+' + str(sigma[i])
            medStr = medStr + str(med[i])
            if i < len(mu) -1:
                tempStr = tempStr + ',      '
                medStr = medStr + ',       '
        tempStr = tempStr + '.'
        medStr = medStr + '.'
        resultStr = tag + ' (mean+std) = ' + tempStr

        print(resultStr)
        fid.write('\n' + resultStr)
    # End of printAccuracySubFunction
    # ------------------------------------------------

    def calculateAccuracyStatsSubFunction(acc, acceptTestResultsMatrix):
        ''' Calculate mu and std of various result types. This is done by column (in a loop)
        because sometimes a test set of a particular domain has 0 pos cases and must be ignored
        when calculating stats.
        Inputs:
            acc: np.array, numSetups x numRuns
            acceptTestResultsMatrix: np.array of booleans, numSetups x numRuns
        Outputs:
            means: vector of floats.
            medians: vector of floats.
            stds: vector of floats.
            '''
        numToLoop = acc.shape[1]
        means = np.zeros([1, numToLoop]).flatten()
        medians = np.zeros([1, numToLoop]).flatten()
        stds = np.zeros([1, numToLoop]).flatten()
        for i in range(numToLoop):
            a = acceptTestResultsMatrix[:, i]  # 'a' = acceptColumn, ie the column of accepts and
            # rejects for this policy domain.
            means[i] = np.round(100*np.mean(acc[a, i]), 1)  # one entry for each domain.
            medians[i] = np.round(100*np.median(acc[a, i]), 1)
            stds[i] = np.round(100*np.std(acc[a, i]), 1)
        return means, medians, stds

    # End of calculateAccuracyStatsSubFunction
    #------------------------------------------------

    # Construct and print each result string by looping over results vectors. Results for all models
    #  (ie for each policy domain) get printed into one line, separated by commas:
    print('\n' + modelStr + ' results (%): ')
    fid.write('\n' + modelStr + ' results (%): ')
    # Raw:
    rawAccTestMeans, rawAccTestMedians, rawAccTestStds = \
    calculateAccuracyStatsSubFunction(accResultDict['rawAcc'], acceptTestResultsMatrix)
    printAccuracySubFunction(rawAccTestMeans, rawAccTestMedians, rawAccTestStds, 'Raw acc')
    # Balanced:
    balAccTestMeans, balAccTestMedians, balAccTestStds = \
    calculateAccuracyStatsSubFunction(accResultDict['balAcc'], acceptTestResultsMatrix)
    printAccuracySubFunction(balAccTestMeans, balAccTestMedians, balAccTestStds, 'Balanced acc')
    # AUC:
    aucTestMeans, aucTestMedians, aucTestStds = \
    calculateAccuracyStatsSubFunction(accResultDict['AUC'], acceptTestResultsMatrix)
    printAccuracySubFunction(aucTestMeans, aucTestMedians, aucTestStds, 'AUC')
    # Skip dict keys ['prec'], ['recall'], ['f1'] since their stats are volatile.

    # Add rows to the results dataframe, either a single combined model or multiple models:
    sp = setupParams
    policyDomainList = sp['policyDomainList']
    rfMed = sp['useIndividualInterestGroupsFlag'] and sp['useRfChosenInterestGroupsFlag'] \
    and not sp['useShortInterestGroupListFlag'] and not sp['useAllInterestGroupsFlag']
    rfShort = sp['useIndividualInterestGroupsFlag'] and sp['useRfChosenInterestGroupsFlag'] \
    and sp['useShortInterestGroupListFlag'] and not sp['useAllInterestGroupsFlag']
    logMed = sp['useIndividualInterestGroupsFlag'] and not sp['useRfChosenInterestGroupsFlag'] \
    and not sp['useShortInterestGroupListFlag'] and not sp['useAllInterestGroupsFlag']
    logShort = sp['useIndividualInterestGroupsFlag'] and not sp['useRfChosenInterestGroupsFlag'] \
    and  sp['useShortInterestGroupListFlag'] and not sp['useAllInterestGroupsFlag']
    for i in range(len(rawAccTestMeans)): # Either 1 (one model) or 5 (one per policy domain).
        if not sp['oneModelPerPolicyDomainFlag']:
            policyDomain = 'All'
        else:
            policyDomain = policyDomainList[i]
        newRow = \
        pd.DataFrame({'modelType':[modelStr], 'balanced':[sp['useBalancedModelFlag']],
                      'policyDomain':[policyDomain], 'startYearForTest':[sp['startYearForTest']],
                      'use90':[sp['use90PercentFlag']], 'use50':[sp['use50PercentFlag']],
                      'use10':[sp['use10PercentFlag']],
                      'use90Minus10':[sp['use90Minus10PercentFlag']], 'useIGNA':[sp['useIGNAFlag']],
                      'useIndividualIGs': [sp['useIndividualInterestGroupsFlag']],
                      'useAllIGs':[sp['useAllInterestGroupsFlag']], 'useRfChosenMediumIGs':[rfMed],
                      'useRfChosenShortIGs':[rfShort], 'useLogChosenMediumIGs':[logMed],
                      'useLogChosenShortIGs': [logShort],
                      'usePolicyDomains':[sp['usePolicyDomainsFlag']],
                      'useOnly90Minus10Disagreements':[sp['useOnly90Minus10Disagreements']],
                      'rawAccTestMeans':[rawAccTestMeans[i]], 'rawAccTestMedians':
                      [rawAccTestMedians[i]], 'rawAccTestStds':[rawAccTestStds[i]],
                      'balAccTestMeans':[balAccTestMeans[i]], 'balAccTestMedians':
                      [balAccTestMedians[i]], 'balAccTestStds':[balAccTestStds[i]],
                      'aucTestMeans':[aucTestMeans[i]], 'aucTestMedians':[aucTestMedians[i]],
                      'aucTestStds':[aucTestStds[i]], 'numRuns':[sp['numRuns']],
                      'trainFraction':[sp['trainFraction']], 'disagreementThreshold':
                      [sp['disagreementThreshold']], 'maxDepth':[sp['maxDepth']],
                      'hiddenLayers':[sp['hiddenLayers']],
                      'IGSubsetUsed':[sp['combinedInterestGroupsToKeep']]})
        resultsDataFrame = resultsDataFrame.append(newRow, ignore_index=True, sort=False)

    return resultsDataFrame

#-----------------------------------------------------------------------------

def calculateDifferenceInAccDueToRfChosenVsLogChosenFunction(rawArray, balArray, aucArray,
                                                             modelType,
                                                             oneModelPerPolicyDomainFlag,
                                                             policyDomainList):
    ''' Calculates and prints (via a subfunction) the delta in accuracy from using RF-chosen
    features vs logistic-chosen features.
    Inputs:
        rawArray: numRuns x 4 array of raw accuracies. Each row is a train/test split (run). The
          columns MUST have the following setup order: 0 = rfShort, 1 = rfMedium, 2 = logShort,
          3 = logMedium').
        balArray: as above, but balanced accuracies.
        aucArray: as above, but AUCs
    Outputs:
        Prints mean (median) and std dev of RF-chosen minus logistic-chosen.
    '''

    def printDiffsInAccSubFunction(this, mediumOrShort, rawOrBal):

        mu = np.mean(this)
        sigma = np.std(this)
        med = np.median(this)
        mu = np.round(100*mu, 1)
        sigma = np.round(100*sigma, 1)
        med = np.round(100*med, 1)
        print(rawOrBal + ' ' + mediumOrShort + ': ' + str(mu) + '+' + str(sigma))

    # ----------------------------------------------------------

    # Print reminder warning:
    print('\nBelow are differences in accuracy RFChosen - logisticChosen. \n Format is: ' \
          + 'mean+std (median) \n CAUTION !!! These results require the following setup order ' \
          + '(four setups total): 0 = rfShort, 1 = rfMedium, 2 = logShort, 3 = logMedium.\n')

    print('Using ' + modelType + ':')
    # Get the right policy domain list:
    if not oneModelPerPolicyDomainFlag:
        domainList = ['All policy domains combined']
    else:
        domainList = policyDomainList

    for i in range(len(domainList)):
        # CAUTION !!! The second indices in each of these assignments and hard-coded and depend
        #  absolutely on the order of the setups, viz 0 = rfShort, 1 = rfMedium, 2 = logShort,
        #  3 = logMedium.
        diffRawShort = rawArray[:, 0, i] - rawArray[:, 2, i]
        diffRawMedium = rawArray[:, 1, i] - rawArray[:, 3, i]
        diffBalShort = balArray[:, 0, i] - balArray[:, 2, i]
        diffBalMedium = balArray[:, 1, i] - balArray[:, 3, i]
        diffAucShort = aucArray[:, 0, i] - aucArray[:, 2, i]
        diffAucMedium = aucArray[:, 1, i] - aucArray[:, 3, i]

        domain = domainList[i]
        print('\n' +  domain + ':')

        # Short IG lists:
        mediumOrShort = 'short'
        print('\n' + mediumOrShort + ': ')
        printDiffsInAccSubFunction(diffRawShort, mediumOrShort, 'Raw')
        printDiffsInAccSubFunction(diffBalShort, mediumOrShort, 'Balanced')
        printDiffsInAccSubFunction(diffAucShort, mediumOrShort, 'Auc')

        # Medium IG lists:
        mediumOrShort = 'medium'
        print('\n' + mediumOrShort + ': ')
        printDiffsInAccSubFunction(diffRawMedium, mediumOrShort, 'Raw')
        printDiffsInAccSubFunction(diffBalMedium, mediumOrShort, 'Balanced')
        printDiffsInAccSubFunction(diffAucMedium, mediumOrShort, 'Auc')

#--------------------------------------------------------------------------

""" END OF FUNCTION DEFS"""

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