9 August 2020

## predictingPolicyOutcomes

This repository contains code for the paper "Predicting United States Policy Outcomes with Random Forests" by Shawn K. McGuire and Charles B. Delahunt, found at arXiv.org.

The predictive models are described in the paper. The dataset (data/gilens_data_sm_copy.csv) is a subset of the dataset procured by Prof. Martin Gilens located at https://www.russellsage.org/economic-inequality-and-political-representation

Thank you for checking in! 
Questions or comments, please contact us at: smcguire@uw.edu, delahunt@uw.edu

## Usage

Python scripts are as follows. Each is ready to run:

1. acc_stats_calculator.py:
		Runs Random Forest data and outputs mean performance values for balanced accuracy and AUC (ROC).  Three User inputs:  model A-D, retrodiction or random draw, number of model runs. Reproduces Table 4 in the paper.

2. run_models_main.py:
		Runs RF, Logistic, and Neural Net models on a variety of feature sets. This script, like 'acc_stats_calculator' above, trains a model(s) and outputs accuracy statistics. However, it has more adjustable user inputs, which enable a wide range of experiments. Case domains and feature sets can be specified. Various outputs and plots can be selected, including feature importance rankings and RF vs Logistic comparisons. Reproduces content of Tables 3 and 5 to 9.

3. intgrp_delta_acc.py:
		Reproduces Table 2 in the paper (accuracy differences between model B and model A by interest group).
 
4. examples_RF_vs_logistic.py:
		Reproduces Figure 2 in the paper (two examples contrasting how RFs and Logistic regression parse the dataset).
  
5. policy_predictor.py:
		Contains support functions for 'acc_stats_calculator' and 'intgrp_delta_acc'. 

6. support_functions.py:
		Contains support functions for 'runModelsMain'.   
 
## MIT License
